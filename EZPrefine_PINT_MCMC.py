# Refinging Fermi data similarly to Mario's EZPrefine, but using PINT.

# numpy
import numpy as np

# Handles commandline inputs
import argparse

# PINT imports
import pint.models
from pint.fermi_toas import load_Fermi_TOAs
from pint.event_toas import load_NICER_TOAs
from pint.event_toas import load_NuSTAR_TOAs
import pint.toa as toa
from pint.eventstats import hmw
from pint.sampler import EmceeSampler
from pint.mcmc_fitter import MCMCFitter, lnlikelihood_chi2
from pint.residuals import Residuals

# astropy imports
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits


# The funciton that actually runs when this script is called
def main():
    """
    Runs a montecarlo optimization on fermi data.

    Maybe some more information about the script.

    Parameters:
        param_1: The first parameter
        param_2: The second parameter

    Keyword Arguments:
        opt_1: The first optional parameter
        opt_2: The second optional parameter
    """

    # Add the docstring to the argument parser
    # This doesn't currently work the way I want it to
    # Try print(__doc__) to see.
    parser = argparse.ArgumentParser(description=__doc__)

    # Add arguments
    parser.add_argument('ft1', help='A barycentered event file or files',
                        nargs='+')
    parser.add_argument('par', help='A .par file')
    parser.add_argument('weightcol',
                        help='The name of the column containing photon \
                              weights')

    # Optional arguments

    # minWeights not Implimented
    # parser.add_argument('--minWeight', nargs='?', default=0.9, type=float)
    parser.add_argument('--nwalkers', nargs='?', default=16, type=int)
    parser.add_argument('--nsteps', nargs='?', default=250, type=int)

    # phs and nbins are related to a gaussian profile
    # parser.add_argument('--nbins', nargs='?', default=256, type=int)
    # parser.add_argument('--phs', nargs='?', default=0.0, type=float)

    # Extract the arguments from the parser
    args = parser.parse_args()

    # Initialize the MCMC class (defined below)
    MCMC_obj = MCMC(args)

    # Print the H-Test
    MCMC_obj.h_test(MCMC_obj.modelin)

    # Run the MCMC
    MCMC_obj.run_MCMC()

    # Print the output of the MCMC
    MCMC_obj.MCMC_output()

    MCMC_obj.h_test(MCMC_obj.fitter.model)

    # Return 0 to show that everything worked okay
    return 0


# I create a class that will be worked with in main()
class MCMC:

    # Initializes model to useful form
    def __init__(self, args):

        # Store the input arguments
        self.args = args

        # Initialize weights and TOAs data
        # These need to be initalized to empty so that make_toas doesn't
        # throw an error of one of these datasets is not provided, and it tries
        # to concatinate an unset variable with a list.
        self.weights_fermi   = []
        self.weights_NICER   = []
        self.weights_NuSTAR  = []
        self.data_fermi      = []
        self.data_NICER      = []
        self.data_NuSTAR     = []

        # If only a string is given, turn it into a list
        if type(self.args.ft1) is str:
            self.args.ft1 = [self.args.ft1]

        # If only a string is given, turn it into a list
        #if type(self.args.weightcol) is str:
            #self.args.weightcol = [self.args.weightcol]

        # Determine if multiple observations should be merged
        for ii in self.args.ft1:

            # Get the name of the telescope
            tele = self.check_tele(ii)

            # Use the appropriate function to load data

            # If the telescope is Fermi (formally GLAST)...
            # Note: should there be an additional check that the data comes
            # from the LAT instrument?
            if tele == 'GLAST':

                # Load the fermi data
                self.read_fermi(ii, self.args.weightcol)

            # If the telescope is NICER...
            elif tele == 'NICER':

                # Load the NICER data
                self.read_NICER(ii)

            # If the telescope is NuSTAR
            if tele == 'NuSTAR':

                # Load the NuSTAR data
                self.read_NuSTAR(ii)

        # Perform the merging
        self.make_TOAs()

        # Add errors to the data (for residual minimization)
        self.add_errors()

        # setup the MCMC to run
        self.init_MCMC()

    # Combine TOAs, TOAs_list, and weights accross different instruments
    def make_TOAs(self):

        # Get the toa object
        self.toas = toa.TOAs(toalist=self.data_fermi + self.data_NICER +
                             self.data_NuSTAR)

        # Get the toa list
        self.toas_list = toa.get_TOAs_list(self.data_fermi + self.data_NICER +
                                           self.data_NuSTAR)

        # Combine weights from individual observatories
        self.weights = np.concatenate((self.weights_fermi, self.weights_NICER,
                                       self.weights_NuSTAR))

    # Check which telescope is in the file header
    def check_tele(self, data):

        # Open the fits file
        fits_file = fits.open(data)

        # Extract the telescope name
        name = fits_file[0].header['TELESCOP']

        # Close the fits file
        fits_file.close()

        # Return the name of the telescope
        return name

    # Store quantities related to Fermi data
    def read_fermi(self, ft1_file, weight_name):

        # Read in model
        self.modelin = pint.models.get_model(self.args.par)

        # Extract target coordinates
        self.t_coord = SkyCoord(self.modelin.RAJ.quantity,
                                self.modelin.DECJ.quantity,
                                frame="icrs")

        # Read in Fermi data
        self.data_fermi = load_Fermi_TOAs(ft1_file, weight_name,
                                          targetcoord=self.t_coord)

        # Convert fermi data to TOAs object
        # I don't understand this. Does load_Fermi_TOAs not already load TOAs?
        # Maybe it loads photon times, then converts to TOA object?
        self.toas_list_fermi = toa.get_TOAs_list(self.data_fermi)
        #self.toas_fermi = toa.TOAs(toalist=self.data)

        # Get the weights
        self.weights_fermi = np.array([w["weight"]
                                 for w in self.toas_list_fermi.table["flags"]])

    # Store quantities related to NICER data
    def read_NICER(self, ft1_file):
        # Read in model
        self.modelin = pint.models.get_model(self.args.par)

        # Extract target coordinates
        self.t_coord = SkyCoord(self.modelin.RAJ.quantity,
                                self.modelin.DECJ.quantity,
                                frame="icrs")

        # Read in Fermi data
        self.data_NICER = load_NICER_TOAs(ft1_file)

        # Convert fermi data to TOAs object
        # I don't understand this. Does load_Fermi_TOAs not already load TOAs?
        # Maybe it loads photon times, then converts to TOA object?
        #self.toas_list_NICER = toa.get_TOAs_list(self.data)
        #self.toas_NICER = toa.TOAs(toalist=self.data)

        self.weights_NICER = np.ones(len(self.data_NICER))

    # Store quantities related to NICER data
    def read_NuSTAR(self, ft1_file):
        # Read in model
        self.modelin = pint.models.get_model(self.args.par)

        # Extract target coordinates
        self.t_coord = SkyCoord(self.modelin.RAJ.quantity,
                                self.modelin.DECJ.quantity,
                                frame="icrs")

        # Read in Fermi data
        self.data_NuSTAR = load_NuSTAR_TOAs(ft1_file)

        # Convert fermi data to TOAs object
        # I don't understand this. Does load_Fermi_TOAs not already load TOAs?
        # Maybe it loads photon times, then converts to TOA object?
        #self.toas_list_NICER = toa.get_TOAs_list(self.data)
        #self.toas_NICER = toa.TOAs(toalist=self.data)

        self.weights_NuSTAR = np.ones(len(self.data_NuSTAR))


    # Add errors for use in minimizing the residuals
    # I imagine taking this out at some point, as I would eventually like to
    # minimize on the H-Test
    def add_errors(self):

        # Introduce a small error so that residuals can be calculated
        self.toas.table["error"] = 1.0
        self.toas.filename = self.args.ft1
        self.toas.compute_TDBs()
        self.toas.compute_posvels(ephem="DE421", planets=False)

    # Initialize the MCMC fitter
    def init_MCMC(self):

        # Make a random seed

        # np.random.seed(0)
        self.state = np.random.mtrand.RandomState()

        # Initialize the sampler
        self.sampler = EmceeSampler(self.args.nwalkers)

        # Initialie PINT's MCMC object
        self.fitter = MCMCFitter(self.toas, self.modelin, self.sampler,
                                 lnlike=self.MCMC_htest)
        # self.fitter = MCMCFitter(self.toas, self.modelin, self.sampler,
        #                          weights=self.weights,
        #                          lnlike=lnlikelihood_chi2)
        self.fitter.sampler.random_state = self.state

    # Returns the H-Test
    def h_test(self, model):

        # Compute model phase for each TOA
        # I believe absolute phase is just a phase offset used to
        # align data from multiple time perods or instruments.
        iphss, phss = model.phase(self.toas_list)  # , abs_phase=True)

        # Ensure all postive
        phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the first H-Test
        htest = hmw(phases, self.weights)

        print(htest)

        return htest

    # Compute the H-Test for a given set of parameters
    # This is what the MCMC maximizes
    def MCMC_htest(self, fitter, params):

        # Update the fitter with the test parameters
        fitter.set_parameters(params)

        # Calcuate phases
        iphss, phss = fitter.model.phase(self.toas_list)

        # Ensure all postive
        phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the H-Test
        htest = hmw(phases, self.weights)

        #print(params)
        #print(htest)

        return htest

    # Run the MCMC
    def run_MCMC(self):
        print()
        print('In the MCMC')
        print()
        self.fitter.fit_toas(maxiter=self.args.nsteps, pos=None)
        self.fitter.set_parameters(self.fitter.maxpost_fitvals)

    # Print MCMC output
    def MCMC_output(self):

        print()
        print(self.fitter.model)
        print()

        # This prints the 16th, 50th, and 84th percentile ranges of the fit.
        samples2 = self.sampler.sampler.chain[:, :, :].reshape((-1, self.fitter.n_fit_params))
        ranges2 = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples2, [16, 50, 84], axis=0)),
        )

        for name, vals in zip(self.fitter.fitkeys, ranges2):
            print("%8s:" % name + "%25.15g (+ %12.5g  / - %12.5g)" % vals)


# If called from the commandline, run this script
if __name__ == '__main__':
    main()
