# Refinging Fermi data similarly to Mario's EZPrefine, but using PINT.

# numpy
import numpy as np

# Handles commandline inputs
import argparse

# PINT imports
import pint.models
from pint.fermi_toas import load_Fermi_TOAs
import pint.toa as toa
from pint.eventstats import hmw
from pint.sampler import EmceeSampler
from pint.mcmc_fitter import MCMCFitter, lnlikelihood_chi2
from pint.residuals import Residuals

# astropy imports
from astropy.coordinates import SkyCoord
import astropy.units as u


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
    parser.add_argument('ft1', help='A barycentered fermi event file')
    parser.add_argument('par', help='A .par file')
    parser.add_argument('weightcol',
                        help='The name of the column containing photon \
                              weights')

    # Optional arguments
    parser.add_argument('--minWeight', nargs='?', default=0.9)
    parser.add_argument('--nwalkers', nargs='?', default=16)
    parser.add_argument('--nsteps', nargs='?', default=50)
    parser.add_argument('--nbins', nargs='?', default=256)
    parser.add_argument('--phs', nargs='?', default=0.0)

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

        # Load the fermi data
        self.read_fermi()

        # Add errors to the fermi data (for residual minimization)
        self.add_errors()

        # setup the MCMC to run
        self.init_MCMC()

    # Store quantities related to Fermi data
    def read_fermi(self):

        # Read in model
        self.modelin = pint.models.get_model(self.args.par)

        # Extract target coordinates
        self.t_coord = SkyCoord(self.modelin.RAJ.quantity,
                                self.modelin.DECJ.quantity,
                                frame="icrs")

        # Read in Fermi data
        self.data = load_Fermi_TOAs(self.args.ft1, self.args.weightcol,
                                    targetcoord=self.t_coord)

        # Convert fermi data to TOAs object
        # I don't understand this. Does load_Fermi_TOAs not already load TOAs?
        # Maybe it loads photon times, then converts to TOA object?
        self.toas_list = toa.get_TOAs_list(self.data)
        self.toas = toa.TOAs(toalist=self.data)

        # Get the weights
        self.weights = np.array([w["weight"]
                                 for w in self.toas_list.table["flags"]])

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

        np.random.seed(0)
        self.state = np.random.mtrand.RandomState()

        # Initialize the sampler
        self.sampler = EmceeSampler(self.args.nwalkers)

        # Initialie PINT's MCMC object
        self.fitter = MCMCFitter(self.toas, self.modelin, self.sampler,
                                lnlike=self.MCMC_htest, weights=self.weights)
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

        return htest

    # Run the MCMC
    def run_MCMC(self):
        print()
        print('In the MCMC')
        print()
        self.fitter.fit_toas(maxiter=self.args.nsteps)
        self.fitter.set_parameters(self.fitter.maxpost_fitvals)

    # Print MCMC output
    def MCMC_output(self):

        print()
        print(self.fitter.model)
        print()

        # samples2 = self.sampler.sampler.chain[:, :, :].reshape((-1, self.fitter.n_fit_params))
        # ranges2 = map(
        #     lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        #     zip(*np.percentile(samples2, [16, 50, 84], axis=0)),
        # )

        # for name, vals in zip(self.fitter.fitkeys, ranges2):
        #     print("%8s:" % name + "%25.15g (+ %12.5g  / - %12.5g)" % vals)


# If called from the commandline, run this script
if __name__ == '__main__':
    main()
