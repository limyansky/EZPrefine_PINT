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
from pint.mcmc_fitter import MCMCFitter
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
    parser.add_argument('--minWeight', nargs='?', const=0.9)
    parser.add_argument('--nwalkers', nargs='?', const=10)
    parser.add_argument('--nsteps', nargs='?', const=50)
    parser.add_argument('--nbins', nargs='?', const=256)
    parser.add_argument('--phs', nargs='?', const=0.0)

    # Extract the arguments from the parser
    args = parser.parse_args()

    # Initialize the MCMC class (defined below)
    MCMC_obj = MCMC(args)

    # Print the H-Test
    MCMC_obj.h_test()

    # Return 0 to show that everything worked okay
    return 0


# I create a class that will be worked with in main()
class MCMC:

    # Initializes model to useful form
    def __init__(self, args):

        # Store the input arguments
        self.args = args

        # Create a known random seed for the MCMC
        self.make_seed()

        # Load the fermi data
        self.read_fermi()

        # Add errors to the fermi data (for residual minimization)
        self.add_errors()

    # Create a random seed
    def make_seed(self, seed=0):

        np.random.seed(0)
        self.state = np.random.mtrand.RandomState()

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

        # Initialize the sampler
        self.sampler = EmceeSampler(self.nwalkers)

        # Initialie PINT's MCMC object
        self.fitter = MCMCFitter(self.toas, self.modelin, self.sampler)
        self.fitter.sampler.random_state = self.state

    # Returns the H-Test
    def h_test(self):

        # Compute model phase for each TOA
        # I believe absolute phase is just a phase offset used to
        # align data from multiple time perods or instruments.
        iphss, phss = self.modelin.phase(self.toas_list)  # , abs_phase=True)

        # Ensure all postive
        phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the first H-Test
        htest = hmw(phases, self.weights)

        print(htest)

        return htest


# Probability fuctions for the MCMC fitter
def lnlikelihood_chi2(ftr, theta):
    ftr.set_parameters(theta)
    # Uncomment to view progress
    # print('Count is: %d' % ftr.numcalls)
    return -Residuals(toas=ftr.toas, model=ftr.model).chi2.value


# If called from the commandline, run this script
if __name__ == '__main__':
    main()
