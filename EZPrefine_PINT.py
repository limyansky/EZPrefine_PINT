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

    # Extract the arguments from the parser
    args = parser.parse_args()

    # Read in model
    modelin = pint.models.get_model(args.par)

    # Extract target coordinates
    t_coord = SkyCoord(modelin.RAJ.quantity, modelin.DECJ.quantity,
                       frame="icrs")

    # Read in Fermi data
    fermi_data = load_Fermi_TOAs(args.ft1, args.weightcol, targetcoord=t_coord)

    # Convert fermi data to TOAs object
    # I don't understand this. Does load_Fermi_TOAs not already load TOAs?
    # Maybe it loads photon times, then converts to TOA object?
    fermi_toas = toa.get_TOAs_list(fermi_data)

    # Get weights
    weights = np.array([w["weight"] for w in fermi_toas.table["flags"]])

    # Compute model phase for each TOA
    # I believe absolute phase is just a phase offset used to align data from
    # multiple time perods or instruments.
    iphss, phss = modelin.phase(fermi_toas)  # , abs_phase=True)

    # Ensure all postive
    phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

    # Pull out the first H-Test
    htest = hmw(phases, weights)

    print(htest)

    # Return 0 to show that everything worked okay
    return 0


# If called from the commandline, run this script
if __name__ == '__main__':
    main()
