# Refinging Fermi data similarly to Mario's EZPrefine, but using PINT.

# Handles commandline inputs
import argparse


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
    parser = argparse.ArgumentParser(description=__doc__)

    # Extract the arguments from the parser
    args = parser.parse_args()

    # Return 0 to show that everything worked okay
    return 0


# If called from the commandline, run this script
if __name__ == '__main__':
    main()
