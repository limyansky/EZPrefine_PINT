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
from pint.eventstats import hmw, h2sig
from pint.sampler import EmceeSampler
from pint.mcmc_fitter import MCMCFitter, lnlikelihood_chi2
from pint.residuals import Residuals
from pint.plot_utils import phaseogram_binned

# astropy imports
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from copy import copy, deepcopy

# Plotting tools
import matplotlib.pylab as plt

from math import floor


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
    parser.add_argument('--minWeight', nargs='?', default=0.0, type=float)
    parser.add_argument('--nwalkers', nargs='?', default=16, type=int)
    parser.add_argument('--nsteps', nargs='?', default=250, type=int)
    parser.add_argument('--minMJD', nargs='?', default=None, type=float)
    parser.add_argument('--maxMJD', nargs='?', default=None, type=float)
    parser.add_argument('--skip', default=False,
                        help='If True, don\'t optimize.')
    parser.add_argument('--ephem', default=None)
    parser.add_argument('--save_pick', default=False)
    parser.add_argument('--load_pick', default=False)

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

    if not args.skip:
        MCMC_obj.run_MCMC()

    # Print the output of the MCMC
        MCMC_obj.MCMC_output()

        MCMC_obj.h_test(MCMC_obj.fitter.model)

    # Return 0 to show that everything worked okay
    return MCMC_obj


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

        if self.args.load_pick is False:
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

        # If load_pick is not False, the user has specified a file
        # Try loading it
        elif self.args.load_pick is not False:
            # load the model
            self.modelin = pint.models.get_model(self.args.par)

            # Read the saved pickle file
            self.toas = toa.TOAs(self.args.load_pick + '.pickle')

            # Create a toas_list from the loaded file
            #self.toas_list = toa.get_TOAs_list([self.toas.toas[:]])

            # This is a bit janky. TOA objects do not include a way to store
            # photon weights. Thus, I am forced to save the weights as a
            # separate file. The other option would be to... Somehow pickle it
            # all together, load that output file, then split it up again in
            # here. However, I don't want to figure that out right now. I also
            # don't want to end up with a propritary file type.

            # Thus, I simply append 'weights_' and move on.

            # self.weights = np.load('weights_' + self.args.load_pick + '.npy')

        if self.args.save_pick is not False:
            self.toas.pickle(filename=self.args.save_pick + '.pickle')
            # np.save('weights_' + self.args.save_pick + '.npy' , self.weights)

        self.mid_save()

        # Check for minMJD
        if self.args.minMJD is not None:
            self.args.minMJD = self.args.minMJD * u.day
            self.cut_minMJD()

        # Check for maxMJD
        if self.args.maxMJD is not None:
            self.args.maxMJD = self.args.maxMJD * u.day
            self.cut_maxMJD()

        # setup the MCMC to run
        self.init_MCMC()

    # Combine TOAs, TOAs_list, and weights accross different instruments
    def make_TOAs(self):

        # Get the toa object
        self.toas = toa.TOAs(toalist=self.data_fermi + self.data_NICER +
                             self.data_NuSTAR)

        # Get the toa list
        #self.toas_list = toa.get_TOAs_list(self.data_fermi + self.data_NICER +
        #                                   self.data_NuSTAR, ephem=self.args.ephem)

        # Combine weights from individual observatories
        weights = np.concatenate((self.weights_fermi, self.weights_NICER,
                                  self.weights_NuSTAR))

        # add the weights to the TOAs object
        for ii in range(len(weights)):
            self.toas.table['flags'][ii]['weights'] = weights[ii]

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
                                          targetcoord=self.t_coord,
                                          minweight=self.args.minWeight)

        # Get the weights
        self.weights_fermi = np.array([w.flags["weight"]
                                 for w in self.data_fermi])

        print('\n')
        print('%d photons from Fermi' % (len(self.weights_fermi)))
        print('%f is the minimum weight' % (min(self.weights_fermi)))
        print('\n')

    # Store quantities related to NICER data
    def read_NICER(self, ft1_file):
        # Read in model
        self.modelin = pint.models.get_model(self.args.par)

        # Extract target coordinates
        self.t_coord = SkyCoord(self.modelin.RAJ.quantity,
                                self.modelin.DECJ.quantity,
                                frame="icrs")

        # Read in NICER data
        self.data_NICER = load_NICER_TOAs(ft1_file)

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

        self.weights_NuSTAR = np.ones(len(self.data_NuSTAR))

    # Add errors for use in minimizing the residuals
    # I imagine taking this out at some point, as I would eventually like to
    # minimize on the H-Test
    def add_errors(self):

        # Introduce a small error so that residuals can be calculated
        self.toas.table["error"] = 1.0
        #self.toas.filename = self.args.ft1
        self.toas.compute_TDBs(ephem=self.args.ephem)
        self.toas.compute_posvels(ephem=self.args.ephem, planets=False)

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
        iphss, phss = model.phase(self.toas)  # , abs_phase=True)

        # Ensure all postive
        phases = np.where(phss < 0.0, phss + 1.0, phss)

        # phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the first H-Test
        htest = hmw(phases, np.array(self.toas.get_flag_value('weights')))

        print(htest)

        return htest

    # Compute the H-Test for a given set of parameters
    # This is what the MCMC maximizes
    def MCMC_htest(self, fitter, params):

        # Update the fitter with the test parameters
        fitter.set_parameters(params)

        # Calcuate phases
        iphss, phss = fitter.model.phase(self.toas)

        # Ensure all postive
        phases = np.where(phss < 0.0, phss + 1.0, phss)

        # Legacy
        # phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the H-Test
        htest = hmw(phases, np.array(self.toas.get_flag_value('weights')))

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

    def mid_save(self):
        # self.all_weights = copy(self.weights)
        self.all_toas     = copy(self.toas)

    # Select data according to the minimum MJD
    def cut_minMJD(self):

        # Create the filter
        selection_filter = self.toas.get_mjds() > self.args.minMJD

        # Apply the filter to weights and the toas_list
        self.toas.select(selection_filter)
        # self.weights = self.weights[selection_filter]

    # Select data according to the maximum MJD
    def cut_maxMJD(self):

        #  Create the filter
        selection_filter = self.toas.get_mjds() < self.args.maxMJD

        # Apply the filter
        self.toas.select(selection_filter)
        # self.weights = self.weights[selection_filter]

    # Restore all the photons to the dataset
    def cut_restore(self):

        self.toas = deepcopy(self.all_toas)
        # self.weights = copy(self.all_weights)

    # Easily make new photon cuts
    def update_cut(self, minMJD=None, maxMJD=None):

        # Restore all photons
        self.cut_restore()

        # Check for minMJD
        if minMJD is not None:
            self.args.minMJD = minMJD * u.day
            self.cut_minMJD()

        # Check for maxMJD
        if maxMJD is not None:
            self.args.maxMJD = maxMJD * u.day
            self.cut_maxMJD()

    # Reset the MCMC object (with a new model), and get it ready to run again
    def update_run(self, minMJD=None, maxMJD=None):

        if minMJD is not None or maxMJD is not None:
            # Return all the photons to the original dataset
            self.update_cut(minMJD, maxMJD)

        # Make the new model the old best fit model
        self.modelin = deepcopy(self.fitter.model)

    # Plot the data in a phaseogram
    def plot(self, bins=100):

        iphss, phss = self.modelin.phase(self.toas)  # , abs_phase=True)

        # Ensure all postive
        phases = np.where(phss < 0.0, phss + 1.0, phss)

        # legacy
        # phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the first H-Test
        htest = hmw(phases, np.array(self.toas.get_flag_value('weights')))

        print(htest)

        phaseogram(self.toas.get_mjds(), phases,
                   weights=np.array(self.toas.get_flag_value('weights')),
                   bins=bins)

    # Plot the data in a binned phaseogram
    def plot_binned(self):

        iphss, phss = self.modelin.phase(self.toas)  # , abs_phase=True)

        # Ensure all postive
        phases = np.where(phss < 0.0, phss + 1.0, phss)
        # Legacy
        # phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)

        # Pull out the first H-Test
        htest = hmw(phases, np.array(self.toas.get_flag_value('weights')))

        print(htest)

        phaseogram_binned(self.toas.get_mjds(),
                          phases, weights=np.array(self.toas.get_flag_value('weights')))
        return 0

    # Plot the weighted H-Test vs time
    def plot_hmw(self):

        # Calculate the phases
        iphss, phss = self.modelin.phase(self.toas)

        # Place to store the H-Test
        h_vec = []
        mjds = []

        for ii in range(0, len(phss), int(floor(len(phss) / 50))):
            h_vec.append(hmw(phss[0:ii],
                             np.array(self.toas.get_flag_value('weights')[0:ii])))
            mjds.append(self.toas.get_mjds()[ii].value)

        plt.plot(mjds, h_vec)
        plt.show()

        # Plot the weighted H-Test vs photon count
    def plot_Phmw(self):

        # Calculate the phases
        iphss, phss = self.modelin.phase(self.toas)

        # Place to store the H-Test
        h_vec = []
        photons = []

        for ii in range(0, len(phss), int(floor(len(phss) / 50))):
            h_vec.append(hmw(phss[0:ii],
                             np.array(self.toas.get_flag_value('weights')[0:ii])))
            photons.append(len(phss[0:ii]))

        plt.plot(photons, h_vec)
        plt.show()

    # Backup the timing model.
    # Store the current model in a value called restore_model
    def backup_timing(self):
        self.restore_model = deepcopy(self.modelin)

    # Take the backed up model (restore_model), and put it back in the working
    # model spot (modelin)
    def restore_timing(self):
        self.modelin = deepcopy(self.restore_model)

    # Manually change F0
    def change_F0(self, new_value):
        self.modelin.F0.quantity = new_value * u.Hz

    # Manually change F1
    def change_F1(self, new_value):
        self.modelin.F1.quantity = new_value * u.Hz / u.s

    # Manually change F2
    def change_F2(self, new_value):
        self.modelin.F2.quantity = new_value * u.Hz / u.s ** 2

    # Manually change EPOCH
    def change_epoch(self, new_value):
        self.modelin.PEPOCH.quantity = str(new_value)
        self.modelin.TZRMJD.quantity = str(new_value)

    # Scan over a range of F2 values
    def scan_F2(self, par_model, values):
        model = deepcopy(par_model)
        significance = []
        # F1_range = np.arange(start, stop, step)
        F2_range = values

        for ii in F2_range:

            # Change F2
            model.F2.quantity = ii * u.Hz / u.s ** 2

            # Calculate the phases
            iphss, phss = model.phase(self.toas)

            # Should I ensure all phases are positive? I don't think so...

            # Calculate the significance, and append it to the list
            significance.append(hmw(phss, np.array(self.toas.get_flag_value('weights'))))

        # plt.plot(F1_range, significance)
        # plt.show()

        return significance

    # Scan over a range of F1 values
    # def scan_F1(self, par_model, start, stop, step):
    def scan_F1(self, par_model, values):
        model = deepcopy(par_model)
        significance = []
        # F1_range = np.arange(start, stop, step)
        F1_range = values

        for ii in F1_range:

            # Change F1
            model.F1.quantity = ii * u.Hz / u.s

            # Calculate the phases
            iphss, phss = model.phase(self.toas)

            # Should I ensure all phases are positive? I don't think so...

            # Calculate the significance, and append it to the list
            significance.append(hmw(phss, np.array(self.toas.get_flag_value('weights'))))

        # plt.plot(F1_range, significance)
        # plt.show()

        return significance

        # Scan over a range of F0 values
    # def scan_F0(self, par_model, start, stop, step):
    def scan_F0(self, par_model, values):
        model = deepcopy(par_model)
        significance = []
        # F0_range = np.arange(start, stop, step)
        F0_range = values

        for ii in F0_range:

            # Change F0
            model.F0.quantity = ii * u.Hz

            # Calculate the phases
            iphss, phss = model.phase(self.toas)

            # Should I ensure all phases are positive? I don't think so...

            # Calculate the significance, and append it to the list
            significance.append(hmw(phss,
                                    np.array(self.toas.get_flag_value('weights'))))

        # plt.plot(F1_range, significance)
        # plt.show()

        return significance

    # Scan through a combination of F0 and F1 values
    def scan_F0_F1(self, par_model, F0_values, F1_values):

        # Make a working copy of the provided model
        model = deepcopy(par_model)

        # Initialize an array of the correct length
        sig_array = np.zeros([1, len(F1_values)])

        # Step through each F0 value
        for ii in F0_values:

            # Set the F0 value
            model.F0.quantity = ii * u.Hz

            # Use an existing method to scan F1
            F1_single = self.scan_F1(model, F1_values)

            # Store the output of the F1 scan
            sig_array = np.append(sig_array, [F1_single], axis=0)

        return np.delete(sig_array, 0, 0)

    # Scan through a combination of F0 and F1 values
    def scan_F1_F2(self, par_model, F1_values, F2_values):

        # Make a working copy of the provided model
        model = deepcopy(par_model)

        # Initialize an array of the correct length
        sig_array = np.zeros([1, len(F2_values)])

        # Step through each F0 value
        for ii in F1_values:

            # Set the F0 value
            model.F1.quantity = ii * u.Hz / u.s

            # Use an existing method to scan F1
            F2_single = self.scan_F2(model, F2_values)

            # Store the output of the F1 scan
            sig_array = np.append(sig_array, [F2_single], axis=0)

        return np.delete(sig_array, 0, 0)

    # Return the highest density time period
    # This is ripped straight from PINT. For some reason, the most recent pip
    # version doesn't include the .value for maxday, which is breaking
    # everything.
    def density(self, ndays=7):

        """print the range of mjds (default 7 days) with the most toas"""
        # TODO: implement sliding window
        nbins = int((max(self.toas.get_mjds()) - min(self.toas.get_mjds())) / (ndays * u.d))
        a = np.histogram(self.toas.get_mjds(), nbins)
        maxday = int((a[1][np.argmax(a[0])]).value)
        diff = int((a[1][1] - a[1][0]).value)
        print(
            "max density range (in steps of {} days -- {} bins) is from MJD {} to {} with {} toas.".format(
                diff, nbins, maxday, maxday + diff, a[0].max()
            )
        )
        return (maxday, maxday + diff)

def plot_scan(array, F0_values, F1_values):

    plt.imshow(array, aspect='auto',
               extent=[min(F1_values), max(F1_values),
                       max(F0_values), min(F0_values)])


# Stolen from PINT
def phaseogram(
    mjds,
    phases,
    weights=None,
    title=None,
    bins=100,
    rotate=0.0,
    size=5,
    alpha=0.25,
    width=6,
    maxphs=2.0,
    plotfile=None,
):
    """Make a nice 2-panel phaseogram"""
    years = (mjds.value - 51544.0) / 365.25 + 2000.0
    phss = phases + rotate
    phss[phss > 1.0] -= 1.0
    plt.figure(figsize=(width, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    wgts = None if weights is None else np.concatenate((weights, weights))
    h, x, p = ax1.hist(
        np.concatenate((phss, phss + 1.0)),
        int(maxphs * bins),
        range=[0, maxphs],
        weights=wgts,
        color="k",
        histtype="step",
        fill=False,
        lw=2,
    )
    ax1.set_xlim([0.0, maxphs])  # show 1 or more pulses

    ### I CHANGE THIS TO ZERO SUPRESS THE TOP PLOT ###
    ax1.set_ylim([0.9 * h.min(), 1.1 * h.max()])

    if weights is not None:
        ax1.set_ylabel("Weighted Counts")
    else:
        ax1.set_ylabel("Counts")
    if title is not None:
        ax1.set_title(title)
    if weights is None:
        ax2.scatter(phss, mjds, s=size, color="k", alpha=alpha)
        ax2.scatter(phss + 1.0, mjds, s=size, color="k", alpha=alpha)
    else:
        colarray = np.array([[0.0, 0.0, 0.0, w] for w in weights])
        ax2.scatter(phss, mjds, s=size, color=colarray)
        ax2.scatter(phss + 1.0, mjds, s=size, color=colarray)
    ax2.set_xlim([0.0, maxphs])  # show 1 or more pulses
    ax2.set_ylim([mjds.min().value, mjds.max().value])
    ax2.set_ylabel("MJD")
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2r = ax2.twinx()
    ax2r.set_ylim([years.min(), years.max()])
    ax2r.set_ylabel("Year")
    ax2r.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2r.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.set_xlabel("Pulse Phase")
    plt.tight_layout()
    if plotfile is not None:
        plt.savefig(plotfile)
        plt.close()
    else:
        plt.show()



# If called from the commandline, run this script
if __name__ == '__main__':
    MCMC_object = main()
