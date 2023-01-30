# Experiment setup for the ICEBEAR radar
# Should be executable in python 2 & 3

"""
Capabilities:

Choose hdf5 file location, select start time, plot time seperation, time avg, ...

Uses functions from icebearModules.py to process and anlayze data collected by ICEBEAR. Based
on the initial script wirtten by Deving Hyugehbeart

"""
#################################################################################
#                               USING THIS FILE                                 #
#   Dependencies: AspectMapper, igrf, ssmf.cu, icebearModules, vizualization    #
#                                                                               #
#       To run fist enter environment using the following code:                 #
#                                                                               #
#       export PATH=/home/icebear-cuda/anaconda2/bin:$PATH                      #
#       source activate carto3env                                               #
#                                                                               #
#       Execute using: python3 noise_gen.py -i canada_gh.ini -y 2015              #
#                                                                               #
#################################################################################

# Import Modules
import matplotlib as mpl
#mpl.use('Agg') # Turn off graphical environment -> plt.show() will not work

import datetime
import sys
import getopt
import time
import pyfftw
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import cmath
import math
import scipy.signal as sig
import scipy.optimize as sci
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyfftw
pyfftw.interfaces.cache.enable()

import calendar
import AspectMapper
import igrf
import digital_rf
import icebearModules as Imod
import vizualization as viz

# ------------------------------------ Initilization ------------------------------------------ #
# Note: Parameters defined here for ease of access and are used througout the script
# Note: Time statements are located throughout code for time tracking purposes, use ts_ tf_

print("\n-----START---------------------START--------")
ts_ini = time.time()

# Predefined input values for command line
input_file = 'canada_gh.ini'
igrf_year = 2015.0

# Cygnus A parameters
RA  = 299.8681525
DEC = 40.73391555

# Radar receiver parameters
LAT = 52.24319
LON = -106.450191
LATr = np.deg2rad(LAT)
LONr = np.deg2rad(LON)

# By default, corrections make no change
phase_corr = [0,0,0,0,0,0,0,0,0,0]
mag_corr   = [1,1,1,1,1,1,1,1,1,1]

# Location of hdf5 files
# '/data/data/icebear_test/'
source = '/ext_data2/icebear_3d_data/'

# Time of first plot (don't use leading zeros)
year    = 2020
month   = 5
day     = 31
hour    = 0
minute  = 0
second  = 0

# Radar and pseudo-random code properties
sample_rate = 200000
codelen     = 20000  # length of pseudo-random code
fdec        = 200    # frequency decimation rate

# Number of ranges, # of averages in each plot(=point), spacing between plots, # of plots to make
nrang        = 2000
averages     = 10                      # number of 0.1 second intervals to combine
plot_spacing = 0.0                     # in seconds
period       = 8                       # number of hours to compute
num_plots    = int(3600/(averages/10+plot_spacing) * period)    # convert hours to number of plots

opt = 'x-spectrum'
noop = 'plot'  # 'save', 'plot' or 'load'

mlat_map = igrf.IGRF(igrf_year)

# Tx Rx locations
Earth_radius = 6378.1
rx_lat       = math.radians(52.24)
rx_lon       = math.radians(253.55-360.00)
tx_lat       = math.radians(50.893)
tx_lon       = math.radians(-109.403)

# FOV plotting
RF_lons, RF_lats, RF_level, RF_contour = Imod.contour_RF(Earth_radius, tx_lat, tx_lon, rx_lat, rx_lon)
Mag_lons, Mag_lats, Mag_level, Mag_contour = Imod.contour_Mag(mlat_map)

# Data Processing Variables
b_code = Imod.generate_bcode()

# Analysis Variables
lam      = 6.06
d        = 6.06           # Spacing of the antennas -> needs update for ICEBEAR-3D
fft_freq = np.fft.fftshift(np.fft.fftfreq(int(codelen/fdec),fdec/float(sample_rate)))

# Plotting Variables
base_time = datetime.datetime(year,month,day,hour,minute,second)
dates = np.array([base_time + datetime.timedelta(seconds = averages/10 * shift) for shift in range(num_plots)])
locator = mdates.AutoDateLocator(minticks=7, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

# Make naming variables
x_lab  = 'Doppler (Hz)'
y_lab  = 'RF Propagation Distance/2.0 (km)'

# Where to store everything
loc   = '/home/icebear-cuda/research/noise_brian/data/'


# Define size of plots (in inches)
mpl.rcParams['figure.figsize'] = [14.5, 12.0]
mpl.rcParams['xtick.labelsize'] = 20.0
mpl.rcParams['ytick.labelsize'] = 20.0

# Initialization
# ----------------------------------------------------------------------------------------------- #


def usage():
    print("usage : %s" % sys.argv[0])
    print("\t -i <config file>  The name of the config ini file to load.")
    print("\t -y <igrf year>    Optional IGRF model year to use in the computation")


# Parse the command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hi:y:')
except:
    usage()
    sys.exit()

for opt, val in opts:
    if opt in ('-h'):
        usage()
        sys.exit()
    elif opt in ('-i'):
        input_file = val
    elif opt in ('-y'):
        igrf_year = float(val)

# This statement is null if input file is defined above
if input_file == None:
    print("An input ini file name must be provided.")
    usage()

# Set variable with path to antenna data
data = digital_rf.DigitalRFReader([source])
channels = data.get_channels()
if len(channels) == 0:
    raise IOError("""Please run one of the example write scripts C: example_rf_write_hdf5, or Python: example_digital_rf_hdf5.py before running this example""")

print('found channels: %s' % (str(channels)))
print('done indexing channels')

# calculate magnitude correction for each antenna
for i in range(0,10):
    # Self-correlate antenna at quiet times
    # time with no ionospheric activity (assumed 4th hour)
    calibrate_time = datetime.datetime(year, month, day, 4, minute, second)
    mag = 0
    time_tuple = (year,month,day,4,minute,second,-1,-1,0)

    # Calculate the start sample (number of samples since epoch less 30 for some reason)
    start_sample = int((calendar.timegm(time_tuple))*sample_rate)-30

    # do processing on some samples (arbitrary, but assumed quiet time)
    antenna_data = (data.read_vector_c81d(start_sample, 1000*10*60/(averages/10),'antenna'+str(i)))

    # knowing average noise power for each antenna lets us normalize to a single antenna
    mag_corr[i] = np.median(np.abs(antenna_data))

print(mag_corr)
print(phase_corr)
# save correction

antenna2 = np.array([1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,3,4,5,6,7,8,9,4,5,6,7,8,9,5,6,7,8,9,6,7,8,9,7,8,9,8,9,9])
antenna1 = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,7,7,8])
for a1 in range(0, len(antenna1)):
    if opt == 'x-spectrum':
        ant = [antenna1[a1], antenna2[a1]]
        print(ant)
        name = '%d-%d-%d_from_%d_to_%d-'%(year,month,day,hour,period)+opt+'_mag_phase_'+str(ant[0])+str(ant[1])
    elif opt == 'single_a':
        ant = 2
        name = '%d-%d-%d_from_%d_to_%d-'%(year,month,day,hour,period)+opt+'_'+str(ant)+'_full'
    else:
        print("Hol' up, this ain't right")

    complex_corr = np.zeros(len(phase_corr), dtype=complex)
    for x in range(len(phase_corr)):
        complex_corr[x] = cmath.rect(1.0/mag_corr[x],cmath.pi*phase_corr[x]/180.0)

    # Pull variables from .ini file
    spacing, altitude, fov_limit, aspect_limit, tx, rx, tx_sites, rx_sites, lat_range, lon_range, corr_num = Imod.load_config(input_file, igrf_year)

    tf_ini = time.time()
    print("Initialization Time: %1.4f"%(tf_ini-ts_ini))
    ts_loop = time.time()

    #############################################################################
    #             DEFINE NOISE VALUES TO SAVE                                   #
    #############################################################################

    noise_array       = np.zeros(num_plots)
    noise_array_phase = np.zeros(num_plots)

    for plot_num in range(num_plots):
        for plot_num in range(num_plots):
            sys.stdout.write("\rBeginning step %5.0d/%5.0d: " % (plot_num, num_plots) + str(
                base_time + datetime.timedelta(seconds=(averages * 0.1 + plot_spacing) * plot_num)))
            sys.stdout.flush()

            # ------------------------------------ Data Processing ---------------------------------------- #

            # Grab one antenna (maybe a cross-correlation)
            if opt == 'x-spectrum':
               xspec, calc_time = Imod.decx(b_code, data, codelen, complex_corr, averages, True, fdec, nrang, base_time,
                                              plot_num, plot_spacing, sample_rate, ant[0], ant[1])
            if opt == 'single_a':
               xspec, calc_time = Imod.dec(b_code, data, codelen, complex_corr, averages, fdec, nrang, base_time,
                                             plot_num, plot_spacing, sample_rate, ant)
            # Calculate noise values
            extend = ''
            # Get a single complex value to represent the 1s average for whole range doppler matrix (all noise)
            n_real = np.average(np.real(xspec.data))
            n_imag = np.average(np.imag(xspec.data))
            n = n_real + 1j * n_imag

            # power and phase of average noise fills in one data point
            noise_array[plot_num] = np.abs(n)
            noise_array_phase[plot_num] = np.angle(n)

        # --------------------------------- Ending Sequence ------------------------------------------- #
        print("\n-----END-------------------------END--------")
        print("Program successfully finished all plots")
        print("Total run time is: %1.4f\n" % (tf_loop - ts_loop))

        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'noise_array.npy', noise_array)
        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'noise_array_phase.npy', noise_array_phase)
        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'dates.npy', dates)

        print(name + extend + 'dates.npy')