# ------- Initialization -------
import datetime
import sys
import getopt
import numpy as np
import cmath
import icebearModules as Imod
import digital_rf
import calendar

print("\n-----START---------------------START--------")

# Default command line values if none given
input_file = 'canada_gh.ini'
igrf_year = 2015.0
source = 'ext_data2/icebear_3d_data'
# Time of first point (don't use leading zeros)
year    = 2020
month   = 5
day     = 31
hour    = 0
minute  = 0
second  = 0
# Number of ranges, # of averages in each plot(=point), spacing between plots, # of plots to make
nrang        = 2000  # Number of ranges
averages     = 10    # Number of 0.1 second intervals to coherently average
plot_spacing = 0.0   # Spacing between points (in seconds)
period       = 8     # number of hours to compute over
num_points    = int(3600/(averages/10+plot_spacing) * period)  # convert hours to number of points

# ------- Parse command line arguments -------


def usage():
    print("usage : %s" % sys.argv[0])
    print("\t --ini-file <config file>  The name of the config ini file to load.")
    print("\t --igrf-year <igrf year>  Optional IGRF model year to use in the computation")
    print("\t --source <filepath>  Filepath to the top-level directory of .h5 files to use")
    print("\t -y <year>  Year value at which to start the calculations")
    print("\t -m <month>  Month value at which to start the calculations")
    print("\t -d <day>  Day value at which to start the calculations")
    print("\t -h <hour>  Hour value at which to start the calculations")
    print("\t --minute <minute>  Minute value at which to start the calculations")
    print("\t -s <second>  Second value at which to start the calculations")


try:
    opts, args = getopt.getopt(sys.argv[1:], 'y:m:d:h:m:s:', ['help', 'ini-file = ', 'igrf-year = ', 'source = ', 'minute = '])
except:
    usage()
    sys.exit()

for opt, val in opts:
    if opt in ('--help'):
        usage()
        sys.exit()
    elif opt in ('--ini-file'):
        input_file = val
    elif opt in ('--igrf-year'):
        igrf_year = float(val)
    elif opt in ('--source'):
        source = int(val)
    elif opt in ('-y'):
        year = int(val)
    elif opt in ('-m'):
        month = int(val)
    elif opt in ('-d'):
        day = int(val)
    elif opt in ('-h'):
        hour = int(val)
    elif opt in ('--minute'):
        minute = int(val)
    elif opt in ('-s'):
        second = int(val)
# Done parsing command line arguments


# By default, corrections make no change
phase_corr = [0,0,0,0,0,0,0,0,0,0]
mag_corr   = [1,1,1,1,1,1,1,1,1,1]

# Radar and pseudo-random code properties
sample_rate = 200000
codelen     = 20000  # length of pseudo-random code
fdec        = 200    # frequency decimation rate
b_code = Imod.generate_bcode()

# Timespan array
start_time = datetime.datetime(year, month, minute, second)
dates = np.array([start_time + datetime.timedelta(seconds=averages/10*shift for shift in range(num_points))])

# Open stream to antenna data
data = digital_rf.DigitalRFReader([source])
channels = data.get_channels()
if len(channels) == 0:
    raise IOError("No channels found.")

print(f'found channels: {str(channels)}')
print('done indexing channels')

# calculate magnitude correction for each antenna by taking the median
# of all measurements recorded by it over some quiet time period (assumed to be 4th hour)
for antenna_num in range(10):
    # assumed time with no ionospheric activity
    calibrate_time = datetime.datetime(year, month, day, 4, minute, second)

    # calculate the number of samples since epoch (subtract 30, not sure why)
    time_tuple = (year, month, day, 4, minute, second, -1, -1, 0)
    start_sample = int((calendar.timegm(time_tuple)) * sample_rate) - 30

    # grab data from an arbitrary number of samples (not sure what this number corresponds to specifically)
    antenna_data = data.read_vector_c81d(start_sample, int(1000*10*60/(averages/10)), 'antenna'+str(antenna_num))

    # antennas will be normalized according to their average noise power over the data
    mag_corr[antenna_num] = np.median(np.abs(antenna_data))

print(mag_corr)
print(phase_corr)

# pull variables from the .ini file
spacing, altitude, fov_limit, aspect_limit, tx, rx, tx_sites, rx_sites, lat_range, lon_range, corr_num = Imod.load_config(input_file, igrf_year)

# begin calculations for each baseline
antenna1 = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,7,7,8])
antenna2 = np.array([1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,3,4,5,6,7,8,9,4,5,6,7,8,9,5,6,7,8,9,6,7,8,9,7,8,9,8,9,9])
for a1 in range(len(antenna1)):
    # generate name for each baseline
    ant = [antenna1[a1], antenna2[a1]]
    print('baseline: ' + ant)
    name = f'{year}-{month}-{day}_from_{hour}_to_{hour+period}-x-spectrum_mag_phase_'+str(ant[0])+str(ant[1])

    # use that magnitude correction we calculated earlier to normalize. phase_corr is all zeros, no effect.
    complex_corr = np.zeros(len(phase_corr), dtype=complex)
    for x in range(len(phase_corr)):
        complex_corr[x] = cmath.rect(1.0 / mag_corr[x], cmath.pi * phase_corr[x] / 180.0)

    # arrays to hold the noise and phase data. these are the final products of this script
    noise_array = np.zeros(num_points)
    noise_array_phase = np.zeros(num_points)

    # for each data point
    for i in range(num_points):
        for point in range(num_points):
            # print progress
            sys.stdout.write(f'\rBeginning step {point}/{num_points}: ' + str(
                start_time + datetime.timedelta(seconds=(averages*0.1 + plot_spacing) * point)))
            sys.stdout.flush()

            # perform cross-correlation to obtain one data point in this baseline
            xspec, calc_time = Imod.decx(b_code, data, codelen, complex_corr, averages, True, fdec, nrang, start_time,
                                         point, plot_spacing, sample_rate, ant[0], ant[1])

            # calculate a single complex value representing the noise in the entire range doppler matrix
            n_real = np.average(np.real(xspec.data))
            n_imag = np.average(np.imag(xspec.data))
            n = n_real + 1j * n_imag

            # power and phase of the average noise fills in one data point
            noise_array[point] = np.abs(n)
            noise_array_phase[point] = np.angle(n)

        # report on the completed baseline and save its associated files
        # todo: why are we nesting for-loops here? what does the outer loop accomplish?
        print("\n-----END-------------------------END--------")
        print(f"Program successfully calculated all points in baseline {ant[0]}-{ant[1]}") # not accurate
        extend = ''
        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'noise_array.npy', noise_array)
        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'noise_array_phase.npy', noise_array_phase)
        np.save('/home/icebear-cuda/research/noise_brian/data/' + name + extend + 'dates.npy', dates)

        print(name + extend + 'dates.npy')

print('script finished.')