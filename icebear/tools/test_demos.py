import numpy as np
import h5py
import matplotlib.pyplot as plt
import datetime

# Pretty plot configuration.
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# files = ['demo_ib3d_level3_20201212.h5', 'demo_ib3d_level3_20201213.h5',
#          'demo_ib3d_level3_20201214.h5', 'demo_ib3d_level3_20201215.h5']
files = ['/beaver/backup/level2b/2020_12_12/ib3d_normal_swht_2020_12_12_prelate_bakker.h5']
# files = ['/beaver/backup/level2_3lambda/2022_02_22/ib3d_normal_swht_2022_02_22_prelate_bakker.h5']
# files = ['/beaver/backup/level2_1lambda/2022_03_05/ib3d_normal_swht_2022_03_05_prelate_bakker.h5']

slant_range = np.array([])
altitude = np.array([])
snr_db = np.array([])
time = np.array([])
rf_distance = np.array([])
doppler_shift = np.array([])
azimuth = np.array([])
elevation = np.array([])
elevation_extent = np.array([])
azimuth_extent = np.array([])
area = np.array([])
lat = np.array([])
lon = np.array([])

# plt.figure()
for file in files:
    f = h5py.File(file, 'r')

    # Example: How to read data in second by second
    # times = f['data']['time'][()]
    # time, start_index, counts = np.unique(times, return_index=True, return_counts=True)
    # for i in range(len(start_index)):
    #     snr_db = f['data']['snr_db'][start_index[i]:start_index[i] + counts[i]]
    #     ts = datetime.datetime.fromtimestamp(time[i])
    #     print(ts.strftime('%Y-%m-%d %H:%M:%S'), snr_db)

    # plt.scatter(f['data']['slant_range'][()], f['data']['altitude'][()], c=f['data']['velocity'][()])
    # plt.colorbar()

    # Example: Reading in data from multiple files (days) into one larger
    slant_range = np.append(slant_range, f['data']['slant_range'][()])
    altitude = np.append(altitude, f['data']['altitude'][()])
    snr_db = np.append(snr_db, f['data']['snr_db'][()])
    time = np.append(time, f['data']['time'][()])
    rf_distance = np.append(rf_distance, f['data']['rf_distance'][()])
    doppler_shift = np.append(doppler_shift, f['data']['doppler_shift'][()])
    azimuth = np.append(azimuth, f['data']['azimuth'][()])
    elevation = np.append(elevation, f['data']['elevation'][()])
    # elevation_extent = np.append(elevation_extent, f['data']['elevation_extent'][()])
    # azimuth_extent = np.append(azimuth_extent, f['data']['azimuth_extent'][()])
    # area = np.append(area, f['data']['area'][()])
    lat = np.append(lat, f['data']['latitude'][()])
    lon = np.append(lon, f['data']['longitude'][()])
    # idx = np.argwhere(time>time[0]+5.0*60.0*60.0)

# print(np.unique(valid, return_counts=True))
print(slant_range.shape)

plt.figure()
# mean_altitude = 93.2 # np.mean(altitude)
total_targets = len(altitude)
n, bins, _ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype='step', label=f'Total Targets {total_targets}', color='k')

counts = n
mids = 0.5*(bins[1:] + bins[:-1])
probs = counts / np.sum(counts)
mean = np.sum(probs * mids)
sd = np.sqrt(np.sum(probs * (mids - mean)**2))
print(f'standard deviation: {sd}, mean: {mean}')

# plt.xscale('log')
# plt.title('E Region Scatter Distribution\nDecember 19, 2019')
# plt.title('Geminids Meteor Trail Distribution\nDecember 12-15, 2020')
plt.xlabel('Count')
plt.ylabel('Altitude [km]')
# plt.ylim((70, 120))
# plt.xlim((10, 4_000))
# plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Peak Altitude {mean_altitude:.1f} [km]')
# plt.legend(loc='lower right')
plt.grid()

plt.figure()

plt.show()

