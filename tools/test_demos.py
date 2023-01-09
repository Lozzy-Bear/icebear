import numpy as np
import h5py
import matplotlib.pyplot as plt
import datetime
import common.pretty_plots
import glob

files = glob.glob('/data/icebear_data/sanitized_data/ib3d_normal_swht_20*.h5')

# files = ['/run/media/arl203/Seagate Expansion Drive/backup/level2b/2020_12_12/ib3d_normal_swht_2020_12_12_prelate_bakker_sanity.h5',
#          '/run/media/arl203/Seagate Expansion Drive/backup/level2b/2020_12_13/ib3d_normal_swht_2020_12_13_prelate_bakker_sanity.h5',
#          '/run/media/arl203/Seagate Expansion Drive/backup/level2b/2020_12_14/ib3d_normal_swht_2020_12_14_prelate_bakker_sanity.h5',
#          '/run/media/arl203/Seagate Expansion Drive/backup/level2b/2020_12_15/ib3d_normal_swht_2020_12_15_prelate_bakker_sanity.h5']
# files = ['/run/media/arl203/Seagate Expansion Drive/backup/level2b/2019_12_19/ib3d_normal_swht_2019_12_19_prelate_bakker.h5']
# files = ['/run/media/arl203/Seagate Expansion Drive/backup/level2b/2021_02_02/ib3d_normal_swht_2021_02_02_prelate_bakker.h5']
# files = ['/run/media/arl203/Seagate Expansion Drive/backup/level2b/ib3d_normal_swht_2021_03_20_prelate_bakker.h5']
# files = ['/run/media/arl203/Seagate Expansion Drive/backup/level2b/ib3d_normal_swht_2020_03_31_prelate_bakker.h5']


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
# print(slant_range.shape)
plt.figure()
plt.hist2d(azimuth, altitude, bins=[np.arange(-45, 46, 1), np.arange(50, 201, 1)])
plt.show()

plt.figure()
total_targets = len(altitude)
bb = np.arange(60, 131, 0.5)
# n, bins, _ = plt.hist(altitude, bins=bb, orientation='horizontal', histtype='step', label=f'Total Targets {len(altitude)}', color='k')
# m, bins, _ = plt.hist(altitude[azimuth>-12.0], bins=bb, orientation='horizontal', histtype='step', label=f'East & Center {len(altitude[azimuth>-12.0])}', color='b')
# p, bins, _ = plt.hist(altitude[azimuth<=-12.0], bins=bb, orientation='horizontal', histtype='step', label=f'Western Beam {len(altitude[azimuth<=-12.0])}', color='r')
n, bins, _ = plt.hist(altitude, bins=bb, orientation='horizontal', histtype='step', label=f'Total Targets {len(altitude)}', color='k')
m, bins, _ = plt.hist(altitude[azimuth>=15.0], bins=bb, orientation='horizontal', histtype='step', label=f'Eastern Beam {len(altitude[azimuth>-12.0])}', color='b')
p, bins, _ = plt.hist(altitude[(azimuth>-12.0) & (azimuth<15.0)], bins=bb, orientation='horizontal', histtype='step', label=f'Center Beam {len(altitude[azimuth<=-12.0])}', color='r')
p, bins, _ = plt.hist(altitude[azimuth<=-12.0], bins=bb, orientation='horizontal', histtype='step', label=f'Western Beam {len(altitude[azimuth<=-12.0])}', color='g')


counts = m
mids = 0.5*(bins[1:] + bins[:-1])
probs = counts / np.sum(counts)
mean = np.sum(probs * mids)
sd = np.sqrt(np.sum(probs * (mids - mean)**2))
print(f'standard deviation: {sd}, mean: {mean}')
mean_altitude = 103.5

# plt.xscale('log')
plt.title('E Region Scatter Distribution\nMarch 20, 2021')
# plt.title('E Region Scatter Distribution\nDecember 19, 2019')
# plt.title('Geminids Meteor Trail Distribution\nDecember 12-15, 2020')
plt.xlabel('Count')
plt.ylabel('Altitude [km]')
# plt.ylim((70, 130))
# plt.xlim((10, 250_000))
# plt.plot([0, 250_000], [mean_altitude, mean_altitude], '--k', label=f'Peak Altitude {mean_altitude:.1f} [km]')
plt.legend(loc='lower right')
plt.grid()

plt.show()

