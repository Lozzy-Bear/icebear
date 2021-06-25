import numpy as np
import h5py
import matplotlib.pyplot as plt

# files = ['demo_ib3d_level3_20201212.h5', 'demo_ib3d_level3_20201213.h5',
#          'demo_ib3d_level3_20201214.h5', 'demo_ib3d_level3_20201215.h5']
# files = ['demo_ib3d_level3_20210202.h5']
files = ['demo_ib3d_level3_20191219.h5']
# files = ['demo_ib3d_level3_20200331.h5']

slant_range = np.array([])
altitude = np.array([])
snr_db = np.array([])
plt.figure()
for file in files:
    f = h5py.File(file, 'r')
    #
    # plt.scatter(f['data']['slant_range'][()], f['data']['altitude'][()], c=f['data']['velocity'][()])
    # plt.colorbar()
    slant_range = np.append(slant_range, f['data']['slant_range'][()])
    altitude = np.append(altitude, f['data']['altitude'][()])
    snr_db = np.append(snr_db, f['data']['snr_db'][()])


plt.figure(figsize=[12, 12])
mean_altitude = np.mean(altitude)
total_targets = len(altitude)
_ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype=u'step', label=f'Total Targets {total_targets}')
plt.xscale('log')
plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude Distribution')
plt.xlabel('Count')
plt.ylabel('Altitude [km]')
plt.ylim((50, 200))
plt.xlim((10, 10_000))
plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
plt.legend(loc='upper right')
plt.grid()
plt.show()

