import h5py
import numpy as np
import matplotlib.pyplot as plt
import icebear.utils as utils
import icebear

filepath = 'E:/icebear/level2a/'  # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2020_12_13', '2020_12_15')  # Enter first sub directory and last
#files = ['E:/icebear/level2/2019_12_19/ib3d_normal_swht_01deg_2019_12_19_05_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
#files = ['E:/icebear/level2a/2019_12_19/ib3d_normal_swht_10deg_2019_12_19_05_prelate_bakker_old.h5']  # Alternate; make a list of files with file paths.
#files = ['E:/icebear/level2a/2019_12_19/ib3d_normal_swht_10deg_2019_12_19_05_prelate_bakker_new.h5']  # Alternate; make a list of files with file paths.
el = np.array([])
rng = np.array([])
dop = np.array([])
snr = np.array([])
az = np.array([])

plt.figure(1)

for file in files:
    f = h5py.File(file, 'r')
    print(file)
    group = f['data']
    keys = group.keys()

    for key in keys:
        if f'{key}' == '052138000':
            break
        data = group[f'{key}']
        rf_distance = data['rf_distance'][()]
        snr_db = data['snr_db'][()]
        doppler_shift = data['doppler_shift'][()]
        azimuth = data['azimuth'][()]
        elevation = data['elevation'][()]
        #elevation_spread = data['elevation_extent'][()]
        #azimuth_spread = data['azimuth_extent'][()]
        area = data['area'][()]
        rng = np.append(rng, rf_distance)
        el = np.append(el, elevation)
        dop = np.append(dop, doppler_shift)
        snr = np.append(snr, snr_db)
        az = np.append(az, azimuth)


#print(np.max(az), np.min(az), np.max(el), np.min(el))
print(len(dop))
# az -= 270
el += 90
el = np.abs(el)
#icebear.plotting.plot.imaging_4plot('E:/icebear/figures/geminid/', 'Comparison', '2019-12-19', dop, rng, np.abs(snr), az, el)

hu = 0.015
rng = rng * 0.75 - 200
alt = -6378 + np.sqrt((6378 + hu) ** 2 + rng ** 2 + 2 * (6378 + hu) * rng * np.sin(np.deg2rad(el)))


d = 20
s = 6

rng = np.where(dop <= d, rng, np.nan)
alt = np.where(dop <= d, alt, np.nan)
dop = np.where(dop <= d, dop, np.nan)
az = np.where(dop <= d, az, np.nan)
el = np.where(dop <= d, el, np.nan)

rng = np.where(dop >= -d, rng, np.nan)
alt = np.where(dop >= -d, alt, np.nan)
dop = np.where(dop >= -d, dop, np.nan)
az = np.where(dop >= -d, az, np.nan)
el = np.where(dop >= -d, el, np.nan)

rng = np.where(snr >= s, rng, np.nan)
alt = np.where(snr >= s, alt, np.nan)
dop = np.where(snr >= s, dop, np.nan)
az = np.where(snr >= s, az, np.nan)
el = np.where(snr >= s, el, np.nan)

plt.subplot(121)
plt.scatter(rng, el, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Elevation [deg]')
#plt.title('Rough Plot -- 2019-12-19')
plt.title('Rough Plot -- Geminids 2020')
plt.subplot(122)
plt.scatter(rng, az, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Azimuth [deg]')
#plt.title('Rough Plot -- 2019-12-19')
plt.title('Rough Plot -- Geminids 2020')

plt.figure()
plt.scatter(rng, alt, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.show()
