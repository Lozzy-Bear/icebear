import h5py
import numpy as np
import matplotlib.pyplot as plt
import icebear.utils as utils

filepath = 'E:/icebear/level2/'  # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_12')  # Enter first sub directory and last
el = np.array([])
rng = np.array([])
dop = np.array([])
snr = np.array([])

plt.figure(1)

for file in files:
    f = h5py.File(file, 'r')
    group = f['data']
    keys = group.keys()

    for key in keys:
        data = group[f'{key}']
        rf_distance = data['rf_distance'][()]
        snr_db = data['snr_db'][()]
        doppler_shift = data['doppler_shift'][()]
        azimuth = data['azimuth'][()]
        elevation = data['elevation'][()]
        elevation_spread = data['elevation_extent'][()]
        azimuth_spread = data['azimuth_extent'][()]
        area = data['area'][()]
        rng = np.append(rng, rf_distance)
        el = np.append(el, elevation)
        dop = np.append(dop, doppler_shift)
        snr = np.append(snr, snr_db)

alt = -6378 + np.sqrt(6378 ** 2 + rng ** 2 - 2 * 6378 * rng * np.cos(np.deg2rad(90 + el)))
rng = np.where(dop <= 20, rng, np.nan)
alt = np.where(dop <= 20, alt, np.nan)
dop = np.where(dop <= 20, dop, np.nan)
rng = np.where(dop >= -20, rng, np.nan)
alt = np.where(dop >= -20, alt, np.nan)
dop = np.where(dop >= -20, dop, np.nan)
rng = np.where(snr >= 6, rng, np.nan)
alt = np.where(snr >= 6, alt, np.nan)
dop = np.where(snr >= 6, dop, np.nan)
plt.scatter(rng, alt, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')
plt.title('Rough Plot -- Geminid 2020/12/12 5 Hours')
plt.show()
