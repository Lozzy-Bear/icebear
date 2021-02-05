import h5py
import numpy as np
import matplotlib.pyplot as plt
import icebear.utils as utils
import icebear

filepath = 'E:/icebear/level2a/'  # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2020_12_13', '2020_12_13')  # Enter first sub directory and last
#files = ['E:/icebear/level2/2019_12_19/ib3d_normal_swht_01deg_2019_12_19_05_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
#files = ['E:/icebear/level2a/2019_12_19/ib3d_normal_swht_10deg_2019_12_19_05_prelate_bakker_old.h5']  # Alternate; make a list of files with file paths.
#files = ['E:/icebear/level2a/2019_12_19/ib3d_normal_swht_10deg_2019_12_19_05_prelate_bakker_new.h5']  # Alternate; make a list of files with file paths.
el = np.array([])
rng = np.array([])
dop = np.array([])
snr = np.array([])
az = np.array([])

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

# az -= 270
#el += 90
#el = np.abs(el)

hu = 0.015
rng = rng * 0.75 - 200
alt = -6378 + np.sqrt((6378 + hu) ** 2 + rng ** 2 + 2 * (6378 + hu) * rng * np.sin(np.deg2rad(el)))
alt_fixed = -6378 + np.sqrt((6378 + hu) ** 2 + rng ** 2 - 2 * (6378 + hu) * rng * np.sin(np.deg2rad(el)))
#alt_old = -6378 + np.sqrt((6378 + hu) ** 2 + rng ** 2 - 2 * (6378 + hu) * rng * np.cos(np.deg2rad(90 + el)))


d = 20
s = 6

rng = np.where(dop <= d, rng, np.nan)
alt = np.where(dop <= d, alt, np.nan)
alt_fixed = np.where(dop <= d, alt_fixed, np.nan)
dop = np.where(dop <= d, dop, np.nan)
az = np.where(dop <= d, az, np.nan)
el = np.where(dop <= d, el, np.nan)

rng = np.where(dop >= -d, rng, np.nan)
alt = np.where(dop >= -d, alt, np.nan)
alt_fixed = np.where(dop >= -d, alt_fixed, np.nan)
dop = np.where(dop >= -d, dop, np.nan)
az = np.where(dop >= -d, az, np.nan)
el = np.where(dop >= -d, el, np.nan)

rng = np.where(snr >= s, rng, np.nan)
alt = np.where(snr >= s, alt, np.nan)
alt_fixed = np.where(snr >= s, alt_fixed, np.nan)
dop = np.where(snr >= s, dop, np.nan)
az = np.where(snr >= s, az, np.nan)
el = np.where(snr >= s, el, np.nan)

rng = np.where(az >= -180, np.nan, rng)
alt = np.where(az >= -180, np.nan, alt)
alt_fixed = np.where(az >= -180, np.nan, alt_fixed)
dop = np.where(az >= -180, np.nan, dop)
az = np.where(az >= -180, np.nan, az)
el = np.where(az >= -180, np.nan, el)

rng = np.where(az <= -320, np.nan, rng)
alt = np.where(az <= -320, np.nan, alt)
alt_fixed = np.where(az <= -320, np.nan, alt_fixed)
dop = np.where(az <= -320, np.nan, dop)
az = np.where(az <= -320, np.nan, az)
el = np.where(az <= -320, np.nan, el)

plt.figure(3)
plt.subplot(121)
plt.scatter(rng, el, c=az)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Elevation [deg]')
#plt.title('Rough Plot -- 2019-12-19')
plt.title('Rough Plot -- Geminids 2020')
plt.subplot(122)
plt.scatter(rng, az, c=el)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Azimuth [deg]')
#plt.title('Rough Plot -- 2019-12-19')
plt.title('Rough Plot -- Geminids 2020')


cr = np.arange(0, 1300, 1)
ca = -6378 + np.sqrt(6378 ** 2 + cr ** 2 + 2 * 6378 * cr * np.sin(np.deg2rad(0)))
caf = -6378 + np.sqrt(6378 ** 2 + cr ** 2 - 2 * 6378 * cr * np.sin(np.deg2rad(0)))

er = np.arange(0, 1300, 1)
ea = np.sqrt(6378 ** 2 - er ** 2) - 6378
ea *= -1

plt.figure(1)
plt.subplot(211)
plt.plot(cr, ca, '--k')
plt.plot(er, ea, '--g')
plt.scatter(rng, alt, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.subplot(212)
plt.plot(cr, caf, '--k')
plt.plot(er, ea, '--g')
plt.scatter(rng, alt_fixed, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.figure(2)
plt.subplot(211)
plt.plot(cr, -ca, '--k')
plt.plot(er, -ea, '--g')
plt.scatter(rng, -alt, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.subplot(212)
plt.plot(cr, -caf, '--k')
plt.plot(er, -ea, '--g')
plt.scatter(rng, -alt_fixed, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.show()
