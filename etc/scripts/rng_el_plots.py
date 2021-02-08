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

az = np.abs(az)
el += 90
el = np.abs(el)
rng = rng * 0.75 - 200
a = 6378.1370
b = 6356.7523
p1 = np.deg2rad(52.1579)
r1 = np.sqrt((a*np.cos(p1))**2 + (b*np.sin(p1))**2)

pre_alt = np.sqrt(r1 ** 2 + rng ** 2 - 2 * r1 * rng * np.cos(np.deg2rad(90 + np.abs(el))))
gamma = np.arccos((rng ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))
el2 = el - np.abs(np.rad2deg(gamma))

p2 = p1 + gamma
r2 = np.sqrt((a*np.cos(p2))**2 + (b*np.sin(p2))**2)

alt_geocentric = -r2 + np.sqrt(r1 ** 2 + rng ** 2 + 2 * r1 * rng * np.sin(np.deg2rad(el2)))
#alt_negative = -r2 + np.sqrt(r1 ** 2 + rng ** 2 + 2 * r1 * rng * np.sin(np.deg2rad(-el)))
alt_negative = -r1 + np.sqrt(r1 ** 2 + rng ** 2 + 2 * r1 * rng * np.sin(np.deg2rad(el2)))
#alt_negative *= 1.017

d = 20
s = 6

rng = np.where(dop <= d, rng, np.nan)
pre_alt = np.where( dop <= d, pre_alt, np.nan)
alt_geocentric = np.where(dop <= d, alt_geocentric, np.nan)
alt_negative = np.where(dop <= d, alt_negative, np.nan)
dop = np.where(dop <= d, dop, np.nan)
az = np.where(dop <= d, az, np.nan)
el = np.where(dop <= d, el, np.nan)

rng = np.where(-d <= dop, rng, np.nan)
pre_alt = np.where(-d <= dop, pre_alt, np.nan)
alt_geocentric = np.where(-d <= dop, alt_geocentric, np.nan)
alt_negative = np.where(-d <= dop, alt_negative, np.nan)
dop = np.where(-d <= dop, dop, np.nan)
az = np.where(-d <= dop, az, np.nan)
el = np.where(-d <= dop, el, np.nan)

rng = np.where(snr >= s, rng, np.nan)
pre_alt = np.where(snr >= s, pre_alt, np.nan)
alt_geocentric = np.where(snr >= s, alt_geocentric, np.nan)
alt_negative = np.where(snr >= s, alt_negative, np.nan)
dop = np.where(snr >= s, dop, np.nan)
az = np.where(snr >= s, az, np.nan)
el = np.where(snr >= s, el, np.nan)

rng = np.where(az >= 315, np.nan, rng)
pre_alt = np.where(az >= 315, np.nan, pre_alt)
alt_geocentric = np.where(az >= 315, np.nan, alt_geocentric)
alt_negative = np.where(az >= 315, np.nan, alt_negative)
dop = np.where(az >= 315, np.nan, dop)
az = np.where(az >= 315, np.nan, az)
el = np.where(az >= 315, np.nan, el)

rng = np.where(az <= 225, np.nan, rng)
pre_alt = np.where(az <= 225, np.nan, pre_alt)
alt_geocentric = np.where(az <= 225, np.nan, alt_geocentric)
alt_negative = np.where(az <= 225, np.nan, alt_negative)
dop = np.where(az <= 225, np.nan, dop)
az = np.where(az <= 225, np.nan, az)
el = np.where(az <= 225, np.nan, el)

plt.figure()
plt.subplot(311)
plt.scatter(rng, np.abs(alt_negative - alt_geocentric), c=dop)
plt.title('Difference between methods')
plt.colorbar(label='Doppler [Hz]')
#plt.scatter(rng, pre_alt-6378, c=dop)
#plt.title('Uncorrected data')
#plt.plot([0, 1300], [90, 90], 'k')
#plt.plot([0, 1300], [130, 130], 'k')

plt.subplot(312)
plt.scatter(rng, alt_geocentric, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.title('Elliprical Earth -- Geocentric angle correction')
plt.plot([0, 1300], [90, 90], 'k')
plt.plot([0, 1300], [130, 130], 'k')

plt.subplot(313)
plt.scatter(rng, alt_negative, c=dop)
plt.colorbar(label='Doppler [Hz]')
plt.title('Circular Earth -- Geocentric angle correction')
#plt.title('Negative angle correction')
plt.plot([0, 1300], [90, 90], 'k')
plt.plot([0, 1300], [130, 130], 'k')

"""
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
"""

plt.show()
