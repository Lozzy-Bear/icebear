import h5py
import numpy as np
import matplotlib.pyplot as plt
import icebear.utils as utils
from scipy.optimize import curve_fit

filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_15')  # Enter first sub directory and last
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

snr = np.abs(snr)
# az = np.abs(az)
# el += 90
# el = np.abs(el)
rng = rng * 0.75 - 200
m = np.ones_like(rng)
m = np.ma.masked_where(dop > 20, m)
m = np.ma.masked_where(dop < -20, m)
m = np.ma.masked_where(snr <= 6.0, m)
# m = np.ma.masked_where(az >= 315, m)
# m = np.ma.masked_where(az <= 225, m)
m = np.ma.masked_where(el >= 30, m)
m = np.ma.masked_where(el <= 1, m)
m = np.ma.masked_where(rng <= 300, m)
m = np.ma.masked_where(rng >= 1200, m)

re = 6378.0
a = 6378.1370
b = 6356.7523
p1 = np.deg2rad(52.1579)
r1 = np.sqrt((a*np.cos(p1))**2 + (b*np.sin(p1))**2)


pre_alt = np.sqrt(re ** 2 + rng ** 2 - 2 * re * rng * np.cos(np.deg2rad(90 + np.abs(el))))
gamma = np.rad2deg(np.arccos((rng ** 2 - (re ** 2) - (pre_alt ** 2)) / (-2 * re * pre_alt)))
p2 = p1 + gamma
r2 = np.sqrt((a*np.cos(p2))**2 + (b*np.sin(p2))**2)

alt_geocentric = -re + np.sqrt(re ** 2 + rng ** 2 + 2 * re * rng * np.sin(np.deg2rad(el - gamma)))
alt_geocentric -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))

alt_normal = -re + np.sqrt(re**2 + rng**2 + 2 * re * rng * np.sin(np.deg2rad(el)))
alt_normal -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))

alt_negative = -re + np.sqrt(re**2 + rng**2 - 2 * re * rng * np.sin(np.deg2rad(el)))

rng = rng * m
dop = dop * m
snr = snr * m
az = az * m
el = el * m
alt_geocentric = alt_geocentric * m
alt_normal = alt_normal * m
alt_negative = alt_negative * m

hist, _, _ = np.histogram2d(rng, alt_normal, bins=[np.arange(300, 1200, 1), np.arange(0, 400, 1)])
h = np.copy(hist)
h[h == 0] = np.nan
mean_hs = np.zeros(h.shape[0])
for i in range(len(mean_hs)):
    mean_hs[i] = np.argmax(hist[i, :])

xdata = np.arange(300, 1199, 1)
xdata = np.ma.masked_where(mean_hs < 10, xdata)
ydata = np.ma.masked_where(mean_hs < 10, mean_hs)
xdata = np.ma.compressed(xdata)
ydata = np.ma.compressed(ydata)


def func(x, a, b, c):
    return a * x**2 + b * x + c


popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
plt.figure()
plt.title('Geminids Meteor Peak Occurrence Altitude vs. Range Bin')
plt.scatter(xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.6f, b=%5.6f, c=%5.6f' % tuple(popt))
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')
plt.legend()

plt.figure()
plt.subplot(211)
plt.scatter(rng, np.abs(alt_geocentric + func(rng, *popt) - 101), marker='o')
plt.scatter(rng, alt_normal, marker='x')
plt.legend(('2nd Order correction - Geocentric correction', 'Normal'))
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.subplot(212)
plt.scatter(rng, np.abs(alt_normal - func(rng, *popt) + 101), marker='o')
plt.scatter(rng, alt_geocentric, marker='x')
plt.legend(('2nd Order correction', 'Geocentric correction'))
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.figure()
plt.title('Difference between Fit and Geocentric')
plt.scatter(rng, (alt_geocentric - np.abs(alt_normal - func(rng, *popt) + 101)), marker='o')
plt.xlabel('Range [km]')
plt.ylabel('Altitude [km]')

plt.show()
