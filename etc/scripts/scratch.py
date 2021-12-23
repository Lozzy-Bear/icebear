import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from scipy.optimize import curve_fit
from scipy.signal import peak_widths, find_peaks


def doppler_spectra(dop, snr):
    """
    Build a Doppler spectra from the targets passed.

    Returns
    -------

    Notes
    -----
    We assume that the passed data is all data in a given range.
    """
    def gaussian(x, x0, sigma, a):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    popt, pcov = curve_fit(gaussian, dop, snr)
    x = np.arange(-500.0, 500.0+10.0, 10.0)
    y = gaussian(x, *popt)
    # peaks, _ = find_peaks(y, rel_height=0.5)
    # spectral_width[i] = peak_widths(y, peaks, rel_height=0.5)[0] * 30.0
    plt.figure()
    plt.plot(x, y, '--k', label='Fit spectra')
    plt.scatter(dop, snr, label='data')
    plt.legend()
    plt.xlabel('Doppler [Hz]')
    plt.ylabel('SNR [dB]')
    plt.show()

    return

# filepath = '/beaver/backup/level2b/ib3d_normal_swht_2020_03_31_prelate_bakker.h5'
# filepath = '/beaver/backup/level2b/ib3d_normal_swht_2020_10_24_prelate_bakker.h5'
# filepath = '/beaver/backup/level2b/ib3d_normal_swht_2019_12_19_prelate_bakker.h5'
filepath = '/beaver/backup/level2_1deg_clutter_corr/2020_03_31/ib3d_normal_swht_2020_03_31_prelate_bakker.h5'
# filepath = '/beaver/backup/level2_1deg/2020_03_31/ib3d_normal_swht_2020_03_31_prelate_bakker.h5'
# filepath = '/beaver/backup/level2b/2020_03_31/ib3d_normal_swht_2020_03_31_prelate_bakker.h5'
f = h5py.File(filepath, 'r')

time = f['data']['time'][()]
dop = f['data']['doppler_shift'][()]
az = f['data']['azimuth'][()]
el = f['data']['elevation'][()]
lat = f['data']['latitude'][()]
lon = f['data']['longitude'][()]
alt = f['data']['altitude'][()]
srng = f['data']['slant_range'][()]
snr = np.abs(f['data']['snr_db'][()])
elr = f['dev']['raw_elevation'][()]
print(len(snr))
before = len(snr)

alt_thresh = 0.0
col_map = 'viridis'
cmi = 1
cma = 200_000
lon_bins = 360
lat_bins = 200
alt_bins = 600
snr_bins = 30
dop_bins = 100
print(np.min(dop), np.max(dop), np.median(dop))
m = np.ones_like(time)
# m = np.ma.masked_where(az < -45.0, m)
# m = np.ma.masked_where(az > 45.0, m)
# m = np.ma.masked_where((az > -45.0) & (az < 45.0) & (el < 45.0), m)
# m = np.ma.masked_where(el < 2.0, m)
# m = np.ma.masked_where(el > 45.0, m)
med_snr = np.nanmedian(snr)
print('median snr:', med_snr)
m = np.ma.masked_where(snr < med_snr, m)
# m = np.ma.masked_where((dop >= -200.0) & (dop <= 200) & (snr <= 10.0) & ((alt < 70) | (alt > 130)), m)
# m = np.ma.masked_where(srng < 300.0, m)

alt = alt * m
az = az * m
lat = lat * m
lon = lon * m
snr = snr * m
srng = srng * m
dop = dop * m
elr = elr * m
alt = alt[~alt.mask]
az = az[~az.mask]
lat = lat[~lat.mask]
lon = lon[~lon.mask]
snr = snr[~snr.mask]
srng = srng[~srng.mask]
dop = dop[~dop.mask]
elr = elr[~elr.mask]
print(alt.shape, (before - alt.shape[0])/before * 100)

# plt.figure()
# plt.scatter(lon, lat)
# plt.axis('equal')
#
# plt.figure()
# plt.hist2d(az, srng)
#
# plt.figure()
# plt.hist2d(az, alt)
#
# plt.show()

plt.figure(figsize=[12, 12])
plt.subplot(221)
plt.hist2d(lon, lat, bins=[lon_bins, lat_bins], cmap=col_map, cmin=cmi, cmax=cma, range=[[-115.0, -97.0], [54.0, 64.0]])
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(222)
plt.hist2d(lon, alt, bins=[lon_bins, alt_bins], cmap=col_map, cmin=cmi, cmax=cma, range=[[-115.0, -97.0], [0.0, 600.0]])
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Altitude [km]')

plt.subplot(223)
plt.hist2d(lat, alt, bins=[lat_bins, alt_bins], cmap=col_map, cmin=cmi, cmax=cma, range=[[54.0, 64.0], [0.0, 600.0]])
plt.colorbar()
plt.xlabel('Latitude')
plt.ylabel('Altitude [km]')

plt.subplot(224)
plt.hist2d(snr, alt, bins=[snr_bins, alt_bins], cmap=col_map, cmin=cmi, cmax=cma, range=[[0.0, 40.0], [0.0, 600.0]])
plt.colorbar()
plt.xlabel('SNR [dB]')
plt.ylabel('Altitude [km]')

plt.tight_layout()

plt.figure()
plt.hist2d(az, alt, bins=[360, alt_bins], cmin=cmi, cmax=cma)

plt.figure()
plt.hist2d(dop, alt, bins=[dop_bins, alt_bins], cmap=col_map, cmin=cmi, cmax=cma, range=[[-500.0, 500.0], [0.0, 600.0]])
plt.show()
