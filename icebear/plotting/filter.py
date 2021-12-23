import glob
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import peak_widths, find_peaks


def linear_fit(xx, yy):
    xx = np.round(xx, decimals=1)
    x = np.arange(-30.0, 30.0+1.0, 0.1)
    x = np.round(x, decimals=1)
    y = np.zeros_like(x)
    for n, i in enumerate(x):
        s = np.argwhere(xx==i)
        h, b = np.histogram(yy[s], bins=np.arange(70, 130+1, 1))
        y[n] = b[np.argmax(h)]
        # p, _ = find_peaks(h)
        # y[n] = np.max(yy[p])
    result = linregress(x, y)
    # plt.figure()
    # plt.scatter(x, y, c='r')
    # plt.plot(x, result.intercept + result.slope * x, '--k')
    # plt.show()
    return result.slope


filepath = '/beaver/backup/level2b/'
files = glob.glob(filepath + 'ib3d_normal_swht_*.h5')
slopes = []
dates = []
for file in files:
    f = h5py.File(file, 'r')
    date = f['info']['date'][()]
    az = f['data']['azimuth'][()]
    alt = f['data']['altitude'][()]
    snr = f['data']['snr_db'][()]
    m = np.ones_like(az)
    m = np.ma.masked_where((snr <= 6.0) & ((alt < 70.0) | (alt > 130.0)) & ((az < -30.0) | (az > 30.0)), m)
    alt = alt * m
    az = az * m
    snr = snr * m
    alt = alt[~alt.mask]
    az = az[~az.mask]
    snr = snr[~snr.mask]
    try:
        s = linear_fit(az, alt)
        slopes.append(s)
        dates.append(date[2] + date[1]*100 + date[0]*10000)
        print(date, s)
    except:
        pass


plt.figure()
plt.scatter(dates, slopes)

plt.figure()
plt.hist(slopes, bins=np.arange(0.0, 0.3, 0.025))
plt.xlabel('Slope M [y=mx+b]')
plt.ylabel('Counts')
plt.title('Histogram of Azimuth Tilt Slopes')
plt.show()





