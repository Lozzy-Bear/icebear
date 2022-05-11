# contains the Matlab algorithm implemented in Cupy.
# reads from data files. Mostly correct, except the
# time read in is not formatted properly.

# todo: figure out how the time array is read it and convert appropriately

import datetime
import numpy as np
try:
   import cupy as xp
   CUDA = True
except ModuleNotFoundError:
   import numpy as xp
   CUDA = False
import h5py
import matplotlib.pyplot as plt
import time as tm
import sys


def haversine(p1, p2, r):
    """

    Parameters
    ----------
    p1 : point [time, latitude, longitude]
    p2 : array of points [time, latitude, longitude]
    r  : distance above surf of earth to calculate (in km)

    Returns
    -------

    Haversine distances between each point in p2 and p1
    """
    # distance from earth's centre
    R = 6371.0 + r

    delta_lat = p2[1, :] - p1[1]
    delta_lon = p2[2, :] - p1[2]

    a = (xp.sin(xp.deg2rad(delta_lat/2))) ** 2 + xp.cos(xp.deg2rad(p2[1])) * xp.cos(xp.deg2rad(p1[1])) * ((xp.sin(xp.deg2rad(delta_lon/2))) ** 2)
    b = 2 * R * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))

    return b

files = ['/beaver/backup/level2b/ib3d_normal_swht_2021_02_20_prelate_bakker.h5']
# files = [str(sys.argv[1])]
print("current device is: " + str(xp.cuda.Device().id))

# slant_range = np.array([])
altitude = np.array([])
# snr_db = np.array([])
time = np.array([])
# rf_distance = np.array([])
# doppler_shift = np.array([])
# azimuth = np.array([])
# elevation = np.array([])
# elevation_extent = np.array([])
# azimuth_extent = np.array([])
# area = np.array([])
lat = np.array([])
lon = np.array([])

for file in files:
    f = h5py.File(file, 'r')

    # todo: change these to xp.append once cupy is updated on ibp3
    # Example: Reading in data from multiple files (days) into one larger
    # slant_range = np.append(slant_range, f['data']['slant_range'][()])
    altitude = np.append(altitude, f['data']['altitude'][()])
    # snr_db = np.append(snr_db, f['data']['snr_db'][()])
    time = np.append(time, f['data']['time'][()])
    # rf_distance = np.append(rf_distance, f['data']['rf_distance'][()])
    # doppler_shift = np.append(doppler_shift, f['data']['doppler_shift'][()])
    # azimuth = np.append(azimuth, f['data']['azimuth'][()])
    # elevation = np.append(elevation, f['data']['elevation'][()])
    # elevation_extent = np.append(elevation_extent, f['data']['elevation_extent'][()])
    # azimuth_extent = np.append(azimuth_extent, f['data']['azimuth_extent'][()])
    # area = np.append(area, f['data']['area'][()])
    lat = np.append(lat, f['data']['latitude'][()])
    lon = np.append(lon, f['data']['longitude'][()])
    # idx = np.argwhere(time>time[0]+5.0*60.0*60.0)

time = xp.asarray(time)
lat = xp.asarray(lat)
lon = xp.asarray(lon)
altitude = xp.asarray(altitude)

alt_idx = (altitude < 150.0) & (altitude > 70.0)

time = time[alt_idx]
lat = lat[alt_idx]
lon = lon[alt_idx]

# lat_idx = (lat < 60.3863458065008) & (lat > 54.7751312601798)
#
# time = time[lat_idx]
# lat = lat[lat_idx]
# lon = lon[lat_idx]
#
# lon_idx = (lon < -102.494825012555) & (lon > -108.732129453555)
#
# time = time[lon_idx]
# lat = lat[lon_idx]
# lon = lon[lon_idx]

starttime = tm.time()
di_r = 512
di_t = int(di_r/2)
# dr holds median spatial distances to nearest di_r neighbours (within tspan hour window)
dr = xp.zeros([len(time)], dtype=xp.float32)
# dt holds median temporal distances to nearest 2*di_t neighbours
dt = xp.zeros([len(time)], dtype=xp.float32)

# p1 will hold the point of interest : p1 = [time, lat, lon]
p1 = xp.ndarray([3, 1], dtype=xp.float32)

for p1_idx in range(0, len(time)):

    p1[0] = time[p1_idx]
    p1[1] = lat[p1_idx]
    p1[2] = lon[p1_idx]

    tspan = 4 # hours
    tspan_seconds = int(tspan*60*60)

    # find indices of all points within the timespan
    p2_idx = (time < (p1[0] + tspan_seconds//2)) & (time > (p1[0] - tspan_seconds//2))

    # populate p2 with all points within the timespan
    p2 = xp.ndarray([3, len(time[p2_idx])], dtype=xp.float32)
    p2[0, :] = time[p2_idx]
    p2[1, :] = lat[p2_idx]
    p2[2, :] = lon[p2_idx]

    # get haversine distances between the point p1 and all points within tspan hours p2
    drs = haversine(p1, p2, 105)

    # calc median space distance of nearest di_r points
    drs = xp.sort(drs)
    dr[p1_idx] = xp.median(drs[0:di_r])

    # calc median time distance of nearest 2*di_t points
    # todo: make this use the actual nearest 2*di_t points instead of the points centred around p1_idx
    earliest = max(0, p1_idx-di_t)
    latest = min(len(time)-1, p1_idx+di_t)
    dt[p1_idx] = xp.median(abs(p1[0] - time[earliest:latest]))

dt[dt < 1] = 1

endtime = tm.time()
print("took about " + str(round(endtime - starttime)) + " seconds to execute, with " + str(len(time)) + " points")

logbinsdt = np.logspace(np.log10(1), np.log10(max(dt.get())), 300)
logbinsdr = np.logspace(np.log10(1), np.log10(max(dr.get())), 300)

fig = plt.figure(1)
gs = fig.add_gridspec(4, 4)
ax1 = fig.add_subplot(gs[1:4, 0:3])
ax2 = fig.add_subplot(gs[0, 0:3])
ax3 = fig.add_subplot(gs[1:4, 3])

ax1.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
ax1.scatter(dt.get(), dr.get(), marker='.')
ax1.set_yscale('log')
ax1.set_xscale('log')

ax2.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
ax2.hist(dt.get(), bins=logbinsdt,  rwidth=3)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.sharex(ax1)

ax3.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
ax3.minorticks_on()
ax3.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
ax3.hist(dr.get(), bins=logbinsdr, orientation='horizontal', rwidth=3)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.sharey(ax1)

plt.show()
