# reads in the dr and dt data from Magnus's matlab calculations and plots it.
# then reads in the lat, long and time data from Magnus's processing and uses
# those values to calculate dr and dt using cupy. Results are good. Some errors
# appear when using xp.float32 arrays. Using xp.float64 removes most errors but
# doubles run time.

import cupy as xp
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import time as tm

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

    delta_lat = p1[1] - p2[1]
    delta_lon = p1[2] - p2[2]

    a = (xp.sin(xp.deg2rad(delta_lat/2))) ** 2 + xp.cos(xp.deg2rad(p2[1])) * xp.cos(xp.deg2rad(p1[1])) * ((xp.sin(xp.deg2rad(delta_lon/2))) ** 2)
    b = 2 * R * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))

    return b

def cluster_plot(fig_num, dt, dr):
    logbinsdt = np.logspace(np.log10(1), np.log10(max(dt)), 300)
    logbinsdr = np.logspace(np.log10(1), np.log10(max(dr)), 300)

    fig = plt.figure(fig_num)
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[1:4, 0:3])
    ax2 = fig.add_subplot(gs[0, 0:3])
    ax3 = fig.add_subplot(gs[1:4, 3])

    ax1.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax1.scatter(dt, dr, marker='.')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax2.hist(dt, bins=logbinsdt, rwidth=3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.sharex(ax1)

    ax3.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax3.hist(dr, bins=logbinsdr, orientation='horizontal', rwidth=3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.sharey(ax1)

    plt.show()

# Matlab calculated dr and dt (these are definitely correct)
f = genfromtxt('/home/brian/Documents/MagnusData/DRDT.csv', delimiter=',')
dr1 = f[:, 0]
dt1 = f[:, 1]

# plot Matlab results
cluster_plot(1, dt1, dr1)

# load in the lat, lon and time data used to calculate the above dr and dt
f = genfromtxt('/home/brian/Documents/MagnusData/LATLON.csv', delimiter=',')
lat = xp.asarray(f[:, 0])
lon = xp.asarray(f[:, 1])
f = genfromtxt('/home/brian/Documents/MagnusData/T.csv', delimiter=',')
time = xp.asarray(f[:])

# convert to seconds
time = time*24.0*60.0*60.0

# start cupy alg

# timer
starttime = tm.time()
# number of points to search for the median
di_r = 512
di_t = int(di_r/2)
# dr holds median spatial distances to nearest di_r neighbours (within tspan hour window)
dr = xp.zeros([len(time)], dtype=xp.float64)
# dt holds median temporal distances to nearest 2*di_t neighbours
dt = xp.zeros([len(time)], dtype=xp.float64)
# p1 will hold the point of interest : p1 = [time, lat, lon] (hopefully this can be expanded to an array later)
p1 = xp.ndarray([3, 1], dtype=xp.float64)
# span of points in time to include when doing spatial median
tspan = 4  # hours
tspan_seconds = int(tspan * 60.0 * 60.0)

# go through every point in time[] array.
for p1_idx in range(0, len(time)):

    p1[0] = time[p1_idx]
    p1[1] = lat[p1_idx]
    p1[2] = lon[p1_idx]

    # find indices of all points within the timespan for spatial average
    p2_idx = (time < (time[p1_idx] + tspan_seconds/2)) & (time > (time[p1_idx] - tspan_seconds/2))

    # populate p2 with all points within the timespan
    p2 = xp.ndarray([3, len(time[p2_idx])], dtype=xp.float64)
    p2[0, :] = time[p2_idx]
    p2[1, :] = lat[p2_idx]
    p2[2, :] = lon[p2_idx]

    # get haversine distances between the point p1 and all points within tspan hours p2
    drs = haversine(p1, p2, 110)

    # calc median space distance of nearest di_r points
    drs = xp.sort(drs)
    dr[p1_idx] = xp.median(drs[0:di_r-1])

    # calc median time distance of nearest 2*di_t points
    # todo: make this use the actual nearest 2*di_t points instead of the points centred around p1_idx
    earliest = max(0, p1_idx-di_t)
    latest = min(len(time)-1, p1_idx+di_t)
    dt[p1_idx] = xp.median(abs(time[p1_idx] - time[earliest:latest]))

dt[dt < 1] = 1

# end timer
endtime = tm.time()
print("took about " + str(round(endtime - starttime)) + " seconds to execute, with " + str(len(time)) + " points")

# plot Cupy results
cluster_plot(2, dt.get(), dr.get())
