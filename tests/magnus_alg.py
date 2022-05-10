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

    a = xp.sin(delta_lat/2) ** 2 + xp.cos(p2[1, :]) * xp.cos(p1[1]) * xp.sin(delta_lon/2) ** 2
    b = 2 * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))

    return R * b

# files = ['/beaver/backup/level2b/ib3d_normal_swht_2021_02_20_prelate_bakker.h5']
files = [str(sys.argv[1])]


# slant_range = np.array([])
# altitude = np.array([])
# snr_db = np.array([])
time = xp.array([])
# rf_distance = np.array([])
# doppler_shift = np.array([])
# azimuth = np.array([])
# elevation = np.array([])
# elevation_extent = np.array([])
# azimuth_extent = np.array([])
# area = np.array([])
lat = xp.array([])
lon = xp.array([])

for file in files:
    f = h5py.File(file, 'r')

    # todo: change these to xp.append once cupy is updated on ibp3
    # Example: Reading in data from multiple files (days) into one larger
    # slant_range = np.append(slant_range, f['data']['slant_range'][()])
    # altitude = np.append(altitude, f['data']['altitude'][()])
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

    #ts = datetime.datetime.fromtimestamp(time[0])

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
        p2_idx = ((time < p1[0] + tspan_seconds//2) & (time > p1[0] - tspan_seconds//2))

        # populate p2 with all points within the timespan
        p2 = xp.ndarray([3, len(time[p2_idx])], dtype=xp.float32)
        p2[0, :] = time[p2_idx]
        p2[1, :] = lat[p2_idx]
        p2[2, :] = lon[p2_idx]

        # get haversine distances between the point p1 and all points within tspan hours p2
        drs = haversine(p1, p2, 110)

        # calc median space distance of nearest di_r points
        drs = xp.sort(drs)
        dr[p1_idx] = xp.median(drs[0:di_r])

        # calc median time distance of nearest 2*di_t points
        # todo: make this use the actual nearest 2*di_t points instead of the points centred around p1_idx
        dt[p1_idx] = xp.median(abs(p1[0] - time[max(0, p1_idx-di_t):min(len(time), p1_idx+di_t)]))

    endtime = tm.time()
    print("took about " + str(round(endtime - starttime)) + " seconds to execute, with " + str(len(time)) + " points")
    plt.scatter(dt.get(), dr.get())
    plt.show()

