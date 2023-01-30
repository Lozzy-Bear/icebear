import cupyx
import numpy as np
from numpy import genfromtxt
import h5py
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False
import matplotlib.pyplot as plt
import time

def beam_finder(azimuth, elevation):
    # linear model val(x) = p1*x + p2
    beam = xp.zeros(len(azimuth))
    coeffs = xp.array([[-0.3906, 0.1125],  # west0 coefficients [p1, p2]
                       [-0.3814, 5.934],   # west1      "          "
                       [-0.5422, 8.47],    # centre0    "          "
                       [-0.6575, 19.79],   # centre1    "          "
                       [-10.38, 32.45],    # east0      "          "
                       [-1.229, 59.03]])   # east1      "          "

    # discretize into azimuth bins
    azimuth_bins = xp.arange(-60.0, 60.0, 0.5)
    azimuth_centres = azimuth_bins[0:len(azimuth_bins)-1] + 0.25
    data_bins = xp.digitize(azimuth, azimuth_bins)

    for data_bin in range(len(azimuth_bins)-1):
        # indices of elements matching current bin
        cur_idx = (data_bins == data_bin)
        if any(cur_idx):
            # temporary elevation and beam vars
            temp_elevation = elevation[cur_idx]
            temp_beam = azimuth[cur_idx]

            # using linear functions to cut out unnecessary data
            el_west0   = float(coeffs[0][0] * azimuth_centres[data_bin] + coeffs[0][1])
            el_west1   = float(coeffs[1][0] * azimuth_centres[data_bin] + coeffs[1][1])
            el_centre0 = float(coeffs[2][0] * azimuth_centres[data_bin] + coeffs[2][1])
            el_centre1 = float(coeffs[3][0] * azimuth_centres[data_bin] + coeffs[3][1])
            el_east0   = float(coeffs[4][0] * azimuth_centres[data_bin] + coeffs[4][1])
            el_east1   = float(coeffs[5][0] * azimuth_centres[data_bin] + coeffs[5][1])

            WEST   = (el_west0   < temp_elevation) & (temp_elevation < el_west1)
            CENTRE = (el_centre0 < temp_elevation) & (temp_elevation < el_centre1)
            EAST   = (el_east0   < temp_elevation) & (temp_elevation < el_east1)

            # numbering to match Magnus
            temp_beam[WEST]   = 3
            temp_beam[CENTRE] = 2
            temp_beam[EAST]   = 1
            beam[cur_idx] = temp_beam

    return beam


def cluster_medians(arr, r=110.0, tspan=4.0, di_r=512):
    """
    Calculates the spatial and temporal medians for each point in the array arr
    using CuPy / CUDA optimization.

    Parameters
    ----------
    arr : float32 ndarray
        array of points [time, latitude, longitude]
            - time is in units of days and sorted earliest to latest
            - latitude and longitude in units of degrees
    r  : float32
        distance above surface of earth in km to calculate haversine (default 110)
    tspan : float32
        total time in hours to window the spatial median (default 4)
    di_r : int
        number of points centred on the point of interest used to calculate the medians (default 512)

    Returns
    -------
    dr: float32 ndarray
        1d array of median spatial distances
    dt: float32 ndarray
        1d array of median temporal distances
    """

    # convert time to seconds
    arr[0, :] *= 24.0*60.0*60.0
    n = arr.shape[1]
    di_t = int(di_r / 2)

    # create GPU arrays

    # dr holds median spatial distances to nearest di_r neighbours (within tspan hour window)
    dr = xp.zeros(n, dtype=xp.float32)
    # dt holds median temporal distances to nearest 2*di_t neighbours
    dt = xp.zeros(n, dtype=xp.float32)

    # convert tspan to seconds
    tspan = int(tspan * 60.0 * 60.0)

    # go through every point in time[] array.
    stride = 50
    for i in range(0, int(xp.ceil(n/stride))):
        p1_idx = xp.arange(stride*i, min(stride*(i+1), arr.shape[1]))
        # populate p1, points of interest
        p1 = arr[:, p1_idx]

        ###################
        # Spatial Average #
        ###################

        idx_shape = (p1_idx.shape[0], arr.shape[1])
        idx_shape_tr = (arr.shape[1], p1_idx.shape[0])

        # find indices of all points within the timespan for spatial average
        p2_idx = (xp.broadcast_to(arr[0, :], idx_shape) < xp.transpose(xp.broadcast_to(arr[0, p1_idx] + tspan/2, idx_shape_tr))) & (xp.broadcast_to(arr[0, :], idx_shape) > xp.transpose(xp.broadcast_to(arr[0, p1_idx] - tspan/2, idx_shape_tr)))
        # populate p2 with those points
        p2_lat = xp.broadcast_to(arr[1, :], p2_idx.shape)
        p2_lon = xp.broadcast_to(arr[2, :], p2_idx.shape)
        # get haversine distances between the point of interest (p1) and all points within tspan hours (p2)
        # sort the distances and calculate median from the nearest di_r
        # drs = xp.partition(xp.asarray(haversine_arrayed(p1[1:, :], p2_lat*p2_idx, p2_lon*p2_idx, r)), min(p2_idx.shape[1]-1, di_r))
        drs = xp.sort(xp.asarray(haversine_arrayed(p1[1:, :], p2_lat*p2_idx, p2_lon*p2_idx, r)))

        dr[p1_idx] = xp.median(drs[0:di_r])

        ####################
        # Temporal Average #
        ####################

        # indices of the earliest and latest points to use in calculating the temporal average
        earliest = xp.maximum(0, p1_idx - di_t)
        latest = xp.minimum(n-1, p1_idx + di_t)

        # calc median time distance of nearest 2*di_t points
        # for j in range(min(stride, p1_idx.shape[0])):
        #     dt[p1_idx[j]] = xp.median(xp.abs(arr[0, p1_idx[j]] - arr[0, earliest[j]:latest[j]]))
            # dt[p1_idx[j]] = xp.median(xp.abs((xp.broadcast_to(arr[0, p1_idx], (arr[0, :].shape[0], stride)) - xp.transpose(xp.broadcast_to(arr[0, :], (stride, arr[0,:].shape[0]))))[earliest[j]:latest[j], j]), axis=0)

    if CUDA:
        dr = dr.get()
        dt = dt.get()

    # minimum temporal and spatial distances
    dt[dt < 1] = 1
    if dr.any():
        dr[dr == 0] = min(dr[dr > 0])
    return dr, dt


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

def haversine_arrayed(p1, p2_lat, p2_lon, r):
    """

    Parameters
    ----------
    p1 : 2d array of points [latitude, longitude]
    p2_lat : 2d array of points where each row represents latitudes corresponding to a point in p1. values of 0 will not be counted
    p2_lon : 2d array of points where each row represents longitudes corresponding to a point in p1. values of 0 will not be counted
    r  : distance above surf of earth to calculate (in km)

    Returns
    -------
    Haversine distances between each point in p2 and p1
    """
    # distance from earth's centre
    R = 6371.0 + r
    n = p1.shape[1]
    m = p2_lat.shape[1]
    lat_broadcast_p1 = np.broadcast_to(p1[0, :], (m, n))
    lon_broadcast_p1 = np.broadcast_to(p1[1, :], (m, n))
    lat_broadcast_p2 = np.transpose(p2_lat)
    lon_broadcast_p2 = np.transpose(p2_lon)

    delta_lat = lat_broadcast_p1 - lat_broadcast_p2
    delta_lon = lon_broadcast_p1 - lon_broadcast_p2
    prod_cos = xp.cos(xp.deg2rad(lat_broadcast_p2)) * xp.cos(xp.deg2rad(lat_broadcast_p1))

    a = (xp.sin(xp.deg2rad(delta_lat/2))) ** 2 + prod_cos * ((xp.sin(xp.deg2rad(delta_lon/2))) ** 2)
    b = 2 * R * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))
    # todo: check this return
    return b[lat_broadcast_p1 != 0]



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


if __name__ == '__main__':
    import h5py
    # filepath = '/run/media/arl203/Seagate Expansion Drive/backup/level2b/ib3d_normal_swht_2021_03_20_prelate_bakker.h5'  # mucho
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_31_prelate_bakker.h5'  # medium
    filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_15_prelate_bakker.h5'  # mini
    f = h5py.File(filepath, 'r')
    la = f['data']['latitude'][()]
    lo = f['data']['longitude'][()]
    ti = f['data']['time'][()]
    arr = xp.array([ti, la, lo])
    print(arr.shape)

    # Do this just to get the algos compiled with numba.njit
    # cluster_medians(arr[:, 0:10])

    print(cupyx.time.repeat(cluster_medians, (arr,), n_repeat=1))
    start = time.perf_counter()
    dr, dt = cluster_medians(arr)
    end = time.perf_counter()
    print(f'time: {end - start}')
    print(dr.shape, dt.shape)

    # Time to beat:
    # 1) cupy 8.2578 s, numpy 0.8100 s -- 7000 pts
    # 2) cupy 13.221 s, numpy 1.312 s -- 10_000 pts
    # 3) cupy 8.3983 s, numpy 1.1332 s -- 10_000 pts
    # 4) cupy 7.0982 s, numpy 1.1332 s -- 10_000 pts
    # 5) numba 0.032 s -- 10_000 pts
    # 6) numba 1.974 s -- 100_000 pts
    # 7) numba 149.231 s -- 976_385 pts (2021_03_31)

    # Partition vs Sort :
    # 1) partition 2.3380 s,   sort 2.2743 s   -- 7000 pts
    # 2) partition 11.7929 s,  sort 11.5286 s  -- 38_000 pts
    # 3) partition 626.9669 s, sort 300.9675 s -- 976_385 pts


# 7000 pts
# sort:
# cluster_medians     :    CPU:2674544.507 us   +/-8741.112 (min:2664067.086 / max:2690357.249) us     GPU-0:2674547.266 us   +/-8742.678 (min:2664068.604 / max:2690363.281) us
# time: 2.668013966991566
# partition:
# cluster_medians     :    CPU:2672041.370 us   +/-1928.972 (min:2668310.690 / max:2675187.748) us     GPU-0:2672043.774 us   +/-1927.995 (min:2668317.627 / max:2675189.453) us
# time: 2.6811051650438458

# 38_000 pts
# sort:
# cluster_medians     :    CPU:26636083.663 us   +/-23450.148 (min:26600045.357 / max:26676676.763) us     GPU-0:26636036.719 us   +/-23449.354 (min:26600007.812 / max:26676632.812) us
# time: 26.78234336094465
# partition:
# cluster_medians     :    CPU:26780097.614 us   +/-117236.884 (min:26718996.304 / max:27124543.236) us     GPU-0:26780065.430 us   +/-117231.618 (min:26719011.719 / max:27124500.000) us
# time: 27.48400734900497


# 38_000 points with time median
#
# stride=1
# cluster_medians     :    CPU:47480225.760 us     GPU-0:47480113.281 us
# time: 47.207301859976724
#
# stride=10
# cluster_medians     :    CPU:26636083.663 us     GPU-0:26636036.719 us
# time: 26.78234336094465
#
# stride=100
# cluster_medians     :    CPU:24408526.812 us     GPU-0:24408474.609 us
# time: 24.406322172028013
#
# stride=500
# cluster_medians     :    CPU:26496091.794 us     GPU-0:26496078.125 us
# time: 29.23689841502346
#
# stride=1000
# cluster_medians     :    CPU:24613840.061 us     GPU-0:24613798.828 us
# time: 26.125514063052833


# 38_000 points without time median

# stride=1
# cluster_medians     :    CPU:38408332.244 us     GPU-0:38408250.000 us
# time: 38.438213059911504

# stride=100
# cluster_medians     :    CPU:16474604.008 us     GPU-0:16474585.937 us
# time: 16.4767437040573

# stride=500
# cluster_medians     :    CPU:16567731.881 us     GPU-0:16567730.469 us
# time: 16.567086618044414

# stride=1000
# cluster_medians     :    CPU:16939801.421 us     GPU-0:16939783.203 us
# time: 16.94057455798611