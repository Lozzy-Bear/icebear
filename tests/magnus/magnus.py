import time
import numba
import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False
import cProfile
import sys

def beam_finder(azimuth, elevation):
    # linear model val(x) = p1*x + p2
    beam = np.zeros(len(azimuth))
    coeffs = np.array([[-0.3906, 0.1125],  # west0 coefficients [p1, p2]
                       [-0.3814, 5.934],   # west1      "          "
                       [-0.5422, 8.47],    # centre0    "          "
                       [-0.6575, 19.79],   # centre1    "          "
                       [-1.038, 32.45],    # east0      "          "
                       [-1.229, 59.03]])   # east1      "          "

    # discretize into azimuth bins
    azimuth_bins = np.arange(-60.0, 60.0, 0.5)
    # todo: these centres are funky. first one is -60.25 instead of -59.75. check with Magnus, but data matches for now
    azimuth_centres = azimuth_bins - 0.25
    data_bins = np.digitize(azimuth, azimuth_bins)

    for data_bin in range(len(azimuth_bins)):
        # indices of elements matching current bin
        cur_idx = (data_bins == data_bin)
        if cur_idx.any():
            # temporary elevation and beam vars
            temp_elevation = elevation[cur_idx]
            temp_beam = beam[cur_idx]

            # using linear functions to cut out unnecessary data
            el_west0   = (coeffs[0][0] * azimuth_centres[data_bin] + coeffs[0][1])
            el_west1   = (coeffs[1][0] * azimuth_centres[data_bin] + coeffs[1][1])
            el_centre0 = (coeffs[2][0] * azimuth_centres[data_bin] + coeffs[2][1])
            el_centre1 = (coeffs[3][0] * azimuth_centres[data_bin] + coeffs[3][1])
            el_east0   = (coeffs[4][0] * azimuth_centres[data_bin] + coeffs[4][1])
            el_east1   = (coeffs[5][0] * azimuth_centres[data_bin] + coeffs[5][1])

            WEST   = (el_west0   < temp_elevation) & (temp_elevation < el_west1)
            CENTRE = (el_centre0 < temp_elevation) & (temp_elevation < el_centre1)
            EAST   = (el_east0   < temp_elevation) & (temp_elevation < el_east1)

            J0 = (EAST.any() or CENTRE.any()) or WEST.any()
            # numbering to match Magnus
            # if not in a beam, set to 0
            if J0.any():
                temp_beam[WEST] = 3
                temp_beam[CENTRE] = 2
                temp_beam[EAST] = 1
                beam[cur_idx] = temp_beam
    return beam


def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.
    :param      ndarray:     The input ndarray
    :type       ndarray:     ndarray
    :param      window_len:  The window length(filter length)
    :type       window_len:  int
    :param      step:        The step(dm rate)
    :type       step:        int
    :returns:   The array with a new view.
    :rtype:     ndarray
    """

    nrows = ((ndarray.shape[-1] - window_len) // step) + 1
    last_dim_stride = ndarray.strides[-1]
    new_shape = ndarray.shape[:-1] + (nrows, window_len)
    new_strides = list(ndarray.strides + (last_dim_stride,))
    new_strides[-2] *= step

    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


def cluster_medians(arr, k, r=110.0, tspan=4, di_r=512):
    """
    Calculates the spatial and temporal medians for each point in the array arr
    using CuPy / CUDA optimization.

    Parameters
    ----------
    arr : float32 ndarray
        array of points [time, latitude, longitude]
            - time is in units of seconds and sorted earliest to latest
            - latitude and longitude in units of degrees
    k : float32
        1/3 number of points in each chunk. Interpreted as the 1/2 the maximum number of points occurring within tspan
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

    def calculate_medians(arr, r=110.0, tspan=4, di_r=512):
        n = arr.shape[1]
        di_t = int(di_r / 2)
        tspan = int(tspan * 60.0 * 60.0)
        dr = xp.zeros(n, dtype=xp.float32)
        dt = xp.zeros(n, dtype=xp.float32)
        stride = 25 # 25ish seems optimal

        # window.shape = (3, ~n/stride, stride)
        window = windowed_view(arr, window_len=stride, step=stride)
        # time_window[0, 0:2*di_t] corresponds to 2*di_t points around idx=255,
        # time_window[1, 0:2*di_t] corresponds to 2*di_t points around idx=256, etc
        time_window = windowed_view(arr[0, :], window_len=min(n, 2*di_t), step=1)

        for i in range(window.shape[1]):
            # points of interest
            idx = xp.arange(stride*i, min(stride*(i+1), stride*i + window.shape[2]))

            # Spatial Median

            # pointspan is tspan/2 hours before the first point, tspan/2 hours after the last point
            pointspan = (arr[0, :] < (window[:, i, :])[0, -1] + tspan / 2) & (arr[0, :] > (window[:, i, :])[0, 0] - tspan / 2)

            # dr[idx] = xp.median(xp.partition(haversine(window[:, i, :], arr[:, pointspan], r), kth=di_r, axis=1)[:, 0:di_r])
            dr[idx] = xp.median(xp.sort(haversine(window[:, i, :], arr[:, pointspan], r))[:, 0:di_r], axis=1)

            # Temporal Median

            # if idx contains only points more than di_t away from the edges, we can use the time_window
            if ((di_t < idx) & (idx < n - di_t)).all():
                dt[idx] = xp.median(xp.abs(window[0, i, :][:, xp.newaxis] - time_window[idx-di_t, :]), axis=1)

            # otherwise, manually do edge cases
            else:
                earliest = xp.maximum(0, idx - di_t)
                latest = xp.minimum(n, idx + di_t)
                for j in range(idx.shape[0]):
                    dt[idx[j]] = xp.median(xp.abs(window[0, i, j] - arr[0, earliest[j]:latest[j]]))

        # minimum values
        if xp.any(dr[dr > 0]):
            dr[dr == 0] = xp.min(dr[dr > 0])
        dt[dt < 1] = 1

        return dr, dt

    def calculate_medians_chunky(arr, k, r=110.0, tspan=4, di_r=512):
        n = arr.shape[1]
        di_t = int(di_r / 2)
        tspan = int(tspan * 60.0 * 60.0)
        dr = xp.zeros(n, dtype=xp.float32)
        dt = xp.zeros(n, dtype=xp.float32)
        stride = 25  # 25ish seems optimal

        # window.shape = (3, ~n/stride, stride)
        window = windowed_view(arr, window_len=stride, step=stride)
        # time_window[0, 0:2*di_t] corresponds to 2*di_t points around idx=255,
        # time_window[1, 0:2*di_t] corresponds to 2*di_t points around idx=256, etc
        time_window = windowed_view(arr[0, :], window_len=2*di_t, step=1)

        for i in range(window.shape[1]):
            # points of interest
            idx = xp.arange(stride*i, min(stride*(i+1), stride*i + window.shape[2]))

            # we only care about the center k points, don't need to calculate the rest
            if xp.amax(idx) >= k and xp.amin(idx) <= 2*k:
                # Spatial Median

                # pointspan is tspan/2 hours before the first point, tspan/2 hours after the last point
                pointspan = (arr[0, :] < (window[:, i, :])[0, -1] + tspan / 2) & (arr[0, :] > (window[:, i, :])[0, 0] - tspan / 2)
                dr[idx] = xp.median(xp.sort(haversine(window[:, i, :], arr[:, pointspan], r))[:, 0:di_r], axis=1)

                # Temporal Median

                # idx contains only points more than di_t away from the edges, so we can use the time_window
                if ((di_t < idx) & (idx < n - di_t)).all():
                    dt[idx] = xp.median(xp.abs(window[0, i, :][:, xp.newaxis] - time_window[idx-di_t, :]), axis=1)

        # minimum values
        if xp.any(dr[dr > 0]):
            dr[dr == 0] = xp.min(dr[dr > 0])
        dt[dt < 1] = 1

        return dr, dt

    dr = xp.zeros(arr.shape[1])
    dt = xp.zeros(arr.shape[1])
    m = int(arr.shape[1] / k)
    if m <= 1:
        dr, dt = calculate_medians(arr)
    else:
        dr_chunk, dt_chunk = calculate_medians(arr[:, 0 * k:2 * k])
        dr[0:k] = dr_chunk[0:k]
        dt[0:k] = dt_chunk[0:k]
        for i in range(m-2):
            dr_chunk, dt_chunk = calculate_medians_chunky(arr[:, k * i:k * (i + 3)], k)
            dr[k*(i+1):k*(i+2)] = dr_chunk[k:2 * k]
            dt[k*(i+1):k*(i+2)] = dt_chunk[k:2 * k]
        dr_chunk, dt_chunk = calculate_medians(arr[:, k * (m - 2):])
        dr[k*(m-1):] = dr_chunk[k:]
        dt[k*(m-1):] = dt_chunk[k:]

    if CUDA:
        dr = dr.get()
        dt = dt.get()
    return dr, dt



def haversine(p1, p2, r):
    """
    Calculates haversine distance between each point in p1 and every point in p2
    Parameters
    ----------
    p1 : float32 ndarray
        2d array of points with shape (3, n). [time, latitude, longitude]
    p2 : float32 ndarray
        2d array of points with shape (3, m). [time, latitude, longitude]
    r  : float32
        distance above surf of earth to calculate (in km)

    Returns
    -------
    dr: float32 ndarray
        2d array of haversine distances between each point in p2 and p1 with shape (n, m).
        dr[n, :] holds the distance between p1[n] and every point in p2

    """
    # distance from earth's centre
    r += 6371.0
    delta_lat = p1[1, :][:, xp.newaxis] - p2[1, :]
    delta_lon = p1[2, :][:, xp.newaxis] - p2[2, :]
    prod_cos = xp.cos(xp.deg2rad(p1[1, :]))[:, xp.newaxis] * xp.cos(xp.deg2rad(p2[1, :]))
    a = (xp.sin(xp.deg2rad(delta_lat / 2))) ** 2 + prod_cos * ((xp.sin(xp.deg2rad(delta_lon / 2))) ** 2)
    b = 2 * r * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))
    return b


def cluster_plot(fig_num, dt, dr):
    logbinsdt = np.logspace(np.log10(1), np.log10(max(dt)), 300)
    logbinsdr = np.logspace(np.log10(1), np.log10(max(dr)), 300)

    fig = plt.figure(fig_num)
    fig.suptitle('February 25, 2021. ' + str(dr.shape[0]) + ' echoes within the beams, ' + str(dr[dr > 40].shape[0]) + ' echoes above 40 km.', fontsize=14)
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
    ax1.set_xlabel('Temporal clustering [s]')
    ax1.set_ylabel('Spatial clustering [km]')
    ax1.set_xlim([1, 3e4])
    ax1.set_ylim([1, 1e3])

    ax2.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax2.hist(dt, bins=logbinsdt, rwidth=3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('Echoes/bin')
    ax2.sharex(ax1)

    ax3.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax3.hist(dr, bins=logbinsdr, orientation='horizontal', rwidth=3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Echoes/bin')
    ax3.sharey(ax1)
    plt.show()


if __name__ == '__main__':
    import h5py
    # filepath = str(sys.argv[1])
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_20_prelate_bakker.h5'  # 9_025_008
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_31_prelate_bakker.h5'  # 976_000
    filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_02_25_prelate_bakker.h5'
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_21_prelate_bakker.h5'  # 159_000
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_15_prelate_bakker.h5'  # 38_000
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_02_20_prelate_bakker.h5'  # Magnus Paper
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2020_11_10_prelate_bakker.h5'  # All meteors

    f = h5py.File(filepath, 'r')
    la = f['data']['latitude'][()]
    lo = f['data']['longitude'][()]
    ti = f['data']['time'][()]
    az = f['data']['azimuth'][()]
    el = f['data']['elevation'][()]
    al = f['data']['altitude'][()]
    start = time.perf_counter()
    beam = beam_finder(az, el)
    beam[al > 150] = 0
    beam[al < 70] = 0
    #beam = beam.get()
    end = time.perf_counter()
    print(f'beam time: {end - start}')
    arr = xp.array([ti, la, lo])
    east_arr = arr[:, beam == 1]
    centre_arr = arr[:, beam == 2]
    west_arr = arr[:, beam == 3]
    print(beam[beam==0].shape)
    #plt.scatter(east_arr.get()[1, :], east_arr.get()[2, :])
    #plt.scatter(centre_arr.get()[1, :], centre_arr.get()[2, :])
    #plt.scatter(west_arr.get()[1, :], west_arr.get()[2, :])
    #plt.show()
    #plt.figure(2)
    #plt.scatter(arr.get()[2, :], arr.get()[1, :])
    #plt.show()
    print(arr.shape, east_arr.shape, centre_arr.shape, west_arr.shape)

    # dr = np.zeros(arr.shape[1], dtype=np.float32)
    # dt = np.zeros(arr.shape[1], dtype=np.float32)
    k = 500_000
    n = int(arr.shape[1]/k)
    #start = time.perf_counter()
    #cProfile.run('cluster_medians(east_arr, k)')
    #cProfile.run('cluster_medians(centre_arr, k)')
    #cProfile.run('cluster_medians(west_arr, k)')
    #end = time.perf_counter()
    #print(f'cluster time: {end - start}')

    #dr1, dt1 = cluster_medians(east_arr, k)
    #dr2, dt2 = cluster_medians(centre_arr, k)
    #dr3, dt3 = cluster_medians(west_arr, k)
    sanitized_arr = arr[:, beam > 0]
    #cluster_plot(1, dt1, dr1)
    #cluster_plot(1, dt2, dr2)
    #cluster_plot(1, dt3, dr3)
    #dr = np.concatenate((dr1, dr2, dr3))
    #dt = np.concatenate((dt1, dt2, dt3))
    #cluster_plot(1, dt, dr)
    dr, dt = cluster_medians(sanitized_arr, k)
    cluster_plot(1, dt, dr)

    # beamfinder benchmarks (k=100_000)
    # 38_000    pts -- 78.0847 s (cupy), 0.06352 s (numpy)
    # 159_000   pts -- 0.1926 s (numpy)

    # benchmarks (k=100_000):
    # 38_000    pts -- 17.8968 s
    # 159_000   pts -- 342.5869 s
    # 976_000   pts -- 9838.8146 s (2.733 hr)

    # k = 50_000
    # 159_000 pts -- 404 s

    # k = 25_000
    # 159_000 pts -- 186 s
