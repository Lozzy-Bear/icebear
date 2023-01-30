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
        stride = 25  # 25ish seems optimal

        # window.shape = (3, ~n/stride, stride)
        window = windowed_view(arr, window_len=stride, step=stride)
        # time_window[0, 0:2*di_t] corresponds to 2*di_t points around idx=255,
        # time_window[1, 0:2*di_t] corresponds to 2*di_t points around idx=256, etc
        time_window = windowed_view(arr[0, :], window_len=2*di_t, step=1)

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

if __name__ == '__main__':
    import h5py
    filepath = str(sys.argv[1])
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_20_prelate_bakker.h5'  # 9_025_008
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_31_prelate_bakker.h5'  # 976_000
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_21_prelate_bakker.h5'  # 159_000
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_15_prelate_bakker.h5'  # 38_000
    f = h5py.File(filepath, 'r')
    la = f['data']['latitude'][()]
    lo = f['data']['longitude'][()]
    ti = f['data']['time'][()]

    arr = xp.array([ti, la, lo])
    print(arr.shape)

    # dr = np.zeros(arr.shape[1], dtype=np.float32)
    # dt = np.zeros(arr.shape[1], dtype=np.float32)
    k = 100_000
    n = int(arr.shape[1]/k)
    start = time.perf_counter()
    cProfile.run('cluster_medians(arr, k)')
    end = time.perf_counter()
    print(f'time: {end - start}')

    # benchmarks (k=100_000):
    # 38_000    pts -- 17.8968 s
    # 159_000   pts -- 342.5869 s
    # 976_000   pts -- 9838.8146 s (2.733 hr)

    # k = 50_000
    # 159_000 pts -- 404 s

    # k = 25_000
    # 159_000 pts -- 186 s

