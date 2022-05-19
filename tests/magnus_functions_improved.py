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

    arr[0, :] *= 24.0 * 60.0 * 60.0
    n = arr.shape[1]
    di_t = int(di_r / 2)
    tspan = int(tspan * 60.0 * 60.0)
    dr = xp.zeros(n, dtype=xp.float32)
    dt = xp.zeros(n, dtype=xp.float32)
    stride = 500

    # shape = (3, ~n/stride, stride)
    window = windowed_view(arr, window_len=stride, step=stride)
    # time_window[0, 0:2*di_t] corresponds to 2*di_t points around idx=255,
    # time_window[1, 0:2*di_t] corresponds to 2*di_t points around idx=256, etc
    time_window = windowed_view(arr[0, :], window_len=2*di_t + 1, step=1)

    for i in range(window.shape[1]):
        # points of interest
        idx = xp.arange(stride*i, min(stride*(i+1), stride*i + window.shape[2]))

        # Spatial Median

        # pointspan is tspan/2 hours before the first point, tspan/2 hours after the last point
        pointspan = (arr[0, :] < (window[:, i, :])[0, -1] + tspan / 2) & (arr[0, :] > (window[:, i, :])[0, 0] - tspan / 2)

        # I think partition should be faster than sort but in practice they seem to be about the same.
        dr[idx] = xp.median(xp.sort(haversine(window[:, i, :], arr[:, pointspan], r))[:, 0:di_r], axis=1)
        # Temporal Median

        # if idx contains only points more than di_t away from the edges, we can use the time_window
        if (xp.min(idx) > di_t) and (xp.max(idx) < n - di_t):
            dt[idx] = xp.median(xp.abs(window[0, i, :][:, xp.newaxis] - time_window[idx-di_t-1, :]), axis=1)

        # otherwise, manually do edge cases
        else:
            earliest = xp.maximum(0, idx - di_t)
            latest = xp.minimum(n-1, idx + di_t)
            for j in range(idx.shape[0]):
                dt[idx[j]] = xp.median(xp.abs(window[0, i, j] - arr[0, earliest[j]:latest[j]]))

    # minimum values
    if xp.any(dr[dr > 0]):
        dr[dr == 0] = xp.min(dr[dr > 0])
    dt[dt < 1] = 1

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
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_31_prelate_bakker.h5'  # 976_000
    # filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_21_prelate_bakker.h5'  # 159_000
    filepath = '/beaver/backup/level2b/ib3d_normal_swht_2021_03_15_prelate_bakker.h5'  # 38_000
    f = h5py.File(filepath, 'r')
    la = f['data']['latitude'][()]
    lo = f['data']['longitude'][()]
    ti = f['data']['time'][()]

    arr = xp.array([ti, la, lo])
    print(arr.shape)

    start = time.perf_counter()
    cProfile.run('cluster_medians(arr)')
    end = time.perf_counter()
    print(f'time: {end - start}')

    # Times (stride=100):
    # 1) cupy 0.8429  s, numpy 0.5449  s -- 38_000 pts
    # 2) cupy 4.4642  s, numpy 9.2786  s -- 159_000 pts
    # 3) cupy 29.8358 s, numpy 72.4494 s -- 976_000 pts

    # Times (stride=250):
    # 1) cupy 0.7161  s, numpy 0.9445  s -- 38_000 pts
    # 2) cupy 4.5760  s, numpy 10.6436 s -- 159_000 pts              winna winna chicken dinna
    # 3) cupy 22.6554 s, numpy 70.4448 s -- 976_000 pts

    # Times (stride=500):
    # 1) cupy 0.9408  s, numpy 1.6564  s -- 38_000 pts
    # 2) cupy 3.8720  s, numpy 13.0249 s -- 159_000 pts
    # 3) cupy 23.9578 s, numpy 80.5099 s -- 976_000 pts
