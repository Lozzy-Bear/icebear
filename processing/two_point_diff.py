import time
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False


def rescale(arr, lower, upper):
    """
    Rescales the input 1d array to the lower and upper boundary. This is used to widen a
    statistical distribution centered about some point to a new boundary.

    Parameters
    ----------
    arr : float32 ndarray
        1d array of points to be rescaled
    lower : float32
        lower boundary point to scale to
    upper : float32
        upper boundary point to scale to

    Returns
    -------
    arr : float32 ndarray
        1d array of points rescaled to within the lower and upper boundary
    """
    arr_min = xp.min(arr)
    arr_max = xp.max(arr)
    arr = lower + ((arr - arr_min) / (arr_max - arr_min)) * (upper - lower)
    return arr


def poisson_points(lat, lon, pad_lat=6.0, pad_lon=12.0, multiplier=2, scaler=1.0e5):
    """
    Generate a 2d poisson distribution centered about the mean of the input latitude and
    longitude array. The output distribution is scaled to a boundary the same size as the
    input arrays, less extreme outliers, and increased by some pad angle.
    The number out points returned by default is twice as many as input.

    Parameters
    ----------
    lat : float32 ndarray
    lon : float32 ndarray
    pad_lat : float32
        amount in degrees to widen the poisson distribution by beyond the bulk latitudes
    pad_lon : float32
        amount in degrees to widen the poisson distribution by beyond the bulk longitudes
    multiplier : float32
        the multiple of more/less points to create over the input number
    scaler : float32
        a value to scale up the poisson centers by; this is required to increase the
        granularity of output as np.random.poisson() outputs integers only.

    Returns
    -------
    arr: float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of poisson distributed latitude and longitude
        points in degrees with shape (lat.shape[0] * multiplier, 2).
    """
    mean_lat = np.median(lat) * scaler  # todo: prefer using the Gaussian_KDE peak but need to write CUDA code for it
    mean_lon = np.median(lon) * scaler
    # np.random.poisson() requires positive numbers and return int, need to scale up input
    # and ensure positive values then scale down and preserve sign after.
    arr = np.random.poisson([np.abs(mean_lat), np.abs(mean_lon)],
                            size=(int(lat.shape[0] * multiplier), 2)) / scaler
    arr[:, 0] = arr[:, 0] * np.sign(mean_lat)
    arr[:, 1] = arr[:, 1] * np.sign(mean_lon)
    arr[:, 0] = rescale(arr[:, 0],
                        np.percentile(lat, 0.01) - pad_lat,
                        np.percentile(lat, 99.99) + pad_lat)
    arr[:, 1] = rescale(arr[:, 1],
                        np.percentile(lon, 0.01) - pad_lon,
                        np.percentile(lon, 99.99) + pad_lon)
    return arr


def clustering_chunk(p1, p2, bins, mode='upper', r=110.0):
    """
    Finds the difference between every element of latitude-longitude point arrays p1 and p2 using the
    Haversine formula. This function is optimized for CUDA processing. The function uses chunking
    to allow for very larger array lengths. For the difference of each element in an array to each other
    element give args p1 == p2.

    Parameters
    ----------
    p1 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude (radians) floats with shape (n, 2).
    p2 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude (radians) floats with shape (m, 2).
    bins : float32 ndarray
        1d array of (km) floats containing the bin edges for binning. Example bins=[0, 1, 5] has two bins, the
        ranges 0 <= x <= 1 and 1 <= x <= 5
    mode : str
        Select return mode
            - 'lower' (default) is p2 - p1
            - 'upper' is p1 - p2, this takes slightly more VRAM than lower
               (doing -1 * lower is suggested for p2.shape = p1.shape).
            - 'full' is the full array flattened
    r : float32
        Altitude in kilometers (km) above Earth surface of assumed shell.

    Returns
    -------
    h : float32 ndarray
        1d array of two-point differences binned.
    """
    r += 6371.0
    bins = xp.asarray(bins)
    h = xp.zeros(bins.shape[0]-1, dtype=float)
    lat1 = xp.array(p1[:, 0])[:, xp.newaxis]
    lon1 = xp.array(p1[:, 1])[:, xp.newaxis]
    lat2 = xp.array(p2[:, 0])[xp.newaxis, :]
    lon2 = xp.array(p2[:, 1])[xp.newaxis, :]

    # Figure out chunk sizes from input sizes
    x = int((xp.cuda.Device().mem_info[0] - (2 * bins.nbytes + p1.nbytes + p2.nbytes))
            / (3 * p2.shape[0] * p1.itemsize))
    chunk_size = int(x/4)  # MUST divide by 4 to allow for intermediate memory expansion in calculations
    chunks = int(p1.shape[0] / chunk_size)
    if chunks * chunk_size < lat1.shape[0]:
        chunks += 1
    print(f'\t\tchunks: {chunks}, chunk length: {chunk_size}, array length:{lat1.shape[0]}, '
          f'chunks * chunk size = array length: {chunks * chunk_size}')

    # print(f'\t\ttotal available VRAM = {xp.cuda.Device().mem_info[0] / 1e6} MiB')
    # print(f'\t\ttotal bytes without chunking = '
    #       f'{(2 * bins.nbytes + p1.nbytes + p2.nbytes + 3 * (p1.shape[0] * p2.shape[0] * p1.itemsize)) / 1e6} MiB')
    # print(f'\t\tarray length for chunking = {x/4}')
    # print(f'\t\ttotal bytes per chunk = '
    #       f'{(2 * bins.nbytes + p1.nbytes + p2.nbytes + 3 * (x * p2.shape[0] * p1.itemsize)) / 1e6} MiB')

    for i in range(chunks):
        # print(f'\t\tcomputing chunk {i}/{chunks}')
        start = i * chunk_size
        end = start + chunk_size
        delta_lat = lat1[start:end, :] - lat2
        delta_lon = lon1[start:end, :] - lon2
        prod_cos = xp.cos(lat1[start:end, :]) * xp.cos(lat2)
        if mode == 'lower':
            # Todo: When CUPY v11 includes tril_indices() in the build revert and remove this function.
            def tril_indices(n, k=0, m=None):
                tri_ = xp.tri(n, m, k=k, dtype=bool)
                return tuple(xp.broadcast_to(inds, tri_.shape)[tri_] for inds in xp.indices(tri_.shape, dtype=int))
            k = -1 - i * chunk_size
            delta_lat = delta_lat[tril_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[tril_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[tril_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'upper':
            # Todo: When CUPY v11 includes triu_indices() in the build revert and remove this function.
            def triu_indices(n, k=0, m=None):
                tri_ = ~xp.tri(n, m, k=k - 1, dtype=bool)
                return tuple(xp.broadcast_to(inds, tri_.shape)[tri_] for inds in xp.indices(tri_.shape, dtype=int))
            k = 1 + i * chunk_size
            delta_lat = delta_lat[triu_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[triu_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[triu_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'full':
            delta_lat = delta_lat.flatten()
            delta_lon = delta_lon.flatten()
            prod_cos = prod_cos.flatten()
        else:
            print(f'mode: {mode} is not an accepted mode choose; lower, upper, or full')
        prod_cos = xp.sin(delta_lat / 2) ** 2 + prod_cos * xp.sin(delta_lon / 2) ** 2
        prod_cos = r * 2 * xp.arctan2(xp.sqrt(prod_cos), xp.sqrt(1 - prod_cos))
        c, _ = xp.histogram(prod_cos, bins)
        h += c
    if CUDA:
        return h.get()
    else:
        return h


if __name__ == '__main__':
    N = 1_000_000
    arr = np.array([np.arange(N), np.arange(N)]).T
    bins = np.arange(0, N, 134)
    ts = time.time()
    h = clustering_chunk(arr, arr, bins, mode='upper', r=110)
    print('chunking time: ', time.time() - ts)
    print('result: ', h)
