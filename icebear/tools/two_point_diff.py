import time
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False


def point_diff(p1, p2, mode='lower'):
    """
    Finds the difference between every element of array p1 and p2, must be the same shape.
    For the difference of each element in an array to each other element give args p1 = p2.
    This function will attempt to use cupy for CUDA processing if available, this  is faster for arrays
    larger than 5_000.

    Parameters
    ----------
    p1 : float32 ndarray
        1d array of floats.
    p2 : float32 ndarray
        1d array of floats.
    mode : str
        Select return mode
            - 'lower' (default) is p2 - p1
            - 'upper' is p1 - p2
            - 'both' is p2 - p1 then appended p1 - p2, no trace
            - 'full' is the full array flattened

    Returns
    -------
    c : float32 ndarray
        1d array of differences between p1 and p2.
    """
    n = p1.shape[0]
    c = xp.tile(p1, n).reshape(n, n) - xp.tile(p2, n).reshape(n, n).T
    if CUDA:
        c = xp.asnumpy(c)
    if mode == 'lower':
        return c[np.tril_indices(n, k=-1)]
    if mode == 'upper':
        return c[np.triu_indices(n, k=1)]
    if mode == 'both':
        c1 = c[np.tril_indices(n, k=-1)]
        c2 = c[np.triu_indices(n, k=1)]
        return np.concatenate(c1, c2)
    if mode == 'full':
        return c.flatten()


def haversine(p1, p2, bins, r=110.0):
    """
    Find the haversine product for all the two-point differences of the input latitude-longitude points.
    Parameters
    ----------
    p1 : float32 ndarray
        [(lat, lon), (), ...] list of latitudes and longitudes in radians.
    p2 : float32 ndarray
        [(lat, lon), (), ...] list of latitudes and longitudes in radians.
    r : float32
        Altitude in kilometers above Earth surface of assumed shell.

    Returns
    -------
    result : float32 ndarray
        Haversine product of the list of points.
    """
    r += 6371.0
    n = p1.shape[0]
    m = p2.shape[0]
    lat1 = np.array(p1[:, 0], dtype=np.float32)
    lon1 = np.array(p1[:, 1], dtype=np.float32)
    lat2 = np.array(p2[:, 0], dtype=np.float32)
    lon2 = np.array(p2[:, 1], dtype=np.float32)
    prod_cos = np.cos(xp.tile(p1[:, 0], n).reshape(n, n)) * np.cos(np.tile(p2[:, 0], m).reshape(m, m)).T
    prod_cos = prod_cos[np.tril_indices(prod_cos.shape[0], k=-1)]
    delta_lat = point_diff(lat1, lat2, mode='lower')
    delta_lon = point_diff(lon1, lon2, mode='lower')
    a = np.sin(delta_lat/2)**2 + prod_cos * np.sin(delta_lon/2)**2
    b = r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    h, _ = np.histogram(b, bins)
    return h


def clustering_chunk(p1, p2, bins, mode='upper', r=110.0, max_chunk=2048):
    """
    Finds the difference between every element of latitude-longitude point arrays p1 and p2 using the
    Haversine formula. This function is optimized for CUDA processing. The function uses chunking
    to allow for very larger array lengths. For the difference of each element in an array to each other
    element give args p1 == p2.

    Parameters
    ----------
    p1 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude floats with shape (n, 2).
    p2 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude floats with shape (m, 2).
    bins : float32 ndarray
        1d array of floats containing the bin edges for binning. Example bins=[0, 1, 5] has two bins, the
        ranges 0 <= x <= 1 and 1 <= x <= 5
    mode : str
        Select return mode
            - 'upper' (default) is p1 - p2
            - 'lower' is p2 - p1
            - 'full' is the full array flattened
    r : float32
        Altitude in kilometers above Earth surface of assumed shell.
    max_chunk : float32
        1/2 to 1/4 the maximum memory in Bytes the GPU can handle (default 2048 Bytes).

    Returns
    -------
    h : float32 ndarray
        1d array of two-point differences binned.
    """
    r += 6371.0
    h = np.zeros(bins.shape[0]-1, dtype=int)
    lat1 = xp.array(p1[:, 0])[:, xp.newaxis]
    lon1 = xp.array(p1[:, 1])[:, xp.newaxis]
    lat2 = xp.array(p2[:, 0])[xp.newaxis, :]
    lon2 = xp.array(p2[:, 1])[xp.newaxis, :]
    chunks = int(xp.ceil((lat1.size * lat1.itemsize) / max_chunk))
    chunk_size = int(p1.shape[0] / chunks)
    if chunks * chunk_size < lat1.shape[0]:
        chunks += 1
    print(f'chunks: {chunks}, chunk size: {chunk_size}, array length:{lat1.shape[0]}, '
          f'chunks * chunk size = array length: {chunks * chunk_size}')
    for i in range(chunks):
        print(f'\tcomputing chunk {i}/{chunks}')
        start = i * chunk_size
        end = start + chunk_size
        delta_lat = lat1[start:end, :] - lat2
        delta_lon = lon1[start:end, :] - lon2
        prod_cos = xp.cos(lat1[start:end, :]) * xp.cos(lat2)

        if mode == 'upper':
            k = 1 + i * chunk_size
            delta_lat = delta_lat[xp.triu_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[xp.triu_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[xp.triu_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'lower':
            k = -1 - i * chunk_size
            delta_lat = delta_lat[xp.tril_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[xp.tril_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[xp.tril_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'both':
            delta_lat = delta_lat
            delta_lon = delta_lon
            prod_cos = prod_cos
        else:
            print(f'mode: {mode} is not an accepted mode choose; lower, upper, or both')

        a = xp.sin(delta_lat / 2) ** 2 + prod_cos * xp.sin(delta_lon / 2) ** 2
        b = r * 2 * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))
        if CUDA:
            b = xp.asnumpy(b)
        c, _ = np.histogram(b, bins)
        h += c
    return h


if __name__ == '__main__':
    N = 200_000
    # arr = np.random.random(2*N).reshape((N, 2))
    arr = np.array([np.arange(N), np.arange(N)]).T
    bins = np.array([0.0, 100.0, 1000.0, 10_000.0])
    limit = 2000
    if CUDA:
        pool = xp.get_default_memory_pool()
        x = xp.arange(1000)
        limit = int(pool.total_bytes()/2)
        del x
    ts = time.time()
    h = clustering_chunk(arr, arr, bins, mode='both', r=110, max_chunk=limit)
    print('chunking time: ', time.time() - ts)
    print('result: ', h)

    # ts = time.time()
    # h = haversine(arr, arr, bins)
    # print('array time: ', time.time() - ts)
    # print('result: ', h)


