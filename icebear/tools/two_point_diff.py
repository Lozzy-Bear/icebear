import time
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False


def clustering_chunk(p1, p2, bins, mode='upper', r=110.0, max_chunk=2048):
    """
    Finds the difference between every element of latitude-longitude point arrays p1 and p2 using the
    Haversine formula. This function is optimized for CUDA processing. The function uses chunking
    to allow for very larger array lengths. For the difference of each element in an array to each other
    element give args p1 == p2.

    Parameters
    ----------
    p1 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude [radians] floats with shape (n, 2).
    p2 : float32 ndarray
        [[lat, lon], [lat, lon], ...] 2d array of latitude and longitude [radians] floats with shape (m, 2).
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
        if CUDA:
            delta_lat = xp.asnumpy(delta_lat)
            delta_lon = xp.asnumpy(delta_lon)
            prod_cos = xp.asnumpy(prod_cos)
        if mode == 'upper':
            k = 1 + i * chunk_size
            delta_lat = delta_lat[np.triu_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[np.triu_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[np.triu_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'lower':
            k = -1 - i * chunk_size
            delta_lat = delta_lat[np.tril_indices(delta_lat.shape[0], k=k, m=delta_lat.shape[1])]
            delta_lon = delta_lon[np.tril_indices(delta_lon.shape[0], k=k, m=delta_lon.shape[1])]
            prod_cos = prod_cos[np.tril_indices(prod_cos.shape[0], k=k, m=prod_cos.shape[1])]
        elif mode == 'both':
            delta_lat = delta_lat
            delta_lon = delta_lon
            prod_cos = prod_cos
        else:
            print(f'mode: {mode} is not an accepted mode choose; lower, upper, or both')

        a = np.sin(delta_lat / 2) ** 2 + prod_cos * np.sin(delta_lon / 2) ** 2
        b = r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        c, _ = np.histogram(b, bins)
        h += c
    return h


if __name__ == '__main__':
    N = 20_000
    arr = np.array([np.arange(N), np.arange(N)]).T
    bins = np.array([0.0, 100.0, 1000.0, 10_000.0])
    limit = 2000
    if CUDA:
        pool = xp.get_default_memory_pool()
        x = xp.arange(1000)
        limit = int(pool.total_bytes()/2)
        del x
    ts = time.time()
    h = clustering_chunk(arr, arr, bins, mode='upper', r=110, max_chunk=limit)
    print('chunking time: ', time.time() - ts)
    print('result: ', h)
