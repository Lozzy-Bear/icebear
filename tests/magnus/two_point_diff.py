import time
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False


def clustering_chunk(p1, p2, bins, mode='lower', r=110.0, max_chunk=2048):
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
            - 'lower' (default) is p2 - p1
            - 'upper' is p1 - p2, this takes slightly more VRAM than lower (doing -1 * lower is suggested).
            - 'full' is the full array flattened
    r : float32
        Altitude in kilometers above Earth surface of assumed shell.
    max_chunk : float32
        1/2 to 1/4 the maximum memory in MB the GPU can handle (default 2048 MB).

    Returns
    -------
    h : float32 ndarray
        1d array of two-point differences binned.
    """
    r += 6371.0
    h = xp.zeros(bins.shape[0]-1, dtype=float)
    lat1 = xp.array(p1[:, 0])[:, xp.newaxis]
    lon1 = xp.array(p1[:, 1])[:, xp.newaxis]
    lat2 = xp.array(p2[:, 0])[xp.newaxis, :]
    lon2 = xp.array(p2[:, 1])[xp.newaxis, :]
    chunks = int(np.ceil((lat1.size * lat1.itemsize) / max_chunk))
    chunk_size = int(p1.shape[0] / chunks)
    if chunks * chunk_size < lat1.shape[0]:
        chunks += 1
    print(f'chunks: {chunks}, chunk size: {chunk_size}, max_chunk: {max_chunk},\n'
          f'array length:{lat1.shape[0]}, chunks * chunk size = array length: {chunks * chunk_size}')
    for i in range(chunks):
        # print(f'\tcomputing chunk {i}/{chunks}')
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
        elif mode == 'both':
            delta_lat = delta_lat.flatten()
            delta_lon = delta_lon.flatten()
            prod_cos = prod_cos.flatten()
        else:
            print(f'mode: {mode} is not an accepted mode choose; lower, upper, or both')
        a = xp.sin(delta_lat / 2) ** 2 + prod_cos * xp.sin(delta_lon / 2) ** 2
        b = r * 2 * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))
        c, _ = xp.histogram(b, bins)
        h += c
    if CUDA:
        return h.get()
    else:
        return h


if __name__ == '__main__':
    N = 200_000
    arr = np.array([np.arange(N), np.arange(N)]).T
    bins = np.array([0.0, 100.0, 1000.0, 10_000.0, 100_000.0])
    limit = 2000
    if CUDA:
        pool = xp.get_default_memory_pool()
        x = xp.arange(1000)
        limit = int(pool.total_bytes()/2)
        del x
    ts = time.time()
    h = clustering_chunk(arr, arr, bins, mode='lower', r=110, max_chunk=limit)
    print('chunking time: ', time.time() - ts)
    print('result: ', h)
