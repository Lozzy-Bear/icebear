import h5py
import skimage.measure as image
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False


def find_nearest(array, value):
    # Returns the index of array that is closest to value
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def beam_finder(data_lat, data_lon, beam_pattern):
    if beam_pattern in ['1lam', '3lam']:
        # load gain data for beam pattern
        with h5py.File("/mnt/icebear/processing_code/icebear/dat/ib3d_link_gain_mask_" + beam_pattern + "_rot13.h5", 'r') as f:
            alt_b = f['altitude'][:]
            gain_b = f['gain_mask'][:]
            lat_b = f['latitude'][:]
            lon_b = f['longitude'][:]
            # Note: gain is indexed as (lat, lon, alt, 0)
        alt_b = alt_b / 1000
        gain_b = np.nan_to_num(gain_b, nan=0.0)
    else:
        return np.ones_like(data_lat) * np.NaN  # Return empty beam array if selected beam pattern is not defined

        # Mask the closest latitudes to the radar (check with Adam or Magnus about if this is reasonable)
    gain_b[0:21, :, :, 0] = 0.0

    # Get gain slice closest to 90 km
    alt_val = 90
    alt_idx = find_nearest(alt_b, alt_val)

    # Grab gain slice at selected altitude and mask the data
    # Currently using mask of 15 dB in beam pattern gain
    # All gain values lower than 15 dB are set to 0
    # Although, I believe image.label requires g to be an array of integers
    mask = 15
    g = gain_b[:, :, alt_idx, 0]
    g = np.where(g < mask, 0, g)
    g = np.where(g >= mask, 30, g)
    l = image.label(g)

    beam = np.zeros_like(data_lat)
    # for every data point in the data set
    for i in range(len(data_lat)):
        # Find the index of the gain pattern lat and long values that is closest to
        # the data point the loop is assigning a beam number to.
        x = find_nearest(lat_b, data_lat[i])
        y = find_nearest(lon_b, data_lon[i])

        # assign beam number (0 if outside beam, 1-3 for the 3 main beams in 3lam)
        # 3-lam I believe it will label east as 1, center as 2, and west as 3. Adjust as needed
        # 1-lam will label the beam as 1, adjust as needed
        beam[i] = l[x, y]

    # relabel outside beam as -1
    beam[beam == 0] = -1
    return beam


def beam_finder_six_lines(azimuth, elevation):
    """
    Classifies each point into a beam corresponding to the 3 lambda Icebear transmitter configuration.

    Parameters
    ----------
    azimuth : 1D ndarray
        Vector array of azimuth values
    elevation : 1D ndarray
        Vector array of elevation values
    Returns
    -------
    beam : 1D ndarray
        Same shape as the input arrays. Classifies each point into a beam.
        Possible values:
            -1 : the corresponding point is not in a beam
            1  : the corresponding point is in the east beam
            2  : the corresponding point is in the centre beam
            3  : the corresponding point is in the west beam

    """

    # linear model val(x) = p1*x + p2
    #              el(az) = p1*az + p2
    beam = np.zeros(len(azimuth))
    coeffs = np.array([[-0.3906, 0.1125],  # west0 coefficients [p1, p2]
                       [-0.3814, 5.934],   # west1      "          "
                       [-0.5422, 8.47],    # centre0    "          "
                       [-0.6575, 19.79],   # centre1    "          "
                       [-1.038, 32.45],    # east0      "          "
                       [-1.229, 59.03]])   # east1      "          "

    # discretize into azimuth bins
    azimuth_bins = np.arange(-60.0, 60.0, 0.5)
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
            # numbering to match Magnus'
            if J0.any():
                temp_beam[WEST] = 3
                temp_beam[CENTRE] = 2
                temp_beam[EAST] = 1
                beam[cur_idx] = temp_beam

    beam[beam == 0] = -1
    return beam


def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.

    Parameters
    ----------
    ndarray : ndarray
        The input ndarray
    window_len : int
        The window length (filter length)
    step : int
        The step size (decimation rate)

    Returns
    -------
     ndarray
        The array ndarray with a new view.
    """

    nrows = ((ndarray.shape[-1] - window_len) // step) + 1
    last_dim_stride = ndarray.strides[-1]
    new_shape = ndarray.shape[:-1] + (nrows, window_len)
    new_strides = list(ndarray.strides + (last_dim_stride,))
    new_strides[-2] *= step

    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


def cluster_medians(arr, k, r=105.0, tspan=4, di_r=512):
    """
    Calculates the spatial and temporal medians for each point in the array arr
    using cupy optimization. If necessary, chunks the dataset to ensure the GPU doesn't run out of memory.

    Parameters
    ----------
    arr : float32 cupy ndarray
        array of points [time, latitude, longitude]
            - time is in units of seconds and sorted earliest to latest
            - latitude and longitude in units of degrees
    k : int
        1/3 number of points in each chunk. Interpreted as the 1/2 the maximum number of points occurring within tspan
    r  : float32
        distance above surface of earth in km to calculate haversine (default 110)
    tspan : float32
        total time in hours to window the spatial median (default 4)
    di_r : int
        number of points centred on the point of interest used to calculate the medians (default 512)

    Returns
    -------
    dr: float32 numpy ndarray
        1d array of median spatial distances
    dt: float32 numpy ndarray
        1d array of median temporal distances
    """

    def calculate_medians(arr, r=105.0, tspan=4, di_r=512):
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

    def calculate_medians_chunky(arr, k, r=105.0, tspan=4, di_r=512):
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
    if m <= 1:  # no chunking necessary
        dr, dt = calculate_medians(arr)
    else:  # have to chunk
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


