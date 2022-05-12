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
import time as tm


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

    for data_bin in range(0, len(azimuth_bins)-1):
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

    # grab points and use GPU arrays
    time = xp.asarray(arr[0, :])
    lat = xp.asarray(arr[1, :])
    lon = xp.asarray(arr[2, :])

    # convert time to seconds
    time = time*24.0*60.0*60.0
    Nt = len(time)

    di_t = int(di_r / 2)

    # create GPU arrays

    # dr holds median spatial distances to nearest di_r neighbours (within tspan hour window)
    dr = xp.zeros(Nt, dtype=xp.float32)
    # dt holds median temporal distances to nearest 2*di_t neighbours
    dt = xp.zeros(Nt, dtype=xp.float32)
    # p1 will hold the point of interest : p1 = [time, lat, lon]
    p1 = xp.ndarray((3, 1), dtype=xp.float32)
    # p2 will hold the span of points around p1 when calculating spatial median
    p2 = xp.ndarray((3, Nt), dtype=xp.float32)

    # convert tspan to seconds
    tspan_seconds = int(tspan * 60.0 * 60.0)

    # go through every point in time[] array.
    for p1_idx in range(0, Nt):

        # populate p1, point of interest
        p1[0] = time[p1_idx]
        p1[1] = lat[p1_idx]
        p1[2] = lon[p1_idx]

        ###################
        # Spatial Average #
        ###################

        # find indices of all points within the timespan for spatial average
        p2_idx = (time < (time[p1_idx] + tspan_seconds / 2)) & (time > (time[p1_idx] - tspan_seconds / 2))

        # populate p2 with all points within that timespan and zero the rest of the points
        # todo: is there a better way to do this? does the remainder have to be zeroed?
        p2[0, 0:len(time[p2_idx])] = time[p2_idx]
        p2[1, 0:len(lat[p2_idx])] = lat[p2_idx]
        p2[2, 0:len(lon[p2_idx])] = lon[p2_idx]
        p2[:, len(time[p2_idx])+1:] = 0.0

        # get haversine distances between the point of interest (p1) and all points within tspan hours (p2)
        drs = xp.asarray(haversine(p1, p2[:, 0:len(time[p2_idx])], r))

        # calc median space distance of nearest di_r points
        drs = xp.sort(drs)
        dr[p1_idx] = xp.median(drs[0:di_r])

        ####################
        # Temporal Average #
        ####################

        # indices of the earliest and latest points to use in calculating the temporal average
        earliest = max(0, p1_idx - di_t)
        latest = min(len(time)-1, p1_idx + di_t)

        # calc median time distance of nearest 2*di_t points
        dt[p1_idx] = xp.median(abs(time[p1_idx] - time[earliest:latest]))

    # minimum temporal and spatial distances
    dt[dt < 1] = 1
    dr[dr == 0] = min(dr[dr > 0])

    if CUDA:
        return dr.get(), dt.get()
    else:
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
