import numpy as np


def uvw_to_rtp(u, v, w):
    """
    Converts u, v, w cartesian baseline coordinates to radius, theta, phi 
    spherical coordinates.
    Args:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.
    Returns:
        r (float np.array): Radius baseline coordinate divided by wavelength.
        t (float np.array): Theta (elevation) baseline coordinate.
        p (float np.array): Phi (azimuthal) baseline coordinate.
    """

    r = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    t = np.pi / 2 - np.arctan2(w, np.sqrt(u ** 2 + v ** 2))
    p = np.arctan2(v, u) + np.pi
    np.nan_to_num(t, copy=False)

    return r, t, p


def rtp_to_uvw(r, t, p):
    """
    Converts radius, theta, phi spherical baseline coordinates to u, v, w
    cartesian coordinates.
    Args:
        r (float np.array): Radius baseline coordinate divided by wavelength.
        t (float np.array): Theta (elevation) baseline coordinate.
        p (float np.array): Phi (azimuthal) baseline coordinate.
    Returns:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.
    """

    u = r * np.sin(t) * np.cos(p)
    v = r * np.sin(t) * np.sin(p)
    w = r * np.cos(t)

    return u, v, w


def baselines(filename, wavelength):
    """
    Given relative antenna positions in cartesian coordinates with units of meters
    and the wavelength in meters determines the u, v, w baselines in cartesian coordinates.
    Args:
        filename (string): File name of .csv for antenna cartersian coordinates in meters.
        wavelength (float): Radar signal wavelength in meters.
    Returns:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.
    Notes:
        * Given N antenna then M=N(N-1)/2 unique baselines exist.
        * M baselines can include conjugates and a origin baseline for M_total = M*2 + 1.
    Todo:
        * Makes options to include or disclude 0th baseline and conjugates.
        * Make array positions load from the calibration.ini file.
        * Error handling for missing antenna position values (like no z).
    """

    coords = np.loadtxt(filename, delimiter=",") / wavelength
    # Baseline for an antenna with itself.
    u = np.array([0])
    v = np.array([0])
    w = np.array([0])
    # Include all possible baseline combinations.
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            u = np.append(u, (coords[i, 0] - coords[j, 0]))
            v = np.append(v, (coords[i, 1] - coords[j, 1]))
            w = np.append(w, (coords[i, 2] - coords[j, 2]))
    # Include the conjugate baselines.
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            u = np.append(u, (-coords[i, 0] + coords[j, 0]))
            v = np.append(v, (-coords[i, 1] + coords[j, 1]))
            w = np.append(w, (-coords[i, 2] + coords[j, 2]))

    return u, v, w


def fov_window(coeffs, resolution=np.array([0.1, 0.1]),
               azimuth=np.array([225.0, 315.0]), elevation=np.array([90.0, 110.0])):
    """
    Set the field-of-view (fov) for a coefficents set. A narrower fov will result in
    faster runtimes.
    Args:
        coeffs (complex64 np.array): Array of pre-calculated SWHT coefficents for full sphere.
        resolution (float np.array): [Azimuth, Elevation] resolution with minimum 0.1 degrees.
        azimuth (float np.array): [[start, stop],...] angles within 0 to 360 degrees.
        elevation (float np.array): [[start, stop],...] angles within 0 to 180 degrees.
    Returns:
        fov_coeffs (complex64 np.array): Array of pre-calculated SWHt coefficents for FOV.
    Notes:
        * All azimuth field of view zones must have a corresponding elevation zone specified.
        * It is advised to specify field of view zones slightly larger than required.
        * Azimuth and elevation resolution are best kept equal.
        * Boresight is at 270 degrees azimuth and 90 degrees elevation.
    """

    fov_coeffs = np.array([])
    for i in range(azimuth.shape[1]):
        az_index = int(azimuth[i, :] / 0.1)
        el_index = int(elevation[i, :] / 0.1)
        az_step = int(resolution[i, 0] / 0.1)
        el_step = int(resolution[i, 1] / 0.1)
        fov_coeffs = np.append(coeffs[el_index[0]:el_index[1]:el_step, \
                               az_index[0]:az_index[1]:az_step, :], axis=0)
    return fov_coeffs


def stats_to_hdf5():
    return


def hdf5_to_stats():
    return


def swhtcoeffs_to_hdf5():
    return


def hdf5_to_swhtcoeffs():
    return


def rawdata_to_hdf5():
    return


def hdf5_to_rawdata():
    return


if __name__ == '__main__':
    print("ICEBEAR: Incoherent Continuous-Wave E-Region Bistatic Experimental Auroral Radar")
    print("\t-icebear.utils.py")
