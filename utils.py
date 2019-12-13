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

    r = np.sqrt(u**2 + v**2 + w**2)
    t = np.pi/2 - np.arctan2(w, np.sqrt(u**2 + v**2))
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

    u = r*np.sin(t)*np.cos(p)
    v = r*np.sin(t)*np.sin(p) 
    w = r*np.cos(t)

    return u, v, w


def xyz_to_uvw(x, y, z, wavelength):
    """
    Given relative antenna positions in cartesian coordinates with units of meters
    and the wavelength in meters determines the u, v, w baselines in cartesian coordinates.

    Args:
        x (float np.array): East-West antenna coordinate in meters.
        y (float np.array): North-South antenna coordinate in meters.
        z (float np.array): Altitude antenna coordinate in meters.

    Returns:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.

    Notes:
        * Given N antenna then M=N(N-1)/2 unique baselines exist.
        * M baselines can include conjugates and a origin baseline for M_total = M*2 + 1.
    """
    
    num_xspectra = 9+8+7+6+5+4+3+2+1+1

    xspectra_x_diff = np.zeros((num_xspectra),dtype=np.float32)
    xspectra_y_diff = np.zeros((num_xspectra),dtype=np.float32)
    xspectra_z_diff = np.zeros((num_xspectra),dtype=np.float32)

    antenna_num_coh_index = 1

    for first_antenna in range(9):
        for second_antenna in range(first_antenna+1,10):
            xspectra_x_diff[antenna_num_coh_index] = x_antenna_loc[first_antenna]-x_antenna_loc[second_antenna]
            xspectra_y_diff[antenna_num_coh_index] = y_antenna_loc[first_antenna]-y_antenna_loc[second_antenna]
            xspectra_z_diff[antenna_num_coh_index] = z_antenna_loc[first_antenna]-z_antenna_loc[second_antenna]
            antenna_num_coh_index+=1

    u=xspectra_x_diff
    v=xspectra_y_diff
    w=xspectra_z_diff

    u_conj = np.concatenate((u, -u))
    v_conj = np.concatenate((v, -v))
    w_conj = np.concatenate((w, -w)) 
    return u, v, w

def stats_to_hdf5():

    return


def hdf5_to_stats():

    return


def swhtcoeffs_to_hdf5():

    return


def hdf5_to_swhtcoeffs():

    return


def rawdata_to hdf5():

    return


def hdf5_to_rawdata():

    return