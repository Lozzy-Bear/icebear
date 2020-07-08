import numpy as np
import scipy.special as special
import time
import icebear.utils as utils


def swht_coeffs(azimuth, elevation, resolution, lmax, wavelength, u, v, w):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Args:
        azimuth (float np.array): [start, stop] angles within 0 to 360 degrees.
        elevation (float np.array): [start, stop] angles within 0 to 180 degrees.
        resolution (float): Angular resolution in degree per pixel.
        lmax (int): The maximum harmonic degree.
        wavelength (float): Radar signal wavelength in meters.
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.

    Returns:
        coeffs (complex64 np.array): Array of pre-calculated SWHT coefficients for full sphere.

    Notes:
        * Maximum harmonic degree is Lmax = 85. Above this scipy crashes.

    Todo:
        * Add functionality to go to harmonic degrees above lmax = 85.
    """

    start_time = time.time()
    print(f"Calculating SWHT coeffs:")
    print(f"\t-azimuth: \t{azimuth[0]} - {azimuth[1]}")
    print(f"\t-elevation: \t{elevation[0]} - {elevation[1]}")
    print(f"\t-resolution: \t{resolution}")
    print(f"\t-degree: \t{lmax}")
    print(f"\t-wavelength: \t{wavelength}")
    
    ko = 2*np.pi/wavelength     #ko = wo/c = 2pi*fo/c = 2pi/wavelength 
    az_step = int(np.abs(azimuth[0] - azimuth[1]) / resolution)
    el_step = int(np.abs(elevation[0] - elevation[1]) / resolution)
    r,t,p = utils.uvw_to_rtp(u, v, w)
    az = np.radians(np.linspace(azimuth[0], azimuth[1], az_step))
    el = np.radians(np.linspace(elevation[0], elevation[1], el_step))
    AZ, EL = np.meshgrid(az, el)
    coeffs = np.zeros((len(el), len(az), len(u)), dtype=np.complex128)
    
    for l in range(lmax+1):
        for m in range(-l, l+1):
            coeffs += ko**2 / (2*np.pi**2*np.round((-1j)**l)) *\
                    np.repeat(special.sph_harm(m, l, AZ, EL)[:,:,np.newaxis], len(u), axis=2) *\
                    np.repeat(np.repeat(special.spherical_jn(l, ko*r) *\
                    np.conjugate(special.sph_harm(m, l, p, t))\
                    [np.newaxis,np.newaxis,:], AZ.shape[0], axis=0), AZ.shape[1],axis=1)
            print(f"Harmonic degree (l) step: {l}\t / {lmax}\r")
        utils.swhtcoeffs_to_hdf5(coeffs) 

    print(f"Complete time: \t{time.time()-start_time}")

    return coeffs


def swht_py(visibilities, coeffs):
    """
    Apply a spherical wave harmonic transforms (Carozzi, 2015) to the given
    visibility values using the pre-calculated transform coefficients.
  
    Args:
        visibilities (complex64 np.array): Data cross-correlation values.
        coeffs (complex64 np.array): Array of pre-calculated SWHT coefficients.

    Returns:
        intensity (complex64 np.array): Array of image domain intensity values.

    Notes:
        * The coeffs is calculated for a specific antenna array pattern and
          wavelength. The visibilities must be from the matching coeffs.
        * np.matmul method is faster than CUDA for array size less than 10e6.
    """

    start_time = time.time()
    intensity = np.matmul(coeffs, visibilities)
    print(f"swht_py time: \t{time.time()-start_time}")
    
    return intensity


def swht_cuda():
    """
    Wrapper to implement the spherical wave harmonic transform (Carozzi, 2015) in CUDA.

    Args:
        visibilities (complex64 np.array): Data cross-correlation values.
        coeffs (complex64 np.array): Array of pre-calculated SWHT coefficients.

    Returns:
        intensity (complex64 np.array): Array of image domain intensity values.
    """
    return

