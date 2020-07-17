import numpy as np
import scipy.special as special
import time
import icebear.utils as utils
import h5py
import yaml


def generate_coeffs(date, array_file, azimuth=(0, 360), elevation=(0, 90), resolution=1.0, lmax=85):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Parameters
    ----------
        date : string
            Date coefficients are calculated in format YYYYMMDD.
        array_file : string
            Filename and path to the receiver array configuration data.
        azimuth : float np.array
            [start, stop] angles within 0 to 360 degrees.
        elevation : float np.array
            [start, stop] angles within 0 to 180 degrees.
        resolution : float
            Angular resolution in degree per pixel.
        lmax : int
            The maximum harmonic degree.

    Returns
    -------
        None

    Notes
    -----
        The array file must contain:
        wavelength : float
            Radar signal wavelength in meters.
        u : float np.array
            East-West baseline coordinate divided by wavelength.
        v : float np.array
            North-South baseline coordinate divided by wavelength.
        w : float np.array
            Altitude baseline coordinate divided by wavelength.
    """

    cfg = yaml.full_load(open(array_file))
    array_name = cfg["general"]["name"]

    ko = 2 * np.pi / wavelength
    az_step = int(np.abs(azimuth[0] - azimuth[1]) / resolution)
    el_step = int(np.abs(elevation[0] - elevation[1]) / resolution)
    r, t, p = utils.uvw_to_rtp(u, v, w)
    az = np.radians(np.linspace(azimuth[0], azimuth[1], az_step))
    el = np.radians(np.linspace(elevation[0], elevation[1], el_step))
    config_name = f"{int(np.round(np.abs(azimuth[0] - azimuth[1]))):03d}-" \
                  f"{int(np.round(np.abs(elevation[0] - elevation[1]))):03d}-" \
                  f"{str(resolution).replace('.', '')}-" \
                  f"{lmax}"

    # Example filename: SWHTCoeffs_icebear_20200713_360-90-10-85
    filename = 'SWHTCoeffs_' + array_name + '_' + date + '_' + config_name

    print(f"Calculating SWHT coeffs:")
    print(f"\t-filename: \t{filename}")
    print(f"\tconfiguration: \t{array_name}")
    print(f"\t-azimuth: \t{azimuth[0]} - {azimuth[1]}")
    print(f"\t-elevation: \t{elevation[0]} - {elevation[1]}")
    print(f"\t-resolution: \t{resolution}")
    print(f"\t-degree: \t{lmax}")
    print(f"\t-wavelength: \t{wavelength}")

    create_coeffs_hdf5(filename)
    calculate_coeffs(filename, az, el, ko, r, t, p, lmax)

    return None


def create_coeffs_hdf5(filename, date, array_name, azimuth, elevation, resolution, lmax):

    coeffs_file = 'none'
    return None


def append_coeffs_hdf5(filename, l, coeffs):

    return None


def calculate_coeffs(filename, az, el, ko, r, t, p, lmax=85):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Parameters
    ----------
        filename : string
            Filename and path to the HDF5 file the calculated coefficients are to be appended.
        az : float np.array
            An array of azimuth angles in radians to calculate coefficients for.
        el : float np.array
            An array of elevation angles in radians to calculate coefficients for.
        lmax : int
            The maximum harmonic degree.
        ko : float
            Radar signal wave number, ko = 2pi/wavelength.
        r : float np.array
            Radius baseline coordinate divided by wavelength.
        t : float np.array
            Theta (elevation) baseline coordinate.
        p : float np.array
            Phi (azimuthal) baseline coordinate.

    Returns
    -------
        None

    Notes
    -----
        * Maximum harmonic degree is Lmax = 85. Above this scipy crashes.

    Todo
        * Add functionality to go to harmonic degrees above lmax = 85.
    """

    start_time = time.time()
    AZ, EL = np.meshgrid(az, el)
    coeffs = np.zeros((len(el), len(az), len(r)), dtype=np.complex128)

    for l in range(lmax+1):
        for m in range(-l, l+1):
            coeffs += ko ** 2 / (2 * np.pi ** 2 * np.round((-1j) ** l)) * \
                      np.repeat(special.sph_harm(m, l, AZ, EL)[:, :, np.newaxis], len(r), axis=2) * \
                      np.repeat(np.repeat(special.spherical_jn(l, ko * r) * \
                      np.conjugate(special.sph_harm(m, l, p, t)) \
                      [np.newaxis, np.newaxis, :], AZ.shape[0], axis=0), AZ.shape[1], axis=1)
            print(f"Harmonic degree (l) step: {l}\t / {lmax}\r")
        append_coeffs_hdf5(filename, l, coeffs)

    print(f"Complete time: \t{time.time()-start_time}")

    return None


def swht_py(visibilities, coeffs):
    """
    Apply a spherical wave harmonic transforms (Carozzi, 2015) to the given
    visibility values using the pre-calculated transform coefficients.
  
    Parameters
    ----------
        visibilities : complex64 np.array
            Data cross-correlation values.
        coeffs : complex64 np.array
            Array of pre-calculated SWHT coefficients.

    Returns
    -------
        intensity : complex64 np.array
            Array of image domain intensity values.

    Notes
    -----
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

    Parameters
    ----------
        visibilities : complex64 np.array
            Data cross-correlation values.
        coeffs : complex64 np.array
            Array of pre-calculated SWHT coefficients.

    Returns
    -------
        intensity : complex64 np.array
            Array of image domain intensity values.
    """
    return

