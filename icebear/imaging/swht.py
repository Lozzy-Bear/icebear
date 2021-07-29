import numpy as np
import scipy.special as special
import time
import icebear.utils as utils
import h5py
import cv2


def generate_coeffs(config, fov=np.array([[0, 360], [0, 90]]), resolution=1.0, lmax=85):
    """
    Makes an array containing all the factors that do not change with Visibility values.
    This array can then be saved to quickly create Brightness values given changing
    Visibilities. The array is then stored as a HDF5 file.

    Parameters
    ----------
        config : Class Object
            Config class instantiation.
        fov : float np.array
            [[start, stop], [start, stop]] azimuth, elevation angles within 0 to 360 and 0 to 180 degrees.
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

    array_name = config.radar_config
    wavelength = 299792458 / config.center_freq
    date_created = config.date_created
    u, v, w = utils.baselines(config.rx_ant_coords[0, :],
                              config.rx_ant_coords[1, :],
                              config.rx_ant_coords[2, :],
                              wavelength)
    if config.check_attr('fov'):
        fov = config.fov
    if config.check_attr('resolution'):
        resolution = config.resolution
    if config.check_attr('lmax'):
        lmax = config.lmax
    ko = 2 * np.pi / wavelength
    az_step = int(np.abs(fov[0, 0] - fov[0, 1]) / resolution)
    el_step = int(np.abs(fov[1, 0] - fov[1, 1]) / resolution)
    r, t, p = utils.uvw_to_rtp(u, v, w)
    r *= wavelength # Since r, t, p was converted from u, v, w we need the * wavelength back to match SWHT algorithm
    az = np.deg2rad(np.linspace(fov[0, 0], fov[0, 1], az_step))
    el = np.deg2rad(np.linspace(fov[1, 0], fov[1, 1], el_step))
    setting_name = f"{int(np.round(np.abs(fov[0, 0] - fov[0, 1]))):03d}az_" \
                  f"{int(np.round(np.abs(fov[1, 0] - fov[1, 1]))):03d}el_" \
                  f"{str(resolution).replace('.', '')}res_" \
                  f"{lmax}lmax"
    filename = f"swhtcoeffs_{array_name}_{date_created[0]:04d}_{date_created[1]:02d}_{date_created[2]:02d}_{setting_name}.h5"

    print(f"Calculating SWHT coeffs:")
    print(f"\t-filename: {filename}")
    print(f"\t-configuration: {array_name}")
    print(f"\t-azimuth: {fov[0, 0]} - {fov[0, 1]}")
    print(f"\t-elevation: {fov[1, 0]} - {fov[1, 1]}")
    print(f"\t-resolution: {resolution}")
    print(f"\t-degree: {lmax}")
    print(f"\t-wavelength: {wavelength}")

    create_coeffs(filename, date_created, array_name, fov, resolution, lmax, wavelength, np.array([u, v, w]))
    calculate_coeffs(filename, az, el, ko, r, t, p, lmax)

    return filename


def create_coeffs(filename, date_created, array_name, fov, resolution, lmax, wavelength, baselines):
    f = h5py.File(filename, 'w')
    f.create_dataset('radar_config', data=np.array(array_name, dtype='S'))
    f.create_dataset('date_created', data=date_created)
    f.create_dataset('fov', data=fov)
    f.create_dataset('resolution', data=resolution)
    f.create_dataset('lmax', data=lmax)
    f.create_dataset('wavelength', data=wavelength)
    f.create_dataset('baselines', data=baselines)
    f.create_group('coeffs')
    f.close()

    return None


def append_coeffs(filename, l, coeffs):
    f = h5py.File(filename, 'a')
    f.create_dataset(f'coeffs/{l:02d}', data=coeffs)
    f.close()

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
        Maximum harmonic degree is Lmax = 85. Above this scipy crashes due to an overflow error. The potential fix is to
        scale the initial Pmm of the recursion by 10^280 sin^m (theta), and then rescale everything back at the end.

        Holmes, S. A., and W. E. Featherstone, A unified approach to the Clenshaw summation and the recursive
        computation of very high degree and order normalised associated Legendre functions,
        J. Geodesy, 76, 279- 299, doi:10.1007/s00190-002-0216-2, 2002.
    """

    start_time = time.time()
    AZ, EL = np.meshgrid(az, el)
    coeffs = np.zeros((len(el), len(az), len(r)), dtype=np.complex128)

    if lmax <= 85:
        for l in range(lmax+1):
            for m in range(-l, l+1):
                coeffs += ko ** 2 / (2 * np.pi ** 2 * np.round((-1j) ** l)) * \
                          np.repeat(special.sph_harm(m, l, AZ, EL)[:, :, np.newaxis], len(r), axis=2) * \
                          np.repeat(np.repeat(special.spherical_jn(l, ko * r) * \
                          np.conjugate(special.sph_harm(m, l, p, t)) \
                          [np.newaxis, np.newaxis, :], AZ.shape[0], axis=0), AZ.shape[1], axis=1)
                print(f"\tharmonic degree (l) = {l:02d}/{lmax:02d}, order (m) = {m:02d}/{l:02d}\r")
            if l in [5, 15, 25, 35, 45, 55, 65, 75, 85]:
                append_coeffs(filename, l, coeffs)

    elif lmax > 85:
        try:
            import pyshtools as pysh
        except ImportError:
            print(f'Error: lmax = {lmax} -- values over 85 requires PySHTOOLS '
                  f'https://github.com/SHTOOLS try pip install pyshtools')
            exit()
        print(f'\twarning: lmax values over 85 generate massive files only 1/10th frames will be stored, evenly distributed')
        ylm_pysh = np.vectorize(pysh.expand.spharm_lm)
        for l in range(lmax+1):
            for m in range(-l, l+1):
                coeffs += ko ** 2 / (2 * np.pi ** 2 * np.round((-1j) ** l)) * \
                          np.repeat(ylm_pysh(l, m, EL, AZ, normalization='ortho', csphase=-1, kind='complex', degrees=False)[:, :, np.newaxis], len(r), axis=2) * \
                          np.repeat(np.repeat(special.spherical_jn(l, ko * r) * \
                          np.conjugate(ylm_pysh(l, m, t, p, normalization='ortho', csphase=-1, kind='complex', degrees=False)) \
                          [np.newaxis, np.newaxis, :], AZ.shape[0], axis=0), AZ.shape[1], axis=1)
                print(f"\tharmonic degree (l) = {l:02d}/{lmax:02d}, order (m) = {m:02d}/{l:02d}\r")
            if l == 85:
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.1):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.2):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.3):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.4):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.5):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.6):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.7):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.8):
                append_coeffs(filename, l, coeffs)
            if l == int(lmax * 0.9):
                append_coeffs(filename, l, coeffs)
        append_coeffs(filename, l, coeffs)

    print(f"Complete time: \t{time.time()-start_time}")

    return None


def unpackage_coeffs(filename, ind):
    """

    Parameters
    ----------
    filename
    ind

    Returns
    -------
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.
    """

    f = h5py.File(filename, 'r')
    try:
        coeffs = np.array(f['coeffs'][f'{ind:02d}'][()], dtype=np.complex64)
    except:
        coeffs = np.array(f['coeffs'][()], dtype=np.complex64)
    print('hdf5 coeffs:', coeffs.shape)
    return coeffs


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
        brightness : complex64 np.array
            Array of image domain intensity values.

    Notes
    -----
        * The coeffs is calculated for a specific antenna array pattern and
          wavelength. The visibilities must be from the matching coeffs.
        * np.matmul method is faster than CUDA for array size less than 10e6.
    """

    #start_time = time.time()
    brightness = np.matmul(coeffs, visibilities)
    #print(f"\t-swht_py time: \t{time.time()-start_time}")
    
    return brightness


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


def swht_method(visibilities, coeffs, resolution, fov, fov_center):
    """

    Parameters
    ----------
    visibilities
    coeffs
    resolution
    fov
    fov_center

    Returns
    -------

    """
    brightness = swht_py(visibilities, coeffs)
    brightness = brightness_cutoff(brightness, threshold=0.9)
    _, _, cx_extent, cy_extent, area = centroid_center(brightness)
    mx, my, _ = max_center(brightness)

    mx = mx * resolution - fov[0, 0] + fov_center[0]
    my = my * resolution - fov[1, 0] + fov_center[1]
    cx_extent *= resolution
    cy_extent *= resolution
    area *= resolution ** 2

    return mx, my, cx_extent, cy_extent, area


def frequency_difference_beamform():
    # This function is to be added. It provides exceptional target locating but sacrifices extent information.
    # Todo
    return


def brightness_cutoff(brightness, threshold=0.5):
    """
    Given a Brightness array this normalizes then removes noise in the image below a power threshold.
    The default threshold is 0.5 (3 dB).

    Parameters
    ----------
        brightness
        threshold

    Returns
    -------

    """
    brightness = np.abs(brightness / np.max(brightness))
    brightness[brightness < threshold] = 0.0
    return brightness


def centroid_center(brightness):
    """
    Given a Brightness array this returns the centroid as x, y index of the array and the area of the largest blob
    that encloses the maximum power pixel.

    Parameters
    ----------
        brightness

    Returns
    -------
        cx
        cy
        cx_extent
        cy_extent
        area
    """

    image = np.array(brightness * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mx, my, _ = max_center(brightness)
    area = 0
    cx = np.nan
    cy = np.nan
    cx_extent = np.nan
    cy_extent = np.nan
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        temp_area = cv2.contourArea(contour)
        if (x <= mx < (x + w)) and (y <= my <= (y + h)):
            if temp_area > area:
                area = temp_area
                moments = cv2.moments(contour)
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                cx_extent = w
                cy_extent = h

    return cx, cy, cx_extent, cy_extent, area


def max_center(brightness):
    """
    Given a Brightness array this returns the x, y index of the array of the brightest point.

    Parameters
    ----------
        brightness

    Returns
    -------
        cx
        cy
        area

    """

    index = np.unravel_index(np.argmax(brightness, axis=None), brightness.shape)

    return index[1], index[0], np.nan
