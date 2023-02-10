import numpy as np
import scipy.special as special
import time
import common.utils as utils
import h5py
import cv2
try:
    import cupy as cp
except:
    print('no cupy')


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


def unpackage_coeffs(filename, index):
    """
    Unpack the coefficient values from a given hdf5 file of pre-calculated SWHT coefficients.

    Parameters
    ----------
        filename : str
            File and path to the HDF5 file containing the SWHT coefficients to unpack.
        index :
            The harmonic order index to unpack.

    Returns
    -------
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.
    """

    f = h5py.File(filename, 'r')
    try:
        coeffs = np.array(f['coeffs'][f'{index:02d}'][()], dtype=np.complex64)
    except:
        coeffs = np.array(f['coeffs'][()], dtype=np.complex64)
    print(f'hdf5 coeffs: index = {index}, shape = {coeffs.shape}')
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

    brightness = np.matmul(coeffs, visibilities)
    
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

    print("ERROR: swht_cuda is not implemented.")
    exit()

    return


def suppressed_swht_py(visibilities, coeffs_filepath, lmax, start=15, stop=85, step=10):
    """
    Python implementation of the Suppressed-SWHT imaging method. Given the visibility values and
    SWHT coefficients that have been processed for various harmonic orders create a brightness map
    and suppressed brightness map for imaging and spatial locating.

    Parameters
    ----------
        visibilities
        coeffs_filepath
        lmax
        start
        stop
        step

    Returns
    -------
        suppressed_brightness : complex64 np.array

        brightness : complex64 np.array

    Notes
    -----
    For details on the Suppressed-SWHT see below paper.

    Lozinsky, A., et. al. ICEBEAR-3D: A low elevation imaging radar using a non-uniform
    coplanar receiver array for E region observations, Radio Science, submitted 2021.
    """

    # Returns a brightness map and suppressed brightness map
    coeffs = unpackage_coeffs(coeffs_filepath, lmax)
    brightness = swht_py(visibilities, coeffs)
    suppressed_brightness = np.copy(brightness)
    for i in range(start, stop, step):
        coeffs = unpackage_coeffs(coeffs_filepath, i)
        suppressed_brightness *= swht_py(visibilities, coeffs)

    return suppressed_brightness, brightness


def suppressed_swht_cuda(visibilities, coeffs):
    """
    CUDA implementation of the Suppressed-SWHT imaging method. Given the visibility values and
    SWHT coefficients that have been processed for various harmonic orders create a brightness map
    and suppressed brightness map for imaging and spatial locating.

    Parameters
    ----------
        visibilities
        coeffs

    Returns
    -------
        brightness : complex64 np.array

    Notes
    -----
    For details on the Suppressed-SWHT see below paper.

    Lozinsky, A., et. al. ICEBEAR-3D: A low elevation imaging radar using a non-uniform
    coplanar receiver array for E region observations, Radio Science, submitted 2021.
    """

    visibilities = cp.tile(cp.array(visibilities, dtype=cp.complex64), (coeffs.shape[-1], 1))
    brightness = cp.prod(cp.einsum('ijkl,lk->ijl', coeffs, visibilities), axis=2)

    return brightness


def swht_method(visibilities, coeffs, resolution, fov, fov_center):
    """
    Todo: This is not implementing the proper algorithm. This is using a shortcut.

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


def swht_method_advanced(visibilities, coeffs_filepath_fov, coeffs_filepath_full, lmax,
                         average_spectra_noise, resolution, fov, fov_center):
    """
    Advanced method of the Suppressed-SWHT with selection filters for identifying noise signal,
    dirty beam fits, and outside of the FoV targets. Additionally, the accuracy of the selection
    is calculated as the difference between the selected target centroid and brightness maximum.

    Parameters
    ----------
        visibilities
        coeffs_filepath : str
            Absolute filepath to the coeffeicents hdf5 file.
        average_spectra_noise : float
            The average noise computed from the cross spectra.
        resolution : float
            Angular resolution in degree per pixel.
        fov : float np.array
            [[start, stop], [start, stop]] azimuth, elevation angles within 0 to 360 and 0 to 180 degrees.
        fov_center : float np.array
            [[], []]

    Returns
    -------

    """

    # Create the brightness map at 1 degree resolution for a spherical cap.
    suppressed_brightness, brightness = suppressed_swht_py(visibilities, coeffs_filepath_full, lmax)

    # Check to see if the spectral flux density relative to the average spectra noise figure meets the threshold to
    # qualify as a real target within the FoV. Noise and outside of the FoV targets are 10x less Jy. Attempts to
    # capture false positives passed by the previous stage.
    mean_jansky = np.mean(np.abs(brightness))
    max_jansky = np.max(np.abs(brightness))
    if mean_jansky <= 0.001 * average_spectra_noise:
        print(f'-\ttarget outside the FoV, noise, or exists in a severe blindspot; skipped')
        return None, None, None, None, None, None, mean_jansky, max_jansky

    # Check if the target is within the FoV using a 1 deg full sphere coeffs.
    brightness = brightness_cutoff(brightness, threshold=0.6)
    suppressed_brightness = brightness_cutoff(suppressed_brightness, threshold=0.0)
    full_mx, full_my, _ = max_center(suppressed_brightness)
    full_mx, full_my, full_acc, full_extent_x, full_extent_y, full_area = contour_map(brightness, full_mx, full_my)
    full_mx = full_mx - 360 + fov_center[0]
    full_my = full_my - 90 + fov_center[1]
    if not(fov[0, 0] <= full_mx <= fov[0, 1]) and not(fov[1, 0] <= full_my <= fov[1, 1]):
        # Record data at 1deg
        print('\t-target outside fov; recording 1 degree accuracy')
        return full_mx, full_my, full_acc, full_extent_x, full_extent_y, full_area, mean_jansky, max_jansky

    # Create the brightness map at desired resolution and FoV.
    suppressed_brightness, brightness = suppressed_swht_py(visibilities, coeffs_filepath_fov, lmax)

    # Apply Suppressed-SWHT to image to spatial locate the target
    mean_jansky = np.mean(np.abs(brightness))
    max_jansky = np.max(np.abs(brightness))
    brightness = brightness_cutoff(brightness, threshold=0.6)
    suppressed_brightness = brightness_cutoff(suppressed_brightness, threshold=0.0)
    fov_mx, fov_my, _ = max_center(suppressed_brightness)
    fov_mx, fov_my, fov_acc, fov_extent_x, fov_extent_y, fov_area = contour_map(brightness, fov_mx, fov_my)
    fov_mx = fov_mx * resolution - fov[0, 0] + fov_center[0]
    fov_my = fov_my * resolution - fov[1, 0] + fov_center[1]
    fov_acc *= resolution
    fov_extent_x *= resolution
    fov_extent_y *= resolution
    fov_area *= resolution ** 2

    return fov_mx, fov_my, fov_acc, fov_extent_x, fov_extent_y, fov_area, mean_jansky, max_jansky


def swht_method_advanced_cuda(visibilities, coeffs_fov, coeffs_full,
                              resolution, fov, fov_center):
    """
    Advanced method of the Suppressed-SWHT with selection filters for identifying noise signal,
    dirty beam fits, and outside of the FoV targets. Additionally, the accuracy of the selection
    is calculated as the difference between the selected target centroid and brightness maximum.

    Parameters
    ----------
        visibilities :
        coeffs_fov :
        coeffs_full :
        resolution : float
            Angular resolution in degree per pixel.
        fov : float np.array
            [[start, stop], [start, stop]] azimuth, elevation angles within 0 to 360 and 0 to 180 degrees.
        fov_center : float np.array
            [[], []]

    Returns
    -------

    """

    # Create the brightness map at 1 degree resolution for a spherical cap.
    brightness_lowres = np.abs(np.matmul(coeffs_full, visibilities))

    # Check to see if the spectral flux density relative to the average spectra noise figure meets the threshold to
    # qualify as a real target within the FoV. Noise and outside of the FoV targets are 10x less Jy. Attempts to
    # capture false positives passed by the previous stage.
    mean_jansky = np.mean(brightness_lowres)
    max_jansky = np.max(brightness_lowres)
    # if mean_jansky <= 0.001 * average_spectra_noise:
    #     print(f'-\ttarget outside the FoV, noise, or exists in a severe blindspot; skipped')
    #     return None, None, mean_jansky, max_jansky

    # Check if the target is within the FoV using a 1 deg full sphere coeffs.
    index = np.unravel_index(np.argmax(brightness_lowres), brightness_lowres.shape)
    full_mx = index[1]
    full_my = index[0]
    # full_mx = full_mx - 360 + fov_center[0]
    # full_my = full_my - 90 + fov_center[1]
    # if not(fov[0, 0] <= full_mx <= fov[0, 1]) and not(fov[1, 0] <= full_my <= fov[1, 1]):
    if not(225.0 <= full_mx <= 315.0) and not(0.0 <= full_my <= 45.0):
        # Record data at 1deg
        print('\t-target outside fov; recording 1 degree accuracy')
        valid = -1
        full_mx = full_mx - 270.0
        full_my = full_my
        return full_mx, full_my, mean_jansky, max_jansky, valid

    # Create the brightness map at desired resolution and FoV.
    visibilities = cp.tile(cp.array(visibilities, dtype=cp.complex64), (coeffs_fov.shape[-1], 1))
    brightness = cp.prod(cp.einsum('ijkl,lk->ijl', coeffs_fov, visibilities), axis=2)
    index = cp.unravel_index(cp.argmax(cp.abs(brightness)), brightness.shape)
    fov_mx = cp.asnumpy(index[1])
    fov_my = cp.asnumpy(index[0])
    fov_mx = fov_mx * resolution - fov[0, 0] + fov_center[0]
    fov_my = fov_my * resolution - fov[1, 0] + fov_center[1]
    valid = 0
    if np.sqrt((full_mx - fov_mx)**2 + (full_my - fov_my)**2) <= 2.0:
        valid = 1
    return fov_mx, fov_my, mean_jansky, max_jansky, valid


def graphic_method(visibilities, coeffs, resolution, fov, fov_center,
                   rf_distance, doppler_shift, snr_db, wavelength=6.056):
    from sanitizing import map_target

    brightness = swht_py(visibilities, coeffs)
    brightness = np.abs(brightness / np.max(brightness))
    brightness[brightness < 0.8] = 0.0
    brightness[brightness > 0.0] = 1.0
    image = np.array(brightness * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertices_geo = []
    for i in range(len(contours)):
        px = contours[i][:, 0, 0] * resolution - fov[0, 0] + fov_center[0]
        py = contours[i][:, 0, 1] * resolution - fov[1, 0] + fov_center[1]
        px += 7.0  # Adjust for RX site pointing direction
        sx, sa, sv = map_target(np.array([50.893, -109.403, 0.0]),
                                np.array([52.243, -106.450, 0.0]),
                                px, py,
                                np.repeat(rf_distance, px.shape),
                                np.repeat(doppler_shift, px.shape), wavelength)
        s = np.zeros((6, sx.shape[1]), dtype=np.float)
        s[0:3, :] = sx
        s[3, :] = np.repeat(np.abs(snr_db), sx.shape[1])
        s[4, :] = sv[2, :]
        s[5, :] = np.repeat(doppler_shift, sx.shape[1])
        vertices_geo.append(s)

    # For this range-doppler bin these are each lists, where each element in the list is a target
    # in the same range-doppler image bin. Each target is represented by an array of vertices.
    return vertices_geo


def contour_map(brightness, px, py):
    """

    Parameters
    ----------
    brightness
    px
    py

    Returns
    -------

    """
    # Image processing of the brightness map using smx, smy to locate correct contour.
    image = np.array(brightness * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    cx = np.nan
    cy = np.nan
    extent_x = np.nan
    extent_y = np.nan
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        temp_area = cv2.contourArea(contour)
        if (x <= px < (x + w)) and (y <= py <= (y + h)):
            if temp_area > area:
                area = temp_area
                moments = cv2.moments(contour)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                extent_x = w
                extent_y = h
                region = [slice(x, x + w), slice(y, y + h)]
    # Todo: region verify works
    mx, my, _ = max_center(brightness[region])
    acc = np.sqrt((cx - mx) ** 2 + (cy - my) ** 2)

    return mx, my, acc, extent_x, extent_y, area


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
