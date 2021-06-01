import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils
import time


def map_target_updated(tx, rx, az, el, rf, dop, wavelength):
    """
    Find the scatter location given tx location, rx location, total rf distance, and target angle-of-arrival
    using the 'WGS84' Earth model. Also determines the bistatic velocity vector and bistatic radar wavelength.

    Parameters
    ----------
        tx : float np.array
            [latitude, longitude, altitude] of tx array in degrees and kilometers
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and kilometers
        az : float np.array
            angle-of-arrival azimuth in degrees
        el : float np.array
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau in kilometers
        dop : float np.array
            doppler shift in hertz
        wavelength : float
            radar signal center wavelength

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        sa : float np.array
            [azimuth, elevation, slant range] of scatter in degrees and kilometers
        sv : float np.array
            [azimuth, elevation, velocity] the bistatic Doppler velocity vector in degrees and kilometers.
            Coordinates given in the scattering targets local frame (azimuth from North, elevation up from
            the plane normal to zenith, Doppler [Hz] * lambda / (2 cos(e/2)) )

    Notes
    -----
    tx : transmitter location
    rx : receiver location
    sx : scatter location
    gx : geometric center of Earth, origin
    u_rt : unit vector rx to tx
    u_rs : unit vector rx to sx
    u_gt : unit vector gx to tx
    u_gr : unit vector gx to rx
    u_gs : unit vector gx to sx
    """

    # Initialize output arrays
    sx = np.zeros((3, len(rf)), dtype=float)
    sa = np.zeros((3, len(rf)), dtype=float)
    sv = np.zeros((3, len(rf)), dtype=float)

    # Setup variables in correct units for pymap3d
    rf = rf * 1.0e3
    az = np.where(az < 0.0, az + 360.0, az)
    az = np.deg2rad(az)
    el = np.deg2rad(np.abs(el))

    # Determine the slant range, r
    bx1, by1, bz1 = pm.geodetic2ecef(rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    v_gr = np.array([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    v_gt = np.array([bx2, by2, bz2])
    raz, rel, b = pm.ecef2aer(bx2, by2, bz2, rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_rt = np.array([np.sin(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.cos(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.sin(np.deg2rad(rel))])
    el -= relaxation_elevation(el, rf, az, b, u_rt)
    u_rs = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.dot(u_rt, u_rs)))

    # WGS84 Model for lat, long, alt
    sx[:, :] = pm.aer2geodetic(np.rad2deg(az), np.rad2deg(el), np.abs(r),
                         np.repeat(rx[0], len(az)),
                         np.repeat(rx[1], len(az)),
                         np.repeat(rx[2], len(az)),
                         ell=pm.Ellipsoid("wgs84"), deg=True)

    # Determine the bistatic Doppler velocity vector
    x, y, z = pm.geodetic2ecef(sx[0, :], sx[1, :], sx[2, :],
                               ell=pm.Ellipsoid('wgs84'), deg=True)
    v_gs = np.array([x, y, z])
    v_bi = (-1 * v_gs.T + v_gt / 2.0 + v_gr / 2.0).T
    u_bi = v_bi / np.linalg.norm(v_bi, axis=0)
    v_sr = (v_gr - v_gs.T).T
    u_sr = v_sr / np.linalg.norm(v_sr, axis=0)
    radar_wavelength = wavelength / np.abs(2.0 * np.einsum('ij,ij->j', u_sr, u_bi))
    doppler_sign = np.sign(dop)
    vaz, vel, _ = pm.ecef2aer(doppler_sign * u_bi[0, :] + x,
                              doppler_sign * u_bi[1, :] + y,
                              doppler_sign * u_bi[2, :] + z,
                              sx[0, :], sx[1, :], sx[2, :],
                              ell=pm.Ellipsoid("wgs84"), deg=True)
    # plt.figure()
    # plt.scatter(vaz, vel, c=dop)
    # plt.colorbar()
    # plt.show()
    # Convert back to conventional units
    sx[2, :] /= 1.0e3
    az = np.rad2deg(az)
    el = np.rad2deg(el)
    sa[:, :] = np.array([az, el, r / 1.0e3])
    sv[:, :] = np.array([vaz, vel, dop * radar_wavelength])

    return sx, sa, sv


def map_target_old(tx, rx, az, el, rf, dop, wavelength):
    """
    Find the scatter location given tx location, rx location, total rf distance, and target angle-of-arrival
    using the 'WGS84' Earth model. Also determines the bistatic velocity vector and bistatic radar wavelength.

    Parameters
    ----------
        tx : float np.array
            [latitude, longitude, altitude] of tx array in degrees and kilometers
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and kilometers
        az : float np.array
            angle-of-arrival azimuth in degrees
        el : float np.array
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau in kilometers
        dop : float np.array
            doppler shift in hertz
        wavelength : float
            radar signal center wavelength

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        sa : float np.array
            [azimuth, elevation, slant range] of scatter in degrees and kilometers
        sv : float np.array
            [azimuth, elevation, velocity] the bistatic Doppler velocity vector in degrees and kilometers.
            Coordinates given in the scattering targets local frame (azimuth from North, elevation up from
            the plane normal to zenith, Doppler [Hz] * lambda / (2 cos(e/2)) )

    Notes
    -----
    tx : transmitter location
    rx : receiver location
    sx : scatter location
    gx : geometric center of Earth, origin
    u_rt : unit vector rx to tx
    u_rs : unit vector rx to sx
    u_gt : unit vector gx to tx
    u_gr : unit vector gx to rx
    u_gs : unit vector gx to sx
    """

    # Initialize output arrays
    sx = np.zeros((3, len(rf)), dtype=float)
    sa = np.zeros((3, len(rf)), dtype=float)
    sv = np.zeros((3, len(rf)), dtype=float)

    # Setup variables in correct units for pymap3d
    rf = rf * 1.0e3
    az = np.where(az < 0.0, az + 360.0, az)
    az = np.deg2rad(az)
    el = np.deg2rad(np.abs(el))

    # Determine the slant range, r
    bx1, by1, bz1 = pm.geodetic2ecef(rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_gr = np.array([bx1, by1, bz1]) / np.linalg.norm([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_gt = np.array([bx2, by2, bz2]) / np.linalg.norm([bx2, by2, bz2])
    raz, rel, b = pm.ecef2aer(bx2, by2, bz2, rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_rt = np.array([np.sin(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.cos(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.sin(np.deg2rad(rel))])
    el -= relaxation_elevation(el, rf, az, b, u_rt)
    u_rs = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.dot(u_rt, u_rs)))

    # WGS84 Model for lat, long, alt
    sx[:, :] = pm.aer2geodetic(np.rad2deg(az), np.rad2deg(el), np.abs(r),
                         np.repeat(rx[0], len(az)),
                         np.repeat(rx[1], len(az)),
                         np.repeat(rx[2], len(az)),
                         ell=pm.Ellipsoid("wgs84"), deg=True)

    # Determine the bistatic Doppler velocity vector
    x, y, z = pm.geodetic2ecef(sx[0, :], sx[1, :], sx[2, :],
                               ell=pm.Ellipsoid('wgs84'), deg=True)
    u_gs = np.array([x, y, z]) / np.linalg.norm([x, y, z], axis=0)
    u_bi = (u_gs.T + u_gt / 2.0 + u_gr / 2.0).T
    radar_wavelength = wavelength / (2.0 * np.einsum('ij,ij->j', u_bi, -1 * u_rs))
    doppler_sign = np.sign(dop)
    vaz, vel, _ = pm.ecef2aer(doppler_sign * (u_bi[0, :] + x),
                              doppler_sign * (u_bi[1, :] + y),
                              doppler_sign * (u_bi[2, :] + z),
                              sx[0, :], sx[1, :], sx[2, :],
                              ell=pm.Ellipsoid("wgs84"), deg=True)

    # Convert back to conventional units
    sx[2, :] /= 1.0e3
    az = np.rad2deg(az)
    el = np.rad2deg(el)
    sa[:, :] = np.array([az, el, r / 1.0e3])
    sv[:, :] = np.array([vaz, vel, np.abs(dop * radar_wavelength)])

    return sx, sa, sv


def package_data(time, snr_db, range, azimuth, elevation, altitude,
                 velocity_azimuth, velocity_elevation, doppler_shift):

    config = ib.utils.Config('X://PythonProjects//icebear//dat//default.yml')

    f = h5py.File('cleaned_2020_06_16.h5', 'w')
    f.create_dataset('date_created', data=np.array([2021, 4, 1]))
    f.create_dataset('version', data=np.array(config.version, dtype='S'))
    f.create_dataset('experiment_name', data=np.array(['meteor winds'], dtype='S'))
    f.create_dataset('radar_config', data=np.array([config.radar_config], dtype='S'))
    f.create_dataset('center_freq', data=config.center_freq)
    # receiver site information
    f.create_dataset('rx_site_name', data=np.array([config.rx_site_name], dtype='S'))
    f.create_dataset('rx_site_lat_long', data=config.rx_site_lat_long)
    f.create_dataset('rx_heading', data=config.rx_heading)
    f.create_dataset('rx_rf_path', data=np.array([config.rx_rf_path], dtype='S'))
    f.create_dataset('rx_ant_type', data=np.array([config.rx_ant_type], dtype='S'))
    f.create_dataset('rx_ant_coords', data=config.rx_ant_coords)
    f.create_dataset('rx_feed_corr', data=config.rx_feed_corr)
    f.create_dataset('rx_feed_corr_date', data=config.rx_feed_corr_date)
    f.create_dataset('rx_feed_corr_type', data=np.array([config.rx_feed_corr_type], dtype='S'))
    f.create_dataset('rx_ant_mask', data=config.rx_ant_mask)
    f.create_dataset('rx_sample_rate', data=config.rx_sample_rate)
    # transmitter site information
    f.create_dataset('tx_site_name', data=np.array([config.tx_site_name], dtype='S'))
    f.create_dataset('tx_site_lat_long', data=config.tx_site_lat_long)
    f.create_dataset('tx_heading', data=config.tx_heading)
    f.create_dataset('tx_rf_path', data=np.array([config.tx_rf_path], dtype='S'))
    f.create_dataset('tx_ant_type', data=np.array([config.tx_ant_type], dtype='S'))
    f.create_dataset('tx_ant_coords', data=config.tx_ant_coords)
    f.create_dataset('tx_feed_corr', data=config.tx_feed_corr)
    f.create_dataset('tx_feed_corr_date', data=config.tx_feed_corr_date)
    f.create_dataset('tx_feed_corr_type', data=np.array([config.tx_feed_corr_type], dtype='S'))
    f.create_dataset('tx_ant_mask', data=config.tx_ant_mask)
    f.create_dataset('tx_sample_rate', data=config.tx_sample_rate)
    # processing settings
    f.create_dataset('decimation_rate', data=config.decimation_rate)
    f.create_dataset('time_resolution', data=config.time_resolution)
    f.create_dataset('coherent_integration_time', data=config.coherent_integration_time)
    f.create_dataset('incoherent_averages', data=config.incoherent_averages)
    f.create_dataset('snr_cutoff_db', data=config.snr_cutoff_db)
    # imaging settings
    f.create_dataset('imaging_method', data=np.array([config.imaging_method], dtype='S'))
    f.create_dataset('swht_coeffs', data=np.array([config.swht_coeffs], dtype='S'))
    f.create_dataset('fov', data=config.fov)
    f.create_dataset('fov_center', data=config.fov_center)
    f.create_dataset('resolution', data=config.resolution)
    f.create_dataset('lmax', data=config.lmax)
    f.create_group('data')

    # append a new group for the current measurement
    f.create_dataset(f'data/time', data=time)
    f.create_dataset(f'data/snr_db', data=snr_db)
    f.create_dataset(f'data/range', data=range)
    f.create_dataset(f'data/azimuth', data=azimuth)
    f.create_dataset(f'data/elevation', data=elevation)
    f.create_dataset(f'data/altitude', data=altitude)
    f.create_dataset(f'data/velocity_azimuth', data=velocity_azimuth)
    f.create_dataset(f'data/velocity_elevation', data=velocity_elevation)
    f.create_dataset(f'data/doppler_shift', data=doppler_shift)
    f.close()

    return


def map_target(tx, rx, az, el, rf, mode='ellipsoidal'):
    """
    Find the scatter location given tx location, rx, location, total rf distance, and target angle-of-arrival.

    Parameters
    ----------
        tx : float np.array
            [latitude, longitude, altitude] of tx array in degrees and kilometers
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and kilometers
        az : float np.array
            angle-of-arrival azimuth in degrees
        el : float np.array
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau in kilometers
        mode : string
            earth model option; 'ellipsoidal' for WGS84 or 'spherical' for simple sphere

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        r : float
            bistatic corrected slant range in kilometers
        el : float np.array
            bistatic low elevation corrected elevation angle-of-arrival in degrees
    """

    # Setup givens in correct units
    rf = rf * 1.0e3
    az = np.where(az < 0.0, az + 360.0, az)
    az = np.deg2rad(az)
    el = np.deg2rad(np.abs(el))
    sx = np.zeros((3, len(rf)))
    uv = np.zeros((3, len(rf)))
    us = np.zeros((3, len(rf)))

    # Determine the slant range, r
    bx1, by1, bz1 = pm.geodetic2ecef(rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    ur = np.array([bx1, by1, bz1]) / np.linalg.norm([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    ut = np.array([bx2, by2, bz2]) / np.linalg.norm([bx2, by2, bz2])
    bx = bx2 - bx1
    by = by2 - by1
    bz = bz2 - bz1
    b = np.linalg.norm([bx, by, bz])
    ub = np.array([bx, by, bz]) / b

    el -= relaxation_elevation(el, rf, az, b, ub)
    ua = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.dot(ub, ua)))

    # WGS84 Model for lat, long, alt.
    if mode == 'ellipsoidal':
        sx[:, :] = pm.aer2geodetic(np.rad2deg(az), np.rad2deg(el), np.abs(r),
                                   np.repeat(rx[0], len(az)),
                                   np.repeat(rx[1], len(az)),
                                   np.repeat(rx[2], len(az)),
                                   ell=pm.Ellipsoid("wgs84"), deg=True)

    # Spherical Earth approximation
    if mode == 'spherical':
        re = 6378.0e3  # Radius of earth in [m]
        sx[2, :] = -re + np.sqrt(re ** 2 + r ** 2 + 2 * re * r * np.sin(el))

    return sx[2, :] / 1.0e3, r / 1.0e3, np.rad2deg(el)


def relaxation_elevation(beta, rf_distance, azimuth, bistatic_distance, bistatic_vector):
    """

    Parameters
    ----------
    beta : float np.array
        Measured elevation angle in degrees
    rf_distance :
    azimuth
    bistatic_distance
    bistatic_vector

    Returns
    -------

    """
    n = 0
    radius_of_earth = 6378.0e3
    err = np.deg2rad(1.0)
    target = np.deg2rad(0.1)
    m = np.zeros((3, len(beta)))
    m[1, :] = 0.1
    v = np.array([np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]), np.sin(beta - m[1, :])])
    r = (rf_distance ** 2 - bistatic_distance ** 2) / (2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
    m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
    while np.nanmean(err) > target:
        m[0, :] = m[1, :]
        m[1, :] = m[2, :]
        v = np.array([np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]), np.sin(beta - m[1, :])])
        r = (rf_distance ** 2 - bistatic_distance ** 2) / (2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
        m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
        err = np.abs((m[1, :] - m[2, :]) ** 2 / (2 * m[1, :] - m[0, :] - m[2, :]))
        n += 1

    m[2, :] = np.where(err >= target, np.nan, m[2, :])
    print('\t-relaxation mean error:', np.rad2deg(np.nanmean(err)), 'iterations:', n)
    return m[2, :]


if __name__ == '__main__':
    # Pretty plot configuration.
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Load the level 2 data file.
    # filepath = '/beaver/backup/level2b/'  # Enter file path to level 1 directory
    filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
    files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_15')  # Enter first sub directory and last
    # files = utils.get_all_data_files(filepath, '2019_12_19', '2019_12_19')  # Enter first sub directory and last
    rf_distance = np.array([])
    snr_db = np.array([])
    doppler_shift = np.array([])
    azimuth = np.array([])
    elevation = np.array([])
    elevation_extent = np.array([])
    azimuth_extent = np.array([])
    area = np.array([])
    t = np.array([])
    for file in files:
        f = h5py.File(file, 'r')
        print(file)
        group = f['data']
        date = f['date']
        keys = group.keys()
        for key in keys:
            data = group[f'{key}']
            # Filter out dropped samples
            if np.any(data['rf_distance'][()] < 250):
                continue
            # Glenn's noise filter
            # if len(data['rf_distance'][()]) <= 5:
            #     continue
            t = np.append(t, np.ones(len(data['rf_distance'][()])) * int(key))
            rf_distance = np.append(rf_distance, data['rf_distance'][()])
            snr_db = np.append(snr_db, np.abs(data['snr_db'][()]))
            doppler_shift = np.append(doppler_shift, data['doppler_shift'][()])
            azimuth = np.append(azimuth, data['azimuth'][()])
            elevation = np.append(elevation, np.abs(data['elevation'][()]))
            elevation_extent = np.append(elevation_extent, data['elevation_extent'][()] / 4)
            azimuth_extent = np.append(azimuth_extent, data['azimuth_extent'][()])
            area = np.append(area, data['area'][()])

            # t = np.ones(len(data['rf_distance'][()])) * int(key)
            # rf_distance = data['rf_distance'][()]
            # snr_db = np.abs(data['snr_db'][()])
            # doppler_shift = data['doppler_shift'][()]
            # azimuth = data['azimuth'][()]
            # elevation = np.abs(data['elevation'][()])
            # elevation_extent = data['elevation_extent'][()] / 4
            # azimuth_extent = data['azimuth_extent'][()]
            # area = data['area'][()]

        print('\t-loading completed')
        azimuth += 7.0
        print('\t-total data', len(rf_distance))
        # Pre-masking
        m = np.ones_like(rf_distance)
        m = np.ma.masked_where(snr_db <= 1.0, m)  # Weak signals close to noise or highly multipathed (meteors are strong)
        # m = np.ma.masked_where(doppler_shift >= 50, m)  # Meteors are less than |150 m/s|
        # m = np.ma.masked_where(doppler_shift <= -50, m)  # Meteors are less than |150 m/s|
        # m = np.ma.masked_where(area >= 5.0, m)  # Meteors should have small scattering cross-sectional area

        rf_distance = rf_distance * m
        snr_db = snr_db * m
        doppler_shift = doppler_shift * m
        azimuth = azimuth * m
        azimuth_extent = azimuth_extent * m
        elevation = elevation * m
        elevation_extent = elevation_extent * m
        area = area * m
        t = t * m
        print('\t-pre-masking completed')

        # Pre-calculate and do altitude earth curvature corrections.
        # altitude, slant_range, elevation = map_target([50.893, -109.403, 0.0],
        #                                               [52.243, -106.450, 0.0],
        #                                                azimuth,
        #                                                elevation,
        #                                                rf_distance)
        sx, sa, sv = map_target_updated([50.893, -109.403, 0.0],
                                        [52.243, -106.450, 0.0],
                                        azimuth, elevation, rf_distance,
                                        doppler_shift, 6.056)

        altitude = sx[2, :]
        slant_range = sa[2, :]
        elevation = sa[1, :]
        doppler_shift = sv[2, :]
        print('\t-mapping completed')

        # Set up a filtering mask.
        m = np.ma.masked_where(elevation == np.nan, m)  # Elevation iterations not converging (noise)
        m = np.ma.masked_where(elevation <= 0.0, m)  # Elevation below the ground
        # m = np.ma.masked_where(((slant_range <= 100.0) & (altitude <= 25.0)), m)  # Beam camping location
        m = np.ma.masked_where(slant_range <= 225, m)  # Man made noise and multipath objects
        m = np.ma.masked_where(altitude <= 50, m)  # Man made noise and multipath objects
        # m = np.ma.masked_where(altitude >= 130, m)  # Man made noise and multipath objects
        rf_distance = rf_distance * m
        snr_db = snr_db * m
        doppler_shift = doppler_shift * m
        azimuth = azimuth * m
        azimuth_extent = azimuth_extent * m
        elevation = elevation * m
        elevation_extent = elevation_extent * m
        area = area * m
        slant_range = slant_range * m
        altitude = altitude * m
        t = t * m
        rf_distance = rf_distance[~rf_distance.mask]
        snr_db = snr_db[~snr_db.mask]
        doppler_shift = doppler_shift[~doppler_shift.mask]
        azimuth = azimuth[~azimuth.mask]
        azimuth_extent = azimuth_extent[~azimuth_extent.mask]
        elevation = elevation[~elevation.mask]
        elevation_extent = elevation_extent[~elevation_extent.mask]
        area = area[~area.mask]
        slant_range = slant_range[~slant_range.mask]
        altitude = altitude[~altitude.mask]
        t = t[~t.mask]
        print('\t-masking completed')

        # if len(slant_range) == 0:
        #     print('-skipped')
        #     continue

        # Pre plotting data management
        mean_altitude = np.mean(altitude)
        total_targets = len(altitude)

        # plt.figure(figsize=[16, 24])
        # plt.subplot(411)
        # # plt.title(f'{key}')
        # plt.scatter(slant_range, altitude, c=doppler_shift, vmin=-1500.0, vmax=1500.0, cmap='jet_r', label=f'Total Targets {total_targets}')
        # plt.colorbar(label='Doppler Shift [m/s]')
        # plt.xlabel('Slant Range [km]')
        # plt.ylabel('Altitude [km]')
        # plt.xlim([0.0, 1200.0])
        # plt.ylim([0.0, 200.0])
        # plt.plot([0.0, 1200.0], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
        # plt.legend(loc='upper right')
        #
        # plt.subplot(412)
        # plt.scatter(azimuth, altitude, c=doppler_shift, vmin=-1500.0, vmax=1500.0, cmap='jet_r', label=f'Total Targets {total_targets}')
        # plt.colorbar(label='Doppler Shift [m/s]')
        # plt.xlabel('Azimuth Angle from North [deg]')
        # plt.ylabel('Altitude [km]')
        # plt.ylim([0.0, 200.0])
        # plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
        # plt.legend(loc='upper right')
        #
        # plt.subplot(413)
        # plt.scatter(slant_range, altitude, c=snr_db, vmin=0.0, vmax=20.0, cmap='plasma_r', label=f'Total Targets {total_targets}')
        # plt.colorbar(label='Signal-to-Noise [dB]')
        # plt.xlabel('Slant Range [km]')
        # plt.ylabel('Altitude [km]')
        # plt.xlim([0.0, 1200.0])
        # plt.ylim([0.0, 200.0])
        # plt.plot([0.0, 1200.0], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
        # plt.legend(loc='upper right')
        #
        # plt.subplot(414)
        # plt.scatter(azimuth, altitude, c=snr_db, vmin=0.0, vmax=20.0, cmap='plasma_r', label=f'Total Targets {total_targets}')
        # plt.colorbar(label='Signal-to-Noise [dB]')
        # plt.xlabel('Azimuth Angle from North [deg]')
        # plt.ylabel('Altitude [km]')
        # plt.ylim([0.0, 200.0])
        # plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
        # plt.legend(loc='upper right')

        # plt.savefig(f'/beaver/backup/geminids/summary/snr_dop_rng_az_filtered_02.png')
        # plt.close()
        # print(f'-time complete: {key}')

        # plt.figure(figsize=[12, 12])
        # _ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype=u'step', label=f'Total Targets {total_targets}')
        # plt.xscale('log')
        # plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude Distribution')
        # plt.xlabel('Count')
        # plt.ylabel('Altitude [km]')
        # plt.ylim((0, 200))
        # plt.xlim((10, 10_000))
        # plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
        # plt.legend(loc='upper right')
        # plt.savefig(f'/beaver/backup/geminids/summary/altitude_histogram_filtered_02.png')

        # import cartopy.crs as ccrs
        # ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.set_extent([np.min(sv[0, :]),np.max(sv[0, :]), np.min(sv[1, :]), np.max(sv[1, :])], crs=ccrs.PlateCarree())
        # ax.lakes
        # ax.coastline
        # ax.rivers
        thresh = 1500.0
        sv[2, :] = np.where(sv[2, :] > thresh, thresh, sv[2, :])
        sv[2, :] = np.where(sv[2, :] < -thresh, -thresh, sv[2, :])

        # plt.figure()
        # plt.hist(sv[2, :])

        plt.figure()
        plt.quiver(sx[1, :], sx[0, :],
                   np.sin(np.deg2rad(sv[0, :])), np.cos(np.deg2rad(sv[0, :])),
                   sv[2, :], cmap='jet_r')
        # plt.title(f'{year}-{month}-{day} {hour}:{minute}:{second}')
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Latitude [deg]')
        plt.colorbar(label='Velocity [m/s]')

        plt.scatter(-109.403, 50.893, c='k')
        plt.annotate('TX', (-109.403, 50.893))
        plt.scatter(-106.450, 52.243, c='k')
        plt.annotate('RX', (-106.450, 52.24))

        plt.xlim([-114.0, -96.0])
        plt.ylim([50.0, 62.0])
        plt.show()
