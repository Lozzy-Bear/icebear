import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils


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

    # Setup givens in correct units for pymap3d
    rf = rf * 1.0e3
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
    filepath = '/beaver/backup/level2b/'  # Enter file path to level 1 directory
    # filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
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

    azimuth += 7.0
    azimuth = np.where(azimuth < 0.0, azimuth + 360.0, azimuth)
    print('\t-loading completed')
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
    altitude, slant_range, elevation = map_target([50.893, -109.403, 0.0],
                                                            [52.243, -106.450, 0.0],
                                                            azimuth,
                                                            elevation,
                                                            rf_distance)
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
    doppler_shift *= 3.0
    mean_altitude = np.mean(altitude)
    total_targets = len(altitude)
