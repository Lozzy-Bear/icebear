import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils


def map_target_spherical(tx, rx, az, el, rf):
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

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        r : float
            bistatic slant range in kilometers
        el : float np.array
            angle-of-arrival elevation in degrees
    """

    # Setup givens in correct units
    re = 6378.0e3  # Radius of earth in [m]
    rf = rf * 1.0e3
    az += 7.0
    az = np.where(az < 0.0, np.deg2rad(az + 360.0), np.deg2rad(az))
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

    for i in range(len(rf)):
        sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    # r = rf/2 - 200e3
    # sx[2, :] = -re + np.sqrt(re ** 2 + r ** 2 + 2 * re * r * np.sin(el))
    # new = -re + np.sqrt(re ** 2 + r ** 2 + 2 * re * r * np.sin(el))
    return sx[2, :] / 1.0e3, r / 1.0e3, np.rad2deg(el)


def relaxation_elevation(beta, rf_distance, azimuth, bistatic_distance, bistatic_vector):
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
    print('relaxation mean error:', np.rad2deg(np.nanmean(err)), 'iterations:', n)
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
    files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_12')  # Enter first sub directory and last
    rf_distance = np.array([])
    snr_db = np.array([])
    doppler_shift = np.array([])
    azimuth = np.array([])
    elevation = np.array([])
    elevation_extent = np.array([])
    azimuth_extent = np.array([])
    area = np.array([])
    for file in files:
        f = h5py.File(file, 'r')
        print(file)
        group = f['data']
        date = f['date']
        keys = group.keys()
        for key in keys:
            data = group[f'{key}']
            # Filter out dropped samples.
            if np.any(data['rf_distance'][()] < 150):
                continue
            rf_distance = np.append(rf_distance, data['rf_distance'][()])
            snr_db = np.append(snr_db, np.abs(data['snr_db'][()]))
            doppler_shift = np.append(doppler_shift, data['doppler_shift'][()])
            azimuth = np.append(azimuth, data['azimuth'][()])
            elevation = np.append(elevation, np.abs(data['elevation'][()]))
            elevation_extent = np.append(elevation_extent, data['elevation_extent'][()] / 4)
            azimuth_extent = np.append(azimuth_extent, data['azimuth_extent'][()])
            area = np.append(area, data['area'][()])

    # Pre-calculate and do altitude earth curvature corrections.
    altitude, slant_range, elevation = map_target_spherical([50.893, -109.403, 0.0],
                                                          [52.243, -106.450, 0.0],
                                                          azimuth,
                                                          elevation,
                                                          rf_distance)

    # Set up a filtering mask.
    m = np.ones_like(slant_range)
    # m = np.ma.masked_where(snr_db <= 3.0, m)
    # m = np.ma.masked_where(azimuth >= 315, m)
    # m = np.ma.masked_where(azimuth <= 225, m)
    # m = np.ma.masked_where(elevation >= 26, m)
    # m = np.ma.masked_where(elevation <= 1, m)
    m = np.ma.masked_where(elevation == np.nan, m)
    # m = np.ma.masked_where(slant_range <= 300, m)
    # m = np.ma.masked_where(slant_range >= 1200, m)
    # m = np.ma.masked_where(altitude <= 70, m)
    # m = np.ma.masked_where(altitude >= 130, m)
    slant_range = slant_range * m
    doppler_shift = doppler_shift * m
    snr_db = snr_db * m
    azimuth = azimuth * m
    azimuth_extent = azimuth_extent * m
    elevation = elevation * m
    elevation_extent = elevation_extent * m
    altitude = altitude * m
    slant_range = slant_range[~slant_range.mask]
    doppler_shift = doppler_shift[~doppler_shift.mask]
    snr_db = snr_db[~snr_db.mask]
    azimuth = azimuth[~azimuth.mask]
    azimuth_extent = azimuth_extent[~azimuth_extent.mask]
    elevation = elevation[~elevation.mask]
    elevation_extent = elevation_extent[~elevation_extent.mask]
    altitude = altitude[~altitude.mask]

    plt.figure()
    plt.subplot(211)
    plt.scatter(slant_range, altitude)
    plt.plot([0, 1200], [90, 90], '--k')
    plt.plot([0, 1200], [100, 100], '--m')
    plt.subplot(212)
    plt.scatter(azimuth, altitude)
    plt.plot([-45, 45], [90, 90], '--k')
    plt.plot([-45, 45], [100, 100], '--m')
    plt.show()
