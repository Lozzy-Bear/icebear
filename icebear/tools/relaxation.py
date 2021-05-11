import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils


def map_target(tx, rx, az, el, rf):
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
            total rf path distance rf = c * tau

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        r : float
            bistatic slant range in kilometers
        el :
        dr :
        dtheta :
    """

    # Setup givens in correct units
    rf = rf * 1.0e3 * 1.5# - 230e3
    az = np.where(az < 0, np.deg2rad(az + 367.0), np.deg2rad(az + 7.0))
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

    dr = np.zeros(100)
    dtheta = np.zeros(100)
    dr[0] = 1501
    dtheta[0] = 2 * np.pi
    r2 = np.copy(rf)
    theta2 = np.ones(len(az)) * 2 * np.pi
    cnt = 0
    el2 = np.copy(el)
    while dtheta[cnt] >= 0.384 and dr[cnt] >= 1500:  #0.002
        ua = np.array([np.sin(az) * np.cos(el2), np.cos(az) * np.cos(el2), np.sin(el2)])
        theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
        r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

        # Correct elevation using geocentric angle gamma and first order ranges, find scatter lat, long, alt
        for i in range(len(rf)):
            bx3, by3, bz3 = pm.aer2ecef(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                        rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
            us[:, i] = np.array([bx3, by3, bz3]) / np.linalg.norm([bx3, by3, bz3])  # gamma
            el2[i] = el[i] - np.arccos(ur[0] * us[0, i] + ur[1] * us[1, i] + ur[2] * us[2, i])  # alpha = chi - gamma

        cnt += 1
        dr[cnt] = np.max(np.abs(r - r2))
        dtheta[cnt] = np.max(np.abs(theta - theta2))
        r2 = np.copy(r)
        theta2 = np.copy(theta)

        print(cnt, dr[cnt]/1e3, np.rad2deg(dtheta[cnt]))

    for i in range(len(rf)):
        sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el2[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    # Set units to degrees and kilometers
    sx[2, :] /= 1.0e3
    r /= 1.0e3

    return sx[2, :], r, el, dr[0:cnt+1], dtheta[0:cnt+1]


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
            total rf path distance rf = c * tau

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        r : float
            bistatic slant range in kilometers
        el :
        dr :
        dtheta :
    """

    # Setup givens in correct units
    re = 6378.0e3  # Radius of earth in [m]
    rf = rf * 1.0e3 * 1.5# - 230e3
    az *= -1
    az = np.where(az < 0, np.deg2rad(az + 367.0), np.deg2rad(az + 7.0))
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
    print(b, ub)

    dr = np.zeros(100)
    dtheta = np.zeros(100)
    dr[0] = 1501
    dtheta[0] = 2 * np.pi
    r2 = np.copy(rf)
    theta2 = np.ones(len(az)) * 2 * np.pi
    cnt = 0
    el2 = np.copy(el)

    for j in range(1):
        ua = np.array([np.sin(az) * np.cos(el2), np.cos(az) * np.cos(el2), np.sin(el2)])
        theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
        r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

        # Correct elevation using geocentric angle gamma and first order ranges, find scatter lat, long, alt
        for i in range(len(rf)):
            bx3, by3, bz3 = pm.aer2ecef(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                        rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
            us[:, i] = np.array([bx3, by3, bz3]) / np.linalg.norm([bx3, by3, bz3])  # gamma
            el2[i] = el[i] - np.arccos(ur[0] * us[0, i] + ur[1] * us[1, i] + ur[2] * us[2, i])  # alpha = chi - gamma

        ua = np.array([np.sin(az) * np.cos(el2), np.cos(az) * np.cos(el2), np.sin(el2)])
        theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
        r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

        cnt += 1
        dr[cnt] = np.mean(np.abs(r - r2))
        dtheta[cnt] = np.mean(np.abs(theta - theta2))
        r2 = np.copy(r)
        theta2 = np.copy(theta)

        print(cnt, dr[cnt]/1e3, np.rad2deg(dtheta[cnt]))

    for i in range(len(rf)):
        sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el2[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)



    # plt.plot(r - (rf/2 -230e3))
    plt.figure()
    col = np.where(theta < 1.57, 'k', 'b')
    plt.scatter(r, sx[2, :], c=col)

    plt.figure()
    plt.scatter(np.rad2deg(az), np.rad2deg(theta), c=col)
    plt.show()

    # r = rf/2 - 230e3
    # sx[2, :] = -re + np.sqrt(re ** 2 + r ** 2 + 2 * re * r * np.sin(el2))



    # Set units to degrees and kilometers
    sx[2, :] /= 1.0e3
    r /= 1.0e3

    return sx[2, :], r, el, dr[0:cnt+1], dtheta[0:cnt+1]


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
    files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_12')  # Enter first sub directory and last
    cnt = 0
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
            rf_distance = np.append(rf_distance, data['rf_distance'][()])
            snr_db = np.append(snr_db, np.abs(data['snr_db'][()]))
            doppler_shift = np.append(doppler_shift, data['doppler_shift'][()])
            azimuth = np.append(azimuth, data['azimuth'][()])
            elevation = np.append(elevation, np.abs(data['elevation'][()]))
            elevation_extent = np.append(elevation_extent, data['elevation_extent'][()] / 4)  # Todo: May need to scale values!
            azimuth_extent = np.append(azimuth_extent, data['azimuth_extent'][()])
            area = np.append(area, data['area'][()])

    # Pre-calculate and do altitude earth curvature corrections.
    altitude, slant_range, elevation, dr, dt = map_target_spherical([50.893, -109.403, 0.0],
                                                          [52.243, -106.450, 0.0],
                                                          azimuth,
                                                          elevation,
                                                          rf_distance)

    # Set up a filtering mask.
    m = np.ones_like(slant_range)
    m = np.ma.masked_where(snr_db <= 3.0, m)
    # # m = np.ma.masked_where(azimuth >= 315, m)
    # # m = np.ma.masked_where(azimuth <= 225, m)
    # m = np.ma.masked_where(elevation >= 26, m)
    # m = np.ma.masked_where(elevation <= 1, m)
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
    plt.subplot(411)
    plt.scatter(slant_range, altitude)
    plt.subplot(412)
    plt.scatter(azimuth, altitude)
    plt.subplot(413)
    plt.plot(np.arange(len(dr)), dr, 'm', label='Slant Range Relaxation')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(np.arange(len(dt)), dt, 'k', label='Theta Relaxation')
    plt.legend(loc='best')
    plt.show()
