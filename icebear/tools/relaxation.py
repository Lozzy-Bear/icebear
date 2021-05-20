import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils


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

    Notes
    -----
    Todo : add velocity vector, compute lat long, process pymap3d faster (vectorize)
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

    # WGS84 Model for lat, long, alt.
    # Todo : This is very slow and needs to be vectorized!!!
    # for i in range(len(rf)):
    #     sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
    #                             rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    # aer2geodetic = np.vectorize(pm.aer2geodetic)
    # sx[:, :] = aer2geodetic(np.rad2deg(az), np.rad2deg(el), np.abs(r),
    #                         rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    # Spherical Earth approximation
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
    altitude, slant_range, elevation = map_target_spherical([50.893, -109.403, 0.0],
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

    plt.figure(figsize=[16, 24])
    plt.subplot(411)
    # plt.title(f'{key}')
    plt.scatter(slant_range, altitude, c=doppler_shift, vmin=-150, vmax=150, cmap='jet_r', label=f'Total Targets {total_targets}')
    plt.colorbar(label='Doppler Shift [m/s]')
    plt.xlabel('Slant Range [km]')
    plt.ylabel('Altitude [km]')
    plt.xlim([0.0, 1200.0])
    plt.ylim([0.0, 200.0])
    plt.plot([0.0, 1200.0], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')

    plt.subplot(412)
    plt.scatter(azimuth, altitude, c=doppler_shift, vmin=-150, vmax=150, cmap='jet_r', label=f'Total Targets {total_targets}')
    plt.colorbar(label='Doppler Shift [m/s]')
    plt.xlabel('Azimuth Angle from North [deg]')
    plt.ylabel('Altitude [km]')
    plt.ylim([0.0, 200.0])
    plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')

    plt.subplot(413)
    plt.scatter(slant_range, altitude, c=snr_db, vmin=0.0, vmax=20.0, cmap='plasma_r', label=f'Total Targets {total_targets}')
    plt.colorbar(label='Signal-to-Noise [dB]')
    plt.xlabel('Slant Range [km]')
    plt.ylabel('Altitude [km]')
    plt.xlim([0.0, 1200.0])
    plt.ylim([0.0, 200.0])
    plt.plot([0.0, 1200.0], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')

    plt.subplot(414)
    plt.scatter(azimuth, altitude, c=snr_db, vmin=0.0, vmax=20.0, cmap='plasma_r', label=f'Total Targets {total_targets}')
    plt.colorbar(label='Signal-to-Noise [dB]')
    plt.xlabel('Azimuth Angle from North [deg]')
    plt.ylabel('Altitude [km]')
    plt.ylim([0.0, 200.0])
    plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')

    # plt.savefig(f'/beaver/backup/geminids/summary/snr_dop_rng_az_filtered_02.png')
    # plt.close()
    # print(f'-time complete: {key}')

    # plt.figure(figsize=[24, 24])
    # plt.subplot(211)
    # plt.scatter(slant_range, altitude, c=area, cmap='inferno', label=f'Total Targets {total_targets}')
    # plt.colorbar(label='Scattering Cross-section')
    # plt.xlabel('Slant Range [km]')
    # plt.ylabel('Altitude [km]')
    # plt.plot([0, np.max(slant_range)], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(212)
    # plt.scatter(azimuth, altitude, c=area, cmap='inferno', label=f'Total Targets {total_targets}')
    # plt.colorbar(label='Scattering Cross-section')
    # plt.xlabel('Azimuth Angle from North [deg]')
    # plt.ylabel('Altitude [km]')
    # plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    # plt.legend(loc='upper right')
    #
    # plt.figure(figsize=[24, 24])
    # plt.subplot(211)
    # plt.scatter(slant_range, altitude, c=t, cmap='Greys', label=f'Total Targets {total_targets}')
    # plt.colorbar(label='Time')
    # plt.xlabel('Slant Range [km]')
    # plt.ylabel('Altitude [km]')
    # plt.plot([0, np.max(slant_range)], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(212)
    # plt.scatter(azimuth, altitude, c=t, cmap='Greys', label=f'Total Targets {total_targets}')
    # plt.colorbar(label='Time')
    # plt.xlabel('Azimuth Angle from North [deg]')
    # plt.ylabel('Altitude [km]')
    # plt.plot([-50+7, 50+7], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    # plt.legend(loc='upper right')
    # print('\t-plotting completed')

    plt.figure(figsize=[12, 12])
    _ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype=u'step', label=f'Total Targets {total_targets}')
    plt.xscale('log')
    plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude Distribution')
    plt.xlabel('Count')
    plt.ylabel('Altitude [km]')
    plt.ylim((0, 200))
    plt.xlim((10, 10_000))
    plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')
    # plt.savefig(f'/beaver/backup/geminids/summary/altitude_histogram_filtered_02.png')

    plt.show()
