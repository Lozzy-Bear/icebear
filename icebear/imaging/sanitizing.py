import numpy as np
import sys
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False
import h5py
import pymap3d as pm
import icebear.utils as utils
import matplotlib.pyplot as plt
import icebear.imaging.clustering as cl
import datetime
from dateutil.tz import tzutc
import pandas

def map_target(tx, rx, az, el, rf, dop, wavelength):
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
    # doppler_sign = np.sign(dop)  # 1 for positive, -1 for negative, and 0 for zero
    doppler_sign = np.where(dop >= 0, 1, -1)  # 1 for positive, -1 for negative, and 0 for zero
    vaz, vel, _ = pm.ecef2aer(doppler_sign * u_bi[0, :] + x,
                              doppler_sign * u_bi[1, :] + y,
                              doppler_sign * u_bi[2, :] + z,
                              sx[0, :], sx[1, :], sx[2, :],
                              ell=pm.Ellipsoid("wgs84"), deg=True)

    # Convert back to conventional units
    sx[2, :] /= 1.0e3
    az = np.rad2deg(az)
    el = np.rad2deg(el)
    sa[:, :] = np.array([az, el, r / 1.0e3])
    sv[:, :] = np.array([vaz, vel, dop * radar_wavelength])

    return sx, sa, sv


def relaxation_elevation(beta, rf_distance, azimuth, bistatic_distance, bistatic_vector):
    """
    Due to the bistatic nature of ICEBEAR the problem of solving range and elevation is transcendental.
    This relaxation method applies numerical relaxation to derive a solution. Usually 2-3 iterations.

    Parameters
    ----------
    beta : float np.array
        Measured elevation angle in radians
    rf_distance : float np.array
        Total rf propagation distance in kilometers
    azimuth : float np.array
        Measured azimuth angle in radians
    bistatic_distance : float
        The distance separating the transmitter and receiver in kilometers
    bistatic_vector : float np.array
        Unit vector from the receiver to the transmitter

    Returns
    -------

    """
    n = 0
    radius_of_earth = 6378.0e3
    err = np.deg2rad(1.0)
    target = np.deg2rad(0.1)
    m = np.zeros((3, len(beta)))
    m[1, :] = 0.1
    v = np.array(
        [np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]), np.sin(beta - m[1, :])])
    r = (rf_distance ** 2 - bistatic_distance ** 2) / (
                2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
    m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
    while np.nanmean(err) > target:
        m[0, :] = m[1, :]
        m[1, :] = m[2, :]
        v = np.array([np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]),
                      np.sin(beta - m[1, :])])
        r = (rf_distance ** 2 - bistatic_distance ** 2) / (
                    2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
        m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
        err = np.abs((m[1, :] - m[2, :]) ** 2 / (2 * m[1, :] - m[0, :] - m[2, :]))
        n += 1

    m[2, :] = np.where(err >= target, np.nan, m[2, :])
    print('\t-relaxation mean error:', np.rad2deg(np.nanmean(err)), 'iterations:', n)
    return m[2, :]


def calculate_clustering(la, lo, ti, az, el, al, k=500_000, beam_pattern='3lam'):
    """
    Calculates the spatial and temporal clustering values for each point in the input arrays. Calculations performed on
    the GPU using cupy. Default spans to use for the calculation are the nearest 512 points in time and the
    nearest 512 points in space within approximately (NOT exactly) 4 hours.

    Parameters
    ----------
    la : 1D ndarray
        Vector array of latitude values in [deg]
    lo : 1D ndarray
        Vector array of longitude values in [deg]
    ti : 1D ndarray
        Vector array of time values in [s]
    az : 1D ndarray
        Vector array of azimuth values in [deg]
    el : 1D ndarray
        Vector array of elevation values in [deg]
    al : 1D ndarray
        Vector array of altitude values in [km]
    k  : int
        Half of the maximum number of points thought to be within tspan hours of each point. Default 500_000. Used
        for chunking purposes. Don't recommend going higher than 1_000_000 or there may be GPU memory issues
    beam_pattern : string
        '1lam' or '3lam' depending on transmitter configuration

    Returns
    -------
    beam : 1D ndarray
        Same shape as the input arrays. Classifies each point into a beam.
        Possible values:
            -1 : the corresponding point is not in a beam
            1  : the corresponding point is in the east beam
            2  : the corresponding point is in the centre beam
            3  : the corresponding point is in the west beam
    dr : 1D ndarray
        Same shape as the input arrays. The spatial clustering values in [km]. -1 indicates a point not in a beam
    dt : 1D ndarray
        Same shape as the input arrays. The temporal clustering values in [s]. -1 indicates a point not in a beam
    """

    # Only assess times that ran with 3-lambda seperation
    try:
        date = pandas.Timestamp(datetime.datetime.fromtimestamp(ti[0], tz=tzutc()))
        dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=tzutc())
        schedule = pandas.read_csv(f'/mnt/icebear/schedule_files/{date.year:04d}-{date.month:02d}.tx_compiled_runtimes.txt', parse_dates=['start time', 'end time'], date_parser=dateparse, na_filter=False)
        idx = np.where((schedule['start time'].dt.day == date.day) | (schedule['end time'].dt.day == date.day))
	
        ti_mask = np.zeros(len(ti))
        for i in range(len(ti)):
            date = pandas.Timestamp(datetime.datetime.fromtimestamp(ti[i], tz=tzutc()))
            for x in idx[0]:
                if (schedule['start time'][x] <= date) and (schedule['end time'][x] >= date):
                    if ('3lam' in schedule['notes'][x]) or ('NA' in schedule['notes'][x]):
                        ti_mask[i] = 1
    
        if np.all(ti_mask == 0):
            return 'NA', 'NA', 'NA'

    except Exception as e:
        print('Assumed to be pre tx schedules')
        print(e)
        ti_mask=np.ones(len(ti))

    #ti = np.ma.masked_where(ti_mask == 0, ti)
    
    beam = cl.beam_finder(la, lo, beam_pattern)
    beam[al > 150] = -1
    beam[al < 70] = -1

    arr = xp.array([ti.data, la.data, lo.data])
    trimmed_arr = arr[:, beam > 0]
    dr = -1 * np.ones(arr.shape[1], dtype=np.float32)
    dt = -1 * np.ones(arr.shape[1], dtype=np.float32)

    # calculate clustering values
    dr[beam > 0], dt[beam > 0] = cl.cluster_medians(trimmed_arr, k)

    return np.where(ti_mask == 0, -2, beam), np.where(ti_mask == 0, -1, dr), np.where(ti_mask == 0, -1, dt)

def create_level2_sanitized_hdf5(config, filename,
                       epoch_time, rf_distance, snr_db, doppler_shift,
                       lattitude, longitude, altitude,
                       azimuth, elevation, slant_range,
                       velocity_azimuth, velocity_elevation, velocity,
                       azimuth_extent, elevation_extent, area, raw_elevation, beam,
                       spatial_cluster, temporal_cluster):
    # Add general information
    # general information
    f = h5py.File(filename, 'w')
    f.create_group('info')
    f.create_dataset('info/date_created', data=np.array(config.date_created))
    f.create_dataset('info/version', data=np.array(config.version, dtype='S'))
    f.create_dataset('info/date', data=config.date)
    f.create_dataset('info/experiment_name', data=np.array([config.experiment_name], dtype='S'))
    f.create_dataset('info/radar_config', data=np.array([config.radar_config], dtype='S'))
    f.create_dataset('info/center_freq', data=config.center_freq)
    # receiver site information
    f.create_dataset('info/rx_site_name', data=np.array([config.rx_site_name], dtype='S'))
    f.create_dataset('info/rx_site_lat_long', data=config.rx_site_lat_long)
    f.create_dataset('info/rx_heading', data=config.rx_heading)
    f.create_dataset('info/rx_rf_path', data=np.array([config.rx_rf_path], dtype='S'))
    f.create_dataset('info/rx_ant_type', data=np.array([config.rx_ant_type], dtype='S'))
    f.create_dataset('info/rx_ant_coords', data=config.rx_ant_coords)
    f.create_dataset('info/rx_feed_corr', data=config.rx_feed_corr)
    f.create_dataset('info/rx_feed_corr_date', data=config.rx_feed_corr_date)
    f.create_dataset('info/rx_feed_corr_type', data=np.array([config.rx_feed_corr_type], dtype='S'))
    f.create_dataset('info/rx_ant_mask', data=config.rx_ant_mask)
    f.create_dataset('info/rx_sample_rate', data=config.rx_sample_rate)
    # transmitter site information
    f.create_dataset('info/tx_site_name', data=np.array([config.tx_site_name], dtype='S'))
    f.create_dataset('info/tx_site_lat_long', data=config.tx_site_lat_long)
    f.create_dataset('info/tx_heading', data=config.tx_heading)
    f.create_dataset('info/tx_rf_path', data=np.array([config.tx_rf_path], dtype='S'))
    f.create_dataset('info/tx_ant_type', data=np.array([config.tx_ant_type], dtype='S'))
    f.create_dataset('info/tx_ant_coords', data=config.tx_ant_coords)
    f.create_dataset('info/tx_feed_corr', data=config.tx_feed_corr)
    f.create_dataset('info/tx_feed_corr_date', data=config.tx_feed_corr_date)
    f.create_dataset('info/tx_feed_corr_type', data=np.array([config.tx_feed_corr_type], dtype='S'))
    f.create_dataset('info/tx_ant_mask', data=config.tx_ant_mask)
    f.create_dataset('info/tx_sample_rate', data=config.tx_sample_rate)
    # processing settings
    f.create_dataset('info/decimation_rate', data=config.decimation_rate)
    f.create_dataset('info/time_resolution', data=config.time_resolution)
    f.create_dataset('info/coherent_integration_time', data=config.coherent_integration_time)
    f.create_dataset('info/incoherent_averages', data=config.incoherent_averages)
    f.create_dataset('info/snr_cutoff_db', data=config.snr_cutoff_db)
    # imaging settings
    f.create_dataset('info/image_method', data=np.array([config.image_method], dtype='S'))
    f.create_dataset('info/swht_coeffs', data=np.array([config.swht_coeffs], dtype='S'))
    f.create_dataset('info/fov', data=config.fov)
    f.create_dataset('info/fov_center', data=config.fov_center)
    f.create_dataset('info/resolution', data=config.resolution)
    f.create_dataset('info/lmax', data=config.lmax)
        # baselines?

    # Create datasets
    f.create_group('data')
    # f.create_dataset('data/time_indices', data=time_indices)
    f.create_dataset('data/time', data=epoch_time)
    f.create_dataset('data/rf_distance', data=rf_distance)
    f.create_dataset('data/snr_db', data=snr_db)
    f.create_dataset('data/doppler_shift', data=doppler_shift)
    # f.create_dataset('data/doppler_spectra', data=doppler_spectra)
    f.create_dataset('data/latitude', data=lattitude)
    f.create_dataset('data/longitude', data=longitude)
    f.create_dataset('data/altitude', data=altitude)
    f.create_dataset('data/azimuth', data=azimuth)
    f.create_dataset('data/elevation', data=elevation)
    f.create_dataset('data/slant_range', data=slant_range)
    f.create_dataset('data/velocity_azimuth', data=velocity_azimuth)
    f.create_dataset('data/velocity_elevation', data=velocity_elevation)
    f.create_dataset('data/velocity_magnitude', data=velocity)
    # dset.attrs['units'] = 'm/s'
    # dset.attrs['description'] = 'magnitude of the target velocity vector'
    # dset.attrs['dtype'] = dset.dtype
    # dset.attrs['flavor'] = 'numpy.array'


    # Create dev datasets
    f.create_group('dev')
    f.create_dataset('dev/raw_elevation', data=raw_elevation)
    f.create_dataset('dev/mean_jansky', data=azimuth_extent)
    f.create_dataset('dev/max_jansky', data=elevation_extent)
    f.create_dataset('dev/valid', data=area)
    # Magnus clustering data
    f.create_dataset('dev/beam', data=beam)
    f.create_dataset('dev/spatial_cluster', data=spatial_cluster)
    f.create_dataset('dev/temporal_cluster', data=temporal_cluster)
    # f.create_dataset('dev/azimuth_extent', data=azimuth_extent)
    # f.create_dataset('dev/elevation_extent', data=elevation_extent)
    # f.create_dataset('dev/area', data=area)
    # f.create_dataset('dev/doppler_spectra', data=doppler_spectra)

    f.close()

    return None


if __name__ == '__main__':
    # Load the level 2 data file.
    # filepath = '/beaver/backup/level2_advanced_cuda/'  # Enter file path to level 2 directory
    filepath = 'HERE'  # Enter file path to level 2 directory
    date_dir = 'HERE'

    for arg in range(len(sys.argv)):
        if sys.argv[arg] == "-d":
            date_dir = str(sys.argv[arg + 1])
        elif sys.argv[arg] == "-p":
            filepath = str(sys.argv[arg + 1])

    if (filepath == 'HERE') or (date_dir == 'HERE'):
        print("Did not specify date and filepath")
        sys.exit()

    files = utils.get_all_data_files(filepath, date_dir, date_dir)  # Enter first sub directory and last
    print(f'files: {files}')

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
        if 'altitude_distribution' in file:
            continue
        if f'swht_{date_dir}' in file:
            continue
        f = h5py.File(file, 'r')
        config = utils.Config(file)
        print(file)
        group = f['data']
        date = f['date']
        keys = group.keys()
        for key in keys:
            data = group[f'{key}']
            # Filter out dropped samples
            if np.any(data['rf_distance'][()] < 250):
                continue
            et = data['time'][()]
            et = np.append(date, et)
            et[5] = int(et[5] / 1000)
            et = utils.epoch_time(et)
            t = np.append(t, np.repeat(et, len(data['rf_distance'][()])))
            rf_distance = np.append(rf_distance, data['rf_distance'][()])
            snr_db = np.append(snr_db, np.abs(data['snr_db'][()]))
            doppler_shift = np.append(doppler_shift, data['doppler_shift'][()])
            azimuth = np.append(azimuth, data['azimuth'][()])
            elevation = np.append(elevation, np.abs(data['elevation'][()]))

            try:
                elevation_extent = np.append(elevation_extent, data['max_jansky'][()])
                azimuth_extent = np.append(azimuth_extent, data['mean_jansky'][()])
                area = np.append(area, data['valid'][()])
            except Exception as e:
                elevation_extent = np.append(elevation_extent, np.zeros_like(data['elevation'][()]))
                azimuth_extent = np.append(azimuth_extent, np.zeros_like(data['elevation'][()]))
                area = np.append(area, np.zeros_like(data['elevation'][()]))
                
            # elevation_extent = np.append(elevation_extent, data['elevation_extent'][()])
            # azimuth_extent = np.append(azimuth_extent, data['azimuth_extent'][()])
            # area = np.append(area, data['area'][()])

    year, month, day = date_dir.split('_')
    filename = f'{filepath}/{year}/{month}/{date_dir}/' \
               f'{config.radar_config}_{config.experiment_name}_swht_' \
               f'{date_dir}_{config.tx_site_name}_{config.rx_site_name}.h5'
    print(filename)
    print('\t-loading completed')
    # Set the azimuth pointing direction

    azimuth += config.rx_heading
    print('\t-total data', len(rf_distance))
    # Pre-masking
    m = np.ones_like(rf_distance)
    m = np.ma.masked_where(snr_db <= 1.0, m)  # Weak signals close to noise or highly multipathed (meteors are strong)

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

    if np.abs(config.rx_site_lat_long[0]) < 10:
        rx_coord = np.rad2deg(config.rx_site_lat_long)
    else:
        rx_coord = config.rx_site_lat_long

    if np.abs(config.tx_site_lat_long[0]) < 10:
        tx_coord = np.rad2deg(config.tx_site_lat_long)
    else:
        tx_coord = config.tx_site_lat_long

    if len(rx_coord) < 3:
        rx_coord = np.append(rx_coord, 0.0)
        print(rx_coord)

    if len(tx_coord) < 3:
        tx_coord = np.append(tx_coord, 0.0)
        print(tx_coord)

    sx, sa, sv = map_target(tx_coord, rx_coord,
                            azimuth, elevation, rf_distance,
                            doppler_shift, 299_762_458.0/config.center_freq)
    lat = sx[0, :]
    lon = sx[1, :]
    altitude = sx[2, :]
    vaz = sv[0, :]
    vel = sv[1, :]
    vma = sv[2, :]
    slant_range = sa[2, :]
    raw_elevation = np.copy(elevation)
    elevation = sa[1, :]
    print('\t-mapping completed')

    # Set up a filtering mask.
    m = np.ma.masked_where(elevation == np.nan, m)  # Elevation iterations not converging (noise)
    m = np.ma.masked_where(elevation <= 0.0, m)  # Elevation below the ground
    m = np.ma.masked_where(slant_range <= 300.0, m)  # Man made noise and multipath objects
    rf_distance = rf_distance * m
    snr_db = snr_db * m
    doppler_shift = doppler_shift * m
    azimuth = azimuth * m
    azimuth_extent = azimuth_extent * m
    raw_elevation = raw_elevation * m
    elevation = elevation * m
    elevation_extent = elevation_extent * m
    area = area * m
    slant_range = slant_range * m
    altitude = altitude * m
    t = t * m
    lat = lat * m
    lon = lon * m
    vaz = vaz * m
    vel = vel * m
    vma = vma * m

    rf_distance = rf_distance[~rf_distance.mask].flatten()
    snr_db = snr_db[~snr_db.mask].flatten()
    doppler_shift = doppler_shift[~doppler_shift.mask].flatten()
    azimuth = azimuth[~azimuth.mask].flatten()
    azimuth_extent = azimuth_extent[~azimuth_extent.mask].flatten()
    raw_elevation = raw_elevation[~raw_elevation.mask].flatten()
    elevation = elevation[~elevation.mask].flatten()
    elevation_extent = elevation_extent[~elevation_extent.mask].flatten()
    area = area[~area.mask].flatten()
    slant_range = slant_range[~slant_range.mask].flatten()
    altitude = altitude[~altitude.mask].flatten()
    t = t[~t.mask].flatten()
    lat = lat[~lat.mask].flatten()
    lon = lon[~lon.mask].flatten()
    vaz = vaz[~vaz.mask].flatten()
    vel = vel[~vel.mask].flatten()
    vma = vma[~vma.mask].flatten()

    print('\t-masking completed')
    print('\t-remaining data', len(rf_distance))

    beam, spatial_cluster, temporal_cluster = calculate_clustering(lat, lon, t, azimuth, elevation, altitude, k=500_000)
    print(beam)

    create_level2_sanitized_hdf5(config, filename, t, rf_distance, snr_db, doppler_shift,
                       lat, lon, altitude, azimuth, elevation, slant_range, vaz, vel, vma,
                       azimuth_extent, elevation_extent, area, raw_elevation, beam, spatial_cluster, temporal_cluster)

    # Draven this little bit of code can be used to make altitude histograms per day if we want one every day!

    plt.figure(figsize=[12, 12])
    mean_altitude = np.mean(altitude)
    total_targets = len(altitude)
    _ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype=u'step', label=f'Total Targets {total_targets}')
    plt.xscale('log')
    plt.title(f'Altitude Distribution {date_dir}')
    plt.xlabel('Count')
    plt.ylabel('Altitude [km]')
    # plt.ylim((50, 200))
    # plt.xlim((10, 10_000))
    plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Mean Altitude {mean_altitude:.1f} [km]')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(f'{filepath}/{year}/{month}/{date_dir}/altitude_distribution_{date_dir}.png')
    # plt.show()
    plt.close()

    # This is how we decided file naming conventions should be.
    # Level 1 Data: ib3d_normal_2020_02_20_01_prelate_bakker.h5  <- one hour
    # Level 2.dev Data: ib3d_normal_dev_swht_2020_02_20_01_prelate_bakker.h5  <- one hour
    # Level 2 Data: ib3d_normal_swht_2020_02_20_prelate_bakker.h5  <- full day

