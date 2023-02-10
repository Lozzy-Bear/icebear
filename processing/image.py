import numpy as np
import common.utils
import h5py
try:
    import cupy as cp
except:
    print('no cupy')
import processing.swht


def generate_level2(config, method='swht'):
    """

    Parameters
    ----------
    config :
    method : str
        Imaging method to use, options include; swht, linear

    Returns
    -------

    """
    if method == 'linear':
        # todo
        calculate_image = _linear_method
        args = () # Devin any specific args you need to handle should be passed as a tuple
    elif method == 'swht':
        calculate_image = _swht_method
        args = (processing.swht.unpackage_coeffs(config.swht_coeffs, int(config.lmax)),
                config.resolution,
                config.fov,
                config.fov_center)
    elif method == 'advanced':
        calculate_image = _swht_method_advanced
        coeffs_full = processing.swht.unpackage_coeffs(config.swht_coeffs_lowres, int(config.lmax))
        coeffs_fov = np.zeros((450, 900, 92, 8), dtype=np.complex64)
        idx = 0
        for i in range(15, 95, 10):
            coeffs_fov[:, :, :, idx] = processing.swht.unpackage_coeffs(config.swht_coeffs, i)
            idx += 1
        coeffs_fov = cp.asarray(coeffs_fov)
        args = (coeffs_fov,
                coeffs_full,
                config.resolution,
                config.fov,
                config.fov_center)
    elif method == 'graphic':
        calculate_image = _graphic_method
        args = (processing.swht.unpackage_coeffs(config.swht_coeffs, int(config.lmax)),
                config.resolution,
                config.fov,
                config.fov_center)
    else:
        print(f'ERROR: the imaging method {method} does not exist.')
        exit()

    print('imaging start:')
    print(f'\t-imaging method: {method}')

    file = h5py.File(config.imaging_source, 'r')
    time = common.utils.Time(config.imaging_start, config.imaging_stop, config.imaging_step)
    temp_hour = [-1, -1, -1, -1]
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        now = time.get_date(t)
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            filename = f'{config.imaging_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/' \
                f'{config.radar_config}_{config.experiment_name}_{config.image_method}_{int(config.resolution * 10):02d}deg_' \
                f'{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}_{int(now.hour):02d}_' \
                f'{config.tx_site_name}_{config.rx_site_name}.h5'
            print(f'\t-created level 2 HDf5: {filename}')
            create_level2_hdf5(config, filename, int(now.year), int(now.month), int(now.day))
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]
        data = file['data'][f'{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}']
        if data['data_flag'][()]:
            # todo
            # Devin notice that the args argument is detupled before passing and everything else is standard.
            # In this way we can pass any amount of additional arguments determined by method.
            # data will pass by reference to the current hdf5 data we are working on.
            calculate_image(filename, int(now.hour), int(now.minute), int(now.second * 1000), data, *args)
            print(f'\t-appended: {int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}')
            exit()

    return None


def create_level2_hdf5(config, filename, year, month, day):
    """

    Parameters
    ----------
    config
    filename
    year
    month
    day

    Returns
    -------

    """
    # general information
    f = h5py.File(filename, 'w')
    f.create_dataset('date_created', data=np.array(config.date_created))
    f.create_dataset('version', data=np.array(config.version, dtype='S'))
    f.create_dataset('date', data=np.array([year, month, day]))
    f.create_dataset('experiment_name', data=np.array([config.experiment_name], dtype='S'))
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
    f.create_dataset('image_method', data=np.array([config.image_method], dtype='S'))
    #f.create_dataset('clean', data=np.array([config.clean], dtype='S'))
    #f.create_dataset('center', data=np.array([config.center], dtype='S'))
    f.create_dataset('swht_coeffs', data=np.array([config.swht_coeffs], dtype='S'))
    f.create_dataset('fov', data=config.fov)
    f.create_dataset('fov_center', data=config.fov_center)
    f.create_dataset('resolution', data=config.resolution)
    f.create_dataset('lmax', data=config.lmax)
    f.create_group('data')
    f.close()

    return None


def append_level2_hdf5(filename, hour, minute, second, doppler_shift, snr_db, rf_distance):
    """
    Appends to the hdf5 the standard data sets.

    Parameters
    ----------
    filename
    hour
    minute
    second
    doppler_shift
    snr_db
    rf_distance

    Returns
    -------

    """
    # append a new group for the current measurement
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    f.create_dataset(f'data/{time}/doppler_shift', data=doppler_shift)
    f.create_dataset(f'data/{time}/snr_db', data=snr_db)
    f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
    f.close()

    return None


def _linear_method(filename, now, data, any, other, args, like, this):
    # todo
    # Devin to do append work here see my _swht_method example.
    # We do the _ in front to indicate this is not a function normally accessible by the user
    return


def _swht_method(filename, hour, minute, second, data, coeffs, resolution, fov, fov_center):
    """
    Sets up the environment for imaging with the SWHT method with standard parameters and appends the level 2 HDF5 file
    for both standard measurements and SWHT specific ones.

    Parameters
    ----------
        filename : string
            Name of the hdf5 file to be appended.
        hour : int
            Hour of the data passed.
        minute : int
            Minute of the data passed.
        second : int
            Second of the data passed.
        data : dict hdf5
            HDF5 structure of the level 1 data
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.

    Returns
    -------
        None
    """

    doppler_shift = data['doppler_shift'][()]
    # This is a little hack to check if we are seeing a dropped sample.
    # Dropped samples always have data for way more range-Doppler bins and that never occurs with real data.
    if len(doppler_shift) >= 9000:#28000:
        print('\t-dropped sample detected; skipped')
        return
    rf_distance = data['rf_distance'][()]
    snr_db = data['snr_db'][()]
    visibilities = np.array(data['spectra'][:, 0], dtype=np.complex64)[:, np.newaxis]
    visibilities = np.append(visibilities, data['xspectra'][:, :], axis=1)
    visibilities = np.append(visibilities, np.conjugate(visibilities), axis=1)
    azimuth = np.empty_like(doppler_shift)
    elevation = np.empty_like(doppler_shift)
    azimuth_extent = np.empty_like(doppler_shift)
    elevation_extent = np.empty_like(doppler_shift)
    area = np.empty_like(doppler_shift)

    for idx, visibility in enumerate(visibilities):
        azimuth[idx], elevation[idx], azimuth_extent[idx], elevation_extent[idx], area[idx] = \
            processing.swht.swht_method(visibility, coeffs, resolution, fov, fov_center)

    # Custom data appending for SWHT image data sets
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    f.create_dataset(f'data/{time}/doppler_shift', data=doppler_shift)
    f.create_dataset(f'data/{time}/snr_db', data=snr_db)
    f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
    f.create_dataset(f'data/{time}/azimuth', data=azimuth)
    f.create_dataset(f'data/{time}/elevation', data=elevation)
    f.create_dataset(f'data/{time}/azimuth_extent', data=azimuth_extent)
    f.create_dataset(f'data/{time}/elevation_extent', data=elevation_extent)
    f.create_dataset(f'data/{time}/area', data=area)
    f.close()

    return


def _swht_method_advanced(filename, hour, minute, second, data, coeffs_fov, coeffs_full, resolution, fov, fov_center):
    """
    Sets up the environment for imaging with the SWHT method with standard parameters and appends the level 2 HDF5 file
    for both standard measurements and SWHT specific ones.

    Parameters
    ----------
        filename : string
            Name of the hdf5 file to be appended.
        hour : int
            Hour of the data passed.
        minute : int
            Minute of the data passed.
        second : int
            Second of the data passed.
        data : dict hdf5
            HDF5 structure of the level 1 data
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.

    Returns
    -------
        None
    """

    doppler_shift = data['doppler_shift'][()]
    # This is a little hack to check if we are seeing a dropped sample.
    # Dropped samples always have data for way more range-Doppler bins and that never occurs with real data.
    if len(doppler_shift) >= 9000:#28000:
        print('\t-dropped sample detected; skipped')
        return
    rf_distance = data['rf_distance'][()]
    snr_db = data['snr_db'][()]
    visibilities = np.array(data['spectra'][:, 0], dtype=np.complex64)[:, np.newaxis]
    visibilities = np.append(visibilities, data['xspectra'][:, :], axis=1)
    visibilities = np.append(visibilities, np.conjugate(visibilities), axis=1)
    azimuth = np.empty_like(doppler_shift)
    elevation = np.empty_like(doppler_shift)
    mean_jansky = np.empty_like(doppler_shift)
    max_jansky = np.empty_like(doppler_shift)
    valid = np.empty_like(doppler_shift)

    for idx, visibility in enumerate(visibilities):
        azimuth[idx], elevation[idx], mean_jansky[idx], max_jansky[idx], valid[idx] = \
            processing.swht.swht_method_advanced_cuda(visibility, coeffs_fov, coeffs_full,
                                                      resolution, fov, fov_center)

    # Custom data appending for SWHT image data sets
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    f.create_dataset(f'data/{time}/doppler_shift', data=doppler_shift)
    f.create_dataset(f'data/{time}/snr_db', data=snr_db)
    f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
    f.create_dataset(f'data/{time}/azimuth', data=azimuth)
    f.create_dataset(f'data/{time}/elevation', data=elevation)
    f.create_dataset(f'data/{time}/mean_jansky', data=mean_jansky)
    f.create_dataset(f'data/{time}/max_jansky', data=max_jansky)
    f.create_dataset(f'data/{time}/valid', data=valid)
    f.close()

    return


def _graphic_method(filename, hour, minute, second, data, coeffs, resolution, fov, fov_center):
    """
    Sets up the environment for imaging with the SWHT method with standard parameters and appends the level 2 HDF5 file
    for both standard measurements and SWHT specific ones.

    Parameters
    ----------
        filename : string
            Name of the hdf5 file to be appended.
        hour : int
            Hour of the data passed.
        minute : int
            Minute of the data passed.
        second : int
            Second of the data passed.
        data : dict hdf5
            HDF5 structure of the level 1 data
        coeffs : complex128 np.array
            Complex matrix of coefficients for the SWHT with dimension fov / resolution.

    Returns
    -------
        None
    """

    doppler_shift = data['doppler_shift'][()]
    # This is a little hack to check if we are seeing a dropped sample.
    # Dropped samples always have data for way more range-Doppler bins and that never occurs with real data.
    if len(doppler_shift) >= 9000:#28000:
        print('\t-dropped sample detected; skipped')
        return
    rf_distance = data['rf_distance'][()]
    snr_db = data['snr_db'][()]
    visibilities = np.array(data['spectra'][:, 0], dtype=np.complex64)[:, np.newaxis]
    visibilities = np.append(visibilities, data['xspectra'][:, :], axis=1)
    visibilities = np.append(visibilities, np.conjugate(visibilities), axis=1)

    vertices_geo = []
    for idx, visibility in enumerate(visibilities):
        vg = processing.swht.graphic_method(visibility, coeffs, resolution, fov, fov_center,
                                            rf_distance[idx],
                                            doppler_shift[idx],
                                            snr_db[idx],
                                            6.056)
        vertices_geo.append(vg)
        print(f"\t{idx:04d} / {visibilities.shape[0]}")

    import pickle
    file = open('important4', 'wb')
    pickle.dump(vertices_geo, file)

    # Custom data appending for SWHT image data sets
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    f.create_dataset(f'data/{time}/doppler_shift', data=doppler_shift)
    f.create_dataset(f'data/{time}/snr_db', data=snr_db)
    f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
    f.create_dataset(f'data/{time}/vertices_geo', data=vertices_geo, dtype=object)
    f.close()

    return


if __name__ == '__main__':
    # file = 'E:/icebear/level1/2022_22_22/ib3d_normal_01dB_1000ms_2019_10_28_06_prelate_bakker.h5'
    # file = '/beaver/backup/level1/2020_03_31/ib3d_normal_01dB_1000ms_2020_03_31_13_prelate_bakker.h5'
    files = ['/data/icebear_datastore/ICEBEAR_Level1_data/2021/02/2021_02_21/ib3d_normal_01dB_1000ms_2021_02_21_05_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
    # files = ['/run/media/arl203/Seagate Expansion Drive/backup/level1/2021_02_02/ib3d_normal_01dB_1000ms_2021_02_02_05_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
    # files = ['/run/media/arl203/Seagate Expansion Drive/backup/level1/2019_12_19/ib3d_normal_01dB_1000ms_2019_12_19_05_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
    # files = ['/run/media/arl203/Seagate Expansion Drive/backup/level1/2020_03_31/ib3d_normal_01dB_1000ms_2020_03_31_03_prelate_bakker.h5']  # Alternate; make a list of files with file paths.
    for file in files:
        config = common.utils.Config(file)
        # config.add_attr('imaging_destination', 'E:/icebear/level2_advanced_cuda/')
        # config.add_attr('imaging_destination', '/beaver/backup/level2_magnus/')
        config.add_attr('imaging_destination', '/data/icebear_data/graphic/')
        config.add_attr('imaging_source', file)
        imaging_start, imaging_stop = common.utils.get_data_file_times(file)
        imaging_start[4] = 41  # 22  # 22
        imaging_start[5] = 55  # 54  # 33
        imaging_step = [0, 0, 0, 1, 0]
        config.add_attr('imaging_start', imaging_start)
        config.add_attr('imaging_stop', imaging_stop)
        config.add_attr('imaging_step', imaging_step)
        config.add_attr('lmax', 85)
        config.add_attr('resolution', 0.1)
        config.add_attr('image_method', 'graphic')
        # config.add_attr('fov', np.array([[0, 360], [0, 180]]))
        # config.add_attr('fov', np.array([[360, 0], [90, 0]]))
        config.add_attr('fov', np.array([[315, 225], [90, 45]]))
        # config.add_attr('fov_center', np.array([90, 90]))
        # config.add_attr('fov_center', np.array([270, 90]))
        config.add_attr('fov_center', np.array([270, 90]))
        config.add_attr('swht_coeffs', '/home/arl203/icebear/icebear/dat/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5')
        # config.add_attr('swht_coeffs_lowres', '/home/arl203/icebear/icebear/dat/swhtcoeffs_ib3d_2021_10_19_360az_090el_10res_85lmax.h5')
        # config.add_attr('swht_coeffs', '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5')
        generate_level2(config, method='graphic')
