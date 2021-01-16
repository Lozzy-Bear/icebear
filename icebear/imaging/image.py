import cv2
import numpy as np
import icebear
import icebear.utils as util
import h5py


def generate_level2(config):
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    print('imaging start:')

    file = h5py.File(config.imaging_source, 'r')
    coeffs = icebear.imaging.swht.unpackage_factors_hdf5(config.swht_coeffs, int(config.lmax))
    time = icebear.utils.Time(config.imaging_start, config.imaging_stop, config.imaging_step)
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
            doppler_shift = data['doppler_shift'][()]
            # This is a little hack to check if we are seeing a dropped sample.
            # Dropped samples always have data for way more range-Doppler bins and that never occurs with real data.
            if len(doppler_shift) >= 28000:
                print('\t-dropped sample detected; skipped')
                continue
            rf_distance = data['rf_distance'][()]
            snr_db = data['snr_db'][()]
            visibilities = np.array(data['spectra'][:, 0], dtype=np.complex64)[:, np.newaxis]
            visibilities = np.append(visibilities, data['xspectra'][:, :], axis=1)
            visibilities = np.append(visibilities, np.conjugate(visibilities), axis=1)
            azimuth = np.empty_like(doppler_shift)
            elevation = np.empty_like(doppler_shift)
            azimuth_spread = np.empty_like(doppler_shift)
            elevation_spread = np.empty_like(doppler_shift)
            area = np.empty_like(doppler_shift)
            for idx, visibility in enumerate(visibilities):
                azimuth[idx], elevation[idx], azimuth_spread[idx], elevation_spread[idx], area[idx] = calculate_image(visibility, coeffs)

            append_level2_hdf5(filename, int(now.hour), int(now.minute), int(now.second * 1000), doppler_shift,
                               snr_db, rf_distance, azimuth, elevation, azimuth_spread, elevation_spread, area)
            print(f'\t-appended: {int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}, '
                  f'targets: {len(doppler_shift)}')

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
    f.create_dataset('version', data=np.array(config.version))
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
    f.create_dataset('clean', data=np.array([config.clean], dtype='S'))
    f.create_dataset('center', data=np.array([config.center], dtype='S'))
    f.create_dataset('swht_coeffs', data=config.swht_coeffs)
    f.create_dataset('fov', data=config.fov)
    f.create_dataset('fov_center', data=config.fov_center)
    f.create_dataset('resolution', data=config.resolution)
    f.create_dataset('lmax', data=config.lmax)
    f.create_group('data')
    f.close()
    return None


def append_level2_hdf5(filename, hour, minute, second, doppler_shift, snr_db, rf_distance,
                       azimuth, elevation, azimuth_spread, elevation_spread, area):
    """
    
    Parameters
    ----------
    filename
    hour
    minute
    second
    doppler_shift
    snr_db
    rf_distance
    azimuth
    elevation
    azimuth_spread
    elevation_spread
    area

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
    f.create_dataset(f'data/{time}/azimuth', data=azimuth)
    f.create_dataset(f'data/{time}/elevation', data=elevation)
    f.create_dataset(f'data/{time}/azimuth_spread', data=azimuth_spread)
    f.create_dataset(f'data/{time}/elevation_spread', data=elevation_spread)
    f.create_dataset(f'data/{time}/area', data=area)
    f.close()
    return None


def calculate_image(visibilities, coeffs):
    """

    Parameters
    ----------
    visibilities
    coeffs

    Returns
    -------

    """
    brightness = icebear.imaging.swht.swht_py(visibilities, coeffs)
    brightness = brightness_cutoff(brightness)
    cx, cy, cx_spread, cy_spread, area = centroid_center(brightness)

    return cx, cy, cx_spread, cy_spread, area


# Cleaning options
def frequency_difference_beamform():
    # This function is to be added. It provides exceptional target locating but sacrifices spread information.
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


# Target location finding options
def centroid_center(brightness):
    """
    Given a Brightness array this returns the centroid as x, y index of the array and the area of the largest blob.

    Parameters
    ----------
        brightness

    Returns
    -------
        cx
        cy
        area

    """
    image = np.array(brightness * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    cx = np.nan
    cy = np.nan
    cx_spread = np.nan
    cy_spread = np.nan
    for index, contour in enumerate(contours):
        temp_area = cv2.contourArea(contour)
        if temp_area > area:
            area = temp_area
            moments = cv2.moments(contour)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            _, _, cx_spread, cy_spread = cv2.boundingRect(contour)

    return cx, cy, cx_spread, cy_spread, area


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


if __name__ == '__main__':
    file = 'E:/icebear/level1/2022_22_22/ib3d_normal_01dB_1000ms_2019_10_28_06_prelate_bakker.h5'
    config = icebear.utils.Config(file)
    config.add_attr('imaging_destination', 'E:/icebear/level2/')
    config.add_attr('imaging_source', file)
    imaging_start, imaging_stop = util.get_data_file_times(file)
    imaging_step = [0, 0, 0, 1, 0]
    config.add_attr('imaging_start', imaging_start)
    config.add_attr('imaging_stop', imaging_stop)
    config.add_attr('imaging_step', imaging_step)
    config.add_attr('lmax', 85)
    config.add_attr('resolution', 0.1)
    config.add_attr('image_method', 'swht')
    config.add_attr('clean', '3db')
    config.add_attr('center', 'centroid')
    config.add_attr('fov', np.array([[0, 360], [0, 180]]))
    config.add_attr('fov_center', np.array([90, 90]))
    config.add_attr('swht_coeffs', 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85')
    generate_level2(config)
