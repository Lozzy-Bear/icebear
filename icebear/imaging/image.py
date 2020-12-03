import cv2
import numpy as np


def generate_level2(config, time, clean='beamform', center='centroid', classify='normal'):
    # Set the image cleaning method to be used.
    if clean == 'beamform':
        clean_function = frequency_difference_beamform
    elif clean == '3db':
        clean_function = brightness_cutoff
    else:
        print(f'generate_level2 clean mode {clean} does not exist.')
        exit()

    # Set the center of target detection method to be used.
    if center == 'centroid':
        center_function = centroid_center
    elif center == 'max':
        center_function = max_center
    else:
        print(f'generate_level2 center mode {center} does not exist.')
        exit()

    # Set the classification method to be used.
    if classify == 'normal':
        classify_function = classify_type
    else:
        print(f'generate_level2 classify mode {classify} does not exist.')
        exit()

    print('imaging start')

    temp_hour = [-1, -1, -1, -1]
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        now = time.get_date(t)
        spectra = np.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras),
            dtype=np.complex128)
        spectra_variance = np.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras),
            dtype=np.complex128)
        xspectra = np.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras),
            dtype=np.complex128)
        xspectra_variance = np.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras),
            dtype=np.complex128)
        power = np.zeros(shape=(int(config.code_length / config.decimation_rate), config.number_ranges),
                         dtype=np.complex128)

        # create new file if new hour
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            filename = f'{config.processing_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/' \
                f'{config.radar_name}_{config.processing_method}_{config.tx_name}_{config.rx_name}_' \
                f'{int(config.snr_cutoff):02d}dB_{int(config.incoherent_averages):02d}00ms_' \
                f'{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}_{int(now.hour):02d}.h5'
            print(f'\t-created level 1 HDf5: {filename}')
            filenames.append(filename)
            create_level1_hdf5(config, filename, int(now.year), int(now.month), int(now.day))
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        calculate_image(clean_function, center_function, classify_function)

        append_level2_hdf5(filename, int(now.hour), int(now.minute), int(now.second * 1000),
                           data_flag, doppler, rf_distance, snr_db, noise,
                           spectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_median, spectra_clutter_corr,
                           xspectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_median, xspectra_clutter_corr)
        print(f'\t-appended {int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}')

    return None


def create_level2_hdf5(config, filename, year, month, day):
    # general information
    f = h5py.File(filename, 'w')
    f.create_dataset('config_updated', data=np.array(config.config_updated))
    f.create_dataset('date', data=np.array([year, month, day]))
    f.create_dataset('processing_method', data=config.processing_method)
    # transmitter site information
    f.create_dataset('tx_name', data=config.tx_name)
    f.create_dataset('tx_coordinates', data=np.array(config.tx_coordinates))
    f.create_dataset('tx_updated', data=config.tx_updated)
    f.create_dataset('tx_pointing', data=config.tx_pointing)
    f.create_dataset('tx_x', data=np.array(config.rx_x))
    f.create_dataset('tx_y', data=np.array(config.rx_y))
    f.create_dataset('tx_z', data=np.array(config.rx_z))
    f.create_dataset('tx_mask', data=np.array(config.tx_mask))
    f.create_dataset('tx_phase', data=np.array(config.tx_phase))
    f.create_dataset('tx_magnitude', data=np.array(config.tx_magnitude))
    f.create_dataset('tx_sample_rate', data=config.tx_sample_rate)
    f.create_dataset('tx_antenna_type', data=config.tx_antenna_type)
    f.create_dataset('tx_rf_path', data=config.tx_rf_path)
    # receiver site information
    f.create_dataset('rx_name', data=config.rx_name)
    f.create_dataset('rx_coordinates', data=np.array(config.rx_coordinates))
    f.create_dataset('rx_updated', data=config.rx_updated)
    f.create_dataset('rx_pointing', data=config.rx_pointing)
    f.create_dataset('rx_x', data=np.array(config.rx_x))
    f.create_dataset('rx_y', data=np.array(config.rx_y))
    f.create_dataset('rx_z', data=np.array(config.rx_z))
    f.create_dataset('rx_mask', data=np.array(config.rx_mask))
    f.create_dataset('rx_phase', data=np.array(config.rx_phase))
    f.create_dataset('rx_magnitude', data=np.array(config.rx_magnitude))
    f.create_dataset('rx_sample_rate', data=config.rx_sample_rate)
    f.create_dataset('rx_antenna_type', data=config.rx_antenna_type)
    f.create_dataset('rx_rf_path', data=config.rx_rf_path)
    # processing settings
    f.create_dataset('wavelength', data=config.wavelength)
    f.create_dataset('center_frequency', data=config.center_frequency)
    f.create_dataset('prn_code_file', data=config.prn_code_file)
    f.create_dataset('raw_sample_rate', data=config.raw_sample_rate)
    f.create_dataset('decimation_rate', data=config.decimation_rate)
    f.create_dataset('incoherent_averages', data=config.incoherent_averages)
    f.create_dataset('time_resolution', data=config.time_resolution)
    f.create_dataset('snr_cutoff', data=config.snr_cutoff)
    # imaging settings
    f.create_dataset('clean', data=config.clean)
    f.create_dataset('center', data=config.center)
    f.create_dataset('classify', data=config.classify)
    f.create_dataset('swht_coeffs', data=config.swht_coeffs)
    f.create_dataset('fov', data=config.fov)
    f.create_dataset('fov_center', data=config.fov_center)
    f.create_dataset('resolution', data=config.resolution)
    f.create_dataset('lmax', data=config.lmax)
    f.create_group('data')
    f.close()
    return None


def append_level2_hdf5(filename, ):
    # append a new group for the current measurement
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    f.create_dataset(f'data/{time}/velocity', data=velocity)
    f.create_dataset(f'data/{time}/snr_db', data=logsnr)
    f.create_dataset(f'data/{time}/spread', data=spread)
    f.create_dataset(f'data/{time}/distance', data=distance)
    f.create_dataset(f'data/{time}/azimuth', data=azimuth)
    f.create_dataset(f'data/{time}/elevation', data=elevation)
    f.create_dataset(f'data/{time}/area', data=area)
    f.create_dataset(f'data/{time}/type', data=scatter_type)
    f.close()
    return None


def calculate_image():
    return


# Cleaning options
def frequency_difference_beamform():
    return


def brightness_cutoff(brightness, threshold=0.5):
    """
    Given a normalized Brightness array this removes noise in the image below a power threshold.
    The default threshold is 0.5 (3 dB).

    Parameters
    ----------
        brightness
        threshold

    Returns
    -------

    """
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
    image = np.array(brightness * 255.0, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    cx = np.nan
    cy = np.nan
    for index, contour in enumerate(contours):
        temp_area = cv2.contourArea(contour)
        if temp_area > area:
            area = temp_area
            moments = cv2.moments(c)
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

    return cx, cy, area


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


# Classification options
def classify_type():
    """
    Type 1 is classified by -
    Returns
    -------

    """
    # Type I, II, III, IV, Meteor, Unknown

    return
