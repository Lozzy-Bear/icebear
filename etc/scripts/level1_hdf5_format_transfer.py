import icebear.utils as utils
import h5py
import os
import numpy as np
import time


filepath = '/beaver/backup/level1/' # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2022_03_06', '2022_03_19') # Enter first sub directory and last
rxfd = [np.nan, np.nan, np.nan]
rxtf = ['data median', 'instrument measured']
txfd = [np.nan, np.nan, np.nan]
txft = ['no information', 'no information']
txmask = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
for file in files:
    ts = time.time()
    f = h5py.File(file, 'r+')
    print(file)

    # Rename the file
    if '3d' in file:
        radar_config = 'ib3d'
    elif 'linear' in file:
        radar_config = 'iblinear'
    else:
        print('ERROR: radar_config does not exist or cannot be found')
        exit()

    # General Attributes (Level 0)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # x                                       date_created                   [year, month, day] date this file was created
    f.create_dataset('date_created', data=np.array([2020, 12, 23]))
    # x                                       version                        software version used to create this file
    f.create_dataset('version', data=np.array([1.0]))
    # date                                    date                           [year, month, day, hour] date of the data
    # experiment_name                         experiment_name                name of experiment (ex; normal, mobile)
    # x                                       radar_config                   name of the radar the data was collected with (ex; ib, ib3d, lofar)
    f.create_dataset('radar_config', data=np.array([radar_config], dtype='S'))
    # center_freq                             center_freq                    radar signal center frequency in Hz
    # rx_name                                 rx_site_name                   name of receiver site
    f.move('rx_name', 'rx_site_name')
    # rx_location_lat_lon                     rx_site_lat_long               [latitude, longitude] coordinates of the receiver antenna array
    f.move('rx_location_lat_lon', 'rx_site_lat_long')
    # rx_pointing_dir                         rx_heading                     receiver array boresight pointing direction in degrees east of north
    try:
        f.move('rx_pointing_dir', 'rx_heading')
    except:
        f.create_dataset('rx_heading', data=np.array([7]))
    # rx_RF_path                              rx_rf_path                     the receiver signal path; all inline elements listed
    f.move('rx_RF_path', 'rx_rf_path')
    # rx_antenna_type                         rx_ant_type                    receiver antenna type
    f.move('rx_antenna_type', 'rx_ant_type')
    # rx_antenna_locations_x_y_z              rx_ant_coords                  receiver antenna relative locations in meters from antenna 0
    f.move('rx_antenna_locations_x_y_z', 'rx_ant_coords')
    # rx_magnitude_corrections_applied        rx_feed_corr                   [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per receiver antenna
    f.create_dataset('rx_feed_corr', data=np.array([f['rx_magnitude_corrections_applied'][()], f['rx_phase_corrections_applied'][()]]))
    del f['rx_magnitude_corrections_applied']
    # rx_phase_corrections_applied            x                              REMOVE
    del f['rx_phase_corrections_applied']
    # x                                       rx_feed_corr_date              [year, month, day] date the feedline corrections were determined
    f.create_dataset('rx_feed_corr_date', data=np.array(rxfd))
    # x                                       rx_feed_corr_type              [magnitude type, phase type] (ex; [data median, instrumental])
    f.create_dataset('rx_feed_corr_type', data=np.array(rxtf, dtype='S'))
    # x                                       rx_ant_mask                    [0, 1, ...] mask indicating which receiver antennas were used and/or available
    f.create_dataset('rx_ant_mask', data=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # raw_recorded_sample_rate                rx_sample_rate                 the raw recorded sample rate at the receiver in Hz
    f.move('raw_recorded_sample_rate', 'rx_sample_rate')
    # tx_name                                 tx_site_name                   name of the transmitter site
    f.move('tx_name', 'tx_site_name')
    # tx_location_lat_lon                     tx_site_lat_long               [latitude, longitude] coordinates of the transmitter antenna array
    f.move('tx_location_lat_lon', 'tx_site_lat_long')
    # tx_pointing_dir                         tx_heading                     transmitter array boresight pointing direction in degrees east of north
    try:
        f.move('tx_pointing_dir', 'tx_heading')
    except:
        f.create_dataset('tx_heading', data=np.array([16]))
    # tx_RF_path                              tx_rf_path                     the transmitter signal path; all inline elements listed
    f.move('tx_RF_path', 'tx_rf_path')
    # tx_antenna_type                         tx_ant_type                    transmitter antenna type
    f.move('tx_antenna_type', 'tx_ant_type')
    # tx_antenna_locations_x_y_z              tx_ant_coords                  transmitter antenna relative locations in meters from antenna 0
    f.move('tx_antenna_locations_x_y_z', 'tx_ant_coords')
    # tx_magnitude_corrections                tx_feed_corr                   [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per transmitter antenna
    f.create_dataset('tx_feed_corr', data=np.array([f['tx_magnitude_corrections'][()], f['tx_phase_corrections'][()]]))
    del f['tx_magnitude_corrections']
    # tx_phase_corrections                    x                              REMOVE
    del f['tx_phase_corrections']
    # x                                       tx_feed_corr_date              [year, month, day] date the feedline corrections were determined
    f.create_dataset('tx_feed_corr_date', data=np.array(txfd))
    # x                                       tx_feed_corr_type              [magnitude type, phase type] (ex; [data median, instrumental])
    f.create_dataset('tx_feed_corr_type', data=np.array(txft, dtype='S'))
    # tx_antennas_used                        tx_ant_mask                    [0, 1, ...] mask indicating which transmitter antennas were used
    f.create_dataset('tx_ant_mask', data=np.array(txmask))
    del f['tx_antennas_used']
    # tx_code_used                            tx_cw_code                     pseudo-random noise like code transmitted (contains full sequence)
    f.move('tx_code_used', 'tx_cw_code')
    # tx_sample_rate                          tx_sample_rate                 sample rate of transmitted code
    # data                                    data                           dataset key for data in file organized per second
    # Processing Attributes (Level 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # x                                       processing_source              file path to directory holding channel separated digital_rf files
    # x                                       processing_destination         file path to directory to store the level1 hdf5 files
    # x                                       processing_start               [year, month, day, hour, minute, second, millisecond] time to start processing
    # x                                       processing_stop                [year, month, day, hour, minute, second, millisecond] time to stop processing
    # x                                       processing_step                [day, hour, minute, second, millisecond] step size for processing
    # software_decimation_rate                decimation_rate
    f.move('software_decimation_rate', 'decimation_rate')
    # time_resolution                         time_resolution
    # x                                       coherent_integration_time
    f.create_dataset('coherent_integration_time', data=np.array([f['time_resolution'][0] / f['incoherent_averages'][0]]))
    # incoherent_averages                     incoherent_averages            number of samples to average over
    # dB_SNR_cutoff                           snr_cutoff_db
    f.move('dB_SNR_cutoff', 'snr_cutoff_db')
    # spectra_descriptors                     spectra_descriptors
    # xspectra_descriptors                    xspectra_descriptors
    del f['spectra_descriptors']
    del f['xspectra_descriptors']
    f.create_dataset('spectra_descriptors',
                     data=np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    f.create_dataset('xspectra_descriptors',
                     data=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8,
                                     9, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]]))
    # data/{time}/time                        time
    # data/{time}/antenna_spectra             spectra
    # data/{time}/antenna_spectra_var         spectra_variance
    # data/{time}/antenna_xspectra            xspectra
    # data/{time}/antenna_xspectra_var        xspectra_variance
    # data/{time}/avg_spectra_noise_value     avg_spectra_noise
    # data/{time}/data_flag                   data_flag
    # data/{time}/doppler_shift               doppler_shift
    # data/{time}/rf_distance                 rf_distance
    # data/{time}/snr_dB                      snr_db
    # data/{time}/spectra_clutter_correction  spectra_clutter_corr
    # data/{time}/spectra_noise_value         spectra_noise                  spectra noise value determined by the median spectra before cutoff
    # data/{time}/xspectra_clutter_correction xspectra_clutter_corr
    # data/{time}/xspectra_noise_value        xspectra_noise
    group = f['data']
    gkeys = group.keys()
    for gkey in gkeys:
        group[f'{gkey}'].move('avg_spectra_noise_value', 'avg_spectra_noise')
        group[f'{gkey}'].move('spectra_noise_value', 'spectra_noise')
        group[f'{gkey}'].move('xspectra_noise_value', 'xspectra_noise')
        try:
            group[f'{gkey}'].move('spectra_clutter_correction', 'spectra_clutter_corr')
        except:
            group[f'{gkey}'].create_dataset('spectra_clutter_corr', data=np.array([np.nan]))

        try:
            group[f'{gkey}'].move('xspectra_clutter_correction', 'xspectra_clutter_corr')
        except:
            group[f'{gkey}'].create_dataset('xspectra_clutter_corr', data=np.array([np.nan]))

        if group[f'{gkey}']['data_flag'][:]:
            group[f'{gkey}'].move('antenna_spectra', 'spectra')
            try:
                group[f'{gkey}'].move('antenna_spectra_var', 'spectra_variance')
            except:
                group[f'{gkey}'].create_dataset('spectra_variance', data=np.array([np.nan]))

            group[f'{gkey}'].move('antenna_xspectra', 'xspectra')
            try:
                group[f'{gkey}'].move('antenna_xspectra_var', 'xspectra_variance')
            except:
                group[f'{gkey}'].create_dataset('xspectra_variance', data=np.array([np.nan]))

            group[f'{gkey}'].move('snr_dB', 'snr_db')
    # Imaging Attributes (Level 2)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # x                                       imaging_source                 file path to directory holding level1 hdf5 files to be imaged
    # x                                       imaging_destination            file path to directory to store the level2 hdf5 files
    # x                                       imaging_start                  [year, month, day, hour, minute, second, millisecond] time to start imaging
    # x                                       imaging_stop                   [year, month, day, hour, minute, second, millisecond] time to stop imaging
    # x                                       imaging_step                   [day, hour, minute, second, millisecond] step size for imaging
    # x                                       imaging_method                 imaging method used (ex; swht)
    # x                                       clean                          image cleaning method applied
    # x                                       center                         target location method applied
    # x                                       classify                       target classification method applied
    # x                                       swht_coeffs                    name of swht_coeffs file used, these files are upward of 4 GB
    # x                                       fov                            [[az min, az max], [el min, el max]] image field of view
    # x                                       fov_center                     [az, el] angles in image which coincide with boresight
    # x                                       resolution                     pixel resolution in degrees
    # x                                       lmax                           maximum harmonic degree the coefficients were calculated
    # x                                       data/{time}/data_flag
    # x                                       data/{time}/doppler_shift
    # x                                       data/{time}/snr_db             target signal strength in dB
    # x                                       data/{time}/rf_distance
    # x                                       data/{time}/azimuth
    # x                                       data/{time}/elevation
    # x                                       data/{time}/azimuth_spread
    # x                                       data/{time}/elevation_spread
    # x                                       data/{time}/area
    # x                                       data/{time}/type               target type; meteor trail, direct feed through, or scatter
    # Plotting Attributes (Level 3)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # x                                       velocity                       speed of target
    # x                                       position                       [lat, long, alt]
    # x                                       time                           time of data point
    # x                                       snr_db                         signal strength
    # x                                       spatial_spread                 [lat spread, long spread, alt spread]
    # x                                       spatial_spread_function        function to determine spreading

    # Rename the file
    if f['experiment_name'][0] == b'normal_operations':
        f['experiment_name'][0] = b'normal'
    experiment_name = f['experiment_name'][0].decode()
    tx_site_name = f['tx_site_name'][0].decode()
    rx_site_name = f['rx_site_name'][0].decode()
    snr_cutoff_db = f['snr_cutoff_db'][0]
    incoherent_averages = f['incoherent_averages'][0]
    year = f['date'][0]
    month = f['date'][1]
    day = f['date'][2]
    data_keys = list(f['data'].keys())
    hour = f[f'data/{data_keys[0]}/time'][0]
    radar_config = f['radar_config'][0].decode()

    new_filename = f'{radar_config}_{experiment_name}_{snr_cutoff_db:02d}dB_{incoherent_averages:02d}00ms_' \
        f'{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{tx_site_name}_{rx_site_name}.h5'
    path = file.split('icebear_')[0]

    f.close()
    os.rename(file, path + new_filename)
    print(path + new_filename)
    print('time:', time.time() - ts)
    print('-----------------------------------------------------------------------------------------------------------')
