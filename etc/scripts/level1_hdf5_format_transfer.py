import icebear.utils as util
import h5py
import os
import numpy as np

filepath = 'E:/icebear/level1/'
files = util.get_all_data_files(filepath, '2022_22_22')
for file in files:
    f = h5py.File(file, 'r+')
    print(file)

    # Rename the file.
    pp = file.split('icebear_3d')
    same_path = pp[0]
    ppp = pp[1].split('vis')
    pppp = ppp[1].split('_prelate')
    snr_cutoff = f['dB_SNR_cutoff'][0]
    incoherent_averages = f['incoherent_averages'][0]
    new_filename = f'{same_path}ib3d_normal_prelate_bakker_' \
        f'{int(snr_cutoff):02d}dB_{int(incoherent_averages):02d}00ms' \
        f'{pppp[0]}.h5'

    # Add new datasets
    f.create_dataset('config_updated', data=[np.nan, np.nan, np.nan])
    f.create_dataset('wavelength', data=[6.05641329])
    f.create_dataset('radar_name', data=np.array(['ib3d'], dtype='S'))
    f.create_dataset('rx_updated', data=[np.nan, np.nan, np.nan])
    f.create_dataset('tx_updated', data=[np.nan, np.nan, np.nan])
    f.create_dataset('rx_mask', data=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    f.create_dataset('tx_mask', data=[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    f.create_dataset('rx_sample_rate', data=[np.nan])
    f.create_dataset('prn_code_file', data=np.array(['dat/pseudo_random_code_test_8_lpf.txt'], dtype='S'))
    
    # Rename datasets
    f["processing_method"] = f["experiment_name"]
    del f["experiment_name"] 
    f["raw_sample_rate"] = f["raw_recorded_sample_rate"]
    del f["raw_recorded_sample_rate"]
    f["rx_rf_path"] = f["rx_RF_path"]
    del f["rx_RF_path"]
    f["tx_rf_path"] = f["tx_RF_path"]
    del f["tx_RF_path"]
    f["rx_x"] = f["rx_antenna_locations_x_y_z"][0,:]
    f["rx_y"] = f["rx_antenna_locations_x_y_z"][1,:]
    f["rx_z"] = f["rx_antenna_locations_x_y_z"][2,:]
    del f["rx_antenna_locations_x_y_z"]
    f["tx_x"] = f["tx_antenna_locations_x_y_z"][0,:]
    f["tx_y"] = f["tx_antenna_locations_x_y_z"][1,:]
    f["tx_z"] = f["tx_antenna_locations_x_y_z"][2,:]
    del f["tx_antenna_locations_x_y_z"]
    del f["tx_antennas_used"]
    f["decimation_rate"] = f["software_decimation_rate"]
    del f["software_decimation_rate"]
    del f["tx_code_used"]
    f["center_frequency"] = f["center_freq"]
    del f["center_freq"]
    f["snr_cutoff"] = f["dB_SNR_cutoff"]
    del f["dB_SNR_cutoff"]
    f["rx_coordinates"] = f["rx_location_lat_lon"]
    del f["rx_location_lat_lon"]
    f["rx_magnitude"] = f["rx_magnitude_corrections_applied"]
    del f["rx_magnitude_corrections_applied"]
    f["rx_phase"] = f["rx_phase_corrections_applied"]
    del f["rx_phase_corrections_applied"]
    f["rx_pointing"] = f["rx_pointing_dir"]
    del f["rx_pointing_dir"]
    f["tx_coordinates"] = f["tx_location_lat_lon"]
    del f["tx_location_lat_lon"]
    f["tx_magnitude"] = f["tx_magnitude_corrections"]
    del f["tx_magnitude_corrections"]
    f["tx_phase"] = f["tx_phase_corrections"]
    del f["tx_phase_corrections"]
    f["tx_pointing"] = f["tx_pointing_dir"]
    del f["tx_pointing_dir"]
    
    # Rename datasets in data group
    group = f['data']
    gkeys = group.keys()
    for gkey in gkeys:
        group[f'{gkey}'].move('avg_spectra_noise_value', 'avg_spectra_noise')
        group[f'{gkey}'].move('spectra_noise_value', 'spectra_median')
        group[f'{gkey}'].move('xspectra_noise_value', 'xspectra_median')
        if group[f'{gkey}']['data_flag'][:]:
            group[f'{gkey}'].move('antenna_spectra', 'spectra')
            group[f'{gkey}'].move('antenna_spectra_var', 'spectra_variance')
            group[f'{gkey}'].move('antenna_xspectra', 'xspectra')
            group[f'{gkey}'].move('antenna_xspectra_var', 'xspectra_variance')
            group[f'{gkey}'].move('snr_dB', 'snr_db')

    # Rename the file
    f.close()
    os.rename(file, new_filename)


"""
                                            - config_updated
experiment_name                             - processing_method
date                                        - date
                                            - radar_name
incoherent_averages                         - incoherent_averages
raw_recorded_sample_rate                    - raw_sample_rate
                                            - rx_updated
                                            - rx_sample_rate
rx_RF_path                                  - rx_rf_path
rx_antenna_locations_x_y_z                  - rx_x, rx_y, rx_z
rx_antenna_type                             - rx_antenna_type
rx_location_lat_lon                         - rx_coordinates
rx_magnitude_corrections_applied            - rx_magnitude
rx_name                                     - rx_name
rx_phase_corrections_applied                - rx_phase
rx_pointing_dir                             - rx_pointing
                                            - rx_mask
software_decimation_rate                    - decimation_rate
spectra_descriptors                         - spectra_descriptors 
time_resolution                             - time_resolution
                                            - tx_updated
tx_RF_path                                  - tx_rf_path
tx_antenna_locations_x_y_z                  - tx_x, tx_y, tx_z
tx_antenna_type                             - tx_antenna_type
tx_antennas_used                            - tx_mask
tx_code_used                                - prn_code_file
tx_location_lat_lon                         - tx_coordinates
tx_magnitude_corrections                    - tx_magnitude
tx_name                                     - tx_name
tx_phase_corrections                        - tx_phase
tx_pointing_dir                             - tx_pointing
tx_sample_rate                              - tx_sample_rate
xspectra_descriptors                        - xspectra_descriptors 
data/035950000                              - data
data/035950000/antenna_spectra              - spectra
data/035950000/antenna_spectra_var          - spectra_variance
data/035950000/antenna_xspectra             - xspectra
data/035950000/antenna_xspectra_var         - xspectra_variance
data/035950000/avg_spectra_noise_value      - avg_spectra_noise
data/035950000/data_flag                    - data_flag
data/035950000/doppler_shift                - doppler_shift
data/035950000/rf_distance                  - rf_distance
data/035950000/snr_dB                       - snr_db
data/035950000/spectra_clutter_correction   - spectra_clutter_correction
data/035950000/spectra_noise_value          - spectra_median
data/035950000/time                         - time
data/035950000/xspectra_clutter_correction  - xspectra_clutter_correction
data/035950000/xspectra_noise_value         - xspectra_median
                                            - wavelength
center_freq                                 - center_frequency
dB_SNR_cutoff                               - snr_cutoff
"""

"""
ATTRIBUTE INFORMATION
=====================
======================================  ============================  ==============================================================================
Original Name                           Changed Name                   Description
General Attributes (Level 0)
----------------------------------------------------------------------------------------------------------------------------------------------------
x                                       date_created                   [year, month, day] date this file was created
x                                       version                        software version used to create this file
date                                    date                           [year, month, day, hour] date of the data
experiment_name                         experiment_name                name of experiment (ex; normal, mobile)
x                                       radar_config                   name of the radar the data was collected with (ex; ib, ib3d, lofar)
center_freq                             center_freq                    radar signal center frequency in Hz
rx_name                                 rx_site_name                   name of receiver site
rx_location_lat_lon                     rx_site_lat_long               [latitude, longitude] coordinates of the receiver antenna array
rx_pointing_dir                         rx_heading                     receiver array boresight pointing direction in degrees east of north
rx_RF_path                              rx_rf_path                     the receiver signal path; all inline elements listed
rx_antenna_type                         rx_ant_type                    receiver antenna type
rx_antenna_locations_x_y_z              rx_ant_coords                  receiver antenna relative locations in meters from antenna 0
rx_magnitude_corrections_applied        rx_feed_corr                   [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per receiver antenna
x                                       rx_feed_corr_date              [year, month, day] date the feedline corrections were determined
x                                       rx_feed_corr_type              [magnitude type, phase type] (ex; [data median, instrumental])
rx_phase_corrections_applied            x                              REMOVE
x                                       rx_ant_mask                    [0, 1, ...] mask indicating which receiver antennas were used and/or available
raw_recorded_sample_rate                rx_sample_rate                 the raw recorded sample rate at the receiver in Hz
tx_name                                 tx_site_name                   name of the transmitter site
tx_location_lat_lon                     tx_site_lat_long               [latitude, longitude] coordinates of the transmitter antenna array
tx_pointing_dir                         tx_heading                     transmitter array boresight pointing direction in degrees east of north
tx_RF_path                              tx_rf_path                     the transmitter signal path; all inline elements listed
tx_antenna_type                         tx_ant_type                    transmitter antenna typ
tx_antenna_locations_x_y_z              tx_ant_coords                  transmitter antenna relative locations in meters from antenna 0
tx_magnitude_corrections                tx_feed_corr                   [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per transmitter antenna
x                                       tx_feed_corr_date              [year, month, day] date the feedline corrections were determined
x                                       tx_feed_corr_type              [magnitude type, phase type] (ex; [data median, instrumental])
tx_phase_corrections                    x                              REMOVE
tx_antennas_used                        tx_ant_mask                    [0, 1, ...] mask indicating which transmitter antennas were used
tx_code_used                            tx_cw_code                     pseudo-random noise like code transmitted (contains full sequence)
tx_sample_rate                          tx_sample_rate                 sample rate of transmitted code
data                                    data                           dataset key for data in file organized per second
Processing Attributes (Level 1)
----------------------------------------------------------------------------------------------------------------------------------------------------
x                                       processing_source              file path to directory holding channel separated digital_rf files
x                                       processing_destination         file path to directory to store the level1 hdf5 files
x                                       processing_start               [year, month, day, hour, minute, second, millisecond] time to start processing
x                                       processing_stop                [year, month, day, hour, minute, second, millisecond] time to stop processing
x                                       processing_step                [day, hour, minute, second, millisecond] step size for processing
software_decimation_rate                decimation_rate
time_resolution                         time_resolution
x                                       coherent_integration_time
incoherent_averages                     incoherent_averages            number of samples to average over
dB_SNR_cutoff                           snr_cutoff_db
spectra_descriptors                     spectra_descriptors
xspectra_descriptors                    xspectra_descriptors
data/{time}/time                        time
data/{time}/antenna_spectra             spectra
data/{time}/antenna_spectra_var         spectra_variance
data/{time}/antenna_xspectra            xspectra
data/{time}/antenna_xspectra_var        xspectra_variance
data/{time}/avg_spectra_noise_value     avg_spectra_noise
data/{time}/data_flag                   data_flag
data/{time}/doppler_shift               doppler_shift
data/{time}/rf_distance                 rf_distance
data/{time}/snr_dB                      snr_db
data/{time}/spectra_clutter_correction  spectra_clutter_corr
data/{time}/spectra_noise_value         spectra_noise                  spectra noise value determined by the median spectra before cutoff
data/{time}/spectra_clutter_correction  xspectra_clutter_corr
data/{time}/xspectra_noise_value        xspectra_noise
Imaging Attributes (Level 2)
----------------------------------------------------------------------------------------------------------------------------------------------------
x                                       imaging_source                 file path to directory holding level1 hdf5 files to be imaged
x                                       imaging_destination            file path to directory to store the level2 hdf5 files
x                                       imaging_start                  [year, month, day, hour, minute, second, millisecond] time to start imaging
x                                       imaging_stop                   [year, month, day, hour, minute, second, millisecond] time to stop imaging
x                                       imaging_step                   [day, hour, minute, second, millisecond] step size for imaging
x                                       imaging_method                 imaging method used (ex; swht)
x                                       clean                          image cleaning method applied
x                                       center                         target location method applied
x                                       classify                       target classification method applied
x                                       swht_coeffs                    name of swht_coeffs file used, these files are upward of 4 GB
x                                       fov                            [[az min, az max], [el min, el max]] image field of view
x                                       fov_center                     [az, el] angles in image which coincide with boresight
x                                       resolution                     pixel resolution in degrees
x                                       lmax                           maximum harmonic degree the coefficients were calculated
x                                       data/{time}/data_flag
x                                       data/{time}/doppler_shift
x                                       data/{time}/snr_db             target signal strength in dB
x                                       data/{time}/rf_distance
x                                       data/{time}/azimuth
x                                       data/{time}/elevation
x                                       data/{time}/azimuth_spread
x                                       data/{time}/elevation_spread
x                                       data/{time}/area
x                                       data/{time}/type               target type; meteor trail, direct feed through, or scatter
Plotting Attributes (Level 3)
----------------------------------------------------------------------------------------------------------------------------------------------------
velocity                                                               speed of target
position                                                               [lat, long, alt]
time                                                                   time of data point
snr_db                                                                 signal strength
spatial_spread                                                         [lat spread, long spread, alt spread]
spatial_spread_function                                                function to determine spreading
======================================  ============================  ==============================================================================

"""