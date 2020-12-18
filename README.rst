README
======
ICEBEAR-3D processing, imaging and plotting python package.

DEPENDENCIES
============
numpy
scipy
matplotlib
time
h5py
ctypes
pyfftw
numba


DATA FILE NAMING CONVENTION
===========================
- Level 1: {radar_config}_{experiment_name}_{tx_site_name}_{rx_site_name}_{snr_cutoff_db}_{incoherent_averages}_{date}.h5
- Level 2: {radar_config}_{experiment_name}_{tx_site_name}_{rx_site_name}_{image_method}_{fov}_{resolution}_{date}.h5

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
