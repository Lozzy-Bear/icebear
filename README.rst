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

ATTRIBUTE INFORMATION
=====================
======================================  ===========================  =========================================================
Original Name                           Changed Name                 Description
General Attributes (Level 0)
------------------------------------------------------------------------------------------------------------------------------
x                                       config_updated               [year, month, day] date the config file used was last updated
experiment_name                         processing_method            name of the processing method used
date                                    date                         [year, month, day] date of the data
x                                       radar_name                   name of the radar the data was collected with (ex, ib, ib3d, lofar, ibsimo)
incoherent_averages                     incoherent_averages
raw_recorded_sample_rate                raw_sample_rate
x                                       rx_updated                   [year, month, day] date the receiver information was updated
x                                       rx_sample_rate
rx_RF_path                              rx_rf_path                   the receiver signal path; all inline elements listed
rx_antenna_locations_x_y_z              rx_x, rx_y, rx_z             receiver antenna relative locations in meters from antenna 0
rx_antenna_type                         rx_antenna_type              receiver antenna type
rx_location_lat_lon                     rx_coordinates               [latitude, longitude] coordinates of the receiver antenna array
rx_magnitude_corrections_applied        rx_magnitude                 [ant 0, ...] magnitude of the complex correction per receiver antenna
rx_name                                 rx_name                      name of receiver site
rx_phase_corrections_applied            rx_phase                     [ant 0, ...] phase of the complex correction per receiver antenna
rx_pointing_dir                         rx_pointing                  receiver array boresight pointing direction in degrees east of north
x                                       rx_mask                      [0, 1, ...] mask indicating which receiver antennas were used
software_decimation_rate                decimation_rate
spectra_descriptors                     spectra_descriptors
time_resolution                         time_resolution
x                                       tx_updated
tx_RF_path                              tx_rf_path
tx_antenna_locations_x_y_z              tx_x, tx_y, tx_z
tx_antenna_type                         tx_antenna_type
tx_antennas_used                        tx_mask
tx_code_used                            prn_code_file
tx_location_lat_lon                     tx_coordinates
tx_magnitude_corrections                tx_magnitude
tx_name                                 tx_name
tx_phase_corrections                    tx_phase
tx_pointing_dir                         tx_pointing
tx_sample_rate                          tx_sample_rate
xspectra_descriptors                    xspectra_descriptors
data                                    data
Processing Attributes (Level 1)
------------------------------------------------------------------------------------------------------------------------------
data/{time}/antenna_spectra             spectra
data/{time}/antenna_spectra_var         spectra_variance
data/{time}/antenna_xspectra            xspectra
data/{time}/antenna_xspectra_var        xspectra_variance
data/{time}/avg_spectra_noise_value     avg_spectra_noise
data/{time}/data_flag                   data_flag
data/{time}/doppler_shift               doppler_shift
data/{time}/rf_distance                 rf_distance
data/{time}/snr_dB                      snr_db
data/{time}/spectra_clutter_correction  spectra_clutter_correction
data/{time}/spectra_noise_value         spectra_median
data/{time}/time                        time
data/{time}/spectra_clutter_correction  xspectra_clutter_correction
data/{time}/xspectra_noise_value        xspectra_median
x                                       wavelength
center_freq                             center_frequency
dB_SNR_cutoff                           snr_cutoff
Imaging Attributes (Level 2)
------------------------------------------------------------------------------------------------------------------------------
x                                       clean
x                                       center
x                                       classify
x                                       swht_coeffs
x                                       fov
x                                       fov_center
x                                       resolution
x                                       lmax
x                                       data/{time}/velocity
x                                       data/{time}/snr_db
x                                       data/{time}/spread
x                                       data/{time}/distance
x                                       data/{time}/azimuth
x                                       data/{time}/elevation
x                                       data/{time}/area
x                                       data/{time}/type
------------------------------------------------------------------------------------------------------------------------------
Plotting Attributes (Level 3)
------------------------------------------------------------------------------------------------------------------------------
======================================  ===========================  =========================================================