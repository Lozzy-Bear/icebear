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

================================  =====================  ===============================================================
Original Name                     Changed Name           Description
--------------------------------  ---------------------  ---------------------------------------------------------------
                                  config_updated
experiment_name                   processing_method
date                              date
                                  radar_name
incoherent_averages               incoherent_averages
raw_recorded_sample_rate          raw_sample_rate
                                  rx_updated
                                  rx_sample_rate
rx_RF_path                        rx_rf_path
rx_antenna_locations_x_y_z        rx_x, rx_y, rx_z
rx_antenna_type                   rx_antenna_type
rx_location_lat_lon               rx_coordinates
rx_magnitude_corrections_applied  rx_magnitude
rx_name                           rx_name
rx_phase_corrections_applied      rx_phase
rx_pointing_dir                   rx_pointing
                                  rx_mask
software_decimation_rate          decimation_rate
spectra_descriptors               spectra_descriptors
time_resolution                   time_resolution
                                  tx_updated
tx_RF_path                        tx_rf_path
tx_antenna_locations_x_y_z        tx_x, tx_y, tx_z
tx_antenna_type                   tx_antenna_type
tx_antennas_used                  tx_mask
tx_code_used                      prn_code_file
tx_location_lat_lon               tx_coordinates
tx_magnitude_corrections          tx_magnitude
tx_name                           tx_name
tx_phase_corrections              tx_phase
tx_pointing_dir                   tx_pointing
tx_sample_rate                    tx_sample_rate
xspectra_descriptors              xspectra_descriptors
data                              data
antenna_spectra                   spectra
antenna_spectra_var               spectra_variance
antenna_xspectra                  xspectra
antenna_xspectra_var              xspectra_variance
avg_spectra_noise_value           avg_spectra_noise
data_flag                         data_flag
doppler_shift                     doppler_shift
rf_distance                       rf_distance
snr_dB                            snr_db
spectra_clutter_correction        spectra_clutter_correction
spectra_noise_value               spectra_median
time                              time
spectra_clutter_correction        xspectra_clutter_correction
xspectra_noise_value              xspectra_median
                                  wavelength
center_freq                       center_frequency
dB_SNR_cutoff                     snr_cutoff
