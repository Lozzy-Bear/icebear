.. image:: ../docs/imgs/ib_logo_long.png
    :align: center

|docs| |license| |python| |release|

.. |docs| image:: https://readthedocs.org/projects/icebear/badge/?version=latest&style=flat-square
    :target: https://icebear.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation (latest build)

.. |license| image:: https://img.shields.io/badge/License-LGPL%20v3-blue.svg?style=flat-square
    :target: https://www.gnu.org/licenses/lgpl-3.0
    :alt: Distribution license

.. |python| image:: https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square
    :target: https://www.python.org/downloads/release/python-370/
    :alt: Python version

.. |release| image:: https://img.shields.io/github/v/release/Lozzy-Bear/icebear?style=flat-square
    :target: https://github.com/Lozzy-Bear/icebear/
    :alt: GitHub release (latest by date)


A Python package for processing, imaging, and plotting Ionospheric Continuous-wave E region Bistatic Experimental Auroral Radar (ICEBEAR) data.

Changelog
=========
Version 0.1-alpha release
-------------------------
Initial development release

- added SWHT imaging
- added common plotting functions

Getting Started
===============
Install Instructions
--------------------
Todo

Running From Command Line
-------------------------
Todo

Other Examples
--------------
todo

Data Information
================
Data File Naming Convention
---------------------------
Level 1: {radar_config}_{experiment_name}_{snr_cutoff_db}_{incoherent_averages}_{date}_{tx_site_name}_{rx_site_name}.h5::

    $ ib3d_normal_01dB_1000ms_2019_10_24_00_prelate_bakker.h5

Level 2: {radar_config}_{experiment_name}_{image_method}_{resolution}_{date}_{tx_site_name}_{rx_site_name}.h5::

    $ ib3d_normal_swht_01deg_2019_10_24_00_prelate_bakker.h5


Attribute Information
---------------------
==================================  ==========================================================================================================
Attribute Name                      Description
General Attributes (Level 0)
----------------------------------------------------------------------------------------------------------------------------------------------
date_created                        [year, month, day] date this file was created
version                             software version used to create this file
date                                [year, month, day] date of the data
experiment_name                     name of experiment (ex; normal, mobile)
radar_config                        name of the radar the data was collected with (ex; ib, ib3d, lofar)
center_freq                         radar signal center frequency in Hz
rx_site_name                        name of receiver site
rx_site_lat_long                    [latitude, longitude] coordinates of the receiver antenna array
rx_heading                          receiver array boresight pointing direction in degrees east of north
rx_rf_path                          the receiver signal path; all inline elements listed
rx_ant_type                         receiver antenna type
rx_ant_coords                       [[x0, ...],[y0, ...],[z0, ...]]receiver antenna relative locations in meters from antenna 0
rx_feed_corr                        [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per receiver antenna
rx_feed_corr_date                   [year, month, day] date the feedline corrections were determined
rx_feed_corr_type                   [magnitude type, phase type] (ex; [data median, instrumental])
rx_ant_mask                         [0, 1, ...] mask indicating which receiver antennas were used and/or functional
rx_sample_rate                      the raw recorded sample rate at the receiver in Hz
tx_site_name                        name of the transmitter site
tx_site_lat_long                    [latitude, longitude] coordinates of the transmitter antenna array
tx_heading                          transmitter array boresight pointing direction in degrees east of north
tx_rf_path                          the transmitter signal path; all inline elements listed
tx_ant_type                         transmitter antenna typ
tx_ant_coords                       transmitter antenna relative locations in meters from antenna 0
tx_feed_corr                        [[ant 0 magnitude, ...],[ant 0 phase, ...]] correction per transmitter antenna
tx_feed_corr_date                   [year, month, day] date the feedline corrections were determined
tx_feed_corr_type                   [magnitude type, phase type] (ex; [data median, instrumental])
tx_ant_mask                         [0, 1, ...] mask indicating which transmitter antennas were used and/or functional
tx_cw_code                          pseudo-random noise like code transmitted (contains full sequence)
tx_sample_rate                      sample rate of transmitted code
data                                dataset key for data in file organized per second
Processing Attributes (Level 1)
----------------------------------------------------------------------------------------------------------------------------------------------
decimation_rate
time_resolution
coherent_integration_time
incoherent_averages                 number of samples to average over
snr_cutoff_db
spectra_descriptors
xspectra_descriptors
data/{time}/time                    [hour, minute, millisecond] data time stamp
data/{time}/spectra
data/{time}/spectra_variance
data/{time}/xspectra
data/{time}/xspectra_variance
data/{time}/avg_spectra_noise
data/{time}/data_flag               flag indicating data above snr_cutoff exists at this time stamp
data/{time}/doppler_shift           target doppler shift in Hz
data/{time}/snr_db                  target signal strength in dB
data/{time}/rf_distance             distance from transmitter to target to receiver in kilometers
data/{time}/spectra_clutter_corr
data/{time}/spectra_noise           spectra noise value determined by the median spectra before cutoff
data/{time}/xspectra_clutter_corr
data/{time}/xspectra_noise
Imaging Attributes (Level 2)
----------------------------------------------------------------------------------------------------------------------------------------------
imaging_method                      imaging method used (ex; swht, linear)
clean                               image cleaning method applied
center                              target location method applied
swht_coeffs                         name of swht_coeffs file used, these files are upward of 4 GB
fov                                 [[az min, az max], [el min, el max]] image field of view
fov_center                          [az, el] angles in image which coincide with receiver boresight
resolution                          pixel resolution in degrees
lmax                                maximum harmonic degree the coefficients were calculated
data/{time}/time                    [year, month, day, hour, minute, second] data time stamp
data/{time}/doppler_shift           target doppler shift in Hz
data/{time}/snr_db                  target signal strength in dB
data/{time}/rf_distance             distance from transmitter to target to receiver in kilometers
data/{time}/azimuth                 targets azimuth position from fov_center in degrees
data/{time}/elevation               targets elevation position from fov_center in degrees
data/{time}/azimuth_spread          targets spread in azimuth given in degrees
data/{time}/elevation_spread        targets spread in elevation given in degrees
data/{time}/area                    area of target in image, highly variant based on imaging settings
Plotting Attributes (Level 3)
----------------------------------------------------------------------------------------------------------------------------------------------
velocity                            speed of target
position                            [lat, long, alt]
time                                time of data point
snr_db                              signal strength
spatial_spread                      [lat spread, long spread, alt spread]
spatial_spread_function             function to determine spreading
Configuration Class Only
----------------------------------------------------------------------------------------------------------------------------------------------
processing_source                   file path to directory holding channel separated digital_rf files
processing_destination              file path to directory to store the level1 hdf5 files
processing_start                    [year, month, day, hour, minute, second, millisecond] time to start processing
processing_stop                     [year, month, day, hour, minute, second, millisecond] time to stop processing
processing_step                     [day, hour, minute, second, millisecond] step size for processing
imaging_source                      file path to directory holding level1 hdf5 files to be imaged
imaging_destination                 file path to directory to store the level2 hdf5 files
imaging_start                       [year, month, day, hour, minute, second, millisecond] time to start imaging
imaging_stop                        [year, month, day, hour, minute, second, millisecond] time to stop imaging
imaging_step                        [day, hour, minute, second, millisecond] step size for imaging
==================================  ==========================================================================================================

