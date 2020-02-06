import numpy as np 
import h5py


def uvw_to_rtp(u, v, w):
    """
    Converts u, v, w cartesian baseline coordinates to radius, theta, phi 
    spherical coordinates.

    Args:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.

    Returns:
        r (float np.array): Radius baseline coordinate divided by wavelength.
        t (float np.array): Theta (elevation) baseline coordinate.
        p (float np.array): Phi (azimuthal) baseline coordinate.
    """

    r = np.sqrt(u**2 + v**2 + w**2)
    t = np.pi/2 - np.arctan2(w, np.sqrt(u**2 + v**2))
    p = np.arctan2(v, u) + np.pi
    np.nan_to_num(t, copy=False)

    return r, t, p


def rtp_to_uvw(r, t, p):
    """
    Converts radius, theta, phi spherical baseline coordinates to u, v, w 
    cartesian coordinates.

    Args:
        r (float np.array): Radius baseline coordinate divided by wavelength.
        t (float np.array): Theta (elevation) baseline coordinate.
        p (float np.array): Phi (azimuthal) baseline coordinate.

    Returns:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.
    """

    u = r*np.sin(t)*np.cos(p)
    v = r*np.sin(t)*np.sin(p) 
    w = r*np.cos(t)

    return u, v, w


def baselines(filename, wavelength):
    """
    Given relative antenna positions in cartesian coordinates with units of meters
    and the wavelength in meters determines the u, v, w baselines in cartesian coordinates.

    Args:
        filename (string): File name of .csv for antenna cartersian coordinates in meters.
        wavelength (float): Radar signal wavelength in meters.

    Returns:
        u (float np.array): East-West baseline coordinate divided by wavelength.
        v (float np.array): North-South baseline coordinate divided by wavelength.
        w (float np.array): Altitude baseline coordinate divided by wavelength.

    Notes:
        * Given N antenna then M=N(N-1)/2 unique baselines exist.
        * M baselines can include conjugates and a origin baseline for M_total = M*2 + 1.

    Todo:
        * Makes options to include or disclude 0th baseline and conjugates.
        * Make array positions load from the calibration.ini file.
        * Error handling for missing antenna position values (like no z).
    """

    coords = np.loadtxt(filename, delimiter=",") / wavelength
    # Baseline for an antenna with itself.
    u = np.array([0])
    v = np.array([0])
    w = np.array([0])
    # Include all possible baseline combinations.
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            u = np.append(u, (coords[i,0] - coords[j,0]))
            v = np.append(v, (coords[i,1] - coords[j,1]))
            w = np.append(w, (coords[i,2] - coords[j,2]))
    # Include the conjugate baselines.
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            u = np.append(u, (-coords[i,0] + coords[j,0]))
            v = np.append(v, (-coords[i,1] + coords[j,1]))
            w = np.append(w, (-coords[i,2] + coords[j,2]))

    return u, v, w


def fov_window():

    return


def stats_to_hdf5():

    return


def hdf5_to_stats():

    return


def swhtcoeffs_to_hdf5():

    return


def hdf5_to_swhtcoeffs():

    return


def rawdata_to_hdf5():

    return

def create_level1_hdf5(year,month,day,rx_name,tx_name,snr_cutoff,averages,fdec,center_freq,sample_rate):

    # inputs: year, month, day, hour
    # figure out way to grab most of these values from external file. antenna location, setup, corrections, type, code used, tx/rx locations
    # ex. coords = np.loadtxt(filename, delimiter=",")

    # read in hardware files corresponding to tx_name and rx_name HERE

    # create hdf5 file for this hour of data
    vis_values_file = h5py.File(f'prototype_vis_values/{year:04d}_{month:02d}_{day:02d}/icebear_{snr_cutoff:02d}dB_{averages:02d}00ms_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{tx_name}_{rx_name}.h5', 'w')
    
    # date and experiment
    vis_values_file.create_dataset('date', data=[year,month,day])
    vis_values_file.create_dataset('experiment_name', data=np.array(['normal_operations'],dtype='S'))
 
    # receiver site information
    vis_values_file.create_dataset('rx_name', data=np.array(['bakker'],dtype='S'))
    vis_values_file.create_dataset('rx_antenna_locations_x_y_z', data=antenna_location_x_y_z)
    vis_values_file.create_dataset('rx_RF_path', data=np.array(['Ant->feed->bulk->BPF->LNA->LNA->X300'],dtype='S'))
    vis_values_file.create_dataset('rx_antenna_type', data=np.array(['Cushcraft 50MHz Superboomer'],dtype='S'))
    vis_values_file.create_dataset('rx_phase_corrections_applied', data=phase_corr)
    vis_values_file.create_dataset('rx_magnitude_corrections_applied', data=mag_corr)
    vis_values_file.create_dataset('rx_location_lat_lon', data=[rx_lat,rx_lon])

    # transmitter site information
    vis_values_file.create_dataset('tx_name', data=np.array(['prelate'],dtype='S'))
    vis_values_file.create_dataset('tx_antenna_locations_x_y_z', data=antenna_location_x_y_z)
    vis_values_file.create_dataset('tx_RF_path', data=np.array(['X300->amplifier->bulk->feed->antenna'],dtype='S'))
    vis_values_file.create_dataset('tx_antenna_type', data=np.array(['Cushcraft A50-5S'],dtype='S'))
    vis_values_file.create_dataset('tx_phase_corrections_applied', data=phase_corr)
    vis_values_file.create_dataset('tx_magnitude_corrections_applied', data=mag_corr)
    vis_values_file.create_dataset('tx_antennas_used', data=tx_antenna_array)
    vis_values_file.create_dataset('tx_location_lat_lon', data=[tx_lat,tx_lon])

    # processing details
    vis_values_file.create_dataset('center_freq', data=[center_freq])
    vis_values_file.create_dataset('raw_recorded_sample_rate', data=[sample_rate])
    vis_values_file.create_dataset('software_decimation_rate', data=[fdec])
    vis_values_file.create_dataset('tx_code_used', data=b_code)
    vis_values_file.create_dataset('incoherent_averages', data=[averages])
    vis_values_file.create_dataset('time_resolution', data=[averages*0.1])
    vis_values_file.create_dataset('dB_SNR_cutoff', data=[snr_cutoff])

    # data information
    vis_values_file.create_dataset('spectra_descriptors', data=np.array(['spec00','spec11','spec22','spec33','spec44','spec55','spec66','spec77','spec88','spec99'],dtype='S'))
    vis_values_file.create_dataset('xspectra_descriptors',data=np.array(['xspec01','xspec02','xspec03','xspec04','xspec05','xspec06','xspec07','xspec08','xspec09','xspec12','xspec13','xspec14','xspec15','xspec16','xspec17','xspec18','xspec19','xspec23','xspec24','xspec25','xspec26','xspec27','xspec28','xspec29','xspec34','xspec35','xspec36','xspec37','xspec38','xspec39','xspec45','xspec46','xspec47','xspec48','xspec49','xspec56','xspec57','xspec58','xspec59','xspec67','xspec68','xspec69','xspec78','xspec79','xspec89'],dtype='S'))
    
    vis_values_file.close

    return

def append_level1_hdf5(hour,minute,second,avg_noise,spec_noise_median,xspec_noise_median,data_flag,doppler,rf_propagation,snr_dB_value,spectra,xspectra):

    # filter by snr before input into this file. pass data flag into file
    # have most of these things input to function as arrays.  Try to pass the full arrays at once, rather than using for loop for writing (single write of data for each group rather than appending)

    print(f'{hours:02d}{minutes:02d}{seconds:02d}')

    # append a new group for the current measurement
    vis_values_file = h5py.File(f'prototype_vis_values/{year:04d}_{month:02d}_{day:02d}/icebear_{snr_cutoff:02d}dB_{averages:02d}00ms_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{tx_name}_{rx_name}.h5', 'a')
    vis_values_file.create_group(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}')
    vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/time',data=[hour,minute,second])

    # create the noise data for the averaged spectra at a given time
    vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/avg_spectra_noise_value',data=[noise/10.0])
    vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/spectra_noise_value',data=[np.median(visi_spec[:,:,0]),np.median(visi_spec[:,:,1]),np.median(visi_spec[:,:,2]),np.median(visi_spec[:,:,3]),np.median(visi_spec[:,:,4]),np.median(visi_spec[:,:,5]),np.median(visi_spec[:,:,6]),np.median(visi_spec[:,:,7]),np.median(visi_spec[:,:,8]),np.median(visi_spec[:,:,9])])
    vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/xspectra_noise_value',data=xspec_median_value)
    vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/data_flag',data=[data_flag])

    # only write data if there are measurements above the SNR threshold
    if data_flag==True:
        vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift', data=doppler_values)
        vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/rf_distance', data=rf_propagation)    
        vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/snr_dB', data=log_snr_value)
        vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra', data=spectra)
        vis_values_file.create_dataset(f'data_index/{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra', data=xspectra)
    
    vis_values_file.close

    return


def hdf5_to_rawdata():

    return


if __name__ == '__main__':
    print("ICEBEAR: Incoherent Continuous-Wave E-Region Bistatic Experimental Auroral Radar")
    print("\t-icebear.utils.py")
