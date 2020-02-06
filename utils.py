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

def icebear_level1_hdf5_create_file():

    # inputs: year, month, day, hour
    # figure out way to grab most of these values from external file. antenna location, setup, corrections, type, code used, tx/rx locations
    # ex. coords = np.loadtxt(filename, delimiter=",")

    vis_values_file = h5py.File(f'/home/icebear-processing/ICEBEAR_3D_software/ICEBEAR-3D/icebear_radar/prototype_vis_values/icebear_3db_1s_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_00.h5', 'w')
    vis_values_file.create_dataset('rx_antenna_locations_x_y_z', data=antenna_location_x_y_z)
    vis_values_file.create_dataset('RF_path', data=np.array(['Ant->feed->bulk->BPF->LNA->LNA->X300'],dtype='S'))
    vis_values_file.create_dataset('date', data=[year,month,day])
    vis_values_file.create_dataset('antenna_spectra_descriptors', data=np.array(['Doppler (Hz)','RF Path Length (km)','spec00','spec11','spec22','spec33','spec44','spec55','spec66','spec77','spec88','spec99','xspec01','xspec02','xspec03','xspec04','xspec05','xspec06','xspec07','xspec08','xspec09','xspec12','xspec13','xspec14','xspec15','xspec16','xspec17','xspec18','xspec19','xspec23','xspec24','xspec25','xspec26','xspec27','xspec28','xspec29','xspec34','xspec35','xspec36','xspec37','xspec38','xspec39','xspec45','xspec46','xspec47','xspec48','xspec49','xspec56','xspec57','xspec58','xspec59','xspec67','xspec68','xspec69','xspec78','xspec79','xspec89'],dtype='S'))
    vis_values_file.create_dataset('phase_corrections', data=phase_corr)
    vis_values_file.create_dataset('magnitude_corrections', data=mag_corr)
    vis_values_file.create_dataset('decimation_rate', data=[fdec])
    vis_values_file.create_dataset('center_freq', data=[49500000.0])
    vis_values_file.create_dataset('initial_sample_rate', data=[sample_rate])
    vis_values_file.create_dataset('rx_antenna_type', data=np.array(['Cushcraft 50MHz Superboomer'],dtype='S'))
    vis_values_file.create_dataset('tx_antenna_type', data=np.array(['Cushcraft A50-5S'],dtype='S'))
    vis_values_file.create_dataset('dB_SNR_cutoff', data=[3.0])
    vis_values_file.create_dataset('time_resolution', data=[averages*0.1])
    vis_values_file.create_dataset('incoherent_averages', data=[averages])
    vis_values_file.create_dataset('code_used', data=b_code) #store the binary code? 10,000 bits?
    vis_values_file.create_dataset('tx_location_lat_lon', data=[tx_lat,tx_lon])
    vis_values_file.create_dataset('rx_location_lat_lon', data=[rx_lat,rx_lon])
    vis_values_file.close

    return

def icebear_level1_hdf5_append_data():

    # filter by snr before input into this file. pass data flag into file
    # have most of these things input to function as arrays.  Try to pass the full arrays at once, rather than using for loop for writing (single write of data for each group rather than appending)

    vis_values_file = h5py.File(f'/home/icebear-processing/ICEBEAR_3D_software/ICEBEAR-3D/icebear_radar/prototype_vis_values/icebear_3db_1s_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_00.h5', 'a')
    print(f'{hours:02d}{minutes:02d}{seconds:02d}')
    vis_values_file.create_group(f'{hours:02d}{minutes:02d}{seconds:02d}')
    vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/time',data=[hour,minute,second])
    vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/avg_noise_value',data=[noise/10.0])
    vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/spec_noise_value',data=[np.median(visi_spec[:,:,0]),np.median(visi_spec[:,:,1]),np.median(visi_spec[:,:,2]),np.median(visi_spec[:,:,3]),np.median(visi_spec[:,:,4]),np.median(visi_spec[:,:,5]),np.median(visi_spec[:,:,6]),np.median(visi_spec[:,:,7]),np.median(visi_spec[:,:,8]),np.median(visi_spec[:,:,9])])
    vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/xspec_median_value',data=xspec_median_value)
    vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/data_flag',data=[data_flag])
    vis_values_file.close

    if first_record==0:
        first_record+=1
        vis_values_file = h5py.File(f'/home/icebear-processing/ICEBEAR_3D_software/ICEBEAR-3D/icebear_radar/prototype_vis_values/icebear_3db_1s_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_00.h5',
 'a')
        vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift', data=np.reshape(doppler_values,(1,1)),chunks=True,maxshape=(None,1))
        vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/snr_dB', data=np.reshape(log_snr_value,(1,1)),chunks=True,maxshape=(None,1))
        vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/rf_distance', data=np.reshape(rf_propagation,(1,1)),chunks=True,maxshape=(None,1))
        vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra', data=np.reshape(antenna_spectra,(1,len(antenna_spectra))),chunks=True,maxshape=(None,len(antenna_spectra)))
        vis_values_file.create_dataset(f'{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra', data=np.reshape(antenna_xspectra,(1,len(antenna_xspectra))),chunks=True,maxshape=(None,len(antenna_xspectra)))
        vis_values_file.close
    else:
        with h5py.File(f'/home/icebear-processing/ICEBEAR_3D_software/ICEBEAR-3D/icebear_radar/prototype_vis_values/icebear_3db_1s_vis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_00.h5', 'a') as vis_values_file:
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift"].resize((vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift"].shape[0]+1,vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift"].shape[1]))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift"][vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/doppler_shift"].shape[0]-1,:] = np.reshape(doppler_values,(1,1))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/snr_dB"].resize((vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/snr_dB"].shape[0]+1,vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/snr_dB"].shape[1]))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/snr_dB"][vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/snr_dB"].shape[0]-1,:] = np.reshape(log_snr_value,(1,1))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/rf_distance"].resize((vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/rf_distance"].shape[0]+1,vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/rf_distance"].shape[1]))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/rf_distance"][vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/rf_distance"].shape[0]-1,:] = np.reshape(rf_propagation,(1,1))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra"].resize((vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra"].shape[0]+1,vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra"].shape[1]))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra"][vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_spectra"].shape[0]-1,:] = np.reshape(antenna_spectra,(1,len(antenna_spectra)))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra"].resize((vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra"].shape[0]+1,vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra"].shape[1]))
            vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra"][vis_values_file[f"{hours:02d}{minutes:02d}{seconds:02d}/antenna_xspectra"].shape[0]-1,:] = np.reshape(antenna_xspectra,(1,len(antenna_xspectra)))
        vis_values_file.close

    return


def hdf5_to_rawdata():

    return


if __name__ == '__main__':
    print("ICEBEAR: Incoherent Continuous-Wave E-Region Bistatic Experimental Auroral Radar")
    print("\t-icebear.utils.py")
