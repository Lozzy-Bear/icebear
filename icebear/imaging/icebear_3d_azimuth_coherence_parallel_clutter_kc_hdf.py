import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['figure.figsize'] = 20, 10
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

loop_processes_value = 250

output = mp.Queue()

import csv

#def coherence_calc_exp(lambda_r,distance,imag_azi,imag_width):
#    coherence_values = np.zeros((len(imag_azi),len(imag_width),len(distance)),dtype=np.complex64)
#    for x in range(9):
#        coherence_values[:,:,x] = np.matmul(np.exp(1.0j*imag_azi*distance[x]/lambda_r).reshape(len(imag_azi),1),np.exp(-imag_width*distance[x]/lambda_r).reshape(1,len(imag_width)))   
#    return coherence_values

def coherence_calc_gauss(lambda_r,distance,imag_azi,imag_width):

    #need to account for different baselines than linear array
    coherence_values = np.zeros((len(imag_azi),len(imag_width),len(distance)),dtype=np.complex64)
    for x in range(len(distance)):
        coherence_values[:,:,x] = np.matmul(np.exp(1.0j*imag_azi*distance[x]/lambda_r).reshape(len(imag_azi),1),np.exp(-imag_width*(distance[x]/lambda_r)**2).reshape(1,len(imag_width)))   
    return coherence_values

def linear_least_square_fit(exp_data,calc_data,weights):
    sigma_2 = np.matmul((np.abs(exp_data[None,None,:]-calc_data))**2,weights)
    return sigma_2

#for the parallelization
def fit_results(x,spectra_values_antennas,xspectra_values,logsnr_single,coherence_values_calc_gauss,baseline_lengths):

    #need to account for different baselines than linear array.  Probably can remove the weighting.
    antenna_coherence = np.zeros(len(xspectra_values),dtype=np.complex64)
    antenna_coherence[0] = 1.0
    temp_ind = 0
    for first_antenna in range(4):
        for second_antenna in range(first_antenna+1,5):
            antenna_coherence[temp_ind] = (xspectra_values[temp_ind]/np.sqrt(spectra_values_antennas[first_antenna]*spectra_values_antennas[second_antenna]))
            temp_ind+=1
    #for coherence_divider in range(8):
    #    antenna_coherence[coherence_divider+1] = antenna_coherence[coherence_divider+1]/np.abs(8-coherence_divider)#/(np.abs(np.power(10,logsnr_single/10))/(np.abs(np.power(10,logsnr_single/10))+1))

    weights = np.ones(len(xspectra_values))

    lsf_calc_gauss = linear_least_square_fit(antenna_coherence,coherence_values_calc_gauss,weights)

    '''
    plt.imshow(lsf_calc_gauss.T,aspect='auto',origin='lower',interpolation='none')
    plt.show()
    '''

    ind_gauss = np.unravel_index(np.argmin(lsf_calc_gauss, axis=None), lsf_calc_gauss.shape)

    gauss_angle = np.arcsin(-azi_values_gauss[ind_gauss[0]]/(2*np.pi))*180/np.pi

    gauss_width = np.abs(np.arcsin((-azi_values_gauss[ind_gauss[0]]-np.sqrt(4*azi_width_values_gauss[ind_gauss[1]]*np.log(2)))/(2*np.pi))-np.arcsin((-azi_values_gauss[ind_gauss[0]]+np.sqrt(4*azi_width_values_gauss[ind_gauss[1]]*np.log(2)))/(2*np.pi)))*180.0/np.pi

    lsf_gauss_value = lsf_calc_gauss[ind_gauss[0],ind_gauss[1]]

    '''
    plt.scatter(baseline_lengths/6.06,np.abs(antenna_coherence),label='data')
    plt.scatter(baseline_lengths/6.06,np.abs(coherence_values_calc_gauss[ind_gauss[0],ind_gauss[1],:]),label='fit')
    #plt.ylim(-1.1,1.1)
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.scatter(baseline_lengths/6.06,np.real(antenna_coherence),label='real')
    plt.scatter(baseline_lengths/6.06,np.imag(antenna_coherence),label='imaginary')
    plt.xlabel(r'Antenna Separation ($\lambda$)')
    plt.title(f'Doppler={doppler[indices[0][x]]} Hz, RF Distance={range_values[indices[0][x]]} km, {year:04d}-{month:02d}-{days:02d} {hours:02d}:{minutes:02d}:{int(seconds/1000):02d} UT')
    plt.ylabel('Coherence')
    plt.scatter(baseline_lengths/6.06,np.real(coherence_values_calc_gauss[ind_gauss[0],ind_gauss[1],:]),label='real fit')
    plt.scatter(baseline_lengths/6.06,np.imag(coherence_values_calc_gauss[ind_gauss[0],ind_gauss[1],:]),label='imaginary fit')
    #plt.ylim(-1.1,1.1)
    plt.legend()
    plt.grid()
    plt.show()
    '''

    output.put((x,gauss_angle,gauss_width,lsf_gauss_value))

year=2020
month=5
day=6
hour=4
minute=0
second=0

dif_azi_values_exp_ind = 0
dif_azi_width_ind = 0
max_azi = 0
max_azi_width = 0
max_fit_value = 0

if second==0:
	second+=1

azi_values_gauss = np.pi*(np.arange(300)-150)/(150)
azi_width_values_gauss = (np.pi*(np.arange(800)/3)/(600))**2

#create HDF5 File
def level2_hdf5_file_write(year,month,day,hours,level1_icebear_file):

    tx_name = 'prelate'
    rx_name = 'bakker'

    imag_values_file = h5py.File(f"prototype_imag_values/{year:04d}_{month:02d}_{day:02d}/icebear_3d_linear_imag_{year:04d}_{month:02d}_{day:02d}_{hours:02d}_{tx_name}_{rx_name}.h5", 'w')

    # date and experiment
    imag_values_file.create_dataset('date', data=[year,month,day])
    imag_values_file.create_dataset('experiment_name', data=level1_icebear_file['experiment_name'])
    
    #read in info for the given date and experiment
    
    #rx-setup file read-in routine
    
    #tx-setup file read-in routine

    # receiver site information (can go in external file)
    imag_values_file.create_dataset('rx_name',data=level1_icebear_file['rx_name'])
    imag_values_file.create_dataset('rx_antenna_locations_x_y_z',data=level1_icebear_file['rx_antenna_locations_x_y_z'])
    imag_values_file.create_dataset('rx_RF_path',data=level1_icebear_file['rx_RF_path'])
    imag_values_file.create_dataset('rx_antenna_type',data=level1_icebear_file['rx_antenna_type'])
    imag_values_file.create_dataset('rx_phase_corrections_applied', data=level1_icebear_file['rx_phase_corrections_applied'])
    imag_values_file.create_dataset('rx_magnitude_corrections_applied', data=level1_icebear_file['rx_magnitude_corrections_applied'])
    imag_values_file.create_dataset('rx_location_lat_lon', data=level1_icebear_file['rx_location_lat_lon'])
    imag_values_file.create_dataset('rx_pointing_dir', data=level1_icebear_file['rx_pointing_dir'])

    # transmitter site information (can go in external file)
    imag_values_file.create_dataset('tx_name', data=level1_icebear_file['tx_name'])
    imag_values_file.create_dataset('tx_antenna_locations_x_y_z', data=level1_icebear_file['tx_antenna_locations_x_y_z'])
    imag_values_file.create_dataset('tx_RF_path', data=level1_icebear_file['tx_RF_path'])
    imag_values_file.create_dataset('tx_antenna_type', data=level1_icebear_file['tx_antenna_type'])
    imag_values_file.create_dataset('tx_phase_corrections', data=level1_icebear_file['tx_phase_corrections'])
    imag_values_file.create_dataset('tx_magnitude_corrections', data=level1_icebear_file['tx_magnitude_corrections'])
    imag_values_file.create_dataset('tx_sample_rate', data=level1_icebear_file['tx_sample_rate'])
    imag_values_file.create_dataset('tx_antennas_used', data=level1_icebear_file['tx_antennas_used'])
    imag_values_file.create_dataset('tx_location_lat_lon', data=level1_icebear_file['tx_location_lat_lon'])
    imag_values_file.create_dataset('tx_pointing_dir', data=level1_icebear_file['tx_pointing_dir'])

    # processing details
    imag_values_file.create_dataset('center_freq', data=level1_icebear_file['center_freq'])
    imag_values_file.create_dataset('raw_recorded_sample_rate', data=level1_icebear_file['raw_recorded_sample_rate'])
    imag_values_file.create_dataset('software_decimation_rate', data=level1_icebear_file['software_decimation_rate'])
    imag_values_file.create_dataset('tx_code_used', data=level1_icebear_file['tx_code_used'])
    imag_values_file.create_dataset('incoherent_averages', data=level1_icebear_file['incoherent_averages'])
    imag_values_file.create_dataset('time_resolution', data=level1_icebear_file['time_resolution'])
    imag_values_file.create_dataset('dB_SNR_cutoff', data=level1_icebear_file['dB_SNR_cutoff'])
    
    imag_values_file.close

#write data to HDF5
def level2_hdf5_data_write(year,month,day,hours,minutes,seconds,snr_cutoff,averages,data_flag,doppler,range_values,logsnr,azimuth,azimuth_extent,least_squares_fit):

    tx_name = 'prelate'
    rx_name = 'bakker'

    # append a new group for the current measurement
    imag_values_file = h5py.File(f'prototype_imag_values/{year:04d}_{month:02d}_{day:02d}/icebear_3d_linear_imag_{year:04d}_{month:02d}_{day:02d}_{hours:02d}_{tx_name}_{rx_name}.h5', 'a')
    imag_values_file.create_group(f'data/{hours:02d}{minutes:02d}{seconds:05d}')

    imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/time',\
                                   data=[hours,minutes,seconds])

    # data flag
    imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/data_flag',data=[data_flag])

    # only write data if there are measurements above the SNR threshold
    if data_flag==True:
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/doppler_shift',data=doppler)
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/rf_distance',data=range_values)    
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/snr_dB',data=logsnr)
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/azimuth', data=azimuth)
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/azimuth_extent', data=azimuth_extent)
        imag_values_file.create_dataset(f'data/{hours:02d}{minutes:02d}{seconds:05d}/least_squares_fit', data=least_squares_fit)
    
    imag_values_file.close

snr_cutoff=1.0

#start of imaging
for temp_days in range(31):
    days=temp_days+day
    for temp_hours in range(24-hour):
        hours = hour+temp_hours
        ib_file = h5py.File(f'data/{year:04d}_{month:02d}_{days:02d}/icebear_3d_01dB_1000ms_vis_{year:04d}_{month:02d}_{days:02d}_{hours:02d}_prelate_bakker.h5','r')
        level2_hdf5_file_write(year,month,day,hours,ib_file)
        for temp_minutes in range(60-minute):
            minutes = minute+temp_minutes
            time_start = time.time()
            for temp_seconds in range(60-second):
                seconds=(second+temp_seconds)*1000
                
                if (ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/data_flag'][:]==False):
                        data_flag = False
                        averages = 10
                        level2_hdf5_data_write(year,month,day,hours,minutes,seconds,snr_cutoff,averages,data_flag,[],[],[],[],[],[])
                else:
                    #example to plot a spectra
                    logsnr = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/snr_dB'][:])
                    #print(logsnr)
                    doppler = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/doppler_shift'][:]
                    range_values = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/rf_distance'][:])
                    xspectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_xspectra'][:]
                    spectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_spectra'][:]
                    xspectra_clutter = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/xspectra_clutter_correction'][:]
                    spectra_clutter = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/spectra_clutter_correction'][:]
                    rx_antenna_location = ib_file[f'rx_antenna_locations_x_y_z'][:]
                    
                    #re-sort the xspectra and spectra values for ease of passing into azimuth fitting routine
                    #need antenna 0,1,3,7,9
                    ant_indices = [0,1,3,7,9]
                    spectra_values = spectra_values[:,ant_indices]
                    spectra_clutter = spectra_clutter[ant_indices]
                    #indicies (0,2,6,8),(10,14,16),(27,29),(43)
                    new_indices = [0,2,6,8,10,14,16,27,29,43]
                    xspectra_values = xspectra_values[:,new_indices]
                    xspectra_clutter = xspectra_clutter[new_indices]
                    #baselines can be calculated from antenna positions
                    baseline_lengths = np.zeros(10)
                    base_tmp_ind = 0
                    for first_antenna in range(4):
                        for second_antenna in range(first_antenna+1,5):
                            baseline_lengths[base_tmp_ind] = rx_antenna_location[0,ant_indices[first_antenna]]-rx_antenna_location[0,ant_indices[second_antenna]]
                            #print(rx_antenna_location[1,ant_indices[first_antenna]]-rx_antenna_location[1,ant_indices[second_antenna]])
                            base_tmp_ind+=1
                            
                    #calculate the expected gaussian fits based on the antenna positions
                    #need to change the calculated Gauss values to match the new baselines
                    coherence_values_calc_gauss = coherence_calc_gauss(6.06,baseline_lengths,azi_values_gauss,azi_width_values_gauss)


                    indices = np.where(logsnr>snr_cutoff)
                    print(len(indices[0]))
                    if (len(indices[0])==0):
                        data_flag = False
                        averages = 10
                        level2_hdf5_data_write(year,month,day,hours,minutes,seconds,snr_cutoff,averages,data_flag,[],[],[],[],[],[])
                    else:
                        if not any((range_values[indices[0][ind_loop]]<100.0 or (range_values[indices[0][ind_loop]]>2900.0)) for ind_loop in indices[0][:]):
                            
                            data_flag = True
                            averages=10
                            doppler_t = np.zeros(len(indices[0]))
                            range_values_t = np.zeros(len(indices[0]))
                            logsnr_t = np.zeros(len(indices[0]))
                            azimuth_t = np.zeros(len(indices[0]))
                            azimuth_extent_t = np.zeros(len(indices[0]))
                            least_squares_fit_t = np.zeros(len(indices[0]))
                    
                            num=0

                            #only processes subsets at a time.  Could be improved to determine the number of cores that are available and split based on that
                            for num in range(int(len(indices[0])/loop_processes_value)):
                                processes = [mp.Process(target=fit_results, args=(x,spectra_values[indices[0][x],:]-spectra_clutter,xspectra_values[indices[0][x],:]-xspectra_clutter,logsnr[indices[0][x]],coherence_values_calc_gauss,baseline_lengths)) for x in range((num)*loop_processes_value,(num+1)*loop_processes_value)]
                                for p in processes:
                                    p.start()
                                # Exit the completed processes
                                for p in processes:
                                    p.join()
                                
                                #moved this to within loop due to hanging.  Might be due to overfilling queue.
                                # Get process results from the output queue
                                ind_temp = [output.get() for p in processes]

                                ind_temp.sort()
                                x_values = [r[0] for r in ind_temp]
                                gauss_angle = [r[1] for r in ind_temp]
                                gauss_width = [r[2] for r in ind_temp] 
                                lsf_gauss_value = [r[3] for r in ind_temp]
                                
                                for counter in range(len(x_values)):
                                    doppler_t[x_values[counter]] = doppler[indices[0][x_values[counter]]]
                                    logsnr_t[x_values[counter]] = logsnr[indices[0][x_values[counter]]]
                                    range_values_t[x_values[counter]] = range_values[indices[0][x_values[counter]]]
                                    azimuth_t[x_values[counter]] = gauss_angle[counter]
                                    azimuth_extent_t[x_values[counter]] = gauss_width[counter]
                                    least_squares_fit_t[x_values[counter]] = lsf_gauss_value[counter]

                            x_value_corr = (int(len(indices[0])/loop_processes_value)*loop_processes_value)
                            processes = [mp.Process(target=fit_results, args=(x,spectra_values[indices[0][x],:]-spectra_clutter,xspectra_values[indices[0][x],:]-xspectra_clutter,logsnr[indices[0][x]],coherence_values_calc_gauss,baseline_lengths)) for x in range(0,len(indices[0])%loop_processes_value)]
                            for p in processes:
                                p.start()
                            # Exit the completed processes
                            for p in processes:
                                p.join()

                            # Get process results from the output queue
                            ind_temp = [output.get() for p in processes]

                            ind_temp.sort()
                            x_values = [r[0] for r in ind_temp]
                            gauss_angle = [r[1] for r in ind_temp]
                            gauss_width = [r[2] for r in ind_temp] 
                            lsf_gauss_value = [r[3] for r in ind_temp]

                            for counter in range(len(x_values)):
                                    doppler_t[x_value_corr+x_values[counter]] = doppler[indices[0][x_value_corr+x_values[counter]]]
                                    logsnr_t[x_value_corr+x_values[counter]] = logsnr[indices[0][x_value_corr+x_values[counter]]]
                                    range_values_t[x_value_corr+x_values[counter]] = range_values[indices[0][x_value_corr+x_values[counter]]]
                                    azimuth_t[x_value_corr+x_values[counter]] = gauss_angle[counter]
                                    azimuth_extent_t[x_value_corr+x_values[counter]] = gauss_width[counter]
                                    least_squares_fit_t[x_value_corr+x_values[counter]] = lsf_gauss_value[counter]

                            level2_hdf5_data_write(year,month,day,hours,minutes,seconds,snr_cutoff,averages,data_flag,doppler_t,range_values_t,logsnr_t,azimuth_t,azimuth_extent_t,least_squares_fit_t)

                        #if there is a significant amount of clutter, dropped samples, and/or noise in the data
                        else:
                            data_flag = False
                            averages = 10
                            level2_hdf5_data_write(year,month,day,hours,minutes,seconds,snr_cutoff,averages,data_flag,[],[],[],[],[],[])
                                           
            second=0
            print('One minute process time: ',time.time()-time_start)
        minute=0
        ib_file.close
    hour=0
