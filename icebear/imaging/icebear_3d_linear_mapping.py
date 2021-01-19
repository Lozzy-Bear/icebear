import h5py
import numpy as np
import time
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

#how many parallel imaging processes to do in a batch.  More than 250 seems to cause errors, though this may be OS/computer hardware dependent.
loop_processes_value = 250

#queue for parallel processing implementation
output = mp.Queue()

#pre-set quantized values for azimuth and azimuth width to fit measured data to
azi_values_gauss = np.pi*(np.arange(300)-150)/(150)
azi_width_values_gauss = (np.pi*(np.arange(800)/3)/(600))**2


#determine the expected spatial coherence of the incoming signal between the baselines for the antennas arranged in a linear array
def coherence_calc_gauss(lambda_r,distance,imag_azi,imag_width):

    #need to account for different baselines
    coherence_values = np.zeros((len(imag_azi),len(imag_width),len(distance)),dtype=np.complex64)
    for x in range(len(distance)):
        coherence_values[:,:,x] = np.matmul(np.exp(1.0j*imag_azi*distance[x]/lambda_r).reshape(len(imag_azi),1),np.exp(-imag_width*(distance[x]/lambda_r)**2).reshape(1,len(imag_width)))   
    return coherence_values


#determine the least square fit value with weights applied
def linear_least_square_fit(exp_data,calc_data,weights):
    sigma_2 = np.matmul((np.abs(exp_data[None,None,:]-calc_data))**2,weights)
    return sigma_2


#for the parallelization
def fit_results(x,spectra_values_antennas,xspectra_values,logsnr_single,coherence_values_calc_gauss,baseline_lengths):

    #only 5 antennas in the icebear 3d antenna configuration.  Calculate
    antenna_coherence = np.zeros(len(xspectra_values),dtype=np.complex64)
    antenna_coherence[0] = 1.0
    temp_ind = 0
    for first_antenna in range(4):
        for second_antenna in range(first_antenna+1,5):
            antenna_coherence[temp_ind] = (xspectra_values[temp_ind]/np.sqrt(spectra_values_antennas[first_antenna]*spectra_values_antennas[second_antenna]))
            temp_ind+=1

    weights = np.ones(len(xspectra_values))

    #determine the least squares fit of the data for all the azimuth/azimuth width combinations
    lsf_calc_gauss = linear_least_square_fit(antenna_coherence,coherence_values_calc_gauss,weights)

    #find the indices corresponding to the minimum in the array of least squares fit
    ind_gauss = np.unravel_index(np.argmin(lsf_calc_gauss, axis=None), lsf_calc_gauss.shape)

    #calculate the azimuth angle and azimuth width based on the fitted Gaussian parameters
    gauss_angle = np.arcsin(-azi_values_gauss[ind_gauss[0]]/(2*np.pi))*180/np.pi
    #azimuth width determined to be FWHM value, though slightly shifted due to converting to angle
    gauss_width = np.abs(np.arcsin((-azi_values_gauss[ind_gauss[0]]-np.sqrt(4*azi_width_values_gauss[ind_gauss[1]]*np.log(2)))/(2*np.pi))-np.arcsin((-azi_values_gauss[ind_gauss[0]]+np.sqrt(4*azi_width_values_gauss[ind_gauss[1]]*np.log(2)))/(2*np.pi)))*180.0/np.pi

    lsf_gauss_value = lsf_calc_gauss[ind_gauss[0],ind_gauss[1]]

    #return the values for azimuth and azimuth width corresponding to the minimum least squares fit
    output.put((x,gauss_angle,gauss_width,lsf_gauss_value))
    
    
def append_level2_hdf5_linear_imaging(filename, hour, minute, second, azimuth, image_data_flag, azimuth_extent, least_squares_fit):
    """
    
    Parameters
    ----------
    filename
    hour
    minute
    second
    image_data_flag
    azimuth
    azimuthal_extent
    least_squares_fit

    Returns
    -------

    """
    # append a new group for the current measurement
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_dataset(f'data/{time}/image_no_data', data=image_data_flag)
    f.create_dataset(f'data/{time}/azimuth', data=azimuth)
    f.create_dataset(f'data/{time}/azimuthal_extent', data=azimuthal_extent)
    f.create_dataset(f'data/{time}/least_squares_fit', data=least_squares_fit)
    f.close()
    return None


def icebear_linear_imaging():

    #set date of data to look at
    year=2020
    month=5
    day=6
    hour=4
    minute=0
    second=0

    #initialize some variables
    dif_azi_values_exp_ind = 0
    dif_azi_width_ind = 0
    max_azi = 0
    max_azi_width = 0
    max_fit_value = 0

    #can set a higher snr threshold than provided in the initial spectra data
    snr_cutoff=1.0

    if second==0:
	    second+=1

    #start of imaging
    for temp_days in range(31):
        days=temp_days+day
        for temp_hours in range(24-hour):
            hours = hour+temp_hours
            ib_file = h5py.File(f'/ib_data/{year:04d}_{month:02d}_{days:02d}/icebear_3d_01dB_1000ms_vis_{year:04d}_{month:02d}_{days:02d}_{hours:02d}_prelate_bakker.h5','r')
            #level2_hdf5_file_write(year,month,day,hours,ib_file)
            for temp_minutes in range(60-minute):
                minutes = minute+temp_minutes
                time_start = time.time()
                for temp_seconds in range(60-second):
                    seconds=(second+temp_seconds)*1000
                    
                    if (ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/data_flag'][:]==False):
                            image_data_flag = False
                            append_level2_hdf5_linear_imaging(filename,hours,minutes,seconds,image_data_flag,[],[],[])
                    else:
                        #read in the data to be fit
                        logsnr = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/snr_dB'][:])
                        doppler = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/doppler_shift'][:]
                        range_values = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/rf_distance'][:])
                        xspectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_xspectra'][:]
                        spectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_spectra'][:]
                        xspectra_clutter = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/xspectra_clutter_correction'][:]
                        spectra_clutter = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/spectra_clutter_correction'][:]
                        rx_antenna_location = ib_file[f'rx_antenna_locations_x_y_z'][:]
                        
                        #re-sort the xspectra and spectra values for ease of passing into azimuth fitting routine
                        #need antenna 0,1,3,7,9 (antennas in the linear array)
                        ant_indices = [0,1,3,7,9]
                        spectra_values = spectra_values[:,ant_indices]
                        spectra_clutter = spectra_clutter[ant_indices]
                        #corresponds to indicies (0,2,6,8),(10,14,16),(27,29),(43) in the xpsectra array
                        new_indices = [0,2,6,8,10,14,16,27,29,43]
                        xspectra_values = xspectra_values[:,new_indices]
                        xspectra_clutter = xspectra_clutter[new_indices]
                        #baselines can be calculated from antenna positions
                        baseline_lengths = np.zeros(10)
                        base_tmp_ind = 0
                        for first_antenna in range(4):
                            for second_antenna in range(first_antenna+1,5):
                                baseline_lengths[base_tmp_ind] = rx_antenna_location[0,ant_indices[first_antenna]]-rx_antenna_location[0,ant_indices[second_antenna]]
                                base_tmp_ind+=1
                                
                        #calculate the expected gaussian fits based on the antenna positions
                        #need to change the calculated Gauss values to match the new baselines
                        coherence_values_calc_gauss = coherence_calc_gauss(6.06,baseline_lengths,azi_values_gauss,azi_width_values_gauss)


                        indices = np.where(logsnr>snr_cutoff)
                        print(len(indices[0]))
                        
                        #determine if there are any data above the snr cutoff
                        if (len(indices[0])==0):
                            #if no data above threshold, write blanks to imaging hdf5 file
                            image_data_flag = False
                            append_level2_hdf5_linear_imaging(filename,hours,minutes,seconds,image_data_flag,[],[],[])
                        else:
                            #rudimentary check for dropped samples.  Potential area for improvement in future iterations
                            if not any((range_values[indices[0][ind_loop]]<100.0 or (range_values[indices[0][ind_loop]]>2900.0)) for ind_loop in indices[0][:]):
                                
                                image_data_flag = True
                                averages=10
                                doppler_t = np.zeros(len(indices[0]))
                                range_values_t = np.zeros(len(indices[0]))
                                logsnr_t = np.zeros(len(indices[0]))
                                azimuth_t = np.zeros(len(indices[0]))
                                azimuth_extent_t = np.zeros(len(indices[0]))
                                least_squares_fit_t = np.zeros(len(indices[0]))
                                
                                num=0

                                #use parallel processing to speed up the data analysis
                                #currently only processes subsets at a time, as errors occurred with too many at once
                                for num in range(int(len(indices[0])/loop_processes_value)):
                                    #fit the measured spatial coherence to the expected values assuming a Gaussian brightness distribution
                                    processes = [mp.Process(target=fit_results, args=(x,spectra_values[indices[0][x],:]-spectra_clutter,xspectra_values[indices[0][x],:]-xspectra_clutter,logsnr[indices[0][x]],coherence_values_calc_gauss,baseline_lengths)) for x in range((num)*loop_processes_value,(num+1)*loop_processes_value)]
                                    for p in processes:
                                        p.start()
                                    # Exit the completed processes
                                    for p in processes:
                                        p.join()
                                    
                                    # Get process results from the output queue
                                    ind_temp = [output.get() for p in processes]

                                    #gather the values from the parallel processing
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

                                #repeat the process above for the last portion of the subset of data
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

                                #write the fitted imaged data to the hdf5 file
                                append_level2_hdf5_linear_imaging(filename,hours,minutes,seconds,image_data_flag,azimuth_t,azimuth_extent_t,least_squares_fit_t)

                            #if there is a significant amount of clutter, dropped samples, and/or noise in the data
                            else:
                                image_data_flag = False
                                append_level2_hdf5_linear_imaging(filename,hours,minutes,seconds,image_data_flag,[],[],[])
                                               
                second=0
                print('One minute process time: ',time.time()-time_start)
            minute=0
        hour=0
