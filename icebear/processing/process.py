import h5py
import numba
import numpy as np
import ctypes as C
import digital_rf
import icebear.utils
import os
import icebear.processing.dsp as dsp


def load_cuda(config):
    if not config.cuda:
        try:
            import cupy as xp
        except ModuleNotFoundError:
            import numpy as xp
            print("cupy module not found, using numpy instead.")
    else:
        import numpy as xp
    return xp


def generate_level1(config):
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    
    xp = load_cuda(config)

    filenames = []
    level0_data = digital_rf.DigitalRFReader(config.processing_source)
    channels = level0_data.get_channels()
    if len(channels) == 0:
        print(f'ERROR: No data channels found in {config.processing_source}')
        exit()
    else:
        print('channels acquired:')
        for i in range(len(channels)):
            print(f'\t-{str(channels[i])}')

    print('processing start:')

    total_xspectras = int(len(channels) * (len(channels) - 1) / 2)
    total_spectras = int(len(channels))
    bcode = generate_bcode(config.prn_code_file)
    config.update_attr('tx_cw_code', bcode)
    complex_correction = 1.0 / config.rx_feed_corr[0] * np.exp(1j * np.deg2rad(config.rx_feed_corr[1]))
    fft_freq = np.fft.fftfreq(int(config.code_length / config.decimation_rate),
                              config.decimation_rate / config.raw_sample_rate)

    if not type(config.processing_step) == type(np.array([])):
        config.processing_step = [0, 0, 0, config.incoherent_averages * config.time_resolution, 0]
    time = icebear.utils.Time(config.processing_start, config.processing_stop, config.processing_step)
    if config.incoherent_averages * config.time_resolution > time.step_epoch:
        print(f'WARNING: averaging time length {config.incoherent_averages * config.time_resolution}s'
              f' is greater than step time length {time.step_epoch}s')
    temp_hour = [-1, -1, -1, -1]
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        # todo : Should this stop actually be stop + step so it ends correctly?
        now = time.get_date(t)
        spectra = xp.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras),
            dtype=xp.complex64)
        spectra_variance = xp.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras),
            dtype=xp.complex64)
        xspectra = xp.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras),
            dtype=xp.complex64)
        xspectra_variance = xp.empty(
            shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras),
            dtype=xp.complex64)
        power = xp.zeros(shape=(int(config.code_length / config.decimation_rate), config.number_ranges),
                         dtype=xp.complex64)

        # create new file if new hour
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            filename = f'{config.processing_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/' \
                       f'{config.radar_name}_{config.experiment_name}_{int(config.snr_cutoff_db):02d}dB_' \
                       f'{int(config.incoherent_averages):02d}00ms_' \
                       f'{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}_{int(now.hour):02d}_' \
                       f'{config.tx_site_name}_{config.rx_site_name}.h5'

            if not os.path.exists(
                    f'{config.processing_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/'):
                os.mkdir(f'{config.processing_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/')

            filenames.append(filename)
            create_level1_hdf5(config, filename, int(now.year), int(now.month), int(now.day))
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]
            print(f'\t-created level 1 HDf5: {filename}')

        # calculate the self-spectra
        for antenna_num in range(total_spectras):
            result, variance = decx(config, t, level0_data, bcode, channels[antenna_num], channels[antenna_num],
                                    complex_correction[antenna_num], complex_correction[antenna_num])
            if not (type(result) == type(xp.array([]))):
                if result == 1:
                    break
            spectra[:, :, antenna_num] = result
            spectra_variance[:, :, antenna_num] = variance
            power[:, :] += result[:, :]
        if not (type(result) == type(xp.array([]))):
            if result == 1:
                continue

        # Perform cross-spectra for each desired baseline or correlation
        cnt = 0
        for j in range(len(channels) - 1):
            for i in range(j + 1, len(channels)):
                result, variance = decx(config, t, level0_data, bcode, channels[j], channels[i],
                                        complex_correction[j], complex_correction[i])
                xspectra[:, :, cnt] = result
                xspectra_variance[:, :, cnt] = variance
                cnt += 1

        snr, noise = dsp.snr(power, 'galeschuk')
        
        if not config.cuda:
            snr = xp.asnumpy(snr) #
       
        snr = np.ma.masked_where(snr < 0.0, snr)
        logsnr = 10 * np.log10(snr.filled(1))
        logsnr = np.ma.masked_where(logsnr < 1.0, logsnr)
        # Find the range-Doppler bins with a SNR above the threshold.
        snr_indices = np.asarray(np.where(logsnr >= config.snr_cutoff_db)).T
        if len(snr_indices) > 0:
            data_flag = True
        else:
            data_flag = False

        # Calculate the spectra noise value.
        spectra_median = xp.zeros(total_spectras, dtype=xp.complex64)
        spectra_clutter_corr = xp.zeros(total_spectras, dtype=xp.complex64)
        for num_spec in range(total_spectras):
            spectra_median[num_spec] = xp.mean(spectra[:, 1900:2000, num_spec])
            spectra_clutter_corr[num_spec] = xp.mean(spectra[:, 0:config.clutter_gates, num_spec])

        # calculate the xspectra 'noise' value
        xspectra_median = xp.zeros(total_xspectras, dtype=xp.complex64)
        xspectra_clutter_corr = xp.zeros(total_xspectras, dtype=xp.complex64)
        for num_xspec in range(total_xspectras):
            xspectra_median[num_xspec] = xp.mean(xspectra[:, 1900:2000, num_xspec])
            xspectra_clutter_corr[num_xspec] = xp.mean(xspectra[:, 0:config.clutter_gates, num_xspec])

        if not config.cuda:
            spectra = spectra.get() #
            spectra_variance = spectra_variance.get() #
            xspectra = xspectra.get() #
            xspectra_variance = xspectra_variance.get() #

        # Calculate noise, range and Doppler values
        doppler = fft_freq[snr_indices[:, 0]]
        rf_distance = config.range_resolution * (snr_indices[:, 1] - config.timestamp_corr)

        if not config.cuda:
            noise = xp.asnumpy(noise / total_spectras) #
        
        noise = noise/ total_spectras
        snr_db = logsnr[snr_indices[:, 0], snr_indices[:, 1]]

        if not config.cuda:
            spectra_median = spectra_median.get() #
            spectra_clutter_corr = spectra_clutter_corr.get() #
            xspectra_median = xspectra_median.get() #
            xspectra_clutter_corr = xspectra_clutter_corr.get() #

        append_level1_hdf5(filename, int(now.hour), int(now.minute), int(now.second * 1000),
                           data_flag, doppler, rf_distance, snr_db, noise,
                           spectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_median, spectra_clutter_corr,
                           xspectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_median, xspectra_clutter_corr)
        print(f'\t-appended {int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}')

    return filenames


def create_level1_hdf5(config, filename, year, month, day):
    """
    Create a level1 HDF5 formatted file for storage of ICEBEAR spectra and cross-spectra

    Parameters
    ----------
        config : Class Object
            Config class instantiation.
        filename:
        year :
        month :
        day :

    Returns
    -------
        None

    Todo
        Make sure everything is put into numpy arrays

    """
    # general information
    f = h5py.File(filename, 'w')
    f.create_dataset('date_created', data=config.date_created)
    f.create_dataset('version', data=np.array(config.version, dtype='S'))
    f.create_dataset('date', data=np.array([year, month, day]))
    f.create_dataset('config_updated', data=np.array(config.config_updated))
    f.create_dataset('experiment_name', data=np.array(config.experiment_name, dtype='S'))
    f.create_dataset('radar_config', data=np.array(config.radar_config, dtype='S'))
    f.create_dataset('center_freq', data=config.center_freq)
    # receiver site information
    f.create_dataset('rx_site_name', data=np.array(config.rx_site_name, dtype='S'))
    f.create_dataset('rx_site_lat_long', data=config.rx_site_lat_long)
    f.create_dataset('rx_heading', data=config.rx_heading)
    f.create_dataset('rx_rf_path', data=np.array(config.rx_rf_path, dtype='S'))
    f.create_dataset('rx_ant_type', data=np.array(config.rx_ant_type, dtype='S'))
    f.create_dataset('rx_ant_coords', data=config.rx_ant_coords)
    f.create_dataset('rx_feed_corr', data=config.rx_feed_corr)
    f.create_dataset('rx_feed_corr_date ', data=config.rx_feed_corr_date)
    f.create_dataset('rx_feed_corr_type', data=np.array(config.rx_feed_corr_type, dtype='S'))
    f.create_dataset('rx_ant_mask', data=config.rx_ant_mask)
    f.create_dataset('rx_sample_rate', data=config.rx_sample_rate)
    # transmitter site information
    f.create_dataset('tx_site_name', data=np.array(config.tx_site_name, dtype='S'))
    f.create_dataset('tx_site_lat_long', data=config.tx_site_lat_long)
    f.create_dataset('tx_heading', data=config.tx_heading)
    f.create_dataset('tx_rf_path', data=np.array(config.tx_rf_path, dtype='S'))
    f.create_dataset('tx_ant_type', data=np.array(config.tx_ant_type, dtype='S'))
    f.create_dataset('tx_ant_coords', data=config.tx_ant_coords)
    f.create_dataset('tx_feed_corr', data=config.tx_feed_corr)
    f.create_dataset('tx_feed_corr_date ', data=config.tx_feed_corr_date)
    f.create_dataset('tx_feed_corr_type', data=np.array(config.tx_feed_corr_type, dtype='S'))
    f.create_dataset('tx_ant_mask', data=config.tx_ant_mask)
    f.create_dataset('tx_cw_code', data=config.tx_cw_code)
    f.create_dataset('tx_sample_rate', data=config.tx_sample_rate)
    # processing settings
    f.create_dataset('decimation_rate', data=config.decimation_rate)
    f.create_dataset('time_resolution', data=config.time_resolution)
    f.create_dataset('coherent_integration_time', data=config.coherent_integration_time)
    f.create_dataset('incoherent_averages', data=config.incoherent_averages)
    f.create_dataset('number_ranges', data=config.number_ranges)
    f.create_dataset('range_resolution', data=config.range_resolution)
    f.create_dataset('timestamp_corr', data=config.timestamp_corr)
    f.create_dataset('clutter_gates', data=config.clutter_gates)
    f.create_dataset('code_length', data=config.code_length)
    f.create_dataset('snr_cutoff_db', data=config.snr_cutoff_db)
    f.create_dataset('spectra_descriptors', data=config.spectra_descriptors)
    f.create_dataset('xspectra_descriptors', data=config.xspectra_descriptors)
    f.create_group('data')
    f.close()
    return None


def append_level1_hdf5(filename, hour, minute, second, data_flag, doppler_shift, rf_distance, snr_db, noise,
                       spectra, spectra_variance, spectra_noise, spectra_clutter_corr,
                       xspectra, xspectra_variance, xspectra_noise, xspectra_clutter_corr):
    """
    Append spectra and cross-spectra ICEBEAR data to previously created HDF5 file

    Parameters
    ----------
        filename
        hour
        minute
        second
        data_flag
        doppler_shift
        rf_distance
        snr_db
        noise
        spectra
        spectra_variance
        spectra_noise
        spectra_clutter_corr
        xspectra
        xspectra_variance
        xspectra_noise
        xspectra_clutter_corr

    Returns
    -------

    Todo
        Make sure everything is put into numpy arrays
    """
    # append a new group for the current measurement
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    # create the noise data for the averaged spectra at a given time
    f.create_dataset(f'data/{time}/avg_spectra_noise', data=noise)
    f.create_dataset(f'data/{time}/spectra_noise', data=spectra_noise)
    f.create_dataset(f'data/{time}/xspectra_noise', data=xspectra_noise)
    f.create_dataset(f'data/{time}/spectra_clutter_corr', data=spectra_clutter_corr)
    f.create_dataset(f'data/{time}/xspectra_clutter_corr', data=xspectra_clutter_corr)
    f.create_dataset(f'data/{time}/data_flag', data=[data_flag])
    # only write data if there are measurements above the SNR threshold
    print(f'\t-data_flag = {data_flag}')
    if data_flag:
        f.create_dataset(f'data/{time}/doppler_shift', data=doppler_shift)
        f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
        f.create_dataset(f'data/{time}/snr_db', data=snr_db)
        f.create_dataset(f'data/{time}/spectra', data=spectra)
        f.create_dataset(f'data/{time}/spectra_variance', data=spectra_variance)
        f.create_dataset(f'data/{time}/xspectra', data=xspectra)
        f.create_dataset(f'data/{time}/xspectra_variance', data=xspectra_variance)
    f.close()

    return None


def generate_bcode(filepath):
    """
       Uses the pseudo-random code file to generate the binary code for signal matching

    todo
        Make function able to take any code length and resample at any rate
    """
    # Array for storing the code to be analyzed
    b_code = np.zeros(20000, dtype=np.float32)

    # Read in code to be tested
    test_sig = np.fromfile(open(str(filepath)), dtype=np.complex64)

    # Sample code at 1/4 of the tx rate
    y = 0
    for x in range(80000):
        if (x + 1) % 4 == 0:
            if test_sig[x] > 0.0:
                b_code[y] = 1.0
                y += 1
            else:
                b_code[y] = -1.0
                y += 1

    return b_code


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    """
    Computes the square of input value for real and imaginary portions of complex values.

    Args:
        x (complex np.array): Complex values.

    Returns:
        return (complex np.array): Squared real and imaginary components.
    """

    return x.real ** 2 + x.imag ** 2


def func():
    """
    Python wrapper for GPU CUDA code in the libssmf.so file.

    Args:
        *

    Returns:
       func (wrapper): Python function wrapper with inputs identical to wrapped CUDA function in the ssmf.cu file.

    Noes:
       * Wrapped function is ssmf.cu is denoted by extern "C" {} tag
       * Inputs are (cufftComplex *meas1, cufftComplex *meas2, cufftComplex *code, cufftComplex *result, size_t measlen, size_t codelen, size_t size, ing avg, ing check)
    """
#    dll = C.CDLL('/mnt/ICEBEAR_datastore/processing_code/icebear/icebear/processing/libssmf.so', mode=C.RTLD_GLOBAL)
    dll = C.CDLL('/mnt/icebear/processing_code/icebear/libssmf.so', mode=C.RTLD_GLOBAL)
    func = dll.ssmf
    func.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
                     C.POINTER(C.c_float), C.c_size_t, C.c_size_t, C.c_size_t, C.c_int, C.c_int]
    return func


"""
    Links wrapped CUDA function to be calleable as python function

    Notes:
        * May be alternative ways to perform C/CUDA wrapping
"""
__fmed = func()


def ssmf(meas, code, averages, nrang, fdec, codelen):
    """
    Formats measured data and CUDA function inputs and calls wrapped function to determine single antenna spectra.

    Args:
        meas (complex64 np.array): Antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 np.array): Transmitted psuedo-random code sequence.
        averages (int): The number of 0.1 second averages to be performed on the GPU.
        nrang (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S    (complex64 np.array): 2D Spectrum output for antenna (Doppler shift x Range).

    Notes:
        * ssmf.cu could be modified to allow code to be a float32 input. This would reduce memory requirements
          on the GPU.
        * 'check' input of __fmed is 0 which indicates a single antenna is being processed.
    """
    nfreq = int(codelen / fdec)
    measlen = codelen * averages + nrang
    result_size = nfreq * nrang
    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)
    m_p = meas.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))
    __fmed(m_p, c_p, r_p, measlen, codelen, result_size, averages, 0)

    return abs2(result)


def ssmfx(meas0, meas1, code, averages, nrang, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        meas0 (complex64 np.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        meas1 (complex64 np.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 np.array): Transmitted psuedo-random code sequence.
        averages (int): The number of 0.1 second averages to be performed on the GPU.
        nrang (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).

    Notes:
        * ssmf.cu could be modified to allow code to be a float32 input. This would reduce memory requirements
          on the GPU.
        * 'check' input of __fmed is 1 which indicates a pair of antennas is being processed.
    """
    nfreq = int(codelen / fdec)
    result_size = nfreq * nrang
    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)
    variance = np.zeros((nrang, nfreq), dtype=np.complex64)
    # Create pointers to convert python tpyes to C types
    m_p0 = meas0.ctypes.data_as(C.POINTER(C.c_float))
    m_p1 = meas1.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))
    v_p = variance.ctypes.data_as(C.POINTER(C.c_float))
    # Runs ssmf.cu on data set using defined pointers
    __fmed(m_p0, m_p1, c_p, r_p, v_p, len(meas0), codelen, result_size, averages, 1)

    return result, variance


def ssmfx_cupy(v0, v1, code, navg, nrng, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        v0 (complex64 xp.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        v1 (complex64 xp.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 xp.array): Transmitted psuedo-random code sequence.
        navg (int): The number of 0.1 second averages to be performed on the GPU.
        nrng (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).
    """

    # todo: improve the integer casting
    nfreq = int(codelen / fdec)
    code = code.astype(xp.complex64)
    #v0_filtered = dsp.unmatched_filtering(v0, code, int(codelen), int(nrng), int(fdec), int(navg))
    #v1_filtered = dsp.unmatched_filtering(v1, code, int(codelen), int(nrng), int(fdec), int(navg))
    #spectra, variance = dsp.wiener_khinchin(v0_filtered, v1_filtered, navg)
    spectra, variance = dsp.wiener_khinchin(dsp.unmatched_filtering(v0, code, int(codelen), int(nrng), int(fdec), int(navg)), dsp.unmatched_filtering(v1, code, int(codelen), int(nrng), int(fdec), int(navg)), navg)

    return spectra, variance

def ssmfx_cupy_v2(v0, v1, code, navg, nrng, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        v0 (complex64 xp.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        v1 (complex64 xp.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 xp.array): Transmitted psuedo-random code sequence.
        navg (int): The number of 0.1 second averages to be performed on the GPU.
        nrng (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).
    """

    nfreq = int(codelen / fdec)
    code = code.astype(xp.complex64)

    variance_samples = xp.zeros((nrng, nfreq, navg))
    for i in range(navg):
        variance_samples[:, :, i] = dsp.wiener_khinchin_v2(dsp.unmatched_filtering_v2(v0, code, int(codelen), int(nrng), int(fdec), int(navg)), dsp.unmatched_filtering_v2(v1, code, int(codelen), int(nrng), int(fdec), int(navg)))

    spectra = xp.sum(variance_samples, axis=2)/navg
    re = xp.sqrt(xp.sum((xp.real(variance_samples) - xp.real(spectra)) * (xp.real(variance_samples) - xp.real(spectra)), axis=0) / navg)
    im = xp.sqrt(xp.sum((xp.imag(variance_samples) - xp.imag(spectra)) * (xp.imag(variance_samples) - xp.imag(spectra)), axis=0) / navg)
    variance = re + 1j*im
    return spectra, variance


def decx(config, time, data, bcode, channel1, channel2, correction1, correction2):
    """
    Performs cross-correlation and decimation for inputed baseline from the radar data

    Parameters
    ----------


    Returns
    -------


    Notes
    -----
        * ssmfx CUDA can only handle number_ranges = 2000 exactly. For farther ranges we loop at step size 2000.
        * Currently the rea_vector command is resulting in an error at the end of execution. This oes not appear to
          affect the output of the script. Issue may be in h5py or digital_rf. This error only appears when using python3
    """
    
    xp = load_cuda(config)

    if config.number_ranges <= 2000:
        start_sample = int(time * config.raw_sample_rate) - config.timestamp_corr
        step_sample = config.code_length * config.incoherent_averages + config.number_ranges
        try:
            data1 = xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1) * correction1)
            data2 = xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2) * correction2)
            if not config.cuda:
                result, variance = ssmfx_cupy(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            else:
                result, variance = ssmfx(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            return xp.transpose(result), xp.transpose(variance)
        except IOError:
            print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                  f'(start {start_sample}, step {step_sample}) and raised an IOError')
            return 1, 1

    else:
        start_sample = int(time * config.raw_sample_rate) - config.timestamp_corr
        step_sample = config.code_length * config.incoherent_averages + 2000
        try:
            data1 = dsp.calibration_correction(xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1)),
                                               xp.asarray(correction1))
            data2 = dsp.calibration_correction(xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2)),
                                               xp.asarray(correction2))
            if not config.cuda:
                result, variance = ssmfx_cupy(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            else: 
                result, variance = ssmfx(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages), 2000,
                                          xp.asarray(config.decimation_rate), xp.asarray(config.code_length))
            for i in range(2000, config.number_ranges, 2000):
                try:
                    start_sample = int(time * config.raw_sample_rate) + i - config.timestamp_corr
                    data1 = dsp.calibration_correction(
                        xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1)), xp.asarray(correction1))
                    data2 = dsp.calibration_correction(
                        xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2)), xp.asarray(correction2))
                    if not config.cuda:
                        r, v = ssmfx_cupy(data1, data2, xp.asarray(bcode), config.incoherent_averages, 2000,
                                      config.decimation_rate, config.code_length)
                    else:
                        r, v = ssmfx(data1, data2, xp.asarray(bcode), config.incoherent_averages, 2000,
                                      config.decimation_rate, config.code_length)

                    result = xp.append(result, r, axis=0)
                    variance = xp.append(variance, v, axis=0)
                except IOError:
                    print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                          f'(start {start_sample}, step {step_sample}) and raised an IOError')
                    return 1, 1
            return xp.transpose(result), xp.transpose(variance)
        except IOError:
            print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                  f'(start {start_sample}, step {step_sample}) and raised an IOError')
            return 1, 1
