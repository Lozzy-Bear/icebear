import h5py
import scipy
import pyfftw
import numba
import numpy as np
import ctypes as C
import digital_rf
import icebear.utils
pyfftw.interfaces.cache.enable()


def generate_level1(config):
    filenames = []
    level0_data = digital_rf.DigitalRFReader(config.processing_source)
    channels = level0_data.get_channels()
    if len(channels) == 0:
        print(f'ERROR: No data channels found in {config.processing_source}')
        exit()
    else:
        print('\tchannels acquired:')
        for i in range(len(channels)):
            print(f'\t\t-{str(channels[i])}')

    total_xspectras = int(len(channels)*(len(channels) - 1) / 2)
    total_spectras = int(len(channels))
    bcode = generate_bcode(config.prn_code_file)
    complex_correction = config.rx_magnitude * np.exp(1j * np.deg2rad(config.rx_phase))
    fft_freq = np.fft.fftfreq(int(config.code_length / config.decimation_rate),
                              config.decimation_rate / config.raw_sample_rate)

    if not config.processing_step:
        config.processing_step = [0, 0, 0, config.incoherent_averages * config.time_resolution, 0]
    time = icebear.utils.Time(config.processing_start, config.processing_stop, config.processing_step)
    if config.incoherent_averages * config.time_resolution > time.step_epoch:
        print(f'WARNING: averaging time length {config.incoherent_averages * config.time_resolution}s'
              f' is greater than step time length {time.step_epoch}s')
    temp_hour = [-1, -1, -1, -1]
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        now = time.get_date(t)
        spectra = np.empty(shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras), dtype=np.complex128)
        spectra_variance = np.empty(shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_spectras), dtype=np.complex128)
        xspectra = np.empty(shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras), dtype=np.complex128)
        xspectra_variance = np.empty(shape=(int(config.code_length / config.decimation_rate), config.number_ranges, total_xspectras), dtype=np.complex128)
        power = np.zeros(shape=(int(config.code_length / config.decimation_rate), config.number_ranges), dtype=np.complex128)

        # create new file if new hour
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            filename = f'{config.processing_destination}{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}/' \
                f'{config.radar_name}_{config.processing_method}_{config.tx_name}_{config.rx_name}_' \
                f'{int(config.snr_cutoff):02d}dB_{int(config.incoherent_averages):02d}00ms_' \
                f'{int(now.year):04d}_{int(now.month):02d}_{int(now.day):02d}_{int(now.hour):02d}.h5'
            print(f'\t-created level 1 HDf5: {filename}')
            filenames.append(filename)
            create_level1_hdf5(config, filename, int(now.year), int(now.month), int(now.day))
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        # calculate the self-spectra
        for antenna_num in range(total_spectras):
            result, variance = decx(config, t, level0_data, bcode, channels[antenna_num], channels[antenna_num],
                                    complex_correction[antenna_num], complex_correction[antenna_num])
            spectra[:, :, antenna_num] = result
            spectra_variance[:, :, antenna_num] = variance
            power[:, :] += result[:, :]

        # Perform cross-spectra for each desired baseline or correlation
        cnt = 0
        for j in range(len(channels)-1):
            for i in range(j + 1, len(channels)):
                result, variance = decx(config, t, level0_data, bcode, channels[j], channels[i],
                                        complex_correction[j], complex_correction[i])
                xspectra[:, :, cnt] = result
                xspectra_variance[:, :, cnt] = variance
                cnt += 1
                
        noise = np.median(power)
        snr = (power - noise) / noise
        snr = np.ma.masked_where(snr < 0.0, snr)
        logsnr = 10 * np.log10(snr.filled(1))
        logsnr = np.ma.masked_where(logsnr < 1.0, logsnr)
        # Find the range-Doppler bins with a SNR above the threshold.
        snr_indices = np.asarray(np.where(logsnr >= config.snr_cutoff)).T
        if len(snr_indices) > 0:
            data_flag = True
        else:
            data_flag = False

        # Calculate the spectra noise value.
        spectra_median = np.zeros(total_spectras, dtype=np.complex64)
        spectra_clutter_corr = np.zeros(total_spectras, dtype=np.complex64)
        for num_spec in range(total_spectras):
            spectra_median[num_spec] = np.median(spectra[:, :, num_spec])
            spectra_clutter_corr[num_spec] = np.mean(spectra[:, 0:config.clutter_gates, num_spec])

        # calculate the xspectra 'noise' value
        xspectra_median = np.zeros(total_xspectras, dtype=np.complex64)
        xspectra_clutter_corr = np.zeros(total_xspectras, dtype=np.complex64)
        for num_xspec in range(total_xspectras):
            xspectra_median[num_xspec] = np.median(xspectra[:, :, num_xspec])
            xspectra_clutter_corr[num_xspec] = np.mean(xspectra[:, 0:config.clutter_gates, num_xspec])

        # Calculate noise, range and Doppler values
        doppler = fft_freq[snr_indices[:, 0]]
        rf_distance = config.range_resolution * (snr_indices[:, 1] - config.timestamp_correction)
        noise /= total_spectras

        append_level1_hdf5(filename, int(now.hour), int(now.minute), int(now.second * 1000),
                           data_flag, doppler, rf_distance, logsnr, noise,
                           spectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           spectra_median, spectra_clutter_corr,
                           xspectra[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_variance[snr_indices[:, 0], snr_indices[:, 1], :],
                           xspectra_median, xspectra_clutter_corr)

    return filenames


def create_level1_hdf5(config, filename, year, month, day):
    """
    Create a level1 HDF5 formatted file for storage of ICEBEAR spectra and cross-spectra

    Parameters
    ----------
        config : Class Object
            Config class instantiation.
        filename:

    Returns
    -------
        None

    """
    # general information
    f = h5py.File(filename, 'w')
    f.create_dataset('config_updated', data=np.array(config.config_updated))
    f.create_dataset('date', data=np.array([year, month, day]))
    f.create_dataset('processing_method', data=config.processing_method)
    # transmitter site information
    f.create_dataset('tx_name', data=config.tx_name)
    f.create_dataset('tx_coordinates', data=np.array(config.tx_coordinates))
    f.create_dataset('tx_updated', data=config.tx_updated)
    f.create_dataset('tx_pointing', data=config.tx_pointing)
    f.create_dataset('tx_x', data=np.array(config.rx_x))
    f.create_dataset('tx_y', data=np.array(config.rx_y))
    f.create_dataset('tx_z', data=np.array(config.rx_z))
    f.create_dataset('tx_mask', data=np.array(config.tx_mask))
    f.create_dataset('tx_phase', data=np.array(config.tx_phase))
    f.create_dataset('tx_magnitude', data=np.array(config.tx_magnitude))
    f.create_dataset('tx_sample_rate', data=config.tx_sample_rate)
    f.create_dataset('tx_antenna_type', data=config.tx_antenna_type)
    f.create_dataset('tx_rf_path', data=config.tx_rf_path)
    # receiver site information
    f.create_dataset('rx_name', data=config.rx_name)
    f.create_dataset('rx_coordinates', data=np.array(config.rx_coordinates))
    f.create_dataset('rx_updated', data=config.rx_updated)
    f.create_dataset('rx_pointing', data=config.rx_pointing)
    f.create_dataset('rx_x', data=np.array(config.rx_x))
    f.create_dataset('rx_y', data=np.array(config.rx_y))
    f.create_dataset('rx_z', data=np.array(config.rx_z))
    f.create_dataset('rx_mask', data=np.array(config.rx_mask))
    f.create_dataset('rx_phase', data=np.array(config.rx_phase))
    f.create_dataset('rx_magnitude', data=np.array(config.rx_magnitude))
    f.create_dataset('rx_sample_rate', data=config.rx_sample_rate)
    f.create_dataset('rx_antenna_type', data=config.rx_antenna_type)
    f.create_dataset('rx_rf_path', data=config.rx_rf_path)
    # processing settings
    f.create_dataset('wavelength', data=config.wavelength)
    f.create_dataset('center_frequency', data=config.center_frequency)
    f.create_dataset('prn_code_file', data=config.prn_code_file)
    f.create_dataset('raw_sample_rate', data=config.raw_sample_rate)
    f.create_dataset('decimation_rate', data=config.decimation_rate)
    f.create_dataset('incoherent_averages', data=config.incoherent_averages)
    f.create_dataset('time_resolution', data=config.time_resolution)
    f.create_dataset('snr_cutoff', data=config.snr_cutoff)
    f.create_dataset('spectra_descriptors', data=np.array(config.spectra_descriptors, dtype='S'))
    f.create_dataset('xspectra_descriptors', data=np.array(config.xspectra_descriptors, dtype='S'))
    f.create_group('data')
    f.close()
    return None


def append_level1_hdf5(filename, hour, minute, second, data_flag, doppler, rf_distance, logsnr, noise,
                       spectra, spectra_variance, spectra_median, spectra_clutter_corr,
                       xspectra, xspectra_variance, xspectra_median, xspectra_clutter_corr):
    """
    Append spectra and cross-spectra ICEBEAR data to previously created HDF5 file

    Parameters
    ----------
        filename
        hour
        minute
        second
        data_flag
        doppler
        rf_distance
        logsnr
        noise
        spectra
        spectra_variance
        spectra_median
        spectra_clutter_corr
        xspectra
        xspectra_variance
        xspectra_median
        xspectra_clutter_corr

    Returns
    -------

    """
    # append a new group for the current measurement
    time = f'{hour:02d}{minute:02d}{second:05d}'
    f = h5py.File(filename, 'a')
    f.create_group(f'data/{time}')
    f.create_dataset(f'data/{time}/time', data=np.array([hour, minute, second]))
    # create the noise data for the averaged spectra at a given time
    f.create_dataset(f'data/{time}/avg_spectra_noise', data=noise)
    f.create_dataset(f'data/{time}/spectra_median', data=spectra_median)
    f.create_dataset(f'data/{time}/xspectra_median', data=xspectra_median)
    f.create_dataset(f'data/{time}/spectra_clutter_correction', data=spectra_clutter_corr)
    f.create_dataset(f'data/{time}/xspectra_clutter_correction', data=xspectra_clutter_corr)
    f.create_dataset(f'data/{time}/data_flag', data=data_flag)
    # only write data if there are measurements above the SNR threshold
    if data_flag:
        f.create_dataset(f'data/{time}/doppler_shift', data=doppler)
        f.create_dataset(f'data/{time}/rf_distance', data=rf_distance)
        f.create_dataset(f'data/{time}/snr_db', data=logsnr)
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
    test_sig = scipy.fromfile(open(filepath), dtype=scipy.complex64)

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
    #dll = C.CDLL('./libssmf.so', mode=C.RTLD_GLOBAL)
    dll = C.CDLL('./icebear/processing/libssmf.so', mode=C.RTLD_GLOBAL)
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


def ssmfx(meas0, meas1, code, averages, nrang, fdec, codelen):
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


def decx(config, time, data, bcode, channel1, channel2, correction1, correction2):
    """
      Performs cross-correlation and decimation for inputed baseline from the radar data

        Currently the rea_vector command is resulting in an error at the end of execution. This
        does not appear to affect the output of the script. Issue may be in h5py or digital_rf.
        Note: This error only appears when using python3
    """
    start_sample = int(time * config.raw_sample_rate) - config.timestamp_correction
    step_sample = config.code_length * config.incoherent_averages + config.number_ranges
    try:
        data1 = data.read_vector_c81d(start_sample, step_sample, channel1) * correction1
        data2 = data.read_vector_c81d(start_sample, step_sample, channel2) * correction2
        result, variance = ssmfx(data1, data2, bcode, config.incoherent_averages, config.number_ranges,
                                 config.decimation_rate, config.code_length)
        return np.transpose(result), np.transpose(variance)
    except IOError:
        print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
              f'(start {start_sample}, step {step_sample}) and raised an IOError')
        exit()

