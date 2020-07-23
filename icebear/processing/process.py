import h5py
import scipy
import calendar
import pyfftw
import numba
import numpy as np
import ctypes as C
import digital_rf
pyfftw.interfaces.cache.enable()


def generate_level1(config):
    time = np.array(config.processing_start)
    bcode = generate_bcode(config.prn_code_file)
    level0_data = digital_rf.DigitalRFReader(config.processing_source)
    channels = level0_data.get_channels()
    if len(channels) == 0:
        print(f'ERROR: No data channels found in {config.processing_source}')
        exit()
    else:
        print('\tchannels acquired:')
        for i in range(len(channels)):
            print(f'\t\t-{str(channels[i])}')
    complex_correction = config.rx_magnitude * np.exp(1j * np.deg2rad(config.rx_phase))
    fft_freq = np.fft.fftfreq(int(config.code_length / config.decimation_rate),
                              config.decimation_rate / config.raw_sample_rate)

    start = time[0]
    filename = f'level1/{time[0]:04d}_{time[1]:02d}_{time[2]:02d}/' \
               f'{config.radar_name}_{config.processing_method}_{config.tx_name}_{config.rx_name}_' \
               f'{config.snr_cutoff:02d}dB_{config.averages:02d}00ms_' \
               f'{time[0]:04d}_{time[1]:02d}_{time[2]:02d}_{time[3]:02d}.h5'

    create_level1_hdf5(config, filename)

    # do work here
    append_level1_hdf5(filename, hour, minute, second, data_flag, doppler, rf_distance, logsnr, noise,
                       spectra, spectra_variance, spectra_median, spectra_clutter_corr,
                       xspectra, xspectra_variance, xspectra_median, xspectra_clutter_corr)
    return None


def create_level1_hdf5(config, filename):
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
    f.create_dataset('date', data=np.array(config.date))
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


class Xdata:
    """
      Class defined to hold cross-spectra data
    """
    def __init__(self, data, b_num, dist):
        """
        Parameters
        ----------
            data
            b_num
            dist
        """
        self.data = data
        self.b_num = b_num
        self.dist = dist


def generate_bcode(filepath):
    """
       Uses the pseudo-random code file to generate the binary code for signal matching
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
    dll = C.CDLL('./libssmf.so', mode=C.RTLD_GLOBAL)
    func = dll.ssmf
    func.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_size_t,
                     C.c_size_t, C.c_size_t, C.c_int, C.c_int]
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

    S = abs2(result)
    return S


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

    measlen0 = len(meas0)
    measlen1 = len(meas1)
    nfreq = int(codelen / fdec)
    result_size = nfreq * nrang

    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)
    var = np.zeros((nrang, nt), dtype=np.complex64)
    # Create pointers to convert python tpyes to C types
    m_p0 = meas0.ctypes.data_as(C.POINTER(C.c_float))
    m_p1 = meas1.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))
    v_p = var.ctypes.data_as(C.POINTER(C.c_float))
    # Runs ssmf.cu on data set using defined pointers
    __fmed(m_p0, m_p1, c_p, r_p, v_p, measlen0, codelen, result_size, averages, 1)

    S = result
    return S, var


def decx(b_code, data, codelen, complex_corr, averages, fdec, nrang, second, minute, hour, day, month, year, plot_num,
         plot_spacing, sample_rate, i, j):
    """
      Performs cross-correlation and decimation for inputed baseline from the radar data

        Currently the rea_vector command is resulting in an error at the end of execution. This
        does not appear to affect the output of the script. Issue may be in h5py or digital_rf.
        Note: This error only appears when using python3
    """
    # Initialize Array
    val = int(20000 / fdec)
    xspec = Xdata(np.zeros((val, nrang), dtype=np.complex64), i, j)

    # Script executes a portion of time averaging on GPU and CPU
    # if range for avg_num is 1, all averaging for variable averages is done on the GPU
    # if range for avg_num is averages, all averaging is done on the CPU
    # divide the averaging appropriately for the task
    seconds = second + (plot_num * (averages * 0.1 + plot_spacing))

    # Calculate the hour and minute from seconds
    minutes = minute + int(seconds / 60.0)
    seconds = seconds % 60.0
    hours = hour + int(minutes / 60.0)
    minutes = minutes % 60.0

    time_tuple = (year, month, day, hours, minutes, seconds, -1, -1, 0)

    # Calculate the start sample
    start_sample = int((calendar.timegm(time_tuple)) * sample_rate) - 30

    # Generate antenna strings
    antenna1 = 'antenna' + str(i)
    antenna2 = 'antenna' + str(j)

    try:
        antenna_data1 = (data.read_vector_c81d(start_sample, codelen * averages + nrang, antenna1)) * complex_corr[i]
        antenna_data2 = (data.read_vector_c81d(start_sample, codelen * averages + nrang, antenna2)) * complex_corr[j]
    except IOError:
        print('Read number %i or %i went beyond existing data and raised an IOError' % (i, j))

    # n, bins, patches = plt.hist(np.real(antenna_data1),log=True)
    # n, bins, patches = plt.hist(np.imag(antenna_data1),log=True)
    # plt.show()

    # Perform cross correlation
    S, var = ssmfx(antenna_data1, antenna_data2, b_code, averages, nrang, fdec, codelen)
    xspec.data = np.transpose(S)
    var = np.transpose(var)

    seconds = second + (plot_num * (averages * 0.1 + plot_spacing)) + (averages * 0.1)

    # Calculate the hour and minute from the seconds for the final time
    minutes = minute + int(seconds / 60.0)
    seconds = seconds % 60.0
    hours = hour + int(minutes / 60.0)
    minutes = minutes % 60.0

    # For quick wave study
    # implement total abs(power) for array for each average
    # take fft after and plot power vs freq

    return xspec, var, hours, minutes, seconds
