import numpy as np
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp


def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.
    :param      ndarray:     The input ndarray
    :type       ndarray:     ndarray
    :param      window_len:  The window length(filter length)
    :type       window_len:  int
    :param      step:        The step(dm rate)
    :type       step:        int
    :returns:   The array with a new view.
    :rtype:     ndarray
    """

    nrows = ((ndarray.shape[-1] - window_len) // step) + 1
    last_dim_stride = ndarray.strides[-1]
    new_shape = ndarray.shape[:-1] + (nrows, window_len)
    new_strides = list(ndarray.strides + (last_dim_stride,))
    new_strides[-2] *= step

    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


def calibration_correction(samples, calibration):
    """
    Applies a complex magnitude and phase correction to all complex voltage samples.
    The complex corrections are often recorded as magnitude and phase [deg], a call of
    calibration = magnitude * np.exp(1j * np.deg2rad(phase)) may be needed.

    Parameters
    ----------
    samples : complex64 ndarray
        A time series of complex (In-phase and Quadrature) voltage samples.
        Shape (n, m) where n is the number of samples and m is number of antennas/channels.
    calibration : complex64 ndarray
        A complex calibration coefficient to be applied to each sample.
        Shape (m,) where there is one calibration term per antenna/channel.

    Returns
    -------
    calibrated_samples : complex64 ndarray
        Calibrated complex voltage samples.
        Shape (n, m) matches input samples shape.
    """
    calibrated_samples = xp.matmul(samples, calibration)
    return calibrated_samples

def unmatched_filtering(samples, code, code_length, nrng, decimation_rate, navg):
    """
    Done on the GPU. Apply the spread spectrum unmatched filter and decimation to the signal. Essentially this
    first decimates the input signal then applies a 'matched filter' like correlation using a
    special psuedo-random code which has been upsampled to match the signal window and contains
    amplitude filtering bits. This essentially demodulates the signal and removes our code.

    See Huyghebaert, (2019). The Ionospheric Continuous-wave E-region Bistatic Experimental
    Auroral Radar (ICEBEAR). https://harvest.usask.ca/handle/10388/12190

    Parameters
    ----------
    samples : complex64 ndarray
        Antenna complex magnitude and phase voltage samples.
    code : float32 ndarray
        Transmitted pseudo-random code sequence.
    code_length : int
        Length of the transmitted psuedo-random code sequence.
    nrng : int
        Number of range gates being processed. Nominally 2000.
    decimation_rate : float32
        Decimation rate (typically 200) to be used by GPU processing, effects Doppler resolution.
    navg : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.

    Returns
    -------
    filtered : complex64 ndarray
        Output decimated and unmatched filtered samples.
    """
    input_samples = xp.lib.stride_tricks.as_strided(samples,
                                                    (navg, int(code_length / decimation_rate), nrng, decimation_rate),
                                                    strides=(code_length * samples.strides[0],
                                                             decimation_rate * samples.strides[0], samples.strides[0],
                                                             samples.strides[0]))
    code_samples = windowed_view(code, window_len=decimation_rate, step=decimation_rate)
    return xp.einsum('lijk,ik->lji', input_samples, xp.conj(code_samples), optimize='greedy')


def unmatched_filtering_v2(samples, code, code_length, nrng, decimation_rate, navg):
    """
    Returns one variance sample instead of 10. Should use significantly less memory
    """
    input_samples = xp.lib.stride_tricks.as_strided(samples,
                                                    (int(code_length / decimation_rate), nrng, decimation_rate),
                                                    strides=(decimation_rate * samples.strides[0], samples.strides[0],
                                                             samples.strides[0]))
    code_samples = windowed_view(code, window_len=decimation_rate, step=decimation_rate)
    return xp.einsum('ijk,ik->ji', input_samples, xp.conj(code_samples), optimize='greedy')


def wiener_khinchin(samples1, samples2, navg):
    """
    Done on the GPU. Apply the Wiener-Khinchin theorem. Do not take the final FFT() as we want the power spectral density (PSD).

    Parameters
    ----------
    samples1 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.
    samples2 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.
    navg : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.

    Returns
    -------
    spectra : complex64 ndarray
        2D Spectrum output for antenna/channel pairs or baseline. Also known as the spectra/
        auto-correlations when samples1 = samples2 or the cross-spectra/cross-correlations when
        samples1 != samples2. These are all called Visibility (the value for a baseline at u,v,w
        sampling space coordinates) for radar imaging.
        Shape (doppler bins, range bins).
        Final spectra is divided by the number of averages provided
    variance : complex64 ndarray
        the un-averaged spectra value. To calculate the variance with the variance function, it is necessary to keep
        these values for each application of the WK function
    clutter_correction : complex64
        mean of the spectra values for the first clutter_gates range gates (averaged)
    """
    # todo: investigate the transpose issue
    # data from CUDA needs to be transposed this may not still be the case np.transpose(result), np.transpose(variance)
    variance_samples = xp.einsum('ijk,ijk->ijk', xp.fft.fft(samples1), xp.conjugate(xp.fft.fft(samples2)))
    spectra = xp.sum(variance_samples, axis=0)/navg
    re = xp.sqrt(xp.sum((xp.real(variance_samples) - xp.real(spectra)) * (xp.real(variance_samples) - xp.real(spectra)), axis=0)/navg)
    im = xp.sqrt(xp.sum((xp.imag(variance_samples) - xp.imag(spectra)) * (xp.imag(variance_samples) - xp.imag(spectra)), axis=0)/navg)
    variance = re + 1j*im
    return spectra, variance


def wiener_khinchin_v2(samples1, samples2, navg):
    """
    Done on the GPU. Apply the Wiener-Khinchin theorem. Do not take the final FFT() as we want the power spectral den>

    Parameters
    ----------
    samples1 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.
    samples2 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.
    navg : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.

    Returns
    -------
    spectra : complex64 ndarray
        2D Spectrum output for antenna/channel pairs or baseline. Also known as the spectra/
        auto-correlations when samples1 = samples2 or the cross-spectra/cross-correlations when
        samples1 != samples2. These are all called Visibility (the value for a baseline at u,v,w
        sampling space coordinates) for radar imaging.
        Shape (doppler bins, range bins).
        Final spectra is divided by the number of averages provided
    variance : complex64 ndarray
        the un-averaged spectra value. To calculate the variance with the variance function, it is necessary to keep
        these values for each application of the WK function
    clutter_correction : complex64
        mean of the spectra values for the first clutter_gates range gates (averaged)
    """
    # data from CUDA needs to be transposed this may not still be the case np.transpose(result), np.transpose(varianc>
    spectra = xp.einsum('ij,ij->ij', xp.fft.fft(samples1), xp.conjugate(xp.fft.fft(samples2)))
    return spectra


def variance(variance_samples, spectra, averages):
    """
    Calculate the variance of the sum of the non-averaged spectra results with respect to the averaged spectra results.

    Parameters
    ----------
    variance_samples : complex64 ndarray
        Shape (averages, doppler bins, range bins)
    spectra : complex64 ndarray
        The incoherently averaged results of the wiener-khinchin calculation
    averages : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.

    Returns
    -------
    variance : complex64 ndarray
        Shape (doppler bins, range bins)
        Contains the variance of each point over the incoherent averages
    """
    variance = xp.sqrt(xp.sum((variance_samples - spectra)**2, axis=0)/averages)
    return variance


def doppler_fft(indices, code_length, decimation_rate, raw_sample_rate):
    """

    Parameters
    ----------
    indices
    code_length
    decimation_rate
    raw_sample_rate

    Returns
    -------

    """
    doppler = xp.fft.fftfreq(int(code_length / decimation_rate), decimation_rate / raw_sample_rate)
    return doppler[indices]


def clutter_correction(spectra, correction):
    """
    Apply self-clutter correction. This is an optional process, typically the self-clutter correction
    term is saved into the level1 data HDF5 file to be used post processing.

    Parameters
    ----------
    spectra : complex64 ndarray
        Shape (doppler bins, range bins, antennas)
    correction : complex64 ndarray
        Shape(antennas,)

    Returns
    -------
    corrected_spectra : complex64 ndarray
        Shape (doppler bins, range bins, antennas)
    """

    corrected_spectra = spectra - correction
    return corrected_spectra


def snr(power, method='mean'):
    """
    Done on the GPU. Calculates the noise floor and snr for a given power spectrum.
    Parameters
    ----------
    power : complex64 ndarray
        Shape (range bins, doppler bins)
    method : str
        There can be several methods for determining the noise value.
        - 'mean' (default) determine noise floor as the mean of power
        - 'median' determine noise floor as the median of power
        - 'galeschuk' determine noise floor as the mean of farthest 100 ranges

    Returns
    -------
    snr : complex64 ndarray
        Shape (range bins, doppler bins). The SNR given by (Power - Noise) / Noise

    noise : float32
        the noise floor as determined by the method given
    """
    if method == 'mean':
        noise = xp.mean(power)
    elif method == 'median':
        noise = xp.median(power)
    elif method == 'galeschuk':
        noise = xp.mean(power[:, -100:-1])
    else:
        raise ValueError('argument \'method\' must be one of: mean, median, galeschuk')

    snr = (power - noise) / noise
    return snr, noise


def generate_bcode(filepath):
    """
       Uses the pseudo-random code file to generate the binary code for signal matching

    todo
        Make function able to take any code length and resample at any rate
    """
    # Array for storing the code to be analyzed
    b_code = np.zeros(20000, dtype=np.float32)

    # Read in code to be tested
    test_sig = np.fromfile(open(filepath), dtype=np.complex64)

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

