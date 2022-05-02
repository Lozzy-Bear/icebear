import numpy as np
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp


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


def unmatched_filtering(samples, code, code_length, averages, ranges, decimation_rate):
    """
    Apply the spread spectrum unmatched filter and decimation to the signal. Essentially this
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
    averages : int
        The number of chip sequence length (typically 0.1 s) incoherent averages to be performed.
    ranges : int
        Number of range gates being processed. Nominally 2000.
    decimation_rate : float32
        Decimation rate (typically 200) to be used by GPU processing, effects Doppler resolution.

    Returns
    -------
    filtered : complex64 ndarray
        Output decimated and unmatched filtered samples.
    """
    filtered = xp.array()  # Todo: Convert Dravens code to cupy
    return filtered


def wiener_khinchin(samples1, samples2):
    """
    Apply the Wiener-Khinchin theorem. Do not take the finally FFT() as we want the power spectral density (PSD).

    Parameters
    ----------
    samples1 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.
    samples2 : complex64 ndarray
        Filtered and decimated complex magnitude and phase voltage samples.

    Returns
    -------
    spectra : complex64 ndarray
        2D Spectrum output for antenna/channel pairs or baseline. Also known as the spectra/
        auto-correlations when samples1 = samples2 or the cross-spectra/cross-correlations when
        samples1 != samples2. These are all called Visibility (the value for a baseline at u,v,w
        sampling space coordinates) for radar imaging.
        Shape (doppler bins, range bins).
    variance : complex64 ndarray
        Shape (doppler bins, range bins)
    """
    spectra = xp.multiply(xp.fft.fft(samples1), xp.conjugate(xp.fft.fft(samples2)))
    variance = xp.zeroes_like(result)  # Todo: Figure out how and why Draven gets variance
    # data from CUDA needs to be transposed this may not still be the case np.transpose(result), np.transpose(variance)
    return spectra, variance, spectra_median, clutter_correction


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
    # Todo: do this
    return


def snr(power, method='mean'):
    """

    Parameters
    ----------
    power : complex64 ndarray
        Shape (range bins, doppler bins)
    method : str
        There can be serveral methods for determing the noise value.
        - 'mean' (default) determine noise floor as the mean of power
        - 'median' determine noise floor as the median of power
        - 'galeschuk' determine noise floor as the mean of farthest 100 ranges

    Returns
    -------
    snr : float32 ndarray
    noise :
    """
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
