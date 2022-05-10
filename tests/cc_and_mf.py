import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy as xp
    CUDA = True
except ModuleNotFoundError:
    import numpy as xp
    CUDA = False
from scipy import signal


def bpskMod(t, data, mod_freq):
    """
    Modulates (BPSK) a carrier frequency with a bit sequence

    Parameters
    ----------
    t    : array of time values
    data : array of bits to be modulated (same length as t)
    freq : frequency of carrier wave

    Returns
    -------
    modulated : modulated signal containing bits in 'data'

    """

    modulated = xp.zeros(t.size, np.complex)
    modulated[data == 1] = xp.sin(2 * xp.pi * mod_freq * t[data == 1])
    modulated[data == -1] = xp.sin(2 * xp.pi * mod_freq * t[data == -1] + xp.pi)

    return modulated

def createBitSequence():
    """
    Creates a random NRZ bit sequence with 10000 bits and one samples per bit

    Parameters
    ----------

    Returns
    -------
    data : a random bit sequence of 10000 bits sampled at 1 sample per bit (NRZ)

    """

    # generate random binary sequence of appropriate length (10000 bits)
    data, state = signal.max_len_seq(14, length=10000)
    data[data == 0] = -1
    return data


def crossCorrelation(f, g):
    """

    Parameters
    ----------
    f : signal to cross correlate with g
    g : signal to cross correlate with f

    Returns
    -------
    result : cross correlation of f and g in the time domain. g lags f by the largest spike (tau) in result
    """

    # zero padding
    N = len(f) + len(g) - 1
    xp.pad(f, (0, N - len(f)), 'constant')
    xp.pad(g, (0, N - len(g)), 'constant')

    # take ffts
    F = xp.fft.fft(f)
    G = xp.fft.fft(g)

    # elementwise multiplication
    xr = xp.multiply(xp.real(F), xp.real(G))
    xi = xp.multiply(xp.imag(F), xp.imag(G))
    yr = xp.multiply(xp.real(F), xp.imag(G))
    yi = xp.multiply(xp.imag(F), xp.real(G))

    # add em up
    fft_result = (xr + yr) + 1j * (yi - xi)
    result = xp.fft.ifft(fft_result)
    result = result / max(result)

    return result

def matchedFilter(received_sig, ideal_sig):

    return crossCorrelation(received_sig, ideal_sig)



