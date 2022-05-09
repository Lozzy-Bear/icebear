import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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
    np.pad(f, (0, N - len(f)), 'constant')
    np.pad(g, (0, N - len(g)), 'constant')

    # take ffts
    F = np.fft.fft(f)
    G = np.fft.fft(g)

    # elementwise multiplication
    xr = np.multiply(np.real(F), np.real(G))
    xi = np.multiply(np.imag(F), np.imag(G))
    yr = np.multiply(np.real(F), np.imag(G))
    yi = np.multiply(np.imag(F), np.real(G))

    # add em up

    fft_result = (xr + yr) + 1j * (yi - xi)

    result = np.fft.ifft(fft_result)

    result = result / max(result)

    return result

def matchedFilter(received_sig, ideal_sig):

    return crossCorrelation(received_sig, ideal_sig)

def nextpow2(i):
    '''
    Find the next power 2 number for FFT
    '''

    n = 1
    while n < i: n *= 2
    return n

def shift_signal_in_frequency_domain(datin, shift):
    '''
    This is function to shift a signal in frequency domain.
    The idea is in the frequency domain,
    we just multiply the signal with the phase shift.
    '''
    Nin = len(datin)

    # get the next power 2 number for fft
    N = nextpow2(Nin +np.max(np.abs(shift)))

    # do the fft
    fdatin = np.fft.fft(datin, N)

    # get the phase shift, shift here is D in the above explanation
    ik = np.array([2j*np.pi*k for k in range(0, N)]) / N
    fshift = np.exp(-ik*shift)

    # multiple the signal with shift and transform it back to time domain
    datout = np.fft.ifft(fshift * fdatin)

    # only get the data have the same length as the input signal
    datout = datout[0:Nin]

    return datout


t1 = np.linspace(-1, 1, 1000, endpoint=False)
fc = 5
i1, q1, e1 = signal.gausspulse(t1, fc=fc, retquad=True, retenv=True)

sig1 = i1 + 1j*q1
noise = np.random.normal(0, 0.5, t1.size)
sig1 = sig1 + noise

# This is the amount we will move
nShift = 150

# generate the 2nd signal time shifted with noise
sig2 = shift_signal_in_frequency_domain(i1, nShift)
noise = np.random.normal(0, 1, t1.size)
sig2 = sig2 + noise


filtered = matchedFilter(sig2, sig1)

print(np.argmax(filtered))
# plot two signals together
plt.plot(sig2, label='received signal')
plt.plot(sig1, 'r', label='transmitted signal')
plt.plot(filtered, 'g', label='matched filtered')
#plt.plot(result1, label='cross-correlation of signal 1 with signal 2')
#plt.plot(result2, label='cross-correlation of signal 2 with signal 1 ')
plt.legend()
plt.show()

