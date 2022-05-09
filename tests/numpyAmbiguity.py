import matplotlib.pyplot as plt
import numpy as np
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

    modulated = np.zeros(t.size, np.complex)
    modulated[data == 1] = np.sin(2 * np.pi * mod_freq * t[data == 1])
    modulated[data == 0] = np.sin(2 * np.pi * mod_freq * t[data == 0] + np.pi)

    return modulated

def createBitSequence(t, sps, seed):
    """
    Creates a random bit sequence with 10 us between bits and a total sequence length of 100 ms

    Parameters
    ----------
    t           : array of time values
    sps         : number of samples per symbol
    seed        : seed for the rng

    Returns
    -------
    data : a random bit sequence

    """

    #  symbol_rate : number of symbols (bits) per second
    symbol_rate = 1e5  # (10 us / bit = 100 000 bits / s)

    #  seed generator
    np.random.seed(seed)

    # generate random binary sequence of appropriate length
    data_unexpanded, state = signal.max_len_seq(14, length=10000)
    # data_unexpanded = np.random.randint(2, size=int(symbol_rate*0.1))

    # expand the sequence to match the samples per symbol (ie match time resolution)
    data = np.repeat(data_unexpanded, int(sps))

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


def ambiguityFunction(u, t, tau, nu, mod_freq, data):

    chi = np.zeros((len(tau), len(nu)), np.complex)
    for i in range(len(tau)):
        for j in range(len(nu)):
            # integrate over all time. manually calculate integral
            u_term = np.multiply(u(t, mod_freq,data), np.conj(u(t-tau[i], mod_freq,data)))
            nu_term = np.exp(1j * 2 * np.pi * nu[j] * t)
            chi[i, j] = np.trapz(np.multiply(u_term, nu_term), x=t)

    result = pow(np.abs(chi), 2)

    return result


def ambiguityFunctionFFT(u, t, nu, mod_freq, data):

    chi = np.zeros((len(tau), len(nu)), np.complex)

    for i in range(len(tau)):

        # instead of manually calculating integral, recognize that it is a fourier transform
        integral = np.fft.fft(np.multiply(u(t, mod_freq, data), np.conj(u(t - tau[i], mod_freq, data))))

        # don't understand this
        integral = np.fft.fftshift(integral)

        # cut down to fit the chi array
        # todo: this might be an issue
        n = len(t)//len(tau)
        integral = integral[::n]

        # populate
        chi[i, :] = integral

    result = pow(np.abs(chi), 2)

    return result


def u(t, mod_freq, data):

    u_func = data

    u_func = bpskMod(t, data, mod_freq)

    
    # u_func = np.exp(-(pow(1000000*t, 2)))

    # single pulse
    # u_func = np.sin(2 * np.pi * (1e4)/2 * t)
    # u_func[t > (1e-4)] = 0

    # continuous pulses
    # u_func = np.cos(2 * np.pi * mod_freq * t)

    # separated continuous pulses
    # u_func = np.sin(2 * np.pi * (1e5) * t)
    # u_func[t%(4*(1e-5)) > (1e-5)] = 0

    return u_func



fig, ax = plt.subplots(2, 1)

symbol_rate = 1e5
sps = 80
tf = 0.1
tlength = 100000
t = np.linspace(0, tf, tlength, endpoint=True)

# create bit sequence
sequence = createBitSequence(t, sps, seed=1)

# repeat the sequence for all time
data = sequence
repeats = int(np.ceil(len(t) / len(sequence)))
for i in range(repeats):
    data = np.append(data, sequence)

# chop off extra values
data = data[0:len(t)]

ax[0].plot(t[0:2000], data[0:2000], label='data')

mod_freq = symbol_rate/2  # = symbol_rate * (any integer, here 1) / 2

ut = u(t, mod_freq, data)

ax[1].plot(t[0:2000], ut[0:2000], label='transmitted signal')
plt.show()

# from -1 ms to 1 ms in 10 us increments
tau = np.linspace(-0.001, 0.001, 10000, endpoint=True)
#tau = np.linspace(0, 1, 1000, endpoint=False)

# from -5kHz to 5kHz in 2 Hz increments
nu = np.linspace(-5000, 5000, 10000, endpoint=True)
#nu = np.linspace(-10, 10, 2000, endpoint=False)

#result = ambiguityFunction(u, t, tau, nu, mod_freq, data)
result = ambiguityFunctionFFT(u, t,  nu, mod_freq, data)

range = 3e5*tau

n = 1

lower_bound = 0.4
upper_bound = 0.6

nu_id = nu[round(lower_bound*len(nu)):round(upper_bound*len(nu))]
range_id = range[round(lower_bound*len(range)):round(upper_bound*len(range))]
result_id = result[round(lower_bound*len(nu)):round(upper_bound*len(nu)), round(lower_bound*len(range)):round(upper_bound*len(range))]
plt.pcolor(nu_id, range_id, result_id)
plt.show()


