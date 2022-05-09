import matplotlib.pyplot as plt
import numpy as np

def createBitSequence(t, symbol_rate, sps):
    """
    Creates a random bit sequence of the same length as array t with number of bits determined by symbol_rate

    Parameters
    ----------
    t           : array of time values
    symbol_rate : number of symbols (bits) per second
    sps         : number of samples per symbol

    Returns
    -------
    data : a random bit sequence

    """

    data_unexpanded = np.random.randint(2, size=(int(symbol_rate * (tf - t0))))  # generates binary array, not same size as t array yet
    data = np.repeat(data_unexpanded, int(sps))  # expands the bit sequence to match samples per symbol

    return data

def bpskMod(t, data, freq, amp):
    """
    Modulates (BPSK) a carrier frequency with a bit sequence

    Parameters
    ----------
    t    : array of time values
    data : array of bits to be modulated (same length as t)
    freq : frequency of carrier wave
    amp  : amplitude of carrier wave

    Returns
    -------
    modulated : modulated signal containing bits in 'data'

    """

    modulated = np.zeros(t.size)
    modulated[data == 1] = amp * np.sin(2 * np.pi * freq * t[data == 1])
    modulated[data == 0] = amp * np.sin(2 * np.pi * freq * t[data == 0] + np.pi)

    return modulated

def bpskDeMod1(t, freq, sps):
    """
    Demodulates a BPSK signal

    Parameters
    ----------
    t    : array of time values
    freq : frequency of carrier wave
    sps  : number of samples per symbol

    Returns
    -------
    noisy_multiplied : signal after multiplying with the reference frequency
    demodulated      : demodulated signal. sequence of bits

    """

    # multiply received signal by reference frequency signal (normally comes from PLL)
    noisy_multiplied = noisy_modulated * np.sin(2 * np.pi * freq * t)

    demodulated = np.zeros(t.size)

    # integrate over symbol period
    for i in range(0, t.size - 1, int(sps)):
        demodulated[i:i + int(sps)] = np.trapz(noisy_multiplied[i:i + int(sps)], x=t[i:i + int(sps)])

    # threshold selector
    demodulated[demodulated > 0] = 1
    demodulated[demodulated < 0] = 0

    return noisy_multiplied, demodulated



def bpskDeMod2(t, freq, sps, samp_rate):
    """
    Demodulates a BPSK signal

    Parameters
    ----------
    t    : array of time values
    freq : frequency of carrier wave
    sps  : number of samples per symbol
    # integrate over symbol period
    for i in range(0, t.size - 1, int(sps)):
        demodulated[i:i + int(sps)] = np.trapz(noisy_multiplied[i:i + int(sps)], x=t[i:i + int(sps)])

    # threshold selector
    demodulated[demodulated > 0] = 1
    demodulated[demodulated < 0] = 0

    Returns
    -------
    noisy_multiplied : signal after multiplying with the reference frequency
    demodulated      : demodulated signal. sequence of bits

    """

    # multiply received signal by reference frequency signal (normally comes from PLL)
    noisy_multiplied = noisy_modulated * np.sin(2 * np.pi * freq * t)

    demodulated = np.zeros(t.size)

    y = lpf(noisy_multiplied, freq, samp_rate)

    # integrate over symbol period
    for i in range(0, t.size - 1, int(sps)):
        demodulated[i:i + int(sps)] = np.trapz(y[i:i + int(sps)], x=t[i:i + int(sps)])

    # threshold selector
    demodulated[demodulated > 0] = 1
    demodulated[demodulated < 0] = 0

    return noisy_multiplied, demodulated


def lpf(f, cutoff_freq, samp_rate):

    N = len(f)

    H = np.zeros(N)

    # create filter in frequency domain
    for i in range (N):
        H[i] = 0
        if (i <= cutoff_freq * N / samp_rate) or (i >= (N - cutoff_freq) * N / samp_rate):
            H[i] = 1

    F = np.fft.fft(f)
    Y = H * F

    y = np.fft.ifft(Y)

    return y


# Numpy
fig, ax = plt.subplots(6, 1)

noise_mag = 2

# Sampling
samp_rate = 1000  # samples per second
symbol_rate = 10  # symbols (bits) per second
sps = samp_rate / symbol_rate  # samples per symbol
t0 = 0.0  # start time
tf = 10.0  # end time
samples = np.arange(samp_rate * tf)  # integer linspace (indices)
t = np.linspace(t0, tf, samples.size)  # timespace (seconds)

# Carrier wave
freq = 5.0  # carrier frequency (hertz)
amp = 1.0  # carrier amplitude
carrier = amp * np.sin(2 * np.pi * freq * t)  # carrier wave

ax[0].plot(t, carrier)
ax[0].set_title('Carrier')

# Bit Sequence
data = createBitSequence(t, symbol_rate, sps)

ax[1].plot(t, data)
ax[1].set_title('Data')

# BPSK mod
modulated = bpskMod(t, data, freq, amp)

ax[2].plot(t, modulated)
ax[2].set_title('Modulated')

# add noise
noise = np.random.normal(0, noise_mag*amp, t.size)
noisy_modulated = modulated + noise

ax[3].plot(t, noisy_modulated)
ax[3].set_title('Added Noise')

# BPSK demod (assume we recover the carrier phase - normally done using PLL or Costas loop)

noisy_multiplied, demodulated = bpskDeMod2(t, freq, sps, samp_rate)

y = lpf(noisy_multiplied, freq, samp_rate)

ax[4].plot(t, y)
ax[4].set_title('LP Filtered')


ax[5].plot(t, demodulated)
ax[5].set_title('Demodulated Data')


plt.show()






