import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def ambiguityFunctionFFT(data, half_tau_length):

    n = len(data)

    # add half_tau_length zeros around data
    padded_data = np.zeros(2*half_tau_length + n)
    padded_data[half_tau_length:half_tau_length + n] = data

    # assuming tau is same spacing as t?
    tau = np.arange(2*half_tau_length)
    chi = np.zeros((2*half_tau_length, n), np.float32)

    for i in range(len(tau)):

        # instead of manually calculating integral, recognize that it is a fourier transform

        # following explanation assumes half_tau_length = 500
        # since chi[0, :] corresponds to the minimum value for tau ( -500 ),
        # index padded_data[0:len(data)] = full tau shift (500 off of data)
        # padded_data[0:len(data)] = 0, 0, ..., 0, data[0:len(data) - 500]
        #                            |-----------|
        #                           half_tau_length = 500
        #
        # chi[len(tau)/2 = 500, :] corresponds to tau = 0
        # index padded_data[500:500 + len(data)]
        # padded_data[500:500 + len(data)] = data[:]
        #
        # chi[len(tau) = 2*500, :] corresponds to tau = 500 (maximum value)
        # index padded_data[2*500:len(data) + 2*500]
        # padded_data[2*500:len(data) + 2*500] = data[500:len(data) - 500], 0, 0, ..., 0
        #                                                                   |-----------|
        #                                                               half_tau_length = 500

        integral = np.fft.fft(data * np.conj(padded_data[i:i+n]))

        # shift to centre
        integral = np.fft.fftshift(integral)

        # populate chi
        chi[i, :] = integral

    result = pow(np.abs(chi), 2)

    # grab frequency values
    nu = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / 100_000.0))  # code length [chips] : 10 000 chips per code, chip length [s] : 100 000 chips per sec

    return result, nu


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

def ambiguity_plot2d(y, f, nrang):
    plt.figure()
    plt.imshow(y[:, 4800:5200], vmax=np.amax(y), vmin=np.amax(y)-36, origin='lower',
               aspect='auto', interpolation='none', cmap='viridis',
               extent=[f[4800] - 5.0, f[5200] - 5.0, (nrang * -1.5) - 0.75, (nrang * 1.5) - 0.75])
    plt.title('Pseudo-Random Noise Auto-correlation Response')
    plt.xlabel('Doppler [Hz]')
    plt.ylabel('Range [km]')
    plt.colorbar(label='Power [dB]')
    plt.xlim((-1500, 1500))
    plt.ylim((-150, 150))
    plt.show()
    return

# create bit sequence (NRZ)
sequence = createBitSequence()

# half_tau_length time steps after t, half_tau_length time steps before t
half_tau_length = 500
result, nu = ambiguityFunctionFFT(sequence, half_tau_length)

ambiguity_plot2d(result, nu, half_tau_length)




