import numpy as np
import matplotlib.pyplot as plt
import cupy as cp


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


def shift_2d_replace(data, dx, dy=0, constant=0.0):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def ambiguity2d(bcode):
    n = len(bcode)
    b = cp.zeros(300 + n)
    b[150:-150] = cp.copy(bcode)
    taus = cp.arange(0, 301, 1)
    m = cp.zeros((len(taus), n))
    for idx, tau in enumerate(taus):
        m[idx, :] = cp.fft.fftshift(cp.fft.fft(
                                    bcode * cp.conj(b[tau:tau+n])
                                    ))
    m = cp.abs(m) ** 2
    m /= cp.max(m)
    m = 10 * cp.log10(m)
    m = cp.asnumpy(m)
    return m


def ambiguity(waveform, bcode):
    y = cp.convolve(bcode, waveform[::-1], mode='same')
    # y = cp.fft.fftshift(cp.fft.fft(y))
    y = cp.abs(y/cp.max(y))
    y = cp.asnumpy(y)
    t = np.arange(len(y))
    return t, y


def ambiguity_plot(t, y):
    plt.figure()
    plt.plot(t, y)
    plt.title('Pseudo-Random Noise Auto-correlation Response')
    plt.xlabel('Normalized Delay Time t/τ')
    plt.ylabel('Normalized Magnitude')
    plt.show()
    return


def ambiguity_plot2d(y):
    plt.figure()
    plt.imshow(y, vmin=0, vmax=-36)
    plt.title('Pseudo-Random Noise Auto-correlation Response')
    plt.xlabel('Doppler [Hz]')
    plt.ylabel('Range [km]')

    plt.xlim((9500, 10500))
    plt.colorbar(label='Power [dB]')
    plt.show()
    return


if __name__ == '__main__':
    # Pretty plot configuration.
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    bcode = generate_bcode('C:/Users/TKOCl/PythonProjects/icebear/dat/pseudo_random_code_test_8_lpf.txt')
    waveform = cp.asarray(bcode)
    freq = 49.5e6
    # t, y = ambiguity(waveform, waveform)
    # ambiguity_plot(t, y)

    y = ambiguity2d(waveform)
    ambiguity_plot2d(y)

