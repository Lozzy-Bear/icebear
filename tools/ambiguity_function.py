import numpy as np
import matplotlib.pyplot as plt


def generate_bcode_10k(filepath):
    # Array for storing the code to be analyzed
    b_code = np.zeros(10000, dtype=np.float32)
    # Read in code to be tested
    test_sig = np.fromfile(open(filepath), dtype=np.complex64)
    # Sample code at 1/4 of the tx rate
    y = 0
    for x in range(80000):
        if (x + 1) % 8 == 0:
            if test_sig[x] > 0.0:
                b_code[y] = 1.0
                y += 1
            else:
                b_code[y] = -1.0
                y += 1

    return b_code


def ambiguity(bcode):
    y = np.convolve(bcode, bcode[::-1], mode='same')
    y = np.abs(y/np.max(y))
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


def ambiguity2d(bcode, nrang):
    n = len(bcode)
    b = np.zeros(2 * nrang + n, dtype=np.complex64)
    b[nrang:nrang+n] = bcode
    taus = np.arange(2*nrang)
    m = np.zeros((2*nrang, n), dtype=float)
    for idx, tau in enumerate(taus):
        m[idx, :] = 20 * np.log10(np.abs(np.fft.fft(np.conj(b[tau:tau+n]) * bcode)) + 0.000000000001)  # log 0 avoided
        m[idx, :] = np.fft.fftshift(m[idx, :])
    m -= np.max(m)
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / 100_000.0))  # code length [chips], chip length [s]
    return m, f


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

    bcode = generate_bcode_10k('../dat/pseudo_random_code_test_8_lpf.txt')
    nrang = 100

    t, y = ambiguity(bcode)
    ambiguity_plot(t, y)

    m, f = ambiguity2d(bcode, nrang)
    ambiguity_plot2d(m, f, nrang)
