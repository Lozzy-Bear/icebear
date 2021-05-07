import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
import scipy.signal as sig
import random
import csv

# Constants
SPEED_OF_LIGHT = sci.c
FREQUENCY = 49500000
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
NUM_ANTENNAS = 10


class JacobsRalston():
    """
    Use 'Ambiguity Resolution in Interferometry (Jacobs,. Ralston, 1981)'
    To determine the best antenna spacing in a crossed configuration.

    Note
    ----
    Units should always be input in terms of wavelengths and degrees

    """

    # UNITS should always be in terms of wavelengths and degrees
    def __init__(self):
        self.fov_azimuth = np.deg2rad(np.array([-45, 45]))
        self.fov_elevation = np.deg2rad(np.array([45, 89]))
                                                          # b12 result = 16.0
        x1, y1 = self.find_position(0.0, 16.0, 0.001, 1.5)  # b13 result = 2.50
        x2, y2 = self.find_position(2.462, 16.0, 0.001, 1.5)  # b14 result = 2.50
        x3, y3 = self.find_position(3.98, 16.0, 0.001, 1.5)  # result = 2.50
        x2, y2 = self.pad_arr(x2, y2, x1)
        x3, y3 = self.pad_arr(x3, y3, x1)
        y1 += 1
        y2 += 1
        y3 += 1

        x = np.copy(x1)
        y = (y1 * y2 * y3) / 3

        plt.figure(figsize=[6, 4])
        plt.plot(x, y, 'k')

        plt.figure(figsize=[6, 4])
        plt.plot(x, y1, 'k', label='Between Antenna 1 and 2')
        # plt.plot(x, y2, 'm', label='Between Antenna 1 and 3')
        # plt.plot(x, y3, 'g', label='Between Antenna 1 and 4')
        plt.title("Minimum Separation vs. Baseline")
        plt.xlabel("Baseline")
        plt.ylabel("Minimum Phase Lines Separation")
        plt.legend(loc='upper right')
        plt.show()
        return

    def pad_arr(self, x, y, ref):
        pads = np.zeros(len(ref) - len(x))
        x = np.append(pads, x)
        y = np.append(pads, y)
        return x, y

    def modulo_integer(self, d, theta_max):
        nmax = int(np.abs(d * np.sin(theta_max) + 0.5))
        n = np.arange(-nmax, nmax, 1, dtype='int')
        return n

    def find_lmin(self, d1, d2, n1, n2):
        y = np.zeros((len(n1), len(n2)))
        for i in range(len(n1)):
            for j in range(len(n2)):
                y[i, j] = d1 / d2 * n2[j] - n1[i]
        y = np.sort(y.flatten())
        lmin = np.min((y[1::] - y[0:-1]) / ((1 + (d1 / d2) ** 2) ** 0.5))

        return lmin

    def find_position(self, x1, x2, dx, xmin):
        """ Find the next position for the antenna in the linear array. """
        x = np.arange(x1+xmin, x2, dx)
        y = np.zeros_like(x)
        theta = np.pi/2
        n1 = self.modulo_integer(np.abs(x2 - x1)/2, theta)
        for i in range(len(x)):
            n2 = self.modulo_integer(x[i], theta)
            y[i] = self.find_lmin(np.abs(x2 - x1)/2, x[i], n1, n2)
        ind1 = np.where(y == np.amax(y))
        print("placed:", x[ind1])

        return x, y


class AntennaArray:
    """
    Construct the antenna's and the array. Outputs the far-field Electric field.
    """

    def __init__(self, array_file):
        self.array_file = array_file
        self.theta = np.linspace(-PI, PI, 1000)
        self.load_array()
        self.plot_antenna()
        self.element_factor()
        self.array_factor()
        self.field_factor()
        return

    def load_array(self):
        """ Load the array to simulate. """
        self.antenna_coords = np.loadtxt(self.array_file, delimiter=",")
        print("Loaded array coordinates: ")
        print(self.antenna_coords)
        print("")
        return

    def element_factor(self):
        """ Calculate the antenna's normalized radiation factor. """
        # Cushcraft 617-6B Superboomer
        FORWARD_GAIN = 14.0
        FRONT_2_BACK = 30.0
        BEAM_WIDTH_E = 2 * 19.0
        BEAM_WIDTH_H = 2 * 20.0
        SIDE_LOBE_ATTEN = 60.0

        # GAIN LINEAR
        forward_gain = 10 ** (FORWARD_GAIN / 10)
        backward_gain = forward_gain / (10 ** (FRONT_2_BACK / 10))
        side_lobe_gain = 10 ** ((FORWARD_GAIN - SIDE_LOBE_ATTEN) / 10)

        # LINEAR GAINS NORMALIZED
        forward_gain = forward_gain / forward_gain
        backward_gain = backward_gain / forward_gain
        side_lobe_gain = side_lobe_gain / forward_gain
        print(forward_gain, backward_gain)

        # RADIATION PATTERN APPROXIMATION
        self.ef = forward_gain * np.cos(self.theta) ** 11 - backward_gain * np.cos(self.theta) ** 3
        return

    def weight_factor(self, d):
        """ Calculate the array's phase distribution. """
        excitation = 1.0
        phase_offset = 0 * PI / 180
        wf = excitation * np.exp(1j * 2 * PI * d * np.sin(phase_offset) / WAVELENGTH)
        return wf

    def array_factor(self):
        """ Calculate the array's array factor. """
        N = self.antenna_coords.shape[0]
        self.af = np.zeros(len(self.theta))
        for n in range(0, N, 1):
            d = self.antenna_coords[n, 0]
            self.af = self.af + np.exp(-1j * 2 * PI * d * np.sin(self.theta) / WAVELENGTH) * self.weight_factor(d)
        self.af = self.af / N
        return

    def field_factor(self):
        self.ff = self.ef * self.af
        plt.figure()
        plt.polar(self.theta, self.ff)
        plt.title("Field Factor")
        plt.ylim(0, 1)
        plt.show()
        return

    def plot_antenna(self):
        plt.figure(5)
        plt.scatter(self.antenna_coords[:, 0], self.antenna_coords[:, 1], marker="v")
        plt.title("Array Pattern")
        plt.xlabel("X-Position [m]")
        plt.ylabel("Y-Position [m]")
        plt.grid(which='both')
        plt.axis('equal')
        plt.show()
        return

    def uv_space(ant_posx, ant_posy):
        u = np.array([])
        v = np.array([])
        for i in range(NUM_ANTENNAS):
            for j in range(i + 1, NUM_ANTENNAS):
                u = np.append(u, (ant_posx[i] - ant_posx[j]) / WAVELENGTH)
                v = np.append(v, (ant_posy[i] - ant_posy[j]) / WAVELENGTH)
            # u = np.append(u, (ant_posx[j] - ant_posx[i]) / WAVELENGTH)
            # v = np.append(v, (ant_posy[j] - ant_posy[i]) / WAVELENGTH)
        return u, v

    """
    u, v = uv_space(ant_posx, ant_posy)


    ps = []
    for i in range(10):
        for j in range(i+1, 10):
            d = ((ant_posx[i]-ant_posx[j])**2 +(ant_posy[i]-ant_posy[j])**2) ** 0.5
            ps.append(d)
    print("Possible Baselines:",len(ps))
    num_same = 0
    for i in range(len(ps)):
        for j in range(i+1, len(ps)):
            if ps[i] == ps[j]:
                num_same += 1
    print("Non-unique Baselines:",num_same)

    plt.figure(2)
    plt.title("U,V Space")
    plt.xlabel("U [wavelength]")
    plt.ylabel("V [wavelength]")
    plt.grid(which='both')
    plt.axis('equal')
    plt.scatter(u,v, color='red')
    #plt.plot(a,b)
    plt.plot(0,0, marker='o')
    """


if __name__ == "__main__":
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    JacobsRalston()
