import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci
import scipy.signal as sig
import random
import csv

# Constants
PI = np.pi
SPEED_OF_LIGHT = sci.c
FREQUENCY = 49500000
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
NUM_ANTENNAS = 10


class JacobsRalston():
    """
    Use 'Ambiguity Resolution in Interferometry (Jacobs,. Ralston, 1981)'
    To determine the best antenna spacing in a crossed configuration.
    """

    # UNITS should always be in terms of wavelengths and degrees
    def __init__(self, array_file):
        self.fov_azimuth = np.array([-45, 45]) / 180 * PI  # This should be defined by recievable fov not desired.
        self.fov_elevation = np.array([45, 89]) / 180 * PI  # This should be defined by recievable fov not desired.
        self.array_file = array_file
        self.antenna_coords = np.array(
            [[0, 0], [9, 0], [19, 0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [9, 5],
             [9, -16], [np.nan, np.nan]])

        self.find_position()
        return

    def modulo_integer(self, d, theta_max):
        nmax = int(np.abs(d * np.sin(theta_max) + 0.5))
        n = np.arange(-nmax, nmax, 1, dtype='int')
        return n

    def find_lmin(self, d1, d2, n1, n2):
        dd = np.abs(d1) + np.abs(d2)
        m = d1 / d2
        B = 1 / ((1 + m ** 2) ** 0.5)
        yi = np.array([])
        # print(dd, m, B)
        for i in n1:
            for j in n2:
                dndn = np.abs(np.abs(d2) * i - np.abs(d1) * j)
                if dndn <= dd:
                    y = m * j - i
                    yi = np.append(yi, y)
                else:
                    pass
        ysorted = np.sort(yi)
        lmin = 1  # This is very lazy
        for i in range(1, len(ysorted)):
            l = (ysorted[i] - ysorted[i - 1]) * B
            if l < lmin:
                lmin = l
        return lmin

    def find_position(self):
        """ Find the next position for the antenna in the linear array. """
        step = 0.001

        start = 2
        d1 = 16 / 2
        stop = d1
        theta = 90 / 180 * PI
        # For the East-West array.
        data_d = np.array([])
        data_l = np.array([])
        n1 = self.modulo_integer(d1, theta)
        d2 = np.arange(start, stop, step)
        for di in d2:
            n2 = self.modulo_integer(di, theta)
            lmin = self.find_lmin(d1, di, n1, n2)
            data_d = np.append(data_d, di)
            data_l = np.append(data_l, lmin)
        ind1 = np.where(data_l == np.amax(data_l))
        print("First placed:", data_d[ind1])

        start = 2.5
        d1 = 16 / 2 - 2.5  # stop -data_d[ind1[0][0]]
        stop = 16 / 2
        theta = 90 / 180 * PI
        # For the East-West array.
        data_d2 = np.array([])
        data_l2 = np.array([])
        n1 = self.modulo_integer(d1, theta)
        d2 = np.arange(start, stop, step)
        for di in d2:
            n2 = self.modulo_integer(di, theta)
            lmin = self.find_lmin(d1, di, n1, n2)
            data_d2 = np.append(data_d2, di)
            data_l2 = np.append(data_l2, lmin)
        ind2 = np.where(data_l2 == np.amax(data_l2))
        data_d2 = data_d2 + data_d[ind1[0]]
        # data_d2 = np.pad(data_d2, (ind1[0][0], (len(data_d)-len(data_d2)-ind1[0][0]) ), 'constant', constant_values=(0,0))
        data_l2 = np.pad(data_l2, (ind1[0][0], (len(data_l) - len(data_l2) - ind1[0][0])), 'constant',
                         constant_values=(0, 0))

        # #dl_mult = np.multiply(data_l+1, data_l2+1)

        # start = 4
        # d1 = 16/2 - 4#stop -4.1
        # stop = 16/2
        # theta = 90 / 180 * PI
        # # For the East-West array.
        # data_d3 = np.array([])
        # data_l3 = np.array([])
        # n1 = self.modulo_integer(d1, theta)
        # d2 = np.arange(start, stop, step)
        # for di in d2:
        # 	n2 = self.modulo_integer(di, theta)
        # 	lmin = self.find_lmin(d1, di, n1, n2)
        # 	data_d3 = np.append(data_d3, di)
        # 	data_l3 = np.append(data_l3, lmin)
        # print(data_d3[np.where(data_l3 == np.amax(data_l3))])
        # data_l3 = np.pad(data_l3, (ind1[0][0], (len(data_l)-len(data_l3)-ind1[0][0]) ), 'constant', constant_values=(0,0))

        plt.figure(figsize=[6, 4])
        plt.scatter(data_d[ind1], data_l[ind1], marker='o', c='k', s=100.0,
                    label=f'Antenna Location {data_d[ind1][0]*WAVELENGTH:.2f} [m]')
        plt.plot(data_d, data_l, 'k', label='Third Antenna Placement')
        plt.plot(data_d, data_l2, '--m', label='Fourth Antenna Placement')
        # plt.plot(data_d, dl_mult)
        # plt.plot(data_d3+4.1, data_l3,'g')
        plt.title("Minimum Separation vs. Baseline")
        plt.xlabel("Baseline")
        plt.ylabel("Minimum Phase Lines Separation")
        plt.legend(loc='upper right')
        plt.savefig('E:/school/masters_thesis/images/jacobs_ralston_plot.png')
        plt.show()
        return


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

    JacobsRalston("pass")
