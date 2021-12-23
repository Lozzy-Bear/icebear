import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy.constants as sci

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
# mpl.rcParams.update({'font.size': 22})
# mpl.rcParams['figure.figsize'] = 20, 10
mpl.rcParams['contour.negative_linestyle'] = 'solid'


def uv_space(ant_posx, ant_posy, wavelength):
    """Generate the U,V space from X,Y,Z"""
    n = 10
    u = np.array([])
    v = np.array([])
    for i in range(n):
        for j in range(i + 1, n):
            u = np.append(u, (ant_posx[i] - ant_posx[j]) / wavelength)
            v = np.append(v, (ant_posy[i] - ant_posy[j]) / wavelength)
            u = np.append(u, (ant_posx[j] - ant_posx[i]) / wavelength)
            v = np.append(v, (ant_posy[j] - ant_posy[i]) / wavelength)
    return u, v


wavelength = sci.c / 49.5e6
ant_posx = np.array([0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9])
ant_posy = np.array([0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.])
u, v = uv_space(ant_posx, ant_posy, wavelength)

sphere_radius = 1E5  # in meters
elevation_values = np.radians((np.arange(361)) * 0.25)
azimuth_values = np.radians(np.arange(1441) * 0.25)
xx, yy = np.meshgrid(azimuth_values, elevation_values, sparse=True)
l_values = np.cos(xx) * np.cos(yy)
m_values = np.sin(xx) * np.cos(yy)
q_values = np.sin(yy)
u_values = np.zeros(45, dtype=np.float32)
v_values = np.zeros(45, dtype=np.float32)
w_values = np.zeros(45, dtype=np.float32)
temp_ind = 0
for first_antenna in range(9):
    for second_antenna in range(first_antenna + 1, 10):
        u_values[temp_ind] = (ant_posx[first_antenna] - ant_posx[second_antenna])
        v_values[temp_ind] = (ant_posy[first_antenna] - ant_posy[second_antenna])
        w_values[temp_ind] = 0
        temp_ind += 1

r_values = np.einsum('b,lm->lmb', u_values, l_values) + np.einsum('b,lm->lmb', v_values, m_values)
antenna_coherence = np.ones(45, dtype=np.complex64)
A_matrix = np.exp(-1.0j * 2.0 * np.pi * r_values / wavelength)
classic_brightness = np.einsum('b,lmb->lm', antenna_coherence, A_matrix)

azl = np.rad2deg(np.arcsin(l_values))
elm = np.rad2deg(np.arcsin(m_values))
amp = 10 * np.log10(np.abs(classic_brightness) / 45.0)

fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
axs[0].set_title("Antenna Positions")
axs[0].set_xlabel("East-West[m]")
axs[0].set_ylabel("North-South [m]")
axs[0].grid(which='both')
axs[0].axis('equal')
axs[0].scatter(ant_posx, ant_posy, marker='v', color='k')
axs[0].set_ylim(-225, 20)
for i, l in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    axs[0].annotate(l, (ant_posx[i]-3, ant_posy[i]-15))

axs[1].set_title("Sampling Space")
axs[1].set_xlabel("u")
axs[1].set_ylabel("v")
axs[1].grid(which='both')
axs[1].axis('equal')
axs[1].scatter(u, v, color='k')
axs[1].scatter(0, 0, color='k')

cs = axs[2].contourf(azl, elm, amp, levels=np.arange(-10, 0 + 1, 1), cmap='Greys')
axs[2].set_title("Dirty Beam")
axs[2].set_xlabel('L [deg]')
axs[2].set_ylabel('M [deg]')
scaler = 3
axs[2].set_ylim(-2 * scaler, 2 * scaler)
axs[2].set_xlim(-1 * scaler, 1 * scaler)
axs[2].grid(which='both')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(axs[2], width='90%', height='3%', loc='upper center')
cb = plt.colorbar(cs, cax=cbaxes, orientation='horizontal')
cb.ax.xaxis.set_ticks_position("bottom")

# plt.subplots_adjust(wspace=0.3)
# plt.tight_layout()
# plt.savefig('array_map.pdf')
plt.show()
