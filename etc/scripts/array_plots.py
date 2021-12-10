import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci


def uv_space(ant_posx, ant_posy, wavelength):
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
ant_posx = np.array([0.,15.10,73.80,24.2,54.5,54.5,42.40,54.5,44.20,96.9])
ant_posy = np.array([0.,0.,-99.90,0.,-94.50,-205.90,-177.2,0.,-27.30,0.])
u, v = uv_space(ant_posx, ant_posy, wavelength)
ps = []

for i in range(10):
    for j in range(i + 1, 10):
        d = ((ant_posx[i] - ant_posx[j]) ** 2 + (ant_posy[i] - ant_posy[j]) ** 2) ** 0.5
        print(i, j, d / 6.06)
        ps.append(d / 6.06)
print("Possible Baselines:", len(ps))
print("Baselines: ", np.sort(ps))
num_same = 0
for i in range(len(ps)):
    for j in range(i + 1, len(ps)):
        if ps[i] == ps[j]:
            num_same += 1
print("Non-unique Baselines:", num_same)

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

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
axs[0].set_title("Antenna Positions")
axs[0].set_xlabel("East-West[m]")
axs[0].set_ylabel("North-South [m]")
axs[0].grid(which='both')
axs[0].axis('equal')
axs[0].scatter(ant_posx, ant_posy, marker='v', color='k')

axs[1].set_title("Sampling Space")
axs[1].set_xlabel("u")
axs[1].set_ylabel("v")
axs[1].grid(which='both')
axs[1].axis('equal')
axs[1].scatter(u, v, color='k')
axs[1].scatter(0, 0, color='k')

# site_image = img.imread('/beaver/backup/images/bakker_built_cropped.jpg')
# axs[2].set_title('Annotated Site Map')
# axs[2].imshow(site_image)
# axs[2].axis('off')


#plt.tight_layout()
plt.show()
#plt.savefig('/beaver/backup/images/array_map.pdf')


