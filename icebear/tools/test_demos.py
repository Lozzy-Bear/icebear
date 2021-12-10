import os.path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime


def velocity_plot(sx, sv, date, filepath):
    year = date[0]
    month = date[1]
    day = date[2]
    hour = time[0:2]
    minute = time[2:4]
    second = time[4:6]

    u = np.sin(np.deg2rad(sv[0, :])) * np.cos(np.deg2rad(sv[1, :]))
    v = np.cos(np.deg2rad(sv[0, :])) * np.cos(np.deg2rad(sv[1, :]))
    n = np.sqrt(u**2 + v**2)
    u /= n
    v /= n
    # u = np.where(sv[2, :] == 0, np.nan, u)
    # v = np.where(sv[2, :] == 0, np.nan, v)

    fig = plt.figure(figsize=[10, 10])
    gs = fig.add_gridspec(10, 10)
    props = dict(boxstyle='square', facecolor='white', alpha=1.0)
    textbox = f'{year}-{month:02d}-{day:02d}  ' \
              f'Records {len(sx[1, :]):3d}'
    fig.suptitle(textbox, y=0.91)
    vel_thresh = 1500.0
    alt_major_ticks = np.arange(70, 130+20, 20)
    alt_minor_ticks = np.arange(70, 130+5, 5)
    lat_major_ticks = np.arange(50, 64+2, 2)
    lat_minor_ticks = np.arange(50, 64+0.25, 0.25)
    lon_major_ticks = np.arange(-113, -99+2, 2)
    lon_minor_ticks = np.arange(-113.0, -99.0+0.25, 0.25)
    facecolor = 'black'
    colormap = 'plasma'#'RdBu'
    latb = lat_minor_ticks
    lonb = lon_minor_ticks
    altb = alt_minor_ticks

    # Altitude-Latitude slice
    ax1 = fig.add_subplot(gs[2:-1, 0:2])
    ax1.set_facecolor(facecolor)
    # plt.scatter(sx[2, :], sx[0, :], c=sv[2, :], marker='D', cmap=colormap, vmin=-vel_thresh, vmax=vel_thresh)
    plt.hist2d(sx[2, :], sx[0, :], bins=[altb, latb], cmin=100, cmap=colormap, norm=colors.LogNorm(vmin=1, vmax=100_00))
    plt.xlabel('Altitude [km]')
    plt.ylabel('Latitude [deg]')
    # plt.clim(-vel_thresh, vel_thresh)
    plt.clim(1, 100_000)
    plt.xlim([130.0, 70.0])
    plt.ylim([50.0, 64.0])
    ax1.set_xticklabels(alt_major_ticks[::-1], rotation=0.0)
    ax1.set_xticks(alt_major_ticks[::-1])
    ax1.set_xticks(alt_minor_ticks[::-1], minor=True)
    ax1.set_yticks(lat_major_ticks)
    ax1.set_yticks(lat_minor_ticks, minor=True)
    ax1.grid(which='minor', linestyle='--', alpha=0.25)
    ax1.grid(which='major', linestyle='-', alpha=0.5)


    # Latitude-Longitude slice
    ax2 = fig.add_subplot(gs[2:-1, 2:-1])
    ax2.set_facecolor(facecolor)
    plt.xlabel('Longitude [deg]')
    # im = plt.scatter(sx[1, :], sx[0, :], c=sv[2, :], marker='D', cmap=colormap, vmin=-vel_thresh, vmax=vel_thresh)
    im = plt.hist2d(sx[1, :], sx[0, :], bins=[lonb, latb], cmin=100, cmap=colormap, norm=colors.LogNorm(vmin=1, vmax=100_00))
    # plt.clim(-vel_thresh, vel_thresh)
    plt.clim(1, 100_000)
    plt.scatter(-109.403, 50.893, c='w')
    # Annotation include setting for drawing connecting arrows that will point to the scatter point.
    # I elected to simply overlap them but moving the xytext offset value will move the box.
    plt.annotate('TX', (-109.403, 50.893),
                 xytext=(0.0, -2.0), textcoords='offset points', ha='center', va='bottom', color='red',
                 bbox=dict(boxstyle='square,pad=0.1', fc='white', alpha=1.0),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='r'))
    plt.scatter(-106.450, 52.243, c='w')
    plt.annotate('RX', (-106.450, 52.24),
                 xytext=(0.0, -2.0), textcoords='offset points', ha='center', va='bottom', color='blue',
                 bbox=dict(boxstyle='square,pad=0.1', fc='white', alpha=1.0),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='b'))
    plt.xlim([-113.0, -99.0])
    plt.ylim([50.0, 64.0])
    ax2.set_yticklabels([])
    ax2.set_xticklabels(lon_major_ticks[:0:-1], rotation=0.0)
    ax2.set_xticks(lon_major_ticks[::-1])
    ax2.set_xticks(lon_minor_ticks[::-1], minor=True)
    ax2.set_yticks(lat_major_ticks)
    ax2.set_yticks(lat_minor_ticks, minor=True)
    ax2.grid(which='minor', linestyle='--', alpha=0.25)
    ax2.grid(which='major', linestyle='-', alpha=0.5)

    # Altitude-Longitutde slice
    ax3 = fig.add_subplot(gs[0:2, 2:-1])
    ax3.set_facecolor(facecolor)
    # plt.scatter(sx[1, :], sx[2, :], c=sv[2, :], marker='D', cmap=colormap, vmin=-vel_thresh, vmax=vel_thresh)
    plt.hist2d(sx[1, :], sx[2, :], bins=[lonb, altb], cmin=100, cmap=colormap, norm=colors.LogNorm(vmin=1, vmax=100_00))
    plt.ylabel('Altitude [km]')
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    # plt.clim(-vel_thresh, vel_thresh)
    plt.clim(1, 100_000)
    plt.xlim([-113.0, -99.0])
    plt.ylim([70.0, 130.0])
    ax3.set_xticklabels([])
    ax3.set_xticks(lon_major_ticks[::-1])
    ax3.set_xticks(lon_minor_ticks[::-1], minor=True)
    ax3.set_yticks(alt_major_ticks)
    ax3.set_yticks(alt_minor_ticks, minor=True)
    ax3.grid(which='minor', linestyle='--', alpha=0.25)
    ax3.grid(which='major', linestyle='-', alpha=0.5)

    # Add the colorbar
    ax4 = fig.add_subplot(gs[2:-1, -1])
    fig.colorbar(im[3], cax=ax4, aspect=0.1, label='Occurrence')

    # Add a doppler window
    ax5 = fig.add_subplot(gs[0:2, 0:2])
    bins = np.arange(-1500, 1500 + 30, 30)
    plt.hist(sv[2, :], bins=bins, histtype=u'step', color='k')
    ax5.set_yscale('log')
    # ax5.set_ylim([1, 1000])
    ax5.grid(which='major', linestyle='--', alpha=0.5)
    ax5.set_yticklabels(np.array(['', '', '$10^1$', '$10^2$', '$10^3$']))
    plt.xlabel('Velocity [km/s]')
    plt.xticks(np.arange(-1000, 1000 + 1000, 1000), np.arange(-1.0, 1.0 + 1.0, 1.0))
    ax5.xaxis.tick_top()
    ax5.xaxis.set_label_position('top')
    ax5.tick_params(axis='x', direction='in', pad=-15)

    plt.savefig(filepath + f'velocity_{year}{month}{day}.pdf', bbox_inches='tight')
    plt.close()
    print('save fig:', filepath + f'velocity_{year}{month}{day}.pdf')

    return


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

files = []
files.append('/beaver/backup/level2b/2020_12_12/ib3d_normal_swht_2020_12_12_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2020_12_13/ib3d_normal_swht_2020_12_13_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2020_12_14/ib3d_normal_swht_2020_12_14_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2020_12_15/ib3d_normal_swht_2020_12_15_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2020_03_31/ib3d_normal_swht_2020_03_31_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2021_02_02/ib3d_normal_swht_2021_02_02_prelate_bakker.h5')
files.append('/beaver/backup/level2b/2019_12_19/ib3d_normal_swht_2019_12_19_prelate_bakker.h5')

slant_range = np.array([])
altitude = np.array([])
snr_db = np.array([])
time = np.array([])
latitude = np.array([])
longitude = np.array([])
altitude = np.array([])
vaz = np.array([])
vel = np.array([])
vma = np.array([])
# plt.figure()
for file in files:
    f = h5py.File(file, 'r')

    # Example: How to read data in second by second
    # times = f['data']['time'][()]
    # time, start_index, counts = np.unique(times, return_index=True, return_counts=True)
    # for i in range(len(start_index)):
    #     snr_db = f['data']['snr_db'][start_index[i]:start_index[i] + counts[i]]
    #     ts = datetime.datetime.fromtimestamp(time[i])
    #     print(ts.strftime('%Y-%m-%d %H:%M:%S'), snr_db)

    # plt.scatter(f['data']['slant_range'][()], f['data']['altitude'][()], c=f['data']['velocity'][()])
    # plt.colorbar()

    # Example: Reading in data from multiple files (days) into one larger
    slant_range = np.append(slant_range, f['data']['slant_range'][()])
    snr_db = np.append(snr_db, f['data']['snr_db'][()])
    time = np.append(time, f['data']['time'][()])

    latitude = np.append(latitude, f['data']['latitude'][()])
    longitude = np.append(longitude, f['data']['longitude'][()])
    altitude = np.append(altitude, f['data']['altitude'][()])
    sx = np.array([latitude, longitude, altitude])

    vaz = np.append(vaz, f['data']['velocity_azimuth'][()])
    vel = np.append(vel, f['data']['velocity_elevation'][()])
    vma = np.append(vma, f['data']['velocity_magnitude'][()])
    sv = np.array([vel, vaz, vma])

    date = f['info']['date'][()]
    filepath = os.path.dirname(file) + '/'
    # idx = np.argwhere(time>time[0]+5.0*60.0*60.0)

velocity_plot(sx, sv, date, filepath)
exit()

plt.figure()
mean_altitude = 93.2 # np.mean(altitude)
total_targets = len(altitude)
n, bins, _ = plt.hist(altitude, bins='auto', orientation='horizontal', histtype='step', label=f'Total Targets {total_targets}', color='k')

counts = n
mids = 0.5*(bins[1:] + bins[:-1])
probs = counts / np.sum(counts)
mean = np.sum(probs * mids)
sd = np.sqrt(np.sum(probs * (mids - mean)**2))
print(f'standard deviation: {sd}, mean: {mean}')

# plt.xscale('log')
# plt.title('E Region Scatter Distribution\nDecember 19, 2019')
# plt.title('E Region Scatter Distribution\nFebruary 2, 2021')
plt.title('Geminids Meteor Trail Distribution\nDecember 12-15, 2020')
plt.xlabel('Count')
plt.ylabel('Altitude [km]')
plt.ylim((70, 130))
plt.xlim((10, 9_000))
plt.plot([0, 10_000], [mean_altitude, mean_altitude], '--k', label=f'Peak Altitude {mean_altitude:.1f} [km]')
plt.legend(loc='lower right')
plt.grid()
plt.show()

