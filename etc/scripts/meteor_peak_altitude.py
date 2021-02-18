import h5py
import numpy as np
import matplotlib.pyplot as plt
import icebear.utils as utils
from mpl_toolkits.mplot3d import Axes3D


def meteor_distribution(alt):
    #plt.figure(figsize=[12, 3])
    #alt = np.ma.masked_where(alt < 70, alt)
    _ = plt.hist(alt, bins='auto', orientation='horizontal', histtype=u'step', label=f'{len(alt)} total targets')
    plt.xscale('log')
    #plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude Distribution')
    plt.xlabel('Count')
    plt.ylabel('Altitude [km]')
    #plt.ylim((80, 130))
    plt.xlim((10, 10_000))
    plt.plot([0, 10_000], [102, 102], '--k', label='102.0 km Altitude')
    plt.legend(loc='best')
    return


def doppler_distribution(dop, alt):
    print(np.min(dop), np.max(dop))
    xbins = np.linspace(-500, 500, 100)
    ybins = np.linspace(50, 150, 100)
    plt.figure()
    h, _, _ = np.histogram2d(dop, alt, bins=(xbins, ybins))
    plt.imshow(h.T, origin="lower", interpolation="bilinear", extent=[-500, 500, 50, 150])
    #plt.title('2020-12-12 to 2020-12-15 Target Distribution')
    plt.xlabel('Doppler [Hz]')
    plt.ylabel('Altitude [km]')
    return


def power_distribution(snr, alt):
    print(np.min(snr), np.max(snr))
    xbins = np.linspace(0, 30, 300)
    ybins = np.linspace(50, 150, 100)
    plt.figure()
    h, _, _ = np.histogram2d(snr, alt, bins=(xbins, ybins))
    plt.imshow(h.T, origin="lower", interpolation="bilinear", extent=[6, 30, 50, 150])
    #plt.title('2020-12-12 to 2020-12-15 Target Distribution')
    plt.xlabel('Signal to Noise [dB]')
    plt.ylabel('Altitude [km]')
    return


def range_altitude_power(rng, alt, snr):
    s = (alt).argsort()
    #alt = np.ma.masked_where(alt < 70, alt)
    plt.figure(figsize=[12, 3])
    plt.xlim((0, 1300))
    plt.ylim((0, 200))
    plt.xticks(np.arange(0, 1400, 100))
    plt.yticks(np.arange(0, 220, 20))
    plt.grid()
    plt.scatter(rng[s], alt[s], c=snr[s], cmap='plasma_r', vmin=6.0, vmax=30.0, alpha=0.25)
    plt.colorbar(label='Signal to Noise [dB]')
    #plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude vs. Range')
    plt.plot([0, 1300], [102, 102], '--k', label='102.0 km Altitude')
    plt.xlabel('Range [km]')
    plt.ylabel('Altitude [km]')
    plt.legend(loc='best')
    return


def range_altitude_doppler(rng, alt, dop):
    s = (alt).argsort()
    #alt = np.ma.masked_where(alt < 70, alt)
    plt.figure(figsize=[12, 3])
    plt.xlim((0, 1300))
    plt.ylim((0, 200))
    plt.xticks(np.arange(0, 1400, 100))
    plt.yticks(np.arange(0, 220, 20))
    plt.grid()
    plt.scatter(rng[s], alt[s], c=dop[s], cmap='jet_r', vmin=-500.0, vmax=500.0, alpha=0.25)
    plt.colorbar(label='Doppler [Hz]')
    #plt.title('Geminids 2020-12-12 to 2020-12-15 Meteor Altitude vs. Range')
    plt.plot([0, 1300], [102, 102], '--k', label='102.0 km Altitude')
    plt.xlabel('Range [km]')
    plt.ylabel('Altitude [km]')
    plt.legend(loc='best')
    return


def plot_3d(az, rng, alt, dop, name, idx):
    #alt = np.ma.masked_where(alt < 70, alt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(rng*np.sin(np.deg2rad(az)), rng*np.cos(np.deg2rad(az)),
                   alt,
                   c=dop, cmap='jet_r',  alpha=0.25, vmin=-500.0, vmax=500.0,)
    ax.set_title(f'Time = {idx*5} [s]')
    ax.set_xlabel('West - East [km]')
    ax.set_ylabel('North - South [km]')
    ax.set_zlabel('Altitude [km]')
    fig.colorbar(p, label='Doppler [Hz]')
    ax.set_xlim(-600, 600)
    ax.set_ylim(0, 1200)
    ax.set_zlim(0, 200)
    ax.view_init(elev=35.0, azim=225.0)
    # for ii in range(0, 360, 1):
    #     ax.view_init(elev=35., azim=ii)
    plt.savefig(f"E:/icebear/figures/2020_03_31/{name}{idx}.png")
    del p
    plt.close()
    return


def perspective(az, rng, alt, dop, name, idx):
    """
    Plots data in 3D with a North-South side view from the East and West breakouts, as well as a top down view.

    Parameters
    ----------
    az
    rng
    alt
    dop
    name
    idx

    Returns
    -------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(rng*np.sin(np.deg2rad(az)), rng*np.cos(np.deg2rad(az)),
                   alt,
                   c=dop, cmap='jet_r',  alpha=0.25, vmin=-500.0, vmax=500.0,)
    ax.set_title(f'Time = {idx*5} [s]')
    ax.set_xlabel('West - East [km]')
    ax.set_ylabel('North - South [km]')
    ax.set_zlabel('Altitude [km]')
    fig.colorbar(p, label='Doppler [Hz]')
    ax.set_xlim(-600, 600)
    ax.set_ylim(0, 1200)
    ax.set_zlim(0, 200)
    ax.view_init(elev=35.0, azim=225.0)
    # for ii in range(0, 360, 1):
    #     ax.view_init(elev=35., azim=ii)
    plt.savefig(f"E:/icebear/figures/2020_03_31/{name}{idx}.png")
    del p
    plt.close()
    return


def movies(file, name):
    f = h5py.File(file, 'r')
    print(file)
    group = f['data']
    keys = group.keys()
    el = np.array([])
    rng = np.array([])
    dop = np.array([])
    snr = np.array([])
    az = np.array([])
    idx = -1
    cnt = 0
    for key in keys:
        data = group[f'{key}']
        rf_distance = data['rf_distance'][()]
        snr_db = data['snr_db'][()]
        doppler_shift = data['doppler_shift'][()]
        azimuth = data['azimuth'][()]
        elevation = data['elevation'][()]
        rng = np.append(rng, rf_distance)
        el = np.append(el, elevation)
        dop = np.append(dop, doppler_shift)
        snr = np.append(snr, np.abs(snr_db))
        az = np.append(az, azimuth)

        cnt += 1
        if cnt >= 5:
            cnt = 0
            idx += 1
            rng = rng * 0.75 - 200
            m = np.ones_like(rng)
            m = np.ma.masked_where(snr <= 3.0, m)
            m = np.ma.masked_where(el >= 25, m)
            m = np.ma.masked_where(el <= 1, m)
            m = np.ma.masked_where(rng <= 25, m)
            m = np.ma.masked_where(rng >= 1200, m)

            a = 6378.1370
            b = 6356.7523
            p1 = np.deg2rad(52.1579)
            r1 = np.sqrt((a * np.cos(p1)) ** 2 + (b * np.sin(p1)) ** 2)
            pre_alt = np.sqrt(r1 ** 2 + rng ** 2 - 2 * r1 * rng * np.cos(np.deg2rad(90 + np.abs(el))))
            gamma = np.arccos((rng ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))
            el2 = el - np.abs(np.rad2deg(gamma))
            alt_geocentric = -r1 + np.sqrt(r1 ** 2 + rng ** 2 + 2 * r1 * rng * np.sin(np.deg2rad(el2)))
            alt_geocentric -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))

            rng = rng * m
            dop = dop * m
            az = az * m
            alt_geocentric = alt_geocentric * m
            alt_geocentric = np.ma.compressed(alt_geocentric)
            rng = np.ma.compressed(rng)
            snr = np.ma.compressed(snr)
            dop = np.ma.compressed(dop)
            az = np.ma.compressed(az)

            plot_3d(az, rng, alt_geocentric, dop, name, idx)

            el = np.array([])
            rng = np.array([])
            dop = np.array([])
            snr = np.array([])
            az = np.array([])

    return


filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
# files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_15')  # Enter first sub directory and last
# files = utils.get_all_data_files(filepath, '2019_12_19', '2019_12_19')  # Enter first sub directory and last
files = utils.get_all_data_files(filepath, '2020_03_31', '2020_03_31')  # Enter first sub directory and last
# el = np.array([])
# rng = np.array([])
# dop = np.array([])
# snr = np.array([])
# az = np.array([])

for file in files:
    file = files[2]
    movies(file, 'march')
    exit()
#     f = h5py.File(file, 'r')
#     print(file)
#     group = f['data']
#     keys = group.keys()
#
#     for key in keys:
#         data = group[f'{key}']
#         rf_distance = data['rf_distance'][()]
#         snr_db = data['snr_db'][()]
#         doppler_shift = data['doppler_shift'][()]
#         azimuth = data['azimuth'][()]
#         elevation = data['elevation'][()]
#         area = data['area'][()]
#         rng = np.append(rng, rf_distance)
#         el = np.append(el, elevation)
#         dop = np.append(dop, doppler_shift)
#         snr = np.append(snr, snr_db)
#         az = np.append(az, azimuth)
#     break
#
# rng = rng * 0.75 - 200
# m = np.ones_like(rng)
# # m = np.ma.masked_where(dop > 20, m)
# # m = np.ma.masked_where(dop < -20, m)
# m = np.ma.masked_where(snr <= 8.0, m)
# m = np.ma.masked_where(el >= 25, m)
# m = np.ma.masked_where(el <= 1, m)
# m = np.ma.masked_where(rng <= 25, m)
# m = np.ma.masked_where(rng >= 1200, m)
#
# a = 6378.1370
# b = 6356.7523
# p1 = np.deg2rad(52.1579)
# r1 = np.sqrt((a*np.cos(p1))**2 + (b*np.sin(p1))**2)
# re = 6378
#
# pre_alt = np.sqrt(r1 ** 2 + rng ** 2 - 2 * r1 * rng * np.cos(np.deg2rad(90 + np.abs(el))))
# gamma = np.arccos((rng ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))
# el2 = el - np.abs(np.rad2deg(gamma))
# p2 = p1 + gamma
# r2 = np.sqrt((a*np.cos(p2))**2 + (b*np.sin(p2))**2)
# alt_geocentric = -r1 + np.sqrt(r1 ** 2 + rng ** 2 + 2 * r1 * rng * np.sin(np.deg2rad(el2)))
# alt_geocentric -= np.tan(np.deg2rad(2.8624)) * (400 + rng*np.sin(np.deg2rad(az)))
#
# # m = np.ma.masked_where(alt_geocentric >= 150, m)
# # m = np.ma.masked_where(alt_geocentric <= 50, m)
# rng = rng * m
# dop = dop * m
# snr = snr * m
# az = az * m
# el = el * m
# alt_geocentric = alt_geocentric * m
# alt_geocentric = np.ma.compressed(alt_geocentric)
#
# print(len(alt_geocentric))
#
# rng = np.ma.compressed(rng)
# snr = np.ma.compressed(snr)
# dop = np.ma.compressed(dop)
# az = np.ma.compressed(az)
#
# #plot_3d(az, rng, alt_geocentric, np.abs(snr))
# #plot_3d(az, rng, alt_geocentric, dop)
# #meteor_distribution(alt_geocentric)
# #doppler_distribution(dop, alt_geocentric)
# #power_distribution(np.abs(snr), alt_geocentric)
# range_altitude_power(rng, alt_geocentric, np.abs(snr))
# range_altitude_doppler(rng, alt_geocentric, dop)
#
#
# plt.show()
