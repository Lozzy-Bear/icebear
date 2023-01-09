"""
Mainly used this to make a summary histogram plot to overlap with the link gain plots
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import common.pretty_plots
import numpy as np
import glob
import h5py


if __name__ == '__main__':
    # files = glob.glob('/data/icebear_data/sanitized_data/ib3d_normal_swht_1lam*.h5')
    files = glob.glob('/data/icebear_data/sanitized_data/ib3d_normal_swht_20*.h5')

    # files = glob.glob('/run/media/arl203/Seagate Expansion Drive/backup/level2b/ib3d_normal_swht_20*.h5')
    # files = glob.glob('/run/media/arl203/Seagate Expansion Drive/backup/level2b/2019_12_19/ib3d_normal_swht_2019_12_19_prelate_bakker_sanity.h5')
    files = files[500::]
    print(len(files))

    snr_db = np.array([])
    doppler_shift = np.array([])
    latitude = np.array([])
    longitude = np.array([])
    altitude = np.array([])
    time = np.array([])
    elevation = np.array([])
    for file in files:
        f = h5py.File(file, 'r')
        snr_db = np.append(snr_db, f['data']['snr_db'][()])
        doppler_shift = np.append(doppler_shift, f['data']['doppler_shift'][()])
        latitude = np.append(latitude, f['data']['latitude'][()])
        longitude = np.append(longitude, f['data']['longitude'][()])
        altitude = np.append(altitude, f['data']['altitude'][()])
        time = np.append(time, f['data']['time'][()])
        elevation = np.append(elevation, f['data']['elevation'][()])
        # print(latitude.max(), latitude.min(), longitude.max(), longitude.min())
        # exit()

    idx = np.nonzero((snr_db > 3.0))# & (altitude > 108.0) & (altitude < 112.0))

    # plt.figure()
    # plt.hist(elevation[idx], bins=45)
    # plt.show()
    # exit()

    records = latitude[idx].shape[0]
    step = 0.1
    lat_bins = np.arange(52, 62 + step, step)
    lon_bins = np.arange(-112, -98 + step, step)
    alt_bins = np.arange(70, 140 + 1, 1)

    # plt.figure()
    # plt.hist2d(latitude[idx], altitude[idx], bins=[lat_bins, alt_bins])
    # plt.hist2d(longitude[idx], altitude[idx], bins=[lon_bins, alt_bins])
    # plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
    plt.title(f"Records = {records}")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_facecolor('black')
    m, xedges, yedges = np.histogram2d(longitude[idx], latitude[idx], bins=[lon_bins, lat_bins])
    m = np.nan_to_num(m, nan=0.0)
    n = np.where(m < 10.0, 10.0, m)
    im = ax.imshow(m.T, origin='lower', extent=[0, 140, 0, 100],
                   cmap='inferno', norm=LogNorm(vmin=1e2, vmax=1e4), interpolation='bilinear')
    plt.colorbar(im, label='Counts', shrink=0.72)
    ax.set_xticks(np.arange(0, 140 + 20, 20), labels=np.arange(-112, -98 + 2, 2))
    ax.set_yticks(np.arange(0, 100 + 20, 20), labels=np.arange(52, 62 + 2, 2))
    ax.grid(which='minor', color='Grey', linestyle=':', linewidth=0.5)
    ax.grid(which='major', color='Grey', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    plt.tight_layout()
    # plt.savefig(f"data_hist_3lam.pdf")
    # plt.show()

    f = h5py.File('/home/arl203/arraytools/nec/ib3d_link_gain_mask_rot13.h5', 'r')
    # f = h5py.File('/home/arl203/arraytools/nec/ib3d_link_gain_mask_1lam_rot13.h5', 'r')
    alt = f['altitude'][()]
    lat = f['latitude'][()]
    lon = f['longitude'][()]
    gain = f['gain_mask'][()]
    gain = np.nan_to_num(gain, nan=0.0)
    gain = np.where(gain < 0.0, 0.0, gain)
    gain[0:21, :, :, 0] = 0.0

    g = gain[1::, 1::, 37, 0].T
    g = np.where(g<= 0.0, 0.0, g)
    g = g / np.max(g)
    m = np.where(m>=1e3, 1e3, m)
    m = m / np.max(m)
    print(np.sum(g - m), np.sum(g), np.sum(m))

    import cv2
    compare_value = cv2.compareHist(g.ravel().astype('float32'), m.ravel().astype('float32'), cv2.HISTCMP_CORREL)
    print(compare_value)
    compare_value = cv2.compareHist(g.ravel().astype('float32'), m.ravel().astype('float32'), cv2.HISTCMP_CHISQR)
    print(compare_value)
    compare_value = cv2.compareHist(g.ravel().astype('float32'), m.ravel().astype('float32'), cv2.HISTCMP_INTERSECT)
    print(compare_value)
    compare_value = cv2.compareHist(g.ravel().astype('float32'), m.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    print(compare_value)
    # g = np.where(g >= 5.0, 1.0, np.nan)
    # print(g.shape, m.shape, np.nansum(np.multiply(m, g)))

    # Option 1
    # im = ax.contour(gain[:, :, 51, 0], origin='lower', extent=[0, 140, 0, 100], vmin=0.0, vmax=30.0,
    #                 levels=[10, 20, 30], linestyles='solid', cmap='Greys')
    # ax.clabel(im, inline=True, inline_spacing=-1, fmt='%1.f', fontsize=12)

    # Option 2
    alt_idx = 37  # 20=90km west, 40=110km center, 60=130km east
    im = ax.contour(gain[:, :, alt_idx, 0], origin='lower', extent=[0, 140, 0, 100], vmin=0.0, vmax=30.0,
                    levels=[10, 20, 30], linestyles='solid', colors='black', linewidths=2.0)
    ax.clabel(im, inline=True, inline_spacing=0, fmt='%1.f', fontsize=12)
    im = ax.contour(gain[:, :, alt_idx, 0], origin='lower', extent=[0, 140, 0, 100], vmin=0.0, vmax=30.0,
                    levels=[10, 20, 30], linestyles='solid', colors='white', linewidths=1.0)
    ax.clabel(im, inline=True, inline_spacing=0, fmt='%1.f', fontsize=12)

    # Option 3
    # im = ax.imshow(gain[:, :, 51, 0], origin='lower', extent=[0, 140, 0, 100],
    #                cmap='gray', vmin=0.0, vmax=30.0, interpolation='bilinear', alpha=0.50)

    plt.show()

