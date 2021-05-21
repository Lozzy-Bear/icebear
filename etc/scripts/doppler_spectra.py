import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm


def map_target(tx, rx, az, el, rf):
    """
    Find the scatter location given tx location, rx, location, total rf distance, and target angle-of-arrival.

    Parameters
    ----------
        tx : float np.array
            [latitude, longitude, altitude] of tx array in degrees and kilometers
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and kilometers
        az : float
            angle-of-arrival azimuth in degrees
        el : float
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        r : float
            bistatic slant range in kilometers
    """

    # Setup givens in correct units
    rf = rf * 1.0e3 * 1.5 - 230e3  # Assumed rf propagation correction in km
    az = np.where(az < 0, np.deg2rad(az + 367.0), np.deg2rad(az + 7.0))
    el = np.deg2rad(np.abs(el))
    sx = np.zeros((3, len(rf)))
    uv = np.zeros((3, len(rf)))
    us = np.zeros((3, len(rf)))

    # Determine the slant range, r
    bx1, by1, bz1 = pm.geodetic2ecef(rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    ur = np.array([bx1, by1, bz1]) / np.linalg.norm([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    ut = np.array([bx2, by2, bz2]) / np.linalg.norm([bx2, by2, bz2])
    bx = bx2 - bx1
    by = by2 - by1
    bz = bz2 - bz1
    b = np.linalg.norm([bx, by, bz])
    ub = np.array([bx, by, bz]) / b

    dr = np.ones(1000)
    dtheta = np.ones(1000)
    dr[0] = 1e6
    dtheta[0] = 1e6
    r2 = np.copy(rf)
    theta2 = np.ones(len(az)) * np.pi
    cnt = 0
    while np.max(dtheta[cnt]) >= 0.002 and dr[cnt] >= 1500:
        ua = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
        theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
        r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

        # Correct elevation using geocentric angle gamma and first order ranges, find scatter lat, long, alt
        for i in range(len(rf)):
            bx3, by3, bz3 = pm.aer2ecef(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                    rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
            us[:, i] = np.array([bx3, by3, bz3]) / np.linalg.norm([bx3, by3, bz3])
            el[i] -= np.arccos(ur[0] * us[0, i] + ur[1] * us[1, i] + ur[2] * us[2, i])
            sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                    rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

        dr[cnt + 1] = np.max(np.abs(r - r2))
        dtheta[cnt + 1] = np.max(np.abs(theta - theta2))

        cnt += 1
        r2 = r
        theta2 = theta

    # Find the bistatic bisector velocity unit vector
    uv = (us + ua) / 2.0
    uv = uv / np.linalg.norm(uv)

    # Set units to degrees and kilometers
    sx[2, :] /= 1.0e3
    r /= 1.0e3

    return sx[2, :], r, dr[0:cnt], dtheta[0:cnt]


if __name__ == '__main__':
    # Pretty plot configuration.
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

    # Load the level 2 data file.
    files = ['/beaver/backup/level2b/2020_03_31/ib3d_normal_swht_01deg_2020_03_31_02_prelate_bakker.h5']
    cnt = 0
    for file in files:
        f = h5py.File(file, 'r')
        print(file)
        group = f['data']
        date = f['date']
        keys = group.keys()
        for key in keys:
            data = group[f'{key}']
            time = data['time'][()]
            time = np.append(date, time)
            rf_distance = data['rf_distance'][()]
            snr_db = np.abs(data['snr_db'][()])
            doppler_shift = data['doppler_shift'][()]
            azimuth = data['azimuth'][()]
            elevation = np.abs(data['elevation'][()])
            elevation_extent = data['elevation_extent'][()] / 4  # Todo: May need to scale values!
            azimuth_extent = data['azimuth_extent'][()]
            area = data['area'][()]
            
            # Pre-calculate and do altitude earth curvature corrections.
            slant_range = rf_distance * 0.75 - 250
            re = 6378.0
            pre_alt = np.sqrt(re ** 2 + slant_range ** 2 - 2 * re * slant_range * np.cos(np.deg2rad(90 + np.abs(elevation))))
            gamma = np.rad2deg(np.arccos((slant_range ** 2 - (re ** 2) - (pre_alt ** 2)) / (-2 * re * pre_alt)))
            elevation -= gamma
            altitude = -re + np.sqrt(re ** 2 + slant_range ** 2 + 2 * re * slant_range * np.sin(np.deg2rad(elevation)))
            altitude -= np.tan(np.deg2rad(2.8624)) * (400 + slant_range * np.sin(np.deg2rad(azimuth)))

            # Set up a filtering mask.
            m = np.ones_like(slant_range)
            # m = np.ma.masked_where(snr_db <= 3.0, m)
            # m = np.ma.masked_where(azimuth >= 315, m)
            # m = np.ma.masked_where(azimuth <= 225, m)
            m = np.ma.masked_where(elevation >= 26, m)
            m = np.ma.masked_where(elevation <= 1, m)
            m = np.ma.masked_where(slant_range <= 300, m)
            m = np.ma.masked_where(slant_range >= 1200, m)
            m = np.ma.masked_where(altitude <= 70, m)
            m = np.ma.masked_where(altitude >= 130, m)
            slant_range = slant_range * m
            doppler_shift = doppler_shift * m
            snr_db = snr_db * m
            azimuth = azimuth * m
            azimuth_extent = azimuth_extent * m
            elevation = elevation * m
            elevation_extent = elevation_extent * m
            altitude = altitude * m
            slant_range = slant_range[~slant_range.mask]
            doppler_shift = doppler_shift[~doppler_shift.mask]
            snr_db = snr_db[~snr_db.mask]
            azimuth = azimuth[~azimuth.mask]
            azimuth_extent = azimuth_extent[~azimuth_extent.mask]
            elevation = elevation[~elevation.mask]
            elevation_extent = elevation_extent[~elevation_extent.mask]
            altitude = altitude[~altitude .mask]

            for rng in np.unique(slant_range):
                dop = doppler_shift[slant_range == rng]
                if len(dop) < 20:
                    continue
                snr = snr_db[slant_range == rng]
                az = azimuth[slant_range == rng]
                az_extent = azimuth_extent[slant_range == rng]
                el = elevation[slant_range == rng]
                el_extent = elevation_extent[slant_range == rng]
                alt = altitude[slant_range == rng]

                resolution = 1.0
                fov = 25
                alt_bins = np.arange(70, 160, 5.0)
                az_bins = np.arange(-fov, fov+resolution, resolution)
                el_bins = np.arange(0, fov+resolution, resolution)
                dop_bins = np.arange(-500, 500+10, 10)

                # D = np.zeros((len(el_bins), len(az_bins), 2))
                # C = np.ones((len(el_bins), len(az_bins)))
                # for i in range(len(dop)):
                #     dop_index = np.argwhere(dop_bins == dop[i])[0, 0]
                #     az_index = np.argwhere(az_bins == az[i])
                #     el_index = np.argwhere(el_bins == el[i])
                #     # Form an ellipse and gaussian given spatial extents
                #     xgrid, ygrid = np.meshgrid(az_bins, el_bins)
                #     gaussian = snr[i] * np.exp(-1 * ((xgrid - az[i]) ** 2 / (2 * az_extent[i] ** 2) +
                #                                      (ygrid - el[i]) ** 2 / (2 * el_extent[i] ** 2)))
                #     ellipse = ((xgrid - az[i]) / az_extent[i]) ** 2 + ((ygrid - el[i]) / el_extent[i]) ** 2 <= 1.0
                #     C += ellipse
                #     ellipse = ellipse * dop[i]
                #     D[:, :, 0] += gaussian / len(dop)
                #     D[:, :, 1] += ellipse

                # D[:, :, 1] = D[:, :, 1] / C
                # # Heat SNR + Doppler Contours
                # zgrid = griddata((xgrid.flatten(), ygrid.flatten()), D[:, :, 1].flatten(), (xgrid, ygrid), method='cubic')
                # zgrid = np.around(zgrid, -1)
                # fig, ax = plt.subplots()
                # PC = ax.pcolormesh(xgrid, ygrid, D[:, :, 0], shading='auto', cmap='inferno')
                # fig.colorbar(PC, label='SNR [db]')
                # CS = ax.contour(xgrid, ygrid, zgrid, cmap='jet')
                # # CS.levels = np.unique(zgrid.flatten())
                # # ax.clabel(CS, CS.levels, inline=True)
                # fig.colorbar(CS, label='Doppler [Hz]')
                # plt.xlabel('Azimuth [deg]')
                # plt.ylabel('Elevation [deg]')
                # plt.title(f'time:{time} rng:{rng:.2f}')
                # plt.show()

                D = np.zeros((len(alt_bins), len(az_bins), len(dop_bins)))
                for i in range(len(dop)):
                    dop_index = np.argwhere(dop_bins == dop[i])[0, 0]
                    az_index = np.argwhere(az_bins == az[i])
                    el_index = np.argwhere(el_bins == el[i])
                    # Form an ellipse and gaussian given spatial extents
                    xgrid, ygrid = np.meshgrid(az_bins, alt_bins)
                    gaussian = snr[i] * np.exp(-1 * ((xgrid - az[i]) ** 2 / (2 * az_extent[i] ** 2) +
                                                     (ygrid - alt[i]) ** 2 / (2 * (rng * np.sin(np.deg2rad(el_extent[i]/2))) ** 2)))
                    D[:, :, dop_index] = gaussian

                # Azimuth slices plots
                agrid, dgrid = np.meshgrid(alt_bins, dop_bins)
                for i in range(len(az_bins)):
                    if np.any(D[:, i, :] > 0.0):
                        plt.figure(1)
                        plt.pcolormesh(dgrid, agrid, D[:, i, :].T, shading='auto', vmin=0.0, vmax=6.0)
                        plt.title(f'range:{rng:.2f} azimuth:{az_bins[i]:.2f}')
                        plt.xlabel('Doppler [Hz]')
                        plt.ylabel('Altitude [km]')
                        plt.colorbar(label='SNR [dB]')
                        plt.savefig(f'/beaver/backup/temp2/2d_dop_{cnt}.png')
                        plt.close(1)
                        cnt += 1

                # Scatter plots
                # for j in range(0, 260, 1):#len(el_bins)):
                #     plt.figure(1)
                #     plt.title(f'az:{el_bins[j]:.2f} rng:{rng:.2f}')
                #     plt.xlabel('Doppler [Hz]')
                #     plt.ylabel('SNR [dB]')
                #     for i in range(len(az_bins)):
                #         plt.scatter(dop_bins, D[j, i, :], c='k')
                #
                #     plt.savefig(f'/beaver/backup/temp2/2d_dop_{cnt}.png')
                #     plt.close()
                #     cnt += 1
