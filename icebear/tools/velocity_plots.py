import numpy as np
import matplotlib.pyplot as plt
import h5py
import pymap3d as pm
import icebear.utils as utils


def map_target(tx, rx, az, el, rf, dop, wavelength):
    """
    Find the scatter location given tx location, rx location, total rf distance, and target angle-of-arrival
    using the 'WGS84' Earth model. Also determines the bistatic velocity vector and bistatic radar wavelength.

    Parameters
    ----------
        tx : float np.array
            [latitude, longitude, altitude] of tx array in degrees and kilometers
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and kilometers
        az : float np.array
            angle-of-arrival azimuth in degrees
        el : float np.array
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau in kilometers
        dop : float np.array
            doppler shift in hertz
        wavelength : float
            radar signal center wavelength

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and kilometers
        sa : float np.array
            [azimuth, elevation, slant range] of scatter in degrees and kilometers
        sv : float np.array
            [azimuth, elevation, velocity] the bistatic Doppler velocity vector in degrees and kilometers.
            Coordinates given in the scattering targets local frame (azimuth from North, elevation up from
            the plane normal to zenith, Doppler [Hz] * lambda / (2 cos(e/2)) )

    Notes
    -----
    tx : transmitter location
    rx : receiver location
    sx : scatter location
    gx : geometric center of Earth, origin
    u_rt : unit vector rx to tx
    u_rs : unit vector rx to sx
    u_gt : unit vector gx to tx
    u_gr : unit vector gx to rx
    u_gs : unit vector gx to sx
    """

    # Initialize output arrays
    sx = np.zeros((3, len(rf)), dtype=float)
    sa = np.zeros((3, len(rf)), dtype=float)
    sv = np.zeros((3, len(rf)), dtype=float)

    # Setup variables in correct units for pymap3d
    rf = rf * 1.0e3
    az = np.where(az < 0.0, az + 360.0, az)
    az = np.deg2rad(az)
    el = np.deg2rad(np.abs(el))

    # Determine the slant range, r
    bx1, by1, bz1 = pm.geodetic2ecef(rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    v_gr = np.array([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    v_gt = np.array([bx2, by2, bz2])
    raz, rel, b = pm.ecef2aer(bx2, by2, bz2, rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_rt = np.array([np.sin(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.cos(np.deg2rad(raz)) * np.cos(np.deg2rad(rel)),
                     np.sin(np.deg2rad(rel))])
    el -= relaxation_elevation(el, rf, az, b, u_rt)
    u_rs = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.dot(u_rt, u_rs)))

    # WGS84 Model for lat, long, alt
    sx[:, :] = pm.aer2geodetic(np.rad2deg(az), np.rad2deg(el), np.abs(r),
                         np.repeat(rx[0], len(az)),
                         np.repeat(rx[1], len(az)),
                         np.repeat(rx[2], len(az)),
                         ell=pm.Ellipsoid("wgs84"), deg=True)

    # Determine the bistatic Doppler velocity vector
    x, y, z = pm.geodetic2ecef(sx[0, :], sx[1, :], sx[2, :],
                               ell=pm.Ellipsoid('wgs84'), deg=True)
    v_gs = np.array([x, y, z])
    v_bi = (-1 * v_gs.T + v_gt / 2.0 + v_gr / 2.0).T
    u_bi = v_bi / np.linalg.norm(v_bi, axis=0)
    v_sr = (v_gr - v_gs.T).T
    u_sr = v_sr / np.linalg.norm(v_sr, axis=0)
    radar_wavelength = wavelength / np.abs(2.0 * np.einsum('ij,ij->j', u_sr, u_bi))
    # doppler_sign = np.sign(dop)  # 1 for positive, -1 for negative, and 0 for zero
    doppler_sign = np.where(dop >= 0, 1, -1)  # 1 for positive, -1 for negative, and 0 for zero
    vaz, vel, _ = pm.ecef2aer(doppler_sign * u_bi[0, :] + x,
                              doppler_sign * u_bi[1, :] + y,
                              doppler_sign * u_bi[2, :] + z,
                              sx[0, :], sx[1, :], sx[2, :],
                              ell=pm.Ellipsoid("wgs84"), deg=True)

    # Convert back to conventional units
    sx[2, :] /= 1.0e3
    az = np.rad2deg(az)
    el = np.rad2deg(el)
    sa[:, :] = np.array([az, el, r / 1.0e3])
    sv[:, :] = np.array([vaz, vel, dop * radar_wavelength])

    return sx, sa, sv


def relaxation_elevation(beta, rf_distance, azimuth, bistatic_distance, bistatic_vector):
    """

    Parameters
    ----------
    beta : float np.array
        Measured elevation angle in degrees
    rf_distance :
    azimuth
    bistatic_distance
    bistatic_vector

    Returns
    -------

    """
    n = 0
    radius_of_earth = 6378.0e3
    err = np.deg2rad(1.0)
    target = np.deg2rad(0.1)
    m = np.zeros((3, len(beta)))
    m[1, :] = 0.1
    v = np.array([np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]), np.sin(beta - m[1, :])])
    r = (rf_distance ** 2 - bistatic_distance ** 2) / (2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
    m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
    while np.nanmean(err) > target:
        m[0, :] = m[1, :]
        m[1, :] = m[2, :]
        v = np.array([np.sin(azimuth) * np.cos(beta - m[1, :]), np.cos(azimuth) * np.cos(beta - m[1, :]), np.sin(beta - m[1, :])])
        r = (rf_distance ** 2 - bistatic_distance ** 2) / (2 * (rf_distance - bistatic_distance * (np.dot(bistatic_vector, v))))
        m[2, :] = 1 / (radius_of_earth / r + np.sin(beta) / 2)
        err = np.abs((m[1, :] - m[2, :]) ** 2 / (2 * m[1, :] - m[0, :] - m[2, :]))
        n += 1

    m[2, :] = np.where(err >= target, np.nan, m[2, :])
    print('\t-relaxation mean error:', np.rad2deg(np.nanmean(err)), 'iterations:', n)
    return m[2, :]


def velocity_plot(sx, sv, date, time, filepath):
    year = date[0]
    month = date[1]
    day = date[2]
    hour = time[0:2]
    minute = time[2:4]
    second = time[4:6]
    # import cartopy.crs as ccrs
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([np.min(sv[0, :]),np.max(sv[0, :]), np.min(sv[1, :]), np.max(sv[1, :])], crs=ccrs.PlateCarree())
    # ax.lakes
    # ax.coastline
    # ax.rivers

    u = np.sin(np.deg2rad(sv[0, :])) * np.cos(np.deg2rad(sv[1, :]))
    v = np.cos(np.deg2rad(sv[0, :])) * np.cos(np.deg2rad(sv[1, :]))
    n = np.sqrt(u**2 + v**2)
    u /= n
    v /= n
    # u = np.where(sv[2, :] == 0, np.nan, u)
    # v = np.where(sv[2, :] == 0, np.nan, v)

    # plt.figure(figsize=[18, 14])
    fig = plt.figure(figsize=[20, 14], constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    fig.suptitle(f'{year}-{month}-{day} {hour}:{minute}:{second}')
    props = dict(boxstyle='square', facecolor='wheat', alpha=1.0)
    vel_thresh = 150.0
    # Altitude slice
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('black')
    plt.scatter(sx[2, :], sx[0, :], c=sv[2, :], marker='D', cmap='RdBu', vmin=-vel_thresh, vmax=vel_thresh)
    plt.xlabel('Altitude [km]')
    plt.ylabel('Latitude [deg]')
    plt.clim(-vel_thresh, vel_thresh)
    plt.xlim([200.0, 0.0])
    plt.ylim([50.0, 62.0])
    plt.grid()

    # Azimuth slice
    ax2 = fig.add_subplot(gs[0, 1::])
    ax2.set_facecolor('black')
    plt.quiver(sx[1, :], sx[0, :], u, v, sv[2, :], cmap='RdBu')
    plt.xlabel('Longitude [deg]')
    plt.colorbar(label='Velocity [m/s]')
    plt.clim(-vel_thresh, vel_thresh)
    plt.scatter(sx[1, :], sx[0, :], c=sv[2, :], marker='D', cmap='RdBu', vmin=-vel_thresh, vmax=vel_thresh)
    plt.scatter(-109.403, 50.893, c='w')
    plt.annotate('TX', (-109.403, 50.893),
                 xytext=(-25.0, 15.0), textcoords='offset points', ha='center', va='bottom', color='red',
                 bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=1.0),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='r'))
    plt.scatter(-106.450, 52.243, c='w')
    plt.annotate('RX', (-106.450, 52.24),
                 xytext=(-25.0, 15.0), textcoords='offset points', ha='center', va='bottom', color='blue',
                 bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=1.0),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='b'))
    plt.text(-100.0, 50.5, f'Records {len(sx[1, :]):3d}\nSNR Cutoff 1.0 dB', bbox=props, color='k')
    plt.xlim([-114.0, -96.0])
    plt.ylim([50.0, 62.0])
    plt.grid()

    plt.savefig(filepath + f'velocity_{year}{month}{day}_{hour}{minute}{second}.png')
    plt.close()

    return


if __name__ == '__main__':
    # Pretty plot configuration.
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Load the level 2 data file.
    filepath = '/beaver/backup/level2b/'  # Enter file path to level 1 directory
    # filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
    files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_15')  # Enter first sub directory and last
    # files = utils.get_all_data_files(filepath, '2019_12_19', '2019_12_19')  # Enter first sub directory and last
    # files = utils.get_all_data_files(filepath, '2020_03_31', '2020_03_31')  # Enter first sub directory and last
    for file in files:
        f = h5py.File(file, 'r')
        print(file)
        group = f['data']
        date = f['date']
        keys = group.keys()
        for key in keys:
            data = group[f'{key}']
            # Filter out dropped samples
            if np.any(data['rf_distance'][()] < 250):
                continue
            t = np.ones(len(data['rf_distance'][()])) * int(key)
            rf_distance = data['rf_distance'][()]
            snr_db = np.abs(data['snr_db'][()])
            doppler_shift = data['doppler_shift'][()]
            azimuth = data['azimuth'][()]
            elevation = np.abs(data['elevation'][()])
            elevation_extent = data['elevation_extent'][()] / 4
            azimuth_extent = data['azimuth_extent'][()]
            area = data['area'][()]

            print(f'time {key}')
            print('\t-loading completed')
            azimuth += 7.0
            print('\t-total data', len(rf_distance))
            # Pre-masking
            m = np.ones_like(rf_distance)
            m = np.ma.masked_where(snr_db <= 1.0, m)  # Weak signals close to noise or highly multipathed (meteors are strong)
            # m = np.ma.masked_where(doppler_shift >= 50, m)  # Meteors are less than |150 m/s|
            # m = np.ma.masked_where(doppler_shift <= -50, m)  # Meteors are less than |150 m/s|
            # m = np.ma.masked_where(area >= 5.0, m)  # Meteors should have small scattering cross-sectional area

            rf_distance = rf_distance * m
            snr_db = snr_db * m
            doppler_shift = doppler_shift * m
            azimuth = azimuth * m
            azimuth_extent = azimuth_extent * m
            elevation = elevation * m
            elevation_extent = elevation_extent * m
            area = area * m
            t = t * m
            print('\t-pre-masking completed')

            sx, sa, sv = map_target([50.893, -109.403, 0.0],
                                    [52.243, -106.450, 0.0],
                                    azimuth, elevation, rf_distance,
                                    doppler_shift, 6.056)
            lat = sx[0, :]
            lon = sx[1, :]
            altitude = sx[2, :]
            vaz = sv[0, :]
            vel = sv[1, :]
            doppler_shift = sv[2, :]
            slant_range = sa[2, :]
            elevation = sa[1, :]
            print('\t-mapping completed')

            # Set up a filtering mask.
            m = np.ma.masked_where(elevation == np.nan, m)  # Elevation iterations not converging (noise)
            m = np.ma.masked_where(elevation <= 0.0, m)  # Elevation below the ground
            m = np.ma.masked_where(slant_range <= 300, m)  # Man made noise and multipath objects
            m = np.ma.masked_where(altitude <= 50, m)  # Man made noise and multipath objects
            # m = np.ma.masked_where(((slant_range <= 100.0) & (altitude <= 25.0)), m)  # Beam camping location
            rf_distance = rf_distance * m
            snr_db = snr_db * m
            doppler_shift = doppler_shift * m
            azimuth = azimuth * m
            azimuth_extent = azimuth_extent * m
            elevation = elevation * m
            elevation_extent = elevation_extent * m
            area = area * m
            slant_range = slant_range * m
            altitude = altitude * m
            t = t * m
            lat = lat * m
            lon = lon * m
            vaz = vaz * m
            vel = vel * m

            rf_distance = rf_distance[~rf_distance.mask].flatten()
            snr_db = snr_db[~snr_db.mask].flatten()
            doppler_shift = doppler_shift[~doppler_shift.mask].flatten()
            azimuth = azimuth[~azimuth.mask].flatten()
            azimuth_extent = azimuth_extent[~azimuth_extent.mask].flatten()
            elevation = elevation[~elevation.mask].flatten()
            elevation_extent = elevation_extent[~elevation_extent.mask].flatten()
            area = area[~area.mask].flatten()
            slant_range = slant_range[~slant_range.mask].flatten()
            altitude = altitude[~altitude.mask].flatten()
            t = t[~t.mask].flatten()
            lat = lat[~lat.mask].flatten()
            lon = lon[~lon.mask].flatten()
            vaz = vaz[~vaz.mask].flatten()
            vel = vel[~vel.mask].flatten()

            print('\t-masking completed')
            print('\t-remaining data', len(rf_distance))

            if len(rf_distance) > 0:
                velocity_plot(np.array([lat, lon, altitude]), np.array([vaz, vel, doppler_shift]), date, key, filepath + 'meteor_2020_12_12-15/')
            else:
                print('\t-0 records')

    exit()  # Needs a clean exit?
