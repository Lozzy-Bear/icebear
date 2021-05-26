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
    u_gr = np.array([bx1, by1, bz1]) / np.linalg.norm([bx1, by1, bz1])
    bx2, by2, bz2 = pm.geodetic2ecef(tx[0], tx[1], tx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
    u_gt = np.array([bx2, by2, bz2]) / np.linalg.norm([bx2, by2, bz2])
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
    u_gs = np.array([x, y, z]) / np.linalg.norm([x, y, z], axis=0)
    u_bi = (u_gs.T + u_gt / 2.0 + u_gr / 2.0).T
    radar_wavelength = wavelength / (2.0 * np.einsum('ij,ij->j', u_bi, -1 * u_rs))
    doppler_sign = np.sign(dop)
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
    sv[:, :] = np.array([vaz, vel, np.abs(dop * radar_wavelength)])

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
    bistatic_vector : float np.array
        unit vector from reciever to transmitter

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


def scatter_map(title, plot_num, masked_map, mlat_map, lon_range, lat_range, rx, tx, RF_lons, RF_lats, Mag_lons,
                Mag_lats, scatter_lon, scatter_lat, scatter_doppler, year, month, day, hours, minutes, seconds,
                RF_level, RF_contour, Mag_level, Mag_contour, loc):

    # setup figure
    data_crs = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.axes([0.025, 0.1, 0.8, 0.8], projection=ccrs.Mercator())
    ax.coastlines(resolution='10m', color='grey')
    ax.set_yticks(np.arange(-60, 80, 5), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 240, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent((np.min(lon_range), np.max(lon_range), np.min(lat_range - 0.15), np.max(lat_range)),
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='none', edgecolor='grey')
    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m'), facecolor='none',
                   edgecolor='black', linestyle=':')

    # Get figure boundaries
    pos = ax.get_position()
    l, b, w, h = pos.bounds

    # Aspect angle contour
    levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lons, lats = np.meshgrid(lon_range, lat_range)
    x, y = lons - 360, lats
    aspect_map = plt.contourf(x, y, masked_map, levels, cmap=matplotlib.cm.gray, alpha=0.4, transform=data_crs)
    cax = plt.axes([0.83, b, 0.01, h])
    cbar = plt.colorbar(drawedges=True, cax=cax)  # draw colorbar

    cbar.ax.set_ylabel("Magnetic Aspect Angle (Degrees)", fontsize=20)
    cax.yaxis.set_ticks_position('left')
    plt.axes(ax)

    # Magnetic latitude contour
    # lons,lats    = Mag_lons,Mag_lats
    # x,y          = lons-360,lats
    # distance_map = plt.contour(x[3:44][8:56], y[3:44][8:56], Mag_contour[3:44][8:56], Mag_level, linewidths = 0.3, colors = 'k', transform = data_crs)
    # plt.clabel(distance_map, inline=1, fmt = '%1.1f\u00b0', fontsize=12)

    # Radio frequency propagation contour  <--  needs fixing to show actual range for bistatic radar
    lons, lats = RF_lons, RF_lats
    x, y = lons - 360, lats
    distance_map = plt.contour(x[43:200][:], y[43:200][:], RF_contour[43:200][:], RF_level, linewidths=0.3, colors='m',
                               transform=data_crs)
    plt.clabel(distance_map, inline=1, fmt=' %i km ', fontsize=10.6)

    # Tx and Rx sites
    rxs_lon = []
    rxs_lat = []
    for rxs in rx:
        rxs_lon.append(rxs.getLongitude())
        rxs_lat.append(rxs.getLatitude())

    txs_lon = []
    txs_lat = []
    for txs in tx:
        txs_lon.append(txs.getLongitude())
        txs_lat.append(txs.getLatitude())

    plt.plot([rxs_lon[0] - 360], [rxs_lat], color='b', marker='o', transform=data_crs)
    plt.plot([txs_lon[0] - 360], [txs_lat], color='g', marker='o', transform=data_crs)

    # Plot the Rx FoV
    rx_beams_lon = np.zeros((200, 2), dtype=float)
    rx_beams_lat = np.zeros((200, 2), dtype=float)
    beam_dirs = np.zeros(2, dtype=float)
    beam_dirs = [-28.0, 42.0]
    for i in range(2):
        for j in range(200):
            rx_beams_lat[j, i] = math.asin(
                math.sin(rx_lat) * math.cos((j * 6.0) / Earth_radius) + math.cos(rx_lat) * math.sin(
                    (j * 6.0) / Earth_radius) * math.cos(math.radians(beam_dirs[i])))
            rx_beams_lon[j, i] = rx_lon + math.atan2(
                math.sin(math.radians(beam_dirs[i])) * math.sin((j * 6.0) / Earth_radius) * math.cos(rx_lat),
                math.cos((j * 6.0) / Earth_radius) - math.sin(rx_lat) * math.sin(rx_beams_lat[j, i]))
            rx_beams_lat[j, i] = math.degrees(rx_beams_lat[j, i])
            rx_beams_lon[j, i] = math.degrees(rx_beams_lon[j, i]) + 360.0

    for i in range(2):
        xd, yd = rx_beams_lon[:, i] - 360, rx_beams_lat[:, i]
        plt.plot(xd, yd, linewidth=3.0, color='b', transform=data_crs)

    # Plot the Tx FoV
    tx_beams_lon = np.zeros((200, 2), dtype=float)
    tx_beams_lat = np.zeros((200, 2), dtype=float)
    beam_dirs = np.zeros(2, dtype=float)
    beam_dirs = [-9.0, 41.0]
    for i in range(2):
        for j in range(200):
            tx_beams_lat[j, i] = math.asin(
                math.sin(tx_lat) * math.cos((j * 7.0) / Earth_radius) + math.cos(tx_lat) * math.sin(
                    (j * 7.0) / Earth_radius) * math.cos(math.radians(beam_dirs[i])))
            tx_beams_lon[j, i] = tx_lon + math.atan2(
                math.sin(math.radians(beam_dirs[i])) * math.sin((j * 7.0) / Earth_radius) * math.cos(tx_lat),
                math.cos((j * 7.0) / Earth_radius) - math.sin(tx_lat) * math.sin(tx_beams_lat[j, i]))
            tx_beams_lat[j, i] = math.degrees(tx_beams_lat[j, i])
            tx_beams_lon[j, i] = math.degrees(tx_beams_lon[j, i]) + 360.0

    for i in range(2):
        xd, yd = tx_beams_lon[:, i] - 360, tx_beams_lat[:, i]
        plt.plot(xd, yd, linewidth=3.0, color='g', transform=data_crs)

    # Plot the scatter Doppler
    x, y = scatter_lon - 360, scatter_lat
    plt.scatter(x, y, c=scatter_doppler, marker='.', cmap=matplotlib.cm.jet_r, edgecolors='none', vmin=-900.0,
                vmax=900.0, zorder=2, transform=data_crs)

    # Make legend and title
    tx_patch = mpatches.Patch(color='g', label='Tx')
    rx_patch = mpatches.Patch(color='b', label='Rx')
    plt.legend(handles=[tx_patch, rx_patch], loc=2)
    plt.title(title + 'Mapped ICEBEAR Results (%04d-%02d-%02d, %02d:%02d:%02d) (2015 IGRF, 100 km altitude)' % (
    year, month, day, hours, minutes, seconds), fontsize=16)  # <- center in figure not plot

    # Generate Colorbar for Doppler
    cax = plt.axes([0.93, b, 0.02, h])
    cbar = plt.colorbar(drawedges=True, cax=cax)  # draw colorbar
    cbar.ax.set_ylabel("Doppler Velocity (m/s)", fontsize=20)
    cax.yaxis.set_ticks_position('left')

    # print 'plotting...'
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.savefig(
        loc + 'map_%04d_%02d_%02d_%02d_%02d_%02d_plot_%04d.png' % (year, month, day, hours, minutes, seconds, plot_num))
    plt.close()

    return 0

