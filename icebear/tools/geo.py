import numpy as np
import nvector as nv
import h5py
import matplotlib.pyplot as plt
import icebear.utils as utils
import pymap3d as pm
import pyproj


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
    rf = rf * 1.0e3 * 1.5 - 230e3
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
    ua = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

    # Correct elevation using geocentric angle gamma and first order ranges, find scatter lat, long, alt
    for i in range(len(rf)):
        bx3, by3, bz3 = pm.aer2ecef(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)
        us[:, i] = np.array([bx3, by3, bz3]) / np.linalg.norm([bx3, by3, bz3])
        #el[i] -= np.arccos(ut[0]*us[0, i] + ut[1]*us[1, i] + ut[2]*us[2, i])
        el[i] -= np.arccos(ur[0] * us[0, i] + ur[1] * us[1, i] + ur[2] * us[2, i])
        sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    # Second order slant range, r
    ua = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    theta = np.arccos(ua[0, :] * ub[0] + ua[1, :] * ub[1] + ua[2, :] * ub[2])
    r = (rf ** 2 - b ** 2) / (2 * (rf - b * np.cos(theta)))

    for i in range(len(rf)):
        sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
                                rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    # Find the bistatic bisector velocity unit vector
    uv = (us + ua) / 2.0
    uv = uv / np.linalg.norm(uv)

    # Set units to degrees and kilometers
    sx[2, :] /= 1.0e3
    r /= 1.0e3

    # d = rf/2 - 200e3
    # r1 = np.sqrt((6378.1370e3 * np.cos(np.deg2rad(52.1579))) ** 2 + (6356.7523e3 * np.sin(np.deg2rad(52.1579))) ** 2)
    # pre_alt = np.sqrt(r1 ** 2 + (d) ** 2 - 2 * r1 * (d) * np.cos(np.pi/2 + el))
    # el -= np.arccos(((d) ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))

    # Find lat, long, alt of target
    # sx = np.zeros((3, len(rf)))
    # for i in range(len(rf)):
    #     sx[:, i] = pm.aer2geodetic(np.rad2deg(az[i]), np.rad2deg(el[i]), np.abs(r[i]),
    #                             rx[0], rx[1], rx[2], ell=pm.Ellipsoid("wgs84"), deg=True)

    #return sx, r, uv
    return sx[2, :], r


def plot_3d(az, rng, alt, dop):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(rng*np.sin(np.deg2rad(az)), rng*np.cos(np.deg2rad(az)),
                   alt,
                   c=dop, cmap='jet_r',  alpha=0.25, vmin=-500.0, vmax=500.0,)
    ax.set_xlabel('West - East [km]')
    ax.set_ylabel('North - South [km]')
    ax.set_zlabel('Altitude [km]')
    fig.colorbar(p, label='Doppler [Hz]')
    ax.set_xlim(-600, 600)
    ax.set_ylim(0, 1200)
    ax.set_zlim(0, 200)
    ax.view_init(elev=35.0, azim=225.0)
    return


def ecef_to_llh(x,y,z):
    llh = pyproj.Proj("epsg:4978")
    ecef = pyproj.Proj("epsg:4326")
    lat, lon, alt = pyproj.transform(llh, ecef, x, y, z, radians=True)
    return [lat,lon,alt]


def llh_to_ecef(lat_lon_alt_input):
    llh = pyproj.Proj("epsg:4978")
    ecef = pyproj.Proj("epsg:4326")
    x, y, z = pyproj.transform(ecef, llh, lat_lon_alt_input[0], lat_lon_alt_input[1], lat_lon_alt_input[2], radians=True)
    return [x/1000.0,y/1000.0,z/1000.0]


def scatter_location_determination(rx_lat_lon_alt,tx_lat_lon_alt,rf_path,azi,elev):
    azi = np.deg2rad(azi)
    elev = np.deg2rad(elev)
    rx_lat_lon_alt[0] = np.deg2rad(rx_lat_lon_alt[0])
    rx_lat_lon_alt[1] = np.deg2rad(rx_lat_lon_alt[1])
    tx_lat_lon_alt[0] = np.deg2rad(tx_lat_lon_alt[0])
    tx_lat_lon_alt[1] = np.deg2rad(tx_lat_lon_alt[1])
    rf_path = rf_path * 1.5 - 200
    djk = rf_path * 1000 / 2 - 100e3
    r1 = np.sqrt((6378.1370e3 * np.cos(np.deg2rad(52.1579))) ** 2 + (6356.7523e3 * np.sin(np.deg2rad(52.1579))) ** 2)
    pre_alt = np.sqrt(r1 ** 2 + (djk) ** 2 - 2 * r1 * (djk) * np.cos(np.pi / 2 + elev))
    elev -= np.arccos(((djk) ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))

    slat = np.sin(rx_lat_lon_alt[0])
    slon = np.sin(rx_lat_lon_alt[1])
    clat = np.cos(rx_lat_lon_alt[0])
    clon = np.cos(rx_lat_lon_alt[1])
    rx_x,rx_y,rx_z = llh_to_ecef(rx_lat_lon_alt)
    tx_x,tx_y,tx_z = llh_to_ecef(tx_lat_lon_alt)
    rx_tx_d = np.sqrt((rx_x-tx_x)**2+(rx_y-tx_y)**2+(rx_z-tx_z)**2)
    #determine if the receiver and transmitter are in the same location (monostatic vs bistatic)
    if rx_tx_d==0:
        r_rs_d = rf_path/2.0
        # if (elev<0).any():
        #     rx_re = np.sqrt(rx_x**2+rx_y**2+rx_z**2)
        #     elev = (np.arccos(((107.0+rx_re)**2 - rx_re**2 - r_rs_d**2)/(-2*rx_re*r_rs_d))-(np.pi/2))
    #if bistatic, use geometry to determine the distance from the receiver to the scattering location
    else:
        #find northward pointing vector from transmitter location
        tx_north_lat_lon_alt=np.asarray(tx_lat_lon_alt)
        tx_north_lat_lon_alt[0]=tx_north_lat_lon_alt[0]+0.05
        tx_north_x,tx_north_y,tx_north_z = llh_to_ecef(tx_north_lat_lon_alt)
        tx_north_x=tx_north_x-tx_x
        tx_north_y=tx_north_y-tx_y
        tx_north_z=tx_north_z-tx_z
        #find vector from transmitter to receiver
        tx_rx_x=rx_x-tx_x
        tx_rx_y=rx_y-tx_y
        tx_rx_z=rx_z-tx_z
        #find angle from north between transmitter and receiver
        tx_to_rx_angle = np.arccos(np.dot([tx_north_x,tx_north_y,tx_north_z],[tx_rx_x,tx_rx_y,tx_rx_z])/(np.sqrt(tx_north_x**2+tx_north_y**2+tx_north_z**2)*np.sqrt(tx_rx_x**2+tx_rx_y**2+tx_rx_z**2)))
        #determine if elevation data was included in the input and make approximate guess for elevation angle if not
        # if (elev<0).any():
        #     rx_re = np.sqrt(rx_x**2+rx_y**2+rx_z**2)
        #     r_rs_temp = (rf_path**2-rx_tx_d**2)/(2*(rf_path-rx_tx_d*np.cos(np.pi-(tx_to_rx_angle-azi))))
        #     elev = (np.arccos(((107.0+rx_re)**2 - rx_re**2 - r_rs_temp**2)/(-2*rx_re*r_rs_temp))-(np.pi/2))
        r_rs = np.zeros((len(elev),3),dtype=np.float32)
        south = -np.cos(azi)*np.cos(elev)
        east = np.sin(azi)*np.cos(elev)
        zenith = np.sin(elev)
        r_rs[:,0] = (slat*clon*south)+(-slon*east)+(clat*clon*zenith)
        r_rs[:,1] = ( slat * slon * south) + ( clon * east) + (clat * slon * zenith)
        r_rs[:,2] = (-clat *        south) + ( slat * zenith)
        r_rt = [(rx_x-tx_x)/rx_tx_d,(rx_y-tx_y)/rx_tx_d,(rx_z-tx_z)/rx_tx_d]
        cos_theta_factor = r_rs[:,0]*r_rt[0]+r_rs[:,1]*r_rt[1]+r_rs[:,2]*r_rt[2]
        cos_theta_factor = np.cos(np.pi+np.arccos(cos_theta_factor))
        r_ts_d = (rx_tx_d**2+rf_path**2-(2*rx_tx_d*rf_path*cos_theta_factor))/(2*rf_path-(2*rx_tx_d*cos_theta_factor))
        r_rs_d = rf_path-r_ts_d
    scat_xyz = np.zeros((len(elev),3),dtype=np.float32)
    south = -np.cos(azi)*np.cos(elev)*r_rs_d
    east = np.sin(azi)*np.cos(elev)*r_rs_d
    zenith = np.sin(elev)*r_rs_d
    scat_xyz[:,0] = rx_x+(slat*clon*south)+(-slon*east)+(clat*clon*zenith)
    scat_xyz[:,1] = rx_y+( slat * slon * south) + ( clon * east) + (clat * slon * zenith)
    scat_xyz[:,2] = rx_z+(-clat * south) + ( slat * zenith)
    scat_lat_lon_alt = ecef_to_llh(scat_xyz[:,0]*1000.0,scat_xyz[:,1]*1000.0,scat_xyz[:,2]*1000.0)
    scat_lat_lon_alt_conv = np.zeros((len(elev),4),dtype=np.float32)
    scat_lat_lon_alt_conv[:,0] = (scat_lat_lon_alt[:][0])*180/np.pi
    scat_lat_lon_alt_conv[:,1] = (scat_lat_lon_alt[:][1])*180/np.pi
    scat_lat_lon_alt_conv[:,2] = scat_lat_lon_alt[:][2]/1000.0
    scat_lat_lon_alt_conv[:,3] = r_rs_d
    return scat_lat_lon_alt[:][2]/1000.0, r_rs_d


def bistatic(tx, rx, az, el, rf):
    """
    Find the scatter location given tx location, rx, location, total rf distance, and target angle-of-arrival.

    Parameters
    ----------
        tx
            [latitude, longitude, altitude] of tx array in degrees and meters
        rx : float np.array
            [latitude, longitude, altitude] of rx array in degrees and meters
        az : float
            angle-of-arrival azimuth in degrees
        el : float
            angle-of-arrival elevation in degrees
        rf : float np.array
            total rf path distance rf = c * tau

    Returns
    -------
        sx : float np.array
            [latitude, longitude, altitude] of scatter in degrees and meters
    """
    rf *= 1000 * 1.5
    rf -= 200e3
    az = np.deg2rad(az + 7)
    el = np.deg2rad(el)
    a = 6378.1370e3
    b = 6356.7523e3
    p1 = np.deg2rad(52.1579)
    r1 = np.sqrt((a * np.cos(p1)) ** 2 + (b * np.sin(p1)) ** 2)
    rpre = rf/2
    pre_alt = np.sqrt(r1 ** 2 + rpre ** 2 - 2 * r1 * rpre * np.cos(np.pi / 2 + el))
    el -= np.arccos((rpre ** 2 - (r1 ** 2) - (pre_alt ** 2)) / (-2 * r1 * pre_alt))

    wgs84 = nv.FrameE(name='wgs84')
    tx = wgs84.GeoPoint(latitude=tx[0], longitude=tx[1], z=tx[2], degrees=True)
    rx = wgs84.GeoPoint(latitude=rx[0], longitude=rx[1], z=rx[2], degrees=True)

    seg_rxtx = rx.delta_to(tx)
    uax, uay, uaz = seg_rxtx.pvector.ravel()
    au = np.linalg.norm([uax, uay, uaz])
    ua = np.array([uax, uay, uaz]) / au
    urx = np.sin(az) * np.cos(el)
    ury = np.cos(az) * np.cos(el)
    urz = np.sin(el)
    ur = np.array([urx, ury, urz])
    theta = np.zeros_like(rf, dtype=np.float)
    for i in range(len(rf)):
        #theta[i] = np.arccos(np.dot(ur[:, i], ua))
        theta[i] = np.arccos(ur[0, i]*ua[0] + ur[1, i]*ua[1] + ur[2, i]*ua[2])
    #theta = np.arccos(np.dot(ur, ua))
    r = (rf**2 - au**2) / (2 * (rf - au*np.cos(theta)))
    print(np.max(r))
    alt = -r1 + np.sqrt(r1 ** 2 + r ** 2 + 2 * r1 * r * np.sin(el))
    r /= 1000
    alt /= 1000
    #sx, _ = rx.displace(distance=r, azimuth=az)
    #alt_geocentric -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))
    #print(r/1000, '[km]')
    # Convert all points to ECEF (earth centered earth fixed) coordinates.
    #sx = np.zeros_like(rx)
    # Find distance d (distance from rx to sx along the ground) and distance r (distance from rx to sx).
    #d =
    #r =
    return r, alt


if __name__ == '__main__':
    filepath = 'E:/icebear/level2b/'  # Enter file path to level 1 directory
    files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_12')  # Enter first sub directory and last
    el = np.array([])
    rng = np.array([])
    dop = np.array([])
    snr = np.array([])
    az = np.array([])

    for file in files:
        f = h5py.File(file, 'r')
        print(file)
        group = f['data']
        keys = group.keys()

        for key in keys:
            if f'{key}' == '052138000':
                break
            data = group[f'{key}']
            rf_distance = data['rf_distance'][()]
            snr_db = data['snr_db'][()]
            doppler_shift = data['doppler_shift'][()]
            azimuth = data['azimuth'][()]
            elevation = data['elevation'][()]
            #elevation_spread = data['elevation_extent'][()]
            #azimuth_spread = data['azimuth_extent'][()]
            area = data['area'][()]

            rng = np.append(rng, rf_distance)
            el = np.append(el, elevation)
            dop = np.append(dop, doppler_shift)
            snr = np.append(snr, snr_db)
            az = np.append(az, azimuth)

    snr = np.abs(snr)
    # az = np.abs(az)
    # el += 90
    # el = np.abs(el)
    m = np.ones_like(rng)
    m = np.ma.masked_where(dop > 20, m)
    m = np.ma.masked_where(dop < -20, m)
    m = np.ma.masked_where(snr <= 6.0, m)
    # m = np.ma.masked_where(az >= 315, m)
    # m = np.ma.masked_where(az <= 225, m)
    #m = np.ma.masked_where(el >= 30, m)
    #m = np.ma.masked_where(el <= 1, m)


    alt_bistatic, rb = map_target([50.893, -109.403, 0.0], [52.243, -106.450, 0.0], np.copy(az), np.copy(el), np.copy(rng))
    #alt_bistatic, rb = scatter_location_determination([52.243, -106.450, 0.0], [50.893, -109.403, 0.0], np.copy(rng), np.copy(az), np.copy(el))

    rng = rng * 0.75 - 200
    #m = np.ma.masked_where(rng <= 300, m)
    #m = np.ma.masked_where(rng >= 1200, m)

    re = 6378.0
    a = 6378.1370
    b = 6356.7523
    p1 = np.deg2rad(52.1579)
    r1 = np.sqrt((a*np.cos(p1))**2 + (b*np.sin(p1))**2)
    pre_alt = np.sqrt(re ** 2 + rng ** 2 - 2 * re * rng * np.cos(np.deg2rad(90 + np.abs(el))))
    gamma = np.rad2deg(np.arccos((rng ** 2 - (re ** 2) - (pre_alt ** 2)) / (-2 * re * pre_alt)))
    p2 = p1 + gamma
    r2 = np.sqrt((a*np.cos(p2))**2 + (b*np.sin(p2))**2)

    alt_geocentric = -re + np.sqrt(re ** 2 + rng ** 2 + 2 * re * rng * np.sin(np.deg2rad(el - gamma)))
    alt_geocentric -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))

    alt_normal = -re + np.sqrt(re**2 + rng**2 + 2 * re * rng * np.sin(np.deg2rad(el)))
    alt_normal -= np.tan(np.deg2rad(2.8624)) * (400 + rng * np.sin(np.deg2rad(az)))

    rng = rng * m
    rb = rb * m
    dop = dop * m
    snr = snr * m
    az = az * m
    el = el * m
    alt_geocentric = alt_geocentric * m
    alt_normal = alt_normal * m
    alt_bistatic = alt_bistatic * m

    plot_3d(az, rb, alt_bistatic, dop)

    plt.figure()
    plt.scatter(rng, alt_normal, c='y', label='No Correction')
    plt.scatter(rng, alt_geocentric, c='r', label='Geocentric Correction')
    plt.scatter(rb, alt_bistatic, c='k', label='Geocentric with Bistatic Correction')
    plt.legend(loc='best')
    plt.xlabel('Range [km]')
    plt.ylabel('Altitude [km]')

    plt.figure()
    plt.scatter(az, alt_normal, c='y', label='No Correction')
    plt.scatter(az, alt_geocentric, c='r', label='Geocentric Correction')
    plt.scatter(az, alt_bistatic, c='k', label='Geocentric with Bistatic Correction')
    plt.legend(loc='best')
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Altitude [km]')

    plt.show()
