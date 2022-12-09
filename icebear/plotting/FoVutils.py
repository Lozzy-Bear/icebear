# plotting & vizualization commands for ICEBEAR experiment ib_main.py


import matplotlib.cm
import matplotlib as mpl
import math
from icebear.plotting import AspectMapper
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import configparser as ConfigParser

# Tx Rx locations
Earth_radius = 6378.1
rx_lat = math.radians(52.24)
rx_lon = math.radians(253.55 - 360.00)
tx_lat = math.radians(50.893)
tx_lon = math.radians(-109.403)

rx_tx_bearing = math.radians(126.27)
rx_tx_distance_arc = 253.21 / Earth_radius

# Predefined input values for command line
input_file = 'canada_gh.ini'
igrf_year = 2015.0


def getArrayList(parser, section, key):
    """
       Convert an ini file array of form [val,val,val...] to a python list.
   """
    val = parser.get(section, key)
    val2 = val.strip('[]')  # Remove delimiters
    return val2.split(',')  # Split by commas


def getFloatArrayList(parser, section, key):
    """
      Convert an ini file array of form [float,float,float...] to a
        floating point numpy array. This can then be reshaped if needed.
   """
    sa = getArrayList(parser, section, key)

    return np.array(sa, dtype=np.float64)


def load_config(input_file, igrf_year):
    """
      Uses getArrayList and getFloatArrayList to parse through in input files
        and reutrns radar parameters
   """
    # Parse the configuration file
    parser = ConfigParser.SafeConfigParser(strict=False)
    parser.read(input_file)

    # Pull map information
    title_str = parser.get('map', 'title') + " [%4.2f]" % (igrf_year)
    extent = getFloatArrayList(parser, 'map', 'extent')
    spacing = getFloatArrayList(parser, 'map', 'spacing')

    # Pull FOV information
    altitude = float(parser.get('map', 'altitude'))
    fov_limit = float(parser.get('map', 'fov_limit'))
    aspect_limit = float(parser.get('map', 'aspect_limit'))

    # Pull site lists
    tx_sites = getArrayList(parser, 'sites', 'tx')
    rx_sites = getArrayList(parser, 'sites', 'rx')

    tx = []

    for tx_name in tx_sites:
        tx_point = getFloatArrayList(parser, 'locations', tx_name)
        tx_loc = AspectMapper.Location(tx_point[0], tx_point[1], tx_point[2])
        tx.append(tx_loc)

    rx = []

    for rx_name in rx_sites:
        rx_point = getFloatArrayList(parser, 'locations', rx_name)
        rx_loc = AspectMapper.Location(rx_point[0], rx_point[1], rx_point[2])
        rx.append(rx_loc)

    lat_range = np.arange(extent[0], extent[1], spacing[0], 'f')
    lon_range = np.arange(extent[2], extent[3], spacing[1], 'f')

    # Pull desired correlation list
    corr_num = getFloatArrayList(parser, 'corr', 'num')

    return spacing, altitude, fov_limit, aspect_limit, tx, rx, tx_sites, rx_sites, lat_range, lon_range, corr_num


def contour_RF(Earth_radius, tx_lat, tx_lon, rx_lat, rx_lon):
    """
      Calculates the Radio Frequency contours for the FoV plot
   """
    var_a1 = 0.0
    var_a2 = 0.0
    var_c1 = 0.0
    var_c2 = 0.0
    mid_lon = 0.0
    mid_lat = 0.0
    distance_map_lat = np.zeros(200)
    distance_map_lon = np.zeros(320)
    distance_contours = np.zeros((200, 320), dtype=float)
    for i in range(320):
        distance_map_lon[i] = -90.0 - (i / 10.0) + 360.0
    for j in range(200):
        distance_map_lat[j] = 45.0 + (j / 10.0)
    for i in range(320):
        for j in range(200):
            mid_lat = math.radians(45.0 + (j / 10.0))
            mid_lon = math.radians(-90.0 - (i / 10.0))
            var_a1 = math.sin((mid_lat - tx_lat) / 2.0) * math.sin((mid_lat - tx_lat) / 2.0) + (
                        math.cos(tx_lat) * math.cos(mid_lat) * math.sin((mid_lon - tx_lon) / 2.0) * math.sin(
                    (mid_lon - tx_lon) / 2.0))
            var_a2 = math.sin((rx_lat - mid_lat) / 2.0) * math.sin((rx_lat - mid_lat) / 2.0) + (
                        math.cos(rx_lat) * math.cos(mid_lat) * math.sin((rx_lon - mid_lon) / 2.0) * math.sin(
                    (rx_lon - mid_lon) / 2.0))
            var_c1 = 2.0 * math.atan2(math.sqrt(var_a1), math.sqrt(1.0 - var_a1))
            var_c2 = 2.0 * math.atan2(math.sqrt(var_a2), math.sqrt(1.0 - var_a2))
            distance_contours[j, i] = Earth_radius * ((var_c1) + (var_c2)) / 2.0

    levels = [200, 400, 600, 800, 1000, 1200]
    lons, lats = np.meshgrid(distance_map_lon, distance_map_lat)
    return lons, lats, levels, distance_contours


def contour_Mag(mlat_map):
    """
      Calculates the Magnetic contours for the FoV plot
   """
    var_a1 = 0.0
    var_a2 = 0.0
    var_c1 = 0.0
    var_c2 = 0.0
    mid_lon = 0.0
    mid_lat = 0.0
    mlat_map_lat = np.zeros(50)
    mlat_map_lon = np.zeros(65)
    mlat_contours = np.zeros((50, 65), dtype=float)
    for i in range(65):
        mlat_map_lon[i] = -91.0 - (i / 2.0) + 360.0
    for j in range(50):
        mlat_map_lat[j] = 45.0 + (j / 2.0)
    for i in range(65):
        for j in range(50):
            igrf_mag_field = mlat_map.B(100.0, mlat_map_lat[j], mlat_map_lon[i])
            mlat_contours[j, i] = math.atan2(igrf_mag_field[0][2], math.sqrt(
                igrf_mag_field[0][1] * igrf_mag_field[0][1] + igrf_mag_field[0][0] * igrf_mag_field[0][
                    0])) * 180.0 / math.pi

    levels = [74.0, 76.0, 78.0, 80.0, 82.0, 84.0]
    lons, lats = np.meshgrid(mlat_map_lon, mlat_map_lat)
    return lons, lats, levels, mlat_contours


def scatter_calc(angle, Earth_radius, rf_dist, rx_lat, rx_lon, doppler):
    lam = 6.06
    scatter_lat = np.zeros(len(rf_dist))
    scatter_lon = np.zeros(len(rf_dist))
    scatter_doppler = np.zeros(len(rf_dist))

    for i in range(0, len(rf_dist)):
        # dist = (rf_dist[i] / 2 - 30)
        # bearing = math.radians(7.0) + angle

        rx_angle = rx_tx_bearing + math.radians(7.0) + angle[i]
        range_factor = (rf_dist[i] - (30 * 1.5)) / Earth_radius

        # find cosine law factors
        factor_1 = math.cos(range_factor) * math.cos(rx_tx_distance_arc) + math.sin(range_factor) * math.sin(
            rx_tx_distance_arc) * math.cos(rx_angle)
        factor_2 = math.sin(range_factor) * math.cos(rx_tx_distance_arc) - math.cos(range_factor) * math.sin(
            rx_tx_distance_arc) * math.cos(rx_angle)

        # calculate the distance from rx to scattering volume
        tx_scatter_arc = math.atan2((1.0 - factor_1), factor_2)
        rx_scatter_d = Earth_radius * (range_factor - tx_scatter_arc)

        scatter_bearing = math.radians(7.0) + angle[i]

        scatter_lat_a = math.asin(
            math.sin(rx_lat) * math.cos(rx_scatter_d / Earth_radius) + math.cos(rx_lat) * math.sin(
                rx_scatter_d / Earth_radius) * math.cos(scatter_bearing))
        scatter_lon_a = rx_lon + math.atan2(
            math.sin(scatter_bearing) * math.sin(rx_scatter_d / Earth_radius) * math.cos(rx_lat),
            math.cos(rx_scatter_d / Earth_radius) - math.sin(rx_lat) * math.sin(scatter_lat_a))

        # scattter_lat[i] = math.asin(math.sin(rx_lat)*math.cos(dist/Earth_radius)+math.cos(rx_lat)*math.sin(dist/Earth_radius)*math.cos(bearing))
        # scatter_lon[i] = rx_lon + math.atan2(math.sin(bearing)*math.sin(dist/Earth_radius)*math.cos(rx_lat),math.cos(dist/Earth_radius)-math.sin(rx_lat)*math.sin(scatter_lat[i]))

        scatter_lat[i] = scatter_lat_a * 180.0 / math.pi
        scatter_lon[i] = scatter_lon_a * 180.0 / math.pi + 360.0
        scatter_doppler[i] = doppler[i]

    return scatter_lon, scatter_lat, scatter_doppler


def plotFoV(tx, rx, lat_range, lon_range, RF_lons, RF_lats, RF_level, RF_contour, Mag_lons, Mag_lats, Mag_level,
            Mag_contour, masked_map, scatter_lon, scatter_lat, scatter_doppler, snr, date, time, loc):
    mpl.rcParams['figure.figsize'] = [16.64, 14.08]
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 25
    mpl.rcParams['axes.titlesize'] = 25
    mpl.rcParams['figure.titlesize'] = 30

    
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

    cbar.ax.set_ylabel("Magnetic Aspect Angle (Degrees)")
    cax.yaxis.set_ticks_position('left')
    plt.axes(ax)

    # Magnetic latitude contour
    lons, lats = Mag_lons, Mag_lats
    x, y = lons - 360, lats
    distance_map = plt.contour(x[3:44][8:56], y[3:44][8:56], Mag_contour[3:44][8:56], Mag_level, linewidths=0.3,
                               colors='k', transform=data_crs)
    plt.clabel(distance_map, inline=1, fmt='%1.1f\u00b0', fontsize=12)

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
    ind = np.argsort(snr)
    plt.scatter(x[ind], y[ind], c=np.abs(snr[ind]), marker='.', edgecolors='none', vmin=0.0, vmax=30.0, s=3, cmap='plasma_r', zorder=2, transform=data_crs)#cmap=matplotlib.cm.jet_r, edgecolors='none', vmin=-900.0, vmax=900.0, zorder=2, transform=data_crs)

    # Make legend and title
    tx_patch = mpatches.Patch(color='g', label='Tx')
    rx_patch = mpatches.Patch(color='b', label='Rx')
    plt.legend(handles=[tx_patch, rx_patch], loc=2)
    plt.suptitle(
        "Mapped ICEBEAR Scatter for {}-{}-{}: 5 second summary starting {:02d}:{:02d}:{:02d}".format(date[0], date[1],
                                                                                                     date[2],
                                                                                                     int(time[0]),
                                                                                                     int(time[1]), int(
                time[2] / 1000)), y=0.94)  # <- center in figure not plot
    plt.figtext(0.095, 0.05,
                "*Mapped scatter positions are derived from a 1.5 lambda baseline and are subject to phase wrapping and a narrowed FoV",
                fontsize=14)

    # Generate Colorbar for Doppler
    cax = plt.axes([0.93, b, 0.02, h])
    cbar = plt.colorbar(drawedges=False, cax=cax)  # draw colorbar
    cbar.ax.set_ylabel("Signal to Noise Ratio (dB)")
    cax.yaxis.set_ticks_position('left')

    # print 'plotting...'
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.savefig(loc + 'map_%04d_%02d_%02d_%02d_%02d_%02d_plot.pdf' % (
    date[0], date[1], date[2], time[0], time[1], int(time[2] / 1000)))
    plt.close()

    return 0

###################################################################
# Here begins the FoV plotting exectution

# Plotting Params
