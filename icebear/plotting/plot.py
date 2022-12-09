import matplotlib.pyplot as plt
import h5py
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math
from icebear.plotting import AspectMapper, igrf
import icebear.plotting.FoVutils as FoV
import matplotlib.patches as mpatches
import pickle

def imaging_4plot(filepath, title, datetime, doppler, rng, snr, az, el):
    """

    Parameters
    ----------
    filepath
    title
    datetime
    doppler
    rng
    snr
    az
    el

    Returns
    -------

    """
    # az *= -1 No longer required with updated coeffs.
    # doppler = np.where(doppler >= 50, (doppler - 100) * 10 * 3, doppler * 10 * 3)

    # Method: remove the arc curvature of the earth.
    pre_alt = np.sqrt(6378**2+(rng*0.75-200)**2 - 2*6378*(rng*0.75-200)*np.cos(np.deg2rad(90 + np.abs(el))))
    gamma = np.arccos(((rng*0.75-200)**2 - (6378**2) - (pre_alt**2))/(-2*6378*pre_alt))
    el = np.abs(el) - np.abs(np.rad2deg(gamma))
    el = np.where(el > 12, np.nan, el)
    alt = -6378+np.sqrt(6378**2+(rng*0.75-200)**2 - 2*6378*(rng*0.75-200)*np.cos(np.deg2rad(90 + np.abs(el))))

    # North-South and East-West determination
    # rng = np.where(az >= 180, (rng * 0.75 - 200) * -1, rng * 0.75 - 200)
    rng = rng * 0.75 - 200
    r = rng * np.cos(np.deg2rad(np.abs(el)))
    horz = np.abs(r) * np.sin(np.deg2rad(az))
    r *= np.cos(np.deg2rad(az))

    # Clutter floor filtering
    #r = np.where(alt < 60, np.nan, r)  # 60 or 85
    #horz = np.where(alt < 60, np.nan, horz)  # 60 or 85
    #alt = np.where(alt < 60, np.nan, alt)  # 60 or 85

    # Setup plotting area.
    plt.figure(1, figsize=[12, 13])
    plt.rcParams.update({'font.size': 20})
    plt.suptitle(title + ' ' + datetime)

    # Top down view with Doppler.
    plt.subplot(221)
    plt.grid()
    plt.ylabel('South-North Distance [km]')
    plt.scatter(horz, r, c=doppler, cmap='jet_r', vmin=-1000, vmax=1000, alpha=0.5)
    plt.xlim((-400, 400))
    plt.ylim((0, 1000))

    # Side view with Doppler.
    plt.subplot(222)
    plt.grid()
    plt.plot(np.ones(len(rng)) * 130, np.linspace(0, 1000, len(r)), '--k', zorder=1)
    plt.plot(np.ones(len(rng)) * 80, np.linspace(0, 1000, len(r)), '--k', zorder=1)
    plt.scatter(alt, r, c=doppler, cmap='jet_r', vmin=-1000, vmax=1000, zorder=2, alpha=0.5)
    plt.colorbar(label='Doppler Velocity [m/s]')
    plt.xlim((0, 200))
    plt.ylim((0, 1000))

    # Top down view with SNR.
    plt.subplot(223)
    plt.grid()
    plt.ylabel('South-North Distance [km]')
    plt.xlabel('West-East Distance [km]')
    plt.scatter(horz, r, c=snr, cmap='plasma_r', vmin=0, vmax=20, alpha=0.5)
    plt.xlim((-400, 400))
    plt.ylim((0, 1000))

    # Side view with SNR.
    plt.subplot(224)
    plt.grid()
    plt.xlabel('Corrected Altitude [km]')
    plt.plot(np.ones(len(rng)) * 130, np.linspace(0, 1000, len(r)), '--k', zorder=1)
    plt.plot(np.ones(len(rng)) * 80, np.linspace(0, 1000, len(r)), '--k', zorder=1)
    plt.scatter(alt, r, c=snr, cmap='plasma_r', vmin=0, vmax=20, zorder=2, alpha=0.5)
    plt.colorbar(label='Signal-to-Noise [dB]')
    plt.xlim((0, 200))
    plt.ylim((0, 1000))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath + str(title + datetime).replace(' ', '_').replace(':', '') + '.png')
    plt.show()
    plt.close()

    return None


def quick_look(config, time):
    """
    Creates a standard Quick Look plot of level 1 data for the specified time frame.

    Parameters
    ----------
        config : Class Object
            Config class instantiation which contains plotting settings.
        time : Class Object
            Time class instantiation for start, stop, step deceleration.

    Returns
    -------
        None

    Notes
    -----
        * Typically a Quick Look plot should be one day of data with a step size equal to the incoherent averages
          time length used to generate the level 1 data used.

    Todo
        * The plt.colorbar() is currently not working.

    """
    plt.figure(1, figsize=[20, 10])
    plt.rcParams.update({'font.size': 22})

    temp_hour = [-1, -1, -1, -1]
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        now = time.get_date(t)
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            try:
                filename = h5py.File(f'{config.plotting_source}{config.radar_name}_{config.processing_method}_'
                                     f'{config.tx_name}_{config.rx_name}_'
                                     f'{int(config.snr_cutoff):02d}dB_{config.incoherent_averages:02d}00ms_'
                                     f'{int(now.year):04d}_'
                                     f'{int(now.month):02d}_'
                                     f'{int(now.day):02d}_'
                                     f'{int(now.hour):02d}.h5', 'r')
            except:
                continue
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        try:
            moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
            if bool(filename[f'{moment}/data_flag']):
                tau = int((now.hour * 60 * 60 + now.minute * 60 + now.second / 1000) / 3600)
                snr_db = np.abs(filename[f'{moment}/snr_db'][:])
                doppler_shift = filename[f'{moment}/doppler_shift'][:]
                rf_distance = np.abs(filename[f'{moment}/rf_distance'][:])

                plt.subplot(212)
                plt.scatter(np.ones(len(rf_distance)) * tau, rf_distance, c=doppler_shift * 3.03,
                            vmin=-900.0, vmax=900.0, s=3, cmap='jet_r')
                plt.colorbar(label='Doppler (m/s)')

                plt.subplot(211)
                plt.scatter(np.ones(len(rf_distance)) * tau, rf_distance, c=snr_db, vmin=0.0,
                            vmax=100.0, s=3, cmap='plasma_r')
                plt.colorbar(label='SNR (dB)')
        except:
            continue

    plt.subplot(211)
    plt.title(f'{int(time.start_human.year):04d}-{int(time.start_human.month):02d}-{int(time.start_human.day):02d}'
              f' {config.radar_name} Quick Look Plot')
    plt.ylabel('RF Distance (km)')
    plt.ylim(0, 2500)
    plt.xlim(0.0, 24.0)
    plt.grid()

    plt.subplot(212)
    plt.xlabel('Time (hours)')
    plt.ylabel('RF Distance (km)')
    plt.ylim(0, 2500)
    plt.xlim(0.0, 24.0)
    plt.grid()

    plt.savefig(f'{config.plotting_destination}quicklook_{config.radar_name}_'
                f'{int(time.start_human.year):04d}-'
                f'{int(time.start_human.month):02d}-'
                f'{int(time.start_human.day):02d}.png')
    plt.close(1)

    return None


def range_doppler_snr(config, time, spacing):
    """
    Creates a standard range-Doppler SNR plot of level 1 data for the specified time frame.

    Parameters
    ----------
        config : Class Object
            Config class instantiation which contains plotting settings.
        time : Class Object
            Time class instantiation for start, stop, step deceleration.
        spacing : int
            The amount of time in seconds to plot in one image.

    Returns
    -------
        None

    Notes
    -----
        * Typically a Quick Look plot should be one day of data with a step size equal to the incoherent averages
          time length used to generate the level 1 data used.

    """
    temp_hour = [-1, -1, -1, -1]
    spacing_counter = 0
    data_flag = False
    sum_counter = 0
    with imageio.get_writer(f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_'
                            f'range_doppler_snr_{spacing}sec_movie_'
                            f'{int(time.start_human.year):04d}_'
                            f'{int(time.start_human.month):02d}_'
                            f'{int(time.start_human.day):02d}_'
                            f'{int(time.start_human.hour)}'
                            f'.mp4', fps=10, mode='I') as writer:
        for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
            now = time.get_date(t)
            if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
                print(f'{config.plotting_source}{config.radar_config}_{config.experiment_name}_'
                                         f'{int(config.snr_cutoff_db):02d}dB_{config.incoherent_averages:02d}00ms_'
                                         f'{int(now.year):04d}_'
                                         f'{int(now.month):02d}_'
                                         f'{int(now.day):02d}_'
                                         f'{int(now.hour):02d}_'
                                         f'{config.tx_site_name}_{config.rx_site_name}.h5')
                try:
                    filename = h5py.File(f'{config.plotting_source}{config.radar_config}_{config.experiment_name}_'
                                         f'{int(config.snr_cutoff_db):02d}dB_{config.incoherent_averages:02d}00ms_'
                                         f'{int(now.year):04d}_'
                                         f'{int(now.month):02d}_'
                                         f'{int(now.day):02d}_'
                                         f'{int(now.hour):02d}_'
                                         f'{config.tx_site_name}_{config.rx_site_name}.h5', 'r')
                except:
                    print('ERROR: File skipped or does not exist')
                    continue
                temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

            spacing_counter += 1
            plt.figure(1)
            # Save the image if over spacing
            if spacing_counter > spacing:
                spacing_counter = 1
                if data_flag:
                    print('\tsaving image:', save_name)
                    plt.savefig(save_name + '.pdf')
                fig = plt.figure(1)
                canvas = FigureCanvas(fig)
                canvas.draw()
                writer.append_data(np.asarray(canvas.buffer_rgba()))
                # plt.savefig(save_name + '.png')
                plt.close(1)
                plt.figure(1)
                data_flag = False
                sum_counter = 0

            # Start a new image
            if spacing_counter == 1:
                plt.scatter(0, -1, c=0.0, vmin=0.0, vmax=30.0, s=3, cmap='plasma_r')
                plt.title(f'ICEBEAR-3D Range-Doppler-SNR Plot\n'
                          f'{int(now.year):04d}-'
                          f'{int(now.month):02d}-'
                          f'{int(now.day):02d} '
                          f'{int(now.hour):02d}:'
                          f'{int(now.minute):02d}:'
                          f'{int(now.second):02d}')
                cb = plt.colorbar(label='SNR (dB)')
                cb.ax.plot([-500, 500], [config.snr_cutoff_db, config.snr_cutoff_db], 'k')
                plt.xlabel('Doppler (Hz)')
                plt.ylabel('Total RF Distance (km)')
                plt.ylim(0, config.number_ranges * config.range_resolution)
                plt.xlim(-500, 500)
                plt.xticks(np.arange(-500, 500 + 100, 100))
                plt.yticks(np.arange(0, int(config.number_ranges * config.range_resolution + 500), 500))
                plt.grid(linestyle=':')
                props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
                plt.text(-450, 150, f'{config.snr_cutoff_db} dB SNR Cutoff', bbox=props)
                plt.text(250, 150, f'{spacing} s Interval', bbox=props)
                save_name = f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_'\
                            f'range_doppler_snr_{spacing}sec_'\
                            f'{int(now.year):04d}_'\
                            f'{int(now.month):02d}_'\
                            f'{int(now.day):02d}_'\
                            f'{int(now.hour):02d}_'\
                            f'{int(now.minute):02d}_'\
                            f'{int(now.second):02d}'

            # Add to the image if under spacing
            if spacing_counter <= spacing:
                try:
                    moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
                    if bool(filename[f'{moment}/data_flag']):
                        dop = filename[f'{moment}/doppler_shift'][:]
                        rng = np.abs(filename[f'{moment}/rf_distance'][:])
                        snr = np.abs(filename[f'{moment}/snr_db'][:])
                        plt.scatter(dop, rng, c=snr, vmin=0.0, vmax=30.0, s=3, cmap='plasma_r')
                        sum_counter += int(len(rng))
                        if sum_counter > 10:
                            data_flag = True
                except:
                    continue
    writer.close()
    return None


def brightness_plot(config, brightness):
    el, az = elevation_azimuth(-20, 20, I.shape[0], -45, 45, I.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title("I'(az,el) Range=" + title)
    plt.xlabel("Azimuth [degrees]")
    plt.ylabel("Elevation [degrees]")
    plt.pcolormesh(np.rad2deg(az), np.rad2deg(el), np.flip(np.abs(I / np.max(I)), axis=(0, 1)), cmap='jet', vmin=0.0,
                   vmax=1.0)
    ax.set_xticks(np.arange(-45, 46, 5))
    ax.set_yticks(np.arange(-20, 21, 5))
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 20)
    plt.grid(which='both')
    plt.colorbar()
    plt.gcf().set_size_inches(12, 4)
    fig.tight_layout()
    return
    
    
def FoV_map(config):
    print("Making FoV map with Cartopy")
    mpl.rcParams['figure.figsize'] = [16.64, 14.08]
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 25
    mpl.rcParams['axes.titlesize'] = 25
    mpl.rcParams['figure.titlesize'] = 30
    
    # Loading Params
    input_file = '/mnt/icebear/processing_code/icebear/dat/canada_gh.ini'
    igrf_year = 2015.0
    space, altitude, fov_limit, aspect_limit, tx, rx, tx_sites, rx_sites, lat_range, lon_range, corr_num = FoV.load_config(
        input_file, igrf_year)

    Earth_radius = 6378.1
    rx_lat = math.radians(52.24)
    rx_lon = math.radians(253.55 - 360.00)
    tx_lat = math.radians(50.893)
    tx_lon = math.radians(-109.403)
    
    # Compute the aspect angle map ---> Can probably be moved to scatter_map function in vizualization.py
    AM = AspectMapper.AspectMapper(tx, rx, lat_range, lon_range, altitude, fov_limit, aspect_limit, igrf_year)
    AM.computeMap()
    map = AM.getMap()
    mm = np.ma.masked_where(np.isnan(map), map)  # Mask the NaN values
    masked_map = np.abs(mm)  # assume 10 dB per degree (i.e. Foster and Erickson @ UHF)

    mlat_map = igrf.IGRF(igrf_year)
    RF_lons, RF_lats, RF_level, RF_contour = FoV.contour_RF(Earth_radius, tx_lat, tx_lon, rx_lat, rx_lon)
    Mag_lons, Mag_lats, Mag_level, Mag_contour = FoV.contour_Mag(mlat_map)
    
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(1)
    space = 0.8
    ax = plt.axes([0.1, 0.1, space, 0.8], projection=ccrs.Mercator())
    ax.coastlines(resolution='10m', color='grey')
    ax.set_yticks(np.arange(-60, 80, 5), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-180, 240, 5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent((np.min(lon_range), np.max(lon_range), np.min(lat_range - 0.15), np.max(lat_range)),
    crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='none',
                   edgecolor='grey')
    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m'),
                   facecolor='none', edgecolor='black', linestyle=':')

    pos = ax.get_position()
    l, b, w, h = pos.bounds

    levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    lons, lats = np.meshgrid(lon_range, lat_range)
    x, y = lons - 360, lats
    aspect_map = plt.contourf(x, y, masked_map, levels, cmap=matplotlib.cm.gray, alpha=0.4,
                              transform=data_crs)
    cax = plt.axes([space + 0.12, b, 0.01, h])
    cbar = plt.colorbar(drawedges=True, cax=cax)

    cbar.ax.set_ylabel("Magnetic Aspect Angle (Degrees)")
    cax.yaxis.set_ticks_position('left')
    plt.axes(ax)

    lons, lats = Mag_lons, Mag_lats
    x, y = lons - 360, lats
    distance_map = plt.contour(x[3:44][8:56], y[3:44][8:56], Mag_contour[3:44][8:56], Mag_level,
                               linewidths=0.3, colors='k', transform=data_crs)
    plt.clabel(distance_map, inline=1, fmt='%1.1f\u00b0', fontsize=12)

    lons, lats = RF_lons, RF_lats
    x, y = lons - 360, lats
    distance_map = plt.contour(x[43:200][:], y[43:200][:], RF_contour[43:200][:], RF_level, linewidths=0.3,
                               colors='m',
                               transform=data_crs)
    plt.clabel(distance_map, inline=1, fmt=' %i km ', fontsize=10.6)

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
            rx_beams_lat[j, i] = math.asin(math.sin(rx_lat) * math.cos((j * 6.0) / Earth_radius) + math.cos(rx_lat) * 
                                           math.sin((j * 6.0) / Earth_radius) * math.cos(math.radians(beam_dirs[i])))
            rx_beams_lon[j, i] = rx_lon + math.atan2(math.sin(math.radians(beam_dirs[i])) * math.sin((j * 6.0) / Earth_radius) * 
                                                     math.cos(rx_lat), math.cos((j * 6.0) / Earth_radius) - math.sin(rx_lat) *  
                                                     math.sin(rx_beams_lat[j, i]))
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
            tx_beams_lat[j, i] = math.asin(math.sin(tx_lat) * math.cos((j * 7.0) / Earth_radius) + math.cos(tx_lat) * 
                                           math.sin((j * 7.0) / Earth_radius) * math.cos(math.radians(beam_dirs[i])))
            tx_beams_lon[j, i] = tx_lon + math.atan2(math.sin(math.radians(beam_dirs[i])) * math.sin((j * 7.0) / Earth_radius) * 
                                                     math.cos(tx_lat), math.cos((j * 7.0) / Earth_radius) - math.sin(tx_lat) *
                                                     math.sin(tx_beams_lat[j, i]))
            tx_beams_lat[j, i] = math.degrees(tx_beams_lat[j, i])
            tx_beams_lon[j, i] = math.degrees(tx_beams_lon[j, i]) + 360.0

    for i in range(2):
        xd, yd = tx_beams_lon[:, i] - 360, tx_beams_lat[:, i]
        plt.plot(xd, yd, linewidth=3.0, color='g', transform=data_crs)

    tx_patch = mpatches.Patch(color='g', label='Tx')
    rx_patch = mpatches.Patch(color='b', label='Rx')
    plt.legend(handles=[tx_patch, rx_patch], loc=2)

    plt.grid(b=True, which='major', color='k', linestyle='--')
    if "snr" in config.plotting_method:
        plt.scatter([0, 0], [-1, 1], c=[0.0, 0.0], s=3, vmin=0.0, vmax=30.0, cmap='plasma_r', transform=data_crs)
        cb = plt.colorbar(label='SNR (dB)')
    else:
        plt.scatter([0, 0], [-1, 1], c=[0.0, 0.0], s=3, vmin=-900.0, vmax=900.0, cmap='jet_r', transform=data_crs)
        cb = plt.colorbar(label='Doppler Velocity (m/s)')

    return fig, ax, data_crs


def FoV_snr(config, time, spacing, source_file):
    """
    Creates a standard FoV SNR plot of level 2 data for the specified time frame.

    Parameters
    ----------
        config : Class Object
            Config class instantiation which contains plotting settings.
        time : Class Object
            Time class instantiation for start, stop, step deceleration.
        spacing : int
            The amount of time in seconds to plot in one image.
        source_file : str
            Path to file containing plot data

    Returns
    -------
        None

    Notes
    -----
        * Typically a Quick Look plot should be one day of data with a step size equal to the incoherent averages
          time length used to generate the level 1 data used.

    """
    fig, ax, data_crs = FoV_map(config)
    data = h5py.File(source_file, 'r')
    plot_time = data['data']['time'][:]
    snr = data['data']['snr_db'][:]
    lat = data['data']['latitude'][:]
    lon = data['data']['longitude'][:]

    print("Check if Lat Lon look correct")
    print(lat)
    print(lon)

    lat = np.ma.masked_where(snr < config.snr_cutoff_db, lat)
    lon = np.ma.masked_where(snr < config.snr_cutoff_db, lon)
    snr = np.ma.masked_where(snr < config.snr_cutoff_db, snr)

    with imageio.get_writer(f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_'
                            f'FoV_snr_{spacing}sec_movie_'
                            f'{int(time.start_human.year):04d}_'
                            f'{int(time.start_human.month):02d}_'
                            f'{int(time.start_human.day):02d}_'
                            f'{int(time.start_human.hour)}'
                            f'.mp4', fps=10, mode='I') as writer:
        for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
            now = time.get_date(t)

            #Plot Data
            time_ind = np.where((plot_time > t) & (plot_time <= t + time.step_epoch))
            plot_lon = lon[time_ind]
            plot_lat = lat[time_ind]
            plot_snr = snr[time_ind]

            ind = np.argsort(plot_snr)
            period = plt.scatter(plot_lon[ind], plot_lat[ind], c=plot_snr[ind], vmin=0.0, vmax=30.0, s=3, cmap='plasma_r', transform=data_crs)

            # Save the image
            plt.title(f'ICEBEAR-3D FoV-SNR Plot\n'
                      f'{int(now.year):04d}-'
                      f'{int(now.month):02d}-'
                      f'{int(now.day):02d} '
                      f'{int(now.hour):02d}:'
                      f'{int(now.minute):02d}:'
                      f'{int(now.second):02d}')

            if len(time_ind[0]) > 10:
                save_name = f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_' \
                            f'FoV_snr_{spacing}sec_' \
                            f'{int(now.year):04d}_' \
                            f'{int(now.month):02d}_' \
                            f'{int(now.day):02d}_' \
                            f'{int(now.hour):02d}_' \
                            f'{int(now.minute):02d}_' \
                            f'{int(now.second):02d}'
                print('\tsaving image:', save_name)
                plt.savefig(save_name + '.pdf')

            canvas = FigureCanvas(fig)
            canvas.draw()
            writer.append_data(np.asarray(canvas.buffer_rgba()))

            #Clear Data
            period.remove()
            #repeat

    writer.close()
    
    return 0


def FoV_dop(config, time, spacing, source_file):
    """
    Creates a standard FoV Doppler plot of level 2 data for the specified time frame.

    Parameters
    ----------
        config : Class Object
            Config class instantiation which contains plotting settings.
        time : Class Object
            Time class instantiation for start, stop, step deceleration.
        spacing : int
            The amount of time in seconds to plot in one image.
        source_file : str
            Path to file containing plot data

    Returns
    -------
        None

    Notes
    -----
        * Typically a Quick Look plot should be one day of data with a step size equal to the incoherent averages
          time length used to generate the level 1 data used.

    """
    fig, ax, data_crs = FoV_map(config)
    data = h5py.File(source_file, 'r')
    plot_time = data['data']['time'][:]
    snr = data['data']['snr_db'][:]
    dop = data['data']['doppler_shift'][:]
    lat = data['data']['latitude'][:]
    lon = data['data']['longitude'][:]

    print("Check if Lat Lon look correct")
    print(lat)
    print(lon)

    lat = np.ma.masked_where(snr < config.snr_cutoff_db, lat)
    lon = np.ma.masked_where(snr < config.snr_cutoff_db, lon)
    dop = np.ma.masked_where(snr < config.snr_cutoff_db, dop)
    snr = np.ma.masked_where(snr < config.snr_cutoff_db, snr)

    with imageio.get_writer(f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_'
                            f'FoV_snr_{spacing}sec_movie_'
                            f'{int(time.start_human.year):04d}_'
                            f'{int(time.start_human.month):02d}_'
                            f'{int(time.start_human.day):02d}_'
                            f'{int(time.start_human.hour)}'
                            f'.mp4', fps=10, mode='I') as writer:
        for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
            now = time.get_date(t)

            # Plot Data
            time_ind = np.where((plot_time > t) & (plot_time <= t + time.step_epoch))
            plot_lon = lon[time_ind]
            plot_lat = lat[time_ind]
            plot_dop = dop[time_ind]
            plot_snr = snr[time_ind]

            ind = np.argsort(plot_snr)
            period = plt.scatter(plot_lon[ind], plot_lat[ind], c=plot_dop[ind], vmin=-900.0, vmax=900.0, s=3,
                                 cmap='jet_r', transform=data_crs)

            # Save the image
            plt.title(f'ICEBEAR-3D FoV-Doppler Plot\n'
                      f'{int(now.year):04d}-'
                      f'{int(now.month):02d}-'
                      f'{int(now.day):02d} '
                      f'{int(now.hour):02d}:'
                      f'{int(now.minute):02d}:'
                      f'{int(now.second):02d}')

            if len(time_ind[0]) > 10:
                save_name = f'{config.plotting_destination}{config.radar_config}_{config.experiment_name}_' \
                            f'FoV_dop_{spacing}sec_' \
                            f'{int(now.year):04d}_' \
                            f'{int(now.month):02d}_' \
                            f'{int(now.day):02d}_' \
                            f'{int(now.hour):02d}_' \
                            f'{int(now.minute):02d}_' \
                            f'{int(now.second):02d}'
                print('\tsaving image:', save_name)
                plt.savefig(save_name + '.pdf')

            canvas = FigureCanvas(fig)
            canvas.draw()
            writer.append_data(np.asarray(canvas.buffer_rgba()))

            # Clear Data
            period.remove()
            # repeat

    writer.close()

    return 0
