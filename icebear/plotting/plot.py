import matplotlib.pyplot as plt
import h5py
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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
