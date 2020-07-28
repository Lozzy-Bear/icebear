import matplotlib.pyplot as plt
import h5py
import numpy as np


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

    todo
        * this will break when i move level 2 data to hdf5 format.
    """
    az *= -1
    doppler = np.where(doppler >= 50, (doppler - 100) * 10 * 3, doppler * 10 * 3)

    # Method: remove the arc curvature of the earth.
    pre_alt = np.sqrt(6378**2+(rng*0.75-200)**2 - 2*6378*(rng*0.75-200)*np.cos(np.deg2rad(90*np.abs(el))))
    gamma = np.arccos(((rng*0.75-200)**2 - (6378**2) - (pre_alt**2))/(-2*6378*pre_alt))
    el = np.abs(el) - np.abs(np.rad2deg(gamma))
    el = np.where(el > 12, np.nan, el)
    alt = -6378+np.sqrt(6378**2+(rng*0.75-200)**2 - 2*6378*(rng*0.75-200)*np.cos(np.deg2rad(90*np.abs(el))))

    # North-South and East-West determination
    rng = np.where(az >= 180, (rng * 0.75 - 200) * -1, rng * 0.75 - 200)
    r = rng * np.cos(np.deg2rad(np.abs(el)))
    horz = np.abs(r) * np.sin(np.deg2rad(az))
    r *= np.cos(np.deg2rad(az))

    # Clutter floor filtering
    r = np.where(alt < 60, np.nan, r)  # 60 or 85
    horz = np.where(alt < 60, np.nan, horz)  # 60 or 85
    alt = np.where(alt < 60, np.nan, alt)  # 60 or 85

    # Setup plotting area.
    plt.figure(x + 1, figsize=[12, 13])
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
    plt.scatter(pm * alt, r, c=doppler, cmap='jet_r', vmin=-1000, vmax=1000, zorder=2, alpha=0.5)
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
    plt.scatter(pm * alt, r, c=snr, cmap='plasma_r', vmin=0, vmax=20, zorder=2, alpha=0.5)
    plt.colorbar(label='Signal-to-Noise [dB]')
    plt.xlim((0, 200))
    plt.ylim((0, 1000))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath + str(title + datetime).replace(' ', '_').replace(':', '') + '.png')
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
                                     f'{config.snr_cutoff:02d}dB_{config.averages:02d}00ms_'
                                     f'{int(now.year):04d}_'
                                     f'{int(now.month):02d}_'
                                     f'{int(now.day):02d}_'
                                     f'{int(now.hour):02d}.h5', 'r')
            except:
                continue
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        try:
            moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
            if filename[f'{moment}/data_flag'][:]:
                tau = int((now.hour * 60 * 60 + now.minute * 60 + now.seconds / 1000) / 3600)
                snr_db = np.abs(filename[f'{moment}/snr_db'][:])
                doppler_shift = filename[f'{moment}/doppler_shift'][:]
                rf_distance = np.abs(filename[f'{moment}/rf_distance'][:])
                plt.subplot(2, 1, 1)
                plt.scatter(np.ones(len(rf_distance)) * tau, rf_distance, c=doppler_shift * 3.03,
                            vmin=-900.0, vmax=900.0, s=3, cmap='jet_r')
                plt.subplot(2, 1, 2)
                plt.scatter(np.ones(len(rf_distance)) * tau, rf_distance, c=snr_db, vmin=0.0,
                            vmax=20.0, s=3, cmap='plasma_r')
        except:
            continue

    plt.subplot(2, 1, 1)
    plt.title(f'{int(time.start_human.year):04d}-{int(time.start_human.month):02d}-{int(time.start_human.day):02d}'
              f' {config.radar_name} Quick Look Plot')
    plt.ylabel('RF Distance (km)')
    plt.colorbar(label='Doppler (m/s)')
    plt.ylim(0, 2500)
    plt.xlim(0, 24.0)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.xlabel('Time (hours)')
    plt.ylabel('RF Distance (km)')
    plt.colorbar(label='SNR (dB')
    plt.ylim(0, 2500)
    plt.xlim(0, 24.0)
    plt.grid()

    plt.savefig(f'{config.plotting_destination}quicklook_{config.radar_name}_'
                f'{int(time.start_human.year):04d}-'
                f'{int(time.start_human.month):02d}-'
                f'{int(time.start_human.day):02d}.png')
    plt.close()

    return None
