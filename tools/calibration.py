import h5py
import numpy as np


def closure_angle(config, time):
    combo = []
    b1 = []
    b2 = []
    b3 = []
    for i in range(1, 9, 1):
        for j in range(i+1, 10, 1):
            combo.append(f'0{i}{j}')
            b1.append(f'xspec0{i}')
            b2.append(f'xspec{i}{j}')
            b3.append(f'xspec0{j}')

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
            except IOError:
                continue
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        xspectra_descriptors = config.xspectra_descriptors
        for i in range(len(combo)):
            try:
                moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
                a = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b1[i])]
                b = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b2[i])]
                c = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b3[i])]
                doppler = np.array(filename[f'{moment}/doppler_shift'])
                #p = np.mod(np.imag(a), 2*np.pi) + np.mod(np.imag(b), 2*np.pi) + np.mod(np.imag(c), 2*np.pi)
                p = np.abs(np.rad2deg(np.angle(a) + np.angle(b) - np.angle(c)))

                p = np.where(p > 350, np.abs(p - 360), p)
                p = np.ma.masked_where(doppler > 20, p)
                p = np.ma.masked_where(doppler < -20, p)

                print(f'antenna_combo: {combo[i]}', b1[i], b2[i], b3[i], np.max(p))
                print(p)
            except IOError:
                continue
    return


def spectra_look(file, time, name):
    spectra_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for spectra_name in spectra_names:
        dop = np.array([])
        rng = np.array([])
        snr = np.array([])

        temp_hour = [-1, -1, -1, -1]
        for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
            now = time.get_date(t)
            if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
                try:
                    filename = h5py.File(file, 'r')
                except:
                    continue
                temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

            try:
                moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
                if bool(filename[f'{moment}/data_flag']):
                    print(filename[f'{moment}/spectra'].shape, filename[f'{moment}/rf_distance'].shape, filename[f'{moment}/doppler_shift'].shape)
                    power = filename[f'{moment}/spectra'][:, name]
                    noise = filename[f'{moment}/spectra_median'][name]
                    snrs = (power - noise) / noise
                    snrs = np.ma.masked_where(snrs < 0.0, snrs)
                    logsnr = 10 * np.log10(snrs.filled(1))
                    logsnr = np.ma.masked_where(logsnr < 1.0, logsnr)
                    doppler = filename[f'{moment}/doppler_shift']
                    rf_distance = filename[f'{moment}/rf_distance']
                    snr_db = logsnr

                    dop = np.append(dop, doppler)
                    rng = np.append(rng, rf_distance)
                    snr = np.append(snr, snr_db)
            except:
                continue

        d = np.argmax(snr)
        print(f'antenna {name}, doppler {dop[d]}, distance {rng[d]}, snr {snr[d]}')
        plt.figure()
        plt.title(f'Antenna {name}, range-Doppler {12} dB SNR Cutoff')
        plt.scatter(dop, rng, c=snr, vmin=0.0, vmax=np.ceil(np.max(snr)), s=3, cmap='plasma_r')
        plt.colorbar(label='SNR (dB)')
        plt.xlabel('Doppler (Hz)')
        plt.ylabel('RF Distance (km)')
        #plt.ylim(-100, 100)
        plt.xlim(-500, 500)
        plt.savefig(f'E:/icebear/figures/vehicle_test/spectralook_antenna{spectra_name}.png')
        plt.close()

    return