import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.tz import tzutc


class Time:
    def __init__(self, start, stop, step):
        """
        Class which hold the iteration time series in both human readable and seconds since epoch (1970-01-01) formats.

        Parameters
        ----------
            start : list int
                Start point of time series in format [year, month, day, hour, minute, second, microsecond]
            stop : list int
                Stop point of time series in format [year, month, day, hour, minute, second, microsecond]
            step : list int
                Step size of time series in format [day, hour, minute, second, microsecond]
        """
        if len(start) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(stop) != 7:
            raise ValueError('Must include [year, month, day, hour, minute, second, microsecond]')
        if len(step) != 5:
            raise ValueError('Must include [day, hour, minute, second, microsecond]')
        self.start_human = datetime.datetime(year=start[0], month=start[1], day=start[2], hour=start[3],
                                             minute=start[4], second=start[5], microsecond=start[6], tzinfo=tzutc())
        self.stop_human = datetime.datetime(year=stop[0], month=stop[1], day=stop[2], hour=stop[3],
                                            minute=stop[4], second=stop[5], microsecond=stop[6], tzinfo=tzutc())
        self.step_human = datetime.timedelta(days=step[0], hours=step[1], minutes=step[2],
                                             seconds=step[3], microseconds=step[4])
        self.start_epoch = self.start_human.timestamp()
        self.stop_epoch = self.stop_human.timestamp()
        self.step_epoch = self.step_human.total_seconds()

    def get_date(self, timestamp):
        return datetime.datetime.fromtimestamp(timestamp, tz=tzutc())


def plot_it(dop, rng, snr, name):
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
    plt.savefig(f'E:/icebear/figures/vehicle_test/{name}.png')
    plt.close()

    return None


def load_it(file, time, name):
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

    return dop, rng, np.real(snr)


if __name__ == '__main__':
    file = 'E:/icebear/code/ib3d_mobile_truck_bakker_12dB_1000ms_2020_08_21_20.h5'
    plotting_start = [2020, 8, 21, 20, 21, 0, 0]
    plotting_stop = [2020, 8, 21, 20, 22, 0, 0]
    plotting_step = [0, 0, 0, 1, 0]
    time = Time(plotting_start, plotting_stop, plotting_step)
    spectra_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for spectra_name in spectra_names:
        d, r, s = load_it(file, time, spectra_name)
        plot_it(d, r, s, spectra_name)