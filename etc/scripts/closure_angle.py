"""
I've solved auto calibration somehwere in here.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.tz import tzutc
import yaml
import os


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


class Config:
    def __init__(self, configuration):
        self.update_config(configuration)
        # Add version attribute
        # here = os.path.abspath(os.path.dirname(__file__))
        # regex = "(?<=__version__..\s)\S+"
        # with open(os.path.join(here, '../icebear/__init__.py'), 'r', encoding='utf-8') as f:
        #     text = f.read()
        # match = re.findall(regex, text)
        # setattr(self, 'version', str(match[0].strip("'")))
        # # Add date_created attribute
        # now = datetime.datetime.now()
        # setattr(self, 'date_created', [now.year, now.month, now.day])

    def update_config(self, file):
        if file.split('.')[1] == 'yml':
            with open(file, 'r') as stream:
                cfg = yaml.full_load(stream)
                for key, value in cfg.items():
                    setattr(self, key, np.array(value))
        if file.split('.')[1] == 'h5':
            stream = h5py.File(file, 'r')
            for key in list(stream.keys()):
                if key == 'data' or key == 'coeffs':
                    pass
                # This horrible little patch fixes strings to UTF-8 from 'S' when loaded from HDF5's
                # and removes unnecessary arrays
                elif '|S' in str(stream[f'{key}'].dtype):
                    temp_value = stream[f'{key}'][()].astype('U')
                    if len(temp_value) == 1:
                        temp_value = temp_value[0]
                    setattr(self, key, temp_value)
                else:
                    temp_value = stream[f'{key}'][()]
                    try:
                        if len(temp_value) == 1:
                            temp_value = temp_value[0]
                        setattr(self, key, temp_value)
                    except:
                        setattr(self, key, temp_value)

    def print_attrs(self):
        print("experiment attributes loaded: ")
        for item in vars(self).items():
            print(f"\t-{item}")
        return None

    def update_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def check_attr(self, key):
        if hasattr(self, key):
            return True
        else:
            return False

    def compare_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            if getattr(self, key) == value:
                return True
            else:
                return False

    def add_attr(self, key, value):
        if self.check_attr(key):
            print(f'ERROR: Attribute {key} already exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def remove_attr(self, key):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            delattr(self, key)
        return None


def load_it(config, time):
    bb = np.zeros(10)
    bad_boys = []
    combo = []
    b1 = []
    b2 = []
    b3 = []
    num_ant = 10
    for i in range(0, num_ant, 1):
        for j in range(i + 1, num_ant, 1):
            for k in range(j + 1, num_ant, 1):
                combo.append([i, j, k])
                b1.append(f'xspec{i}{j}')
                b2.append(f'xspec{j}{k}')
                b3.append(f'xspec{i}{k}')

    temp_hour = [-1, -1, -1, -1]
    cnt = 0
    for t in range(int(time.start_epoch), int(time.stop_epoch), int(time.step_epoch)):
        now = time.get_date(t)
        if [int(now.year), int(now.month), int(now.day), int(now.hour)] != temp_hour:
            filename = f'{config.plotting_source}' \
                       f'{int(now.year):04d}_' \
                       f'{int(now.month):02d}_' \
                       f'{int(now.day):02d}/' \
                       f'{config.radar_config}_{config.experiment_name}_' \
                       f'{int(config.snr_cutoff_db):02d}dB_{config.incoherent_averages:02d}00ms_' \
                       f'{int(now.year):04d}_' \
                       f'{int(now.month):02d}_' \
                       f'{int(now.day):02d}_' \
                       f'{int(now.hour):02d}_' \
                       f'{config.tx_site_name}_{config.rx_site_name}' \
                       f'.h5'
            try:
                filename = h5py.File(filename, 'r')
            except IOError:
                print("You are getting this IO error again!", filename)
                continue
            temp_hour = [int(now.year), int(now.month), int(now.day), int(now.hour)]

        xspectra_descriptors = config.xspectra_descriptors.tolist()
        s = np.zeros(num_ant)
        flg = False
        for i in range(len(combo)):
            try:
                moment = f'data/{int(now.hour):02d}{int(now.minute):02d}{int(now.second * 1000):05d}'
                if filename[f'{moment}/data_flag'][()] == True:
                    if filename[f'{moment}/xspectra'].shape[0]:# < 100:
                        a = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b1[i])]
                        b = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b2[i])]
                        c = filename[f'{moment}/xspectra'][:, xspectra_descriptors.index(b3[i])]
                        # doppler = np.array(filename[f'{moment}/doppler_shift'])
                        # snr = np.array(filename[f'{moment}/snr_db'])
                        p = np.rad2deg(np.angle(a) + np.angle(b) - np.angle(c)) % 360
                        p = ((p + 180) % 360) - 180
                        # p = np.ma.masked_where(doppler > 30, p)
                        # p = np.ma.masked_where(doppler < -30, p)
                        # p = np.ma.masked_where(snr < 6.0, p)

                        # If everything was good the p should = 0. But it won't.
                        # so what I want is how much off of 0 am I
                        # If it is flippy floppy np.abs(np.average(p)) = 0 and
                        # np.averages(np.abs(p)) = maximum
                        p = np.average(np.abs(p))
                        s[combo[i][0]] += p
                        s[combo[i][1]] += p
                        s[combo[i][2]] += p
                        # print(f'antenna_combo: {s} {combo[i]}', b1[i], b2[i], b3[i], p)
                        flg = True
            except:
                print("its busted")
                continue

        if flg:
            s = s / ((num_ant - 1) * (num_ant - 2) / 2)
            s = s - np.mean(s)
            # bad_boys.append(np.argmax(s))
            bb += s
            cnt += 1
            flg = False

        if cnt == 10:
            break
    # values, counts = np.unique(np.asarray(bad_boys), return_counts=True)
    # print("Bad antennas!", cnt, values, counts)
    # print(bb/cnt)
    # print(config.rx_feed_corr[1, :] - bb/cnt)
    return bb/cnt


def closure_angle_plot():
    from datetime import datetime
    from matplotlib.dates import DateFormatter
    dates = np.load('dat_rx_feed.npy')
    d = [datetime(dates[i, 0], dates[i, 1], dates[i, 2]) for i in range(dates.shape[0])]
    corr = np.load('old_rx_feed_corr.npy')
    cals = np.load('new_rx_feed_cal.npy')
    fig, ax = plt.subplots()
    myFmt = DateFormatter("%d")
    ax.xaxis.set_major_formatter(myFmt)
    for i in range(10):
        plt.scatter(d, corr[:, i])
        plt.scatter(d, cals[:, i])

    plt.show()

    return


if __name__ == '__main__':
    import glob
    hour = 9
    files = glob.glob(f'/run/media/arl203/Seagate Expansion Drive/backup/level1/2020*/*0{hour}_prelate_bakker.h5')
    print(files)
    old = []
    new = []
    dat = []
    for file in sorted(files):
        config = Config(file)
        old.append(config.rx_feed_corr[1, :])
        config.plotting_source = '/run/media/arl203/Seagate Expansion Drive/backup/level1/'
        date = np.zeros(7, dtype=int)
        date[0:3] = config.date
        if date[1] == 12:
            break
        dat.append(date)
        date[3] = hour
        config.plotting_start = date
        date2 = np.copy(date)
        date2[4] = 59
        date2[5] = 59
        config.plotting_stop = date2
        config.plotting_step = [0, 0, 0, 1, 0]
        t = Time(config.plotting_start, config.plotting_stop, config.plotting_step)
        cal = load_it(config, t)
        new.append(cal)
        print(config.date, config.rx_feed_corr[1, :] - cal)
        # break

    np.save('old_rx_feed_corr.npy', np.asarray(old))
    np.save('new_rx_feed_cal.npy', np.asarray(new))
    np.save('dat_rx_feed.npy', np.asarray(dat))


    # file = '/run/media/arl203/Seagate Expansion Drive/backup/level1/2020_07_14/ib3d_normal_01dB_1000ms_2020_07_14_09_prelate_bakker.h5'
    # config = Config(file)
    # print(config.rx_feed_corr[1, :])
    # config.plotting_source = '/run/media/arl203/Seagate Expansion Drive/backup/level1/'
    # config.plotting_start = [2020, 7, 14, 9, 0, 0, 0]  # [year, month, day, hour, minute, second, millisecond] time to start plotting
    # config.plotting_stop = [2020, 7, 14, 9, 59, 59, 0]  # [year, month, day, hour, minute, second, millisecond] time to stop plotting
    # config.plotting_step = [0, 0, 0, 1, 0]
    # time = Time(config.plotting_start, config.plotting_stop, config.plotting_step)
    # load_it(config, time)
