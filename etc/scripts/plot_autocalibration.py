import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
from dateutil.tz import tzutc
from scipy.interpolate import interp1d, make_interp_spline
import common.pretty_plots

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def closure_angle_plot():
    dates = np.load('dat_rx_feed.npy')
    d = []
    for i in range(dates.shape[0]):
        dd = datetime(dates[i, 0], dates[i, 1], dates[i, 2])
        d.append(dd.timetuple().tm_yday)
    # d = [datetime(dates[i, 0], dates[i, 1], dates[i, 2]).timetuple().tm_yday() for i in range(dates.shape[0])]
    corr = np.load('old_rx_feed_corr.npy')
    corr = corr[:-1, :]
    cals = np.load('new_rx_feed_cal.npy')
    fig, ax = plt.subplots(figsize=[12, 4])
    myFmt = DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(myFmt)
    cs = ['k', 'g', '--g', '-.g', 'r', '--r', '-.r', 'b', '--b', '-.b']
    wd = [1.5, 1, 1, 1, 1, 1, 1.5, 1.5, 1, 1]
    for i in range(10):
        c = np.nan_to_num(cals[:, i], 0.0)
        f = interp1d(d, c, fill_value='extrapolate')
        x = np.arange(14, 280, 1)
        y = f(x)
        y = moving_average(y, 15)
        plt.plot(x[7:-7], y, cs[i], label=f'Antenna {i}', linewidth=wd[i])

        if i == 0:
            print('dasd')
            plt.fill_between(x[7:-7], y+15, y-15, alpha=0.1, color='g')

    ax.set_ylabel('Phase Calibration Correction [deg]')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
              ncol=1, fancybox=True, shadow=True)
    ax.grid(True)
    plt.ylim(-30, 30)
    plt.xlim(x[7], x[-8])

    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    closure_angle_plot()
