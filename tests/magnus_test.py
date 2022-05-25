import cupy as xp
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import time
import magnus_chunky as mfi

def cluster_plot(fig_num, dt, dr):
    logbinsdt = np.logspace(np.log10(1), np.log10(max(dt)), 300)
    logbinsdr = np.logspace(np.log10(1), np.log10(max(dr)), 300)

    fig = plt.figure(fig_num)
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[1:4, 0:3])
    ax2 = fig.add_subplot(gs[0, 0:3])
    ax3 = fig.add_subplot(gs[1:4, 3])

    ax1.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax1.scatter(dt, dr, marker='.')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Temporal clustering [s]')
    ax1.set_ylabel('Spatial clustering [km]')

    ax2.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax2.hist(dt, bins=logbinsdt, rwidth=3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('Echoes/bin')
    ax2.sharex(ax1)

    ax3.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.3)
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', linewidth=0.1)
    ax3.hist(dr, bins=logbinsdr, orientation='horizontal', rwidth=3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Echoes/bin')
    ax3.sharey(ax1)
    plt.show()


if __name__ == '__main__':
    f = genfromtxt('./DRDT.csv', delimiter=',')
    dr = np.asarray(f[:, 0])
    print(dr[dr > 40].shape[0])
    f = genfromtxt('./LATLON.csv', delimiter=',')
    la = xp.asarray(f[:, 0])
    lo = xp.asarray(f[:, 1])
    f = genfromtxt('./T.csv', delimiter=',')
    ti = xp.asarray(f[:])
    ti *= 24.0 * 60.0 * 60.0

    arr = xp.array([ti, la, lo])
    print(arr.shape)
    start = time.perf_counter()
    dr, dt = mfi.cluster_medians(arr, 100_000)
    print(dr[dr > 40].shape[0])
    end = time.perf_counter()
    print(f'time: {end - start}')
    cluster_plot(1, dt, dr)

    # 807 echoes above dr=40 for legitimate data
    # 742 echoes above dr=40 for tspan=3.75 hr, stride = 50
    # 733 echoes above dr=40 for tspan=4 hr,    stride = 100
    # 742 echoes above dr=40 for tspan=4 hr,    stride = 50
    # 742 echoes above dr=40 for tspan=4.25 hr, stride = 50
    # 727 echoes above dr=40 for tspan=5 hr,    stride = 50
    # 773 echoes above dr=40 for tspan=4 hr,    stride = 25 (290s)
