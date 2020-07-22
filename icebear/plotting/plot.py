import matplotlib.pyplot as plt
import numpy as np


def ib3d_4plot(filepath, title, datetime, doppler, rng, snr, az, el):
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
    return


# Quick look plots Devin stuff
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['figure.figsize'] = 20, 10
#plt.rcParams['axes.facecolor'] = 'black'

#cmap = colors.LinearSegmentedColormap.from_list(
#        'incr_alpha', [(0, (*colors.to_rgb('RdYlBu'),0)), (1, 'RdYlBu')])

year=2019
month=12
day=18
hour=0
minute=0
second=0

if second==0:
        second+=1

for temp_days in range(31):
    days=temp_days+day
    for temp_hours in range(24-hour):
        hours = hour+temp_hours
        try:
            ib_file = h5py.File(f'{year:04d}_{month:02d}_{days:02d}/icebear_linear_01dB_1000ms_vis_{year:02d}_{month:02d}_{days:02d}_{hours:02d}_prelate_bakker.h5','r')
        except:
            continue
        print(hours)
        #print(ib_file.keys())

        for temp_minutes in range(60-minute):
            minutes = minute+temp_minutes
            for temp_seconds in range(60-second):
                seconds=(second+temp_seconds)*1000

                try:
                    if (ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/data_flag'][:]==True):
                        #example to plot a spectra
                        icebear_time = (hours*60*60+minutes*60+(seconds/1000))/(60*60)
                        logsnr = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/snr_dB'][:])
                        doppler = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/doppler_shift'][:]
                        range_values = np.abs(ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/rf_distance'][:])
                        xspectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_xspectra'][:]
                        spectra_values = ib_file[f'data/{hours:02d}{minutes:02d}{seconds:05d}/antenna_spectra'][:]

                        plt.subplot(2,1,1)
                        plt.scatter(np.ones(len(range_values))*icebear_time,range_values,c=doppler*3.03,vmin=-900.0,vmax=900.0,s=3,cmap='jet_r')#,alpha=(logsnr/60+0.5))
                        plt.subplot(2,1,2)
                        plt.scatter(np.ones(len(range_values))*icebear_time,range_values,c=logsnr,vmin=0.0,vmax=20.0,s=3,cmap='plasma_r')#,alpha=(logsnr/60+0.5))
                except:
                    break
            second=0
        minute=0
    hour=0

    plt.subplot(2,1,1)
    cbar = plt.colorbar()
    cbar.set_label('Doppler (m/s)')
    plt.title(f'{year:04d}-{month:02d}-{days:02d} ICEBEAR Summary Plot')
    plt.ylabel('RF Distance (km)')
    plt.ylim(0,2500)
    plt.xlim(0,24.0)
    plt.grid()

    plt.subplot(2,1,2)
    cbar = plt.colorbar()
    cbar.set_label('SNR (dB)')
    plt.ylabel('RF Distance (km)')
    plt.ylim(0,2500)
    plt.xlim(0,24.0)
    plt.grid()
    plt.xlabel('Time (hours)')

    plt.savefig(f'{year:04d}_{month:02d}_{days:02d}.png')
    plt.close()
    print('plot saved')

