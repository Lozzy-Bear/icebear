import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import sys

mpl.rcParams.update({'font.size': 22})
mpl.rcParams['figure.figsize'] = 20, 10
#plt.rcParams['axes.facecolor'] = 'black'


year = 0
month = 0
day = 0
milisecond_step = 1000
start_hour = 0
start_minute = 0
start_second = 0 * 1000
end_hour = 24
end_minute = 60
end_second = 60 * 1000
snr_lim = 1.0
prefix = "ib3d_normal_01db_1000ms"
suffix = "prelate_bakker"
path = '/mnt/icebear/ICEBEAR_plots/summary_plots/'
source = 'Not Set'
errors = False

for arg in range(len(sys.argv)):
    if sys.argv[arg] == '-y':
        year = int(sys.argv[arg+1])
    if sys.argv[arg] == '-m':
        month = int(sys.argv[arg+1])
    if sys.argv[arg] == '-d':
        day = int(sys.argv[arg+1])
    if (sys.argv[arg] == '-ms') or (sys.argv[arg] == '--ms-step'):
        milisecond_step = int(sys.argv[arg+1])
    if sys.argv[arg] == '-sh':
        start_hour = int(sys.argv[arg+1])
    if sys.argv[arg] == '-sm':
        start_minute = int(sys.argv[arg+1])
    if sys.argv[arg] == '-ss':
        start_second = int(sys.argv[arg+1]) * 1000
    if sys.argv[arg] == '-eh':
        end_hour = int(sys.argv[arg+1])
    if sys.argv[arg] == '-em':
        end_minute = int(sys.argv[arg+1])
    if sys.argv[arg] == '-es':
        end_second = int(sys.argv[arg+1]) * 1000
    if sys.argv[arg] == '--snr':
        snr_lim = float(sys.argv[arg+1])
    if sys.argv[arg] == '--prefix':
        prefix = str(sys.argv[arg+1])
    if sys.argv[arg] == '--suffix':
        suffix = str(sys.argv[arg+1])
    if sys.argv[arg] == '--path':
        path = str(sys.argv[arg+1])
    if sys.argv[arg] == '--errors':
        errors = True
    if sys.argv[arg] == '--source':
        source = str(sys.argv[arg+1])

if (year == 0) or (month == 0) or (day == 0):
    print(f"Improper date passed {year} {month} {day}")
    print("Use the following options to set year month and day:")
    print("\t-y: year\n\t-m: month\n\t-d: day")

if source == 'Not Set':
    source = f'/mnt/icebear/ICEBEAR_Level1_data/{year:04d}/{month:02d}/'

print(f"Beginning summary plotting for {year:04d}-{month:02d}-{day:02d}")
for hour in range(start_hour, end_hour):
    try:
        ib_file = h5py.File(f'{source}/{year:04d}_{month:02d}_{day:02d}/{prefix}_{year:02d}_{month:02d}_{day:02d}_{hour:02d}_{suffix}.h5','r')
    except Exception as e:
        print(f"No file found or problem reading for {year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00 from {source}/{year:04d}_{month:02d}_{day:02d}/")
        if errors:
            print(e)
        continue
    print(f"processing hour: {hour:02d}")

    for minute in range(start_minute, end_minute):
        for second in range(start_second, end_second, milisecond_step):
            try:
                if (ib_file[f'data/{hour:02d}{minute:02d}{second:05d}/data_flag'][:]==True):
                    icebear_time = (hour*60*60+minute*60+(second/1000))/(60*60)
                    logsnr = np.abs(ib_file[f'data/{hour:02d}{minute:02d}{second:05d}/snr_db'][:])
                    doppler = ib_file[f'data/{hour:02d}{minute:02d}{second:05d}/doppler_shift'][:]
                    range_values = np.abs(ib_file[f'data/{hour:02d}{minute:02d}{second:05d}/rf_distance'][:])

                    inx = np.where(logsnr>snr_lim)

                    plt.subplot(2,1,2)
                    plt.scatter(np.ones(len(range_values[inx[0]]))*icebear_time,range_values[inx[0]],c=doppler[inx[0]]*3.03,vmin=-900.0,vmax=900.0,s=3,cmap='jet_r')#,alpha=(logsnr/60+0.5))
                    plt.subplot(2,1,1)
                    plt.scatter(np.ones(len(range_values[inx[0]]))*icebear_time,range_values[inx[0]],c=logsnr[inx[0]],vmin=0.0,vmax=20.0,s=3,cmap='plasma_r')#,alpha=(logsnr/60+0.5))
            except Exception as e:
                print("Issue reading or plotting data")
                if errors:
                    print(e)
                break


ax = plt.subplot(2,1,2)
cbar = plt.colorbar()
cbar.set_label('Doppler (m/s)')
plt.xlabel('Time (hours)')
plt.ylabel('RF Distance (km)')
plt.ylim(0,3000)
plt.xlim(0,24.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.grid()

ax = plt.subplot(2,1,1)
cbar = plt.colorbar()
cbar.set_label('SNR (dB)')
plt.ylabel('RF Distance (km)')
plt.title(f'{year:04d}-{month:02d}-{day:02d} ICEBEAR Summary Plot')
plt.ylim(0,3000)
plt.xlim(0,24.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.grid()

plt.savefig(f'{path}/{year:04d}_{month:02d}_{day:02d}.png')
plt.close()
print('plot saved')


























