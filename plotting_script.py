import matplotlib as mpl
#mpl.use('Agg') # Turns of graphical environment

import sys
import os
import icebear.utils as utils
import icebear.plotting.plot as ib3d
from pathlib import Path
import matplotlib.pyplot as plt

import h5py

year = 0
month = 0
day = 0
hour = -1
prefix = "ib3d_normal_01db_1000ms"
suffix = "prelate_bakker"
img_path = '/mnt/icebear/ICEBEAR_plots/'
temp_dir = 'NA'
source = 'Not Set'
errors = False
mode = 'NA'

for arg in range(len(sys.argv)):
    if sys.argv[arg] == '-y':
        year = int(sys.argv[arg+1])
    if sys.argv[arg] == '-m':
        month = int(sys.argv[arg+1])
    if sys.argv[arg] == '-d':
        day = int(sys.argv[arg+1])
    if sys.argv[arg] == '-h':
        hour = int(sys.argv[arg+1])
    if sys.argv[arg] == '--prefix':
        prefix = str(sys.argv[arg+1])
    if sys.argv[arg] == '--suffix':
        suffix = str(sys.argv[arg+1])
    if sys.argv[arg] == '--temp':
        temp_dir = str(sys.argv[arg+1])
    if sys.argv[arg] == '--path':
        img_path = str(sys.argv[arg+1])
    if sys.argv[arg] == '--errors':
        errors = True
    if sys.argv[arg] == '--source':
        source = str(sys.argv[arg+1])
    if sys.argv[arg] == '-fovd':
        mode = 'FoV Doppler'
        source = '/mnt/icebear/ICEBEAR_Level2_data/'
        prefix = "ib3d_normal_swht"
    elif sys.argv[arg] == '-fovs':
        mode = 'FoV SNR'
        source = '/mnt/icebear/ICEBEAR_Level2_data/'
        prefix = "ib3d_normal_swht"
    elif sys.argv[arg] == '-rngd':
        mode = 'range Doppler'
        source = '/mnt/icebear/ICEBEAR_Level1_data/'

if (year == 0) or (month == 0) or (day == 0) or (hour == -1):
    print(f"Improper date passed y:{year} m:{month} d:{day} h:{hour}")
    print("Use the following options to set year month day and hour:")
    print("\t-y: year\n\t-m: month\n\t-d: day\n\t-h: hour")
    sys.exit()

if mode == 'NA':
    print(f"Need to specify plotting mode:")
    print("\t-fovs: plot FoV SNR\n\t-fovd: plot FoV Doppler\n\t-rngd: plot range-Doppler\n")
    print("Note: If you want to define a custom source it must be listed after the plotting mode\nand preceeded by --source")
    sys.exit()

if temp_dir == 'NA':
    print("Need to set a temp directory to hold intermediary files")
    print("\t--temp: /path/to/dir/")

if mode == 'range Doppler':
    source_file = f"{source}/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}/{prefix}_{year:04d}_{month:02d}_{day:02d}_{hour:02d}_{suffix}.h5"
else:
    source_file = f"{source}/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}/{prefix}_{year:04d}_{month:02d}_{day:02d}_{suffix}.h5"
print(source_file)
if os.path.exists(source_file):
    config = utils.Config('/mnt/icebear/processing_code/icebear/dat/default_processing.yml')
    if hour+1 >= 24:
        day_add = 1
        hour_add = -23
    else:
        day_add = 0
        hour_add = 1
    if mode == 'range Doppler':
        print("Starting Range Doppler")
        
        config.update_attr('processing_start', [year, month, day, hour, 0, 0, 0])
        config.update_attr('processing_stop', [year, month, day+day_add, hour+hour_add, 0, 1, 0])
        config.update_attr('plotting_method', 'range_doppler_snr')
        config.update_attr('plotting_source', source+f'/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}/')
        config.add_attr('snr_cutoff', 3.0)
        
        # Make 1~s pplots
        print('1s plots')

        config.update_attr('processing_step', [0, 0, 0, 1, 0])
        config.update_attr('plot_cadence', 1)

        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/1sec/range-Doppler_snr'
        Path(dest).mkdir(parents=True, exist_ok=True)
        try:
            config.update_attr('plotting_destination', dest+'/')
        except:
            config.add_attr('plotting_destination', dest+'/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)
        
        ib3d.range_doppler_snr(config, time, config.plot_cadence)
        plt.close('all')
        
        # Make 5~s pplots
        print('5s plots')
        config.update_attr('processing_step', [0, 0, 0, 5, 0])
        config.update_attr('plot_cadence', 5)
        
        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/5sec/range-Doppler_snr'
        Path(dest).mkdir(parents=True, exist_ok=True)
        config.update_attr('plotting_destination', dest+'/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)
        
        ib3d.range_doppler_snr(config, time, config.plot_cadence)
        plt.close('all')

    elif mode == 'FoV SNR':
        print("Starting SNR FoV")
        
        config.update_attr('processing_start', [year, month, day, hour, 0, 0, 0])
        config.update_attr('processing_stop', [year, month, day+day_add, hour+hour_add, 0, 1, 0])
        config.update_attr('plotting_method', 'FoV_snr')
        config.update_attr('plotting_source', source+f'/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}/')
        config.add_attr('snr_cutoff', 1.0)
        
        # Make 1~s pplots
        print('1s plots')

        config.update_attr('processing_step', [0, 0, 0, 1, 0])
        config.update_attr('plot_cadence', 1)

        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/1sec/FoV_snr'
        Path(dest).mkdir(parents=True, exist_ok=True)
        try:
            config.update_attr('plotting_destination', dest+'/')
        except:
            config.add_attr('plotting_destination', dest+'/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)
        
        ib3d.FoV_snr(config, time, config.plot_cadence, source_file)
        plt.close('all')
        
        # Make 5~s pplots
        print('5s plots')
        config.update_attr('processing_step', [0, 0, 0, 5, 0])
        config.update_attr('plot_cadence', 5)
        
        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/5sec/FoV_snr'
        Path(dest).mkdir(parents=True, exist_ok=True)
        config.update_attr('plotting_destination', dest+'/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)
        
        ib3d.FoV_snr(config, time, config.plot_cadence, source_file)
        plt.close('all')
        
    else:
        print("Starting Doppler FoV")

        config.update_attr('processing_start', [year, month, day, hour, 0, 0, 0])
        config.update_attr('processing_stop', [year, month, day+day_add, hour+hour_add, 0, 1, 0])
        config.update_attr('plotting_method', 'FoV_dop')
        config.update_attr('plotting_source', source+f'/{year:04d}/{month:02d}/{year:04d}_{month:02d}_{day:02d}/')
        config.add_attr('snr_cutoff', 1.0)
        
        # Make 1~s pplots
        print('1s plots')

        config.update_attr('processing_step', [0, 0, 0, 1, 0])
        config.update_attr('plot_cadence', 1)

        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/1sec/FoV_doppler'
        Path(dest).mkdir(parents=True, exist_ok=True)
        try:
            config.update_attr('plotting_destination', dest+'/')
        except:
            config.add_attr('plotting_destination', dest+'/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)

        ib3d.FoV_dop(config, time, config.plot_cadence, source_file)
        plt.close('all')

        # Make 5~s pplots
        print('5s plots')
        config.update_attr('processing_step', [0, 0, 0, 5, 0])
        config.update_attr('plot_cadence', 5)

        dest = f'{img_path}/{year:04d}/{month:02d}/{day:02d}/5sec/FoV_doppler'
        Path(dest).mkdir(parents=True, exist_ok=True)
        config.update_attr('plotting_destination', dest + '/')
        time = utils.Time(config.processing_start, config.processing_stop, config.processing_step)

        ib3d.FoV_dop(config, time, config.plot_cadence, source_file)
        plt.close('all')






