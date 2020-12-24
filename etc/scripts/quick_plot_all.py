import icebear
import icebear.utils as util
import os

# Set plotting step size
plotting_step = [0, 0, 0, 1, 0]
plotting_spacing = 5

# Gather all the level1 data file paths
filepath = '/beaver/backup/level1/'
files = util.get_all_data_files(filepath, '2017_12_06', '2020_11_16')
# Create plots
for file in files:
    config = util.Config(file)
    config.add_attr('number_ranges', 2000)
    config.add_attr('range_resolution', 1.5)
    config.add_attr('timestamp_correction', 30)
    config.add_attr('clutter_gates', 100)
    config.add_attr('code_length', 20000)
    config.plotting_source = '/'.join(file.split('/')[0:-1])+'/'
    config.plotting_destination = config.plotting_source + f'plots/{plotting_spacing}sec/'

    os.makedirs(config.plotting_source + f'plots/{plotting_spacing}sec', exist_ok=True)

    plotting_start, plotting_stop = util.get_data_file_times(file)
    print(config.plotting_destination, plotting_start, plotting_stop)
    time = util.Time(plotting_start, plotting_stop, plotting_step)
    icebear.range_doppler_snr(config, time, plotting_spacing)


