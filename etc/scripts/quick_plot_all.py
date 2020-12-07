import icebear
import icebear.utils as util
import os

# Set plotting step size
plotting_step = [0, 0, 0, 1, 0]
plotting_spacing = 5

# Load the config file
configuration = 'X:/PythonProjects/icebear/dat/default.yml'
settings = 'X:/PythonProjects/icebear/dat/default.yml'
config = util.Config(configuration, settings)
config.print_attrs()

# Gather all the level1 data file paths
filepath = 'E:/icebear/level1/'
files = util.get_all_data_files(filepath, '2019_10_25', '2020_10_25')
# Create plots
for file in files:
    config.plotting_source = file.split('\\')[0] + '/'
    os.makedirs(config.plotting_source + f'plots/{plotting_spacing}sec', exist_ok=True)
    config.plotting_destination = config.plotting_source + f'plots/{plotting_spacing}sec/'
    plotting_start, plotting_stop = util.get_data_file_times(file)
    print(config.plotting_destination, plotting_start, plotting_stop)
    time = util.Time(plotting_start, plotting_stop, plotting_step)
    icebear.range_doppler_snr(config, time, plotting_spacing)


