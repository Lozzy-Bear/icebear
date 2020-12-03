import icebear
import icebear.utils as util


# Set plotting step size
plotting_step = [0, 0, 0, 1, 0]

# Load the config file
configuration = 'X:/PythonProjects/icebear/dat/default.yml'
settings = 'X:/PythonProjects/icebear/dat/default.yml'
config = util.Config(configuration, settings)
config.print_attrs()

# Gather all the level1 data file paths
filepath = 'E:/icebear/level1/'
files = util.get_all_data_files(filepath, '2019_10_24', '2020_07_14')

# Create plots
for file in files:
    config.plotting_source = file.split('\\')[0] + '/'
    print(config.plotting_source)
    plotting_start, plotting_stop = util.get_data_file_times(file)
    #time = util.Time(plotting_start, plotting_stop, plotting_step)
    #icebear.range_doppler_snr(config, time)
