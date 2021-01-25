import icebear.utils as utils
import icebear
import numpy as np


filepath = 'E:/icebear/level1/'  # Enter file path to level 1 directory
files = utils.get_all_data_files(filepath, '2020_12_12', '2020_12_15')  # Enter first sub directory and last
for file in files:
    config = icebear.utils.Config(file)
    config.lmax = 85
    config.add_attr('swht_coeffs', 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_01_17_090az_045el_01res_218lmax.h5')

    config.add_attr('resolution', 0.1)
    config.add_attr('fov', np.array([[135, 45], [90, 45]]))
    config.add_attr('fov_center', np.array([90, 90]))   # Should be removed with future coeff files that have fov_center

    imaging_start, imaging_stop = utils.get_data_file_times(file)
    imaging_step = [0, 0, 0, 1, 0]
    config.add_attr('imaging_start', imaging_start)
    config.add_attr('imaging_stop', imaging_stop)
    config.add_attr('imaging_step', imaging_step)
    config.add_attr('imaging_destination', 'E:/icebear/level2/')
    config.add_attr('imaging_source', file)
    config.add_attr('image_method', 'swht')

    icebear.imaging.image.generate_level2(config, method='swht')
