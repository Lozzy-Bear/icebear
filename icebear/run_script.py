import icebear.utils as utils
import numpy as np


import icebear.imaging.image as ibi
filepath = 'F:/icebear/level1/'  # Enter file path to level 1 directory
# filepath = '/beaver/backup/level1/'
date_dir = '2020_12_12'
files = utils.get_all_data_files(filepath, date_dir, date_dir)
print(f'files: {files}')

for file in files:
    config = utils.Config(file)
    config.add_attr('imaging_destination', 'F:/icebear/level2_advanced_cuda/')
    # config.add_attr('imaging_destination', '/beaver/backup/level2_1lambda/')
    config.add_attr('imaging_source', file)
    imaging_start, imaging_stop = utils.get_data_file_times(file)
    imaging_step = [0, 0, 0, 1, 0]
    config.add_attr('imaging_start', imaging_start)
    config.add_attr('imaging_stop', imaging_stop)
    config.add_attr('imaging_step', imaging_step)
    config.add_attr('lmax', 85)
    config.add_attr('resolution', 0.1)
    config.add_attr('image_method', 'sswht')
    config.add_attr('fov', np.array([[315, 225], [90, 45]]))
    config.add_attr('fov_center', np.array([270, 90]))
    config.add_attr('swht_coeffs', 'C:/Users/TKOCl/PythonProjects/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5')
    config.add_attr('swht_coeffs_lowres','C:/Users/TKOCl/PythonProjects/icebear/swhtcoeffs_ib3d_2021_10_19_360az_090el_10res_85lmax.h5')
    # config.add_attr('swht_coeffs', '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5')
    ibi.generate_level2(config, method='advanced')
