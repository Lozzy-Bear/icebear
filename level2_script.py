import sys
import numpy as np

import icebear.utils as utils
import icebear.imaging.image as ibi

# Set user inputs
if len(sys.argv) != 7:
    print("needs 6 inputs: year, month, day, raw dir, L1 dir, swht coeff path, low res swht coeff path")
    sys.exit()

year=int(sys.argv[0])
month=int(sys.argv[1])
day=int(sys.argv[2])
L1_data_path=str(sys.argv[3])
L2_data_path=str(sys.arv[4])
swht_coeffs=str(sys.argv[5])
low_res_coeffs=str(sys.argv[6])
config_file="./dat/default_processing.yml"

config = utils.Config(config_file)
config.update_attr("imaging_start", [year,month,day,0,0,0,0])
config.update_attr("imaging_stop", [year,month,day,23,0,0,0])
config.update_attr("processing_source", L1_data_path)
config.update_attr("processing_destination", L2_data_path)
config.update_attr("swht_coeffs", swht_coeffs)
config.add_attr("swht_coeffs_lowres", low_res_coeffs)
ibi.generate_level2(config, method='advanced')
