import sys

import icebear.utils as utils
import icebear.imaging.image as ibi

# Set user inputs
year = -1
month = -1
day = -1
start_minute = 0
end_minute = 30
start_hour = 1
end_hour = 1
L1_data_path = "HERE"
L2_data_path = "HERE"
swht_coeffs = "FILE_PATH"
low_res_coeffs = "FILE_PATH"
config_file = "/mnt/icebear/processing_code/icebear/dat/default_processing.yml"

for arg in range(len(sys.argv)):
    if sys.argv[arg] == "-y":
        year = int(sys.argv[arg+1])
    elif sys.argv[arg] == "-m":
        month = int(sys.argv[arg+1])
    elif sys.argv[arg] == "-d":
        day = int(sys.argv[arg+1])
    elif sys.argv[arg] == "-sm":
        start_minute = int(sys.argv[arg + 1])
    elif sys.argv[arg] == "-sh":
        start_hour = int(sys.argv[arg + 1])
    elif sys.argv[arg] == "-em":
        end_minute = int(sys.argv[arg + 1])
    elif sys.argv[arg] == "-eh":
        end_hour = int(sys.argv[arg + 1])
    elif sys.argv[arg] == "--path-L1":
        L1_data_path = str(sys.argv[arg+1])
    elif sys.argv[arg] == "--path-L2":
        L2_data_path = str(sys.argv[arg+1])
    elif sys.argv[arg] == "--swht":
        swht_coeffs = str(sys.argv[arg+1])
    elif sys.argv[arg] == "--low-res":
        low_res_coeffs = str(sys.argv[arg+1])

if (year == -1) or (month == -1) or (day == -1) or (L1_data_path == "HERE") or (L2_data_path == "HERE") \
        or (swht_coeffs == "FILE_PATH") or (low_res_coeffs == "FILE_PATH"):
    print("needs 6 inputs: \n\t-y year \n\t-m month \n\t-d day \n\t--path-L1 L1 dir \n\t--path-L2 L2 dir "
          "\n\t--swht path to swht coeffs \n\t--low-res path to low resolution swht coeffs")
    sys.exit()

config = utils.Config(config_file)
config.update_attr("imaging_start", [year, month, day, start_hour, start_minute, 0, 0])
config.update_attr("imaging_stop", [year, month, day, end_hour, end_minute, 59, 0])
config.update_attr("imaging_source", L1_data_path)
config.update_attr("imaging_destination", L2_data_path)
config.update_attr("swht_coeffs", swht_coeffs)
config.add_attr("swht_coeffs_lowres", low_res_coeffs)
ibi.generate_level2(config, method='advanced')
