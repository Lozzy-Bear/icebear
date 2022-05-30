import sys
import numpy as np

import icebear.utils as utils
import icebear.processing.process as proc

# Set user inputs

year=1995
month=9
day=1
raw_data_path="HERE"
L1_data_path="HERE"
config_file="/mnt/ICEBEAR_datastore/processing_code/icebear/dat/default_processing.yml"

for arg in range(len(sys.argv)):
	if sys.argv[arg] == "-y":
		year = int(sys.argv[arg+1])
	elif sys.argv[arg] == "-m":
		month = int(sys.argv[arg+1])
	elif sys.argv[arg] == "-d":
		day = int(sys.argv[arg+1])
	elif sys.argv[arg] == "--path-raw":
		raw_data_path = str(sys.argv[arg+1])
	elif sys.argv[arg] == "--path-L1":
		L1_data_path = str(sys.argv[arg+1])
	
print(year)
print(month)
print(day)
print(raw_data_path)
print(L1_data_path)

config = utils.Config(config_file)
config.update_attr("processing_start", [year,month,day,1,0,0,0])
config.update_attr("processing_stop", [year,month,day,2,0,0,0])
config.update_attr("processing_source", raw_data_path)
config.update_attr("processing_destination", L1_data_path)
proc.generate_level1(config)
