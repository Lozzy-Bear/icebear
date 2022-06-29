import sys
import subprocess
import glob

import icebear.utils as utils
import icebear.processing.process as proc
import time
import cProfile

# Set user inputs
year = -1
month = -1
day = -1
start_minute = 0
end_minute = 30
start_hour = 1
end_hour = 1
hashing = False
profile = False
cuda = False
raw_data_path = "HERE"
L1_data_path = "HERE"
config_file = "/mnt/icebear/processing_code/icebear/dat/default_processing.yml"

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
	elif sys.argv[arg] == "-sm":
		start_minute = int(sys.argv[arg+1])
	elif sys.argv[arg] == "-sh":
		start_hour = int(sys.argv[arg+1])
	elif sys.argv[arg] == "-em":
		end_minute = int(sys.argv[arg+1])
	elif sys.argv[arg] == "-eh":
		end_hour = int(sys.argv[arg+1])
	elif sys.argv[arg] == "--hash":
		hashing = True
	elif sys.argv[arg] == "--profile":
		profile = True
	elif sys.argv[arg] == "--cuda":
		cuda = True

if (year == -1) or (month == -1) or (day == -1) or (raw_data_path == "HERE") or (L1_data_path == "HERE"):
	print("needs 5 inputs: \n\t-y year \n\t-m month \n\t-d day \n\t--path-raw raw dir \n\t--path-L1 L1 dir")
	sys.exit()

print(year)
print(month)
print(day)
print(raw_data_path)
print(L1_data_path)

config = utils.Config(config_file)
config.update_attr("processing_start", [year, month, day, start_hour, start_minute, 0, 0])
config.update_attr("processing_stop", [year, month, day, end_hour, end_minute, 59, 0])
config.update_attr("processing_source", raw_data_path)
config.update_attr("processing_destination", L1_data_path)
config.update_attr("cuda", cuda)  # cuda switch (If true uses cuda libssmf.so code, else uses cupy code)

start = time.perf_counter()
if profile:
	cProfile.run('proc.generate_level1(config)')
else:
	proc.generate_level1(config)
end = time.perf_counter()
print(f'time: {end - start}')


#These are to add to a hash file once the processing for a day finishes.
if hashing:
	l1_files = glob.glob(f"{year:0>4}_{month:0>2}_20/*")
	with open(f"{year:0>4}{month:0>2}_level1.hash", "a") as hash_file:
		for f in l1_files:
			hash_command = f"sha1sum {f}"
			process = subprocess.Popen(hash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			output, error = process.communicate()
			hash_file.write(output.decode('ascii'))

