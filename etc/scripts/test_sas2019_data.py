import pydarnio
import pydarn
import matplotlib.pyplot as plt
import h5py
import numpy as np

# file = '/home/glatteis/SuperDARN/code/borealis_postprocessors/20190827.0200.01.sas.1.rawacf.hdf5.array'
file = '/home/glatteis/SuperDARN/code/borealis_postprocessors/20190428.0000.00.sas.0.rawacf.hdf5.array.hdf5'

# f = h5py.File(file, 'a')
# f.create_dataset/('agc_status_word', data=0)
# f.attrs['agc_status_word'] = np.int(0)
# f.attrs['borealis_git_hash'] = 'v0.3-143-g0ed7503'
# print(githash)
# githash = githash.split('.')
# githash = '.'.join(githash[0]) + '6'.join(githash[2::])
# print(githash)
# exit()
# f.close()
# keys = list(f.keys())
# print(keys, f.attrs.keys())
# print(f.attrs['borealis_git_hash'])
# # exit()
# data = pydarn.SuperDARNRead().read_borealis(file, 0)

dmap = pydarnio.BorealisConvert(file, 'rawacf', file+'.dmap', 0, borealis_file_structure='array')
data = dmap.sdarn_dmap_records
pydarn.RTP.plot_range_time(data, beam_num=0, parameter='pwr0', cmap='gnuplot')

