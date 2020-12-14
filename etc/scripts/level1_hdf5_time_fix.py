import icebear.utils as util
import h5py
import os
import numpy as np

filepath = 'E:/icebear/level1/'
files = util.get_all_data_files(filepath, '2022_22_22')
for file in files:
    f = h5py.File(file, 'r+')
    print(file)

    # Rename datasets in data group
    group = f['data']
    gkeys = group.keys()
    for gkey in gkeys:
        #print(gkey, int(gkey) - 1000)
        print(group[f'{gkey}']['time'][()])
        t = group[f'{gkey}']['time'][()]
        t[2] -= 1000
        group[f'{gkey}']['time'][...] = t
        print(group[f'{gkey}']['time'][()])

