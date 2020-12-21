import icebear.utils as util
import h5py
import os


filepath = 'E:/icebear/level1/'
files = util.get_all_data_files(filepath, '2019_10_26')
remove_flag = False
for file in files:
    f = h5py.File(file, 'r+')
    start1, stop1 = util.get_data_file_times(file)
    group = f['data']
    gkeys = group.keys()
    for gkey in gkeys:
        # print('initial:', group[f'{gkey}']['time'][()])
        t = group[f'{gkey}']['time'][()]
        t[2] -= 1000
        if t[2] < 0:
            t[2] = 59000
            t[1] -= 1
            if t[1] < 0:
                if t[0] == 24:
                    remove_flag = True
                t[1] = 59
                t[0] -= 1
                group[f'{gkey}']['time'][...] = t
                new_gkey = f'{t[0]:02d}{t[1]:02d}{t[2]:05d}'
                g = h5py.File(last_file, 'r+')
                g['data'].create_group(new_gkey)
                src_keys = group[f'{gkey}'].keys()
                for src_key in src_keys:
                    g['data'][f'{new_gkey}'].create_dataset(src_key, data=f['data'][f'{gkey}'][f'{src_key}'][()])
                del group[f'{gkey}']
                # print('changed:', g['data'][f'{new_gkey}']['time'][()])
                # print('--')
            else:
                group[f'{gkey}']['time'][...] = t
                new_gkey = f'{t[0]:02d}{t[1]:02d}{t[2]:05d}'
                group.move(f'{gkey}', f'{new_gkey}')
                # print('changed:', group[f'{new_gkey}']['time'][()])
                # print('--')
        else:
            group[f'{gkey}']['time'][...] = t
            new_gkey = f'{t[0]:02d}{t[1]:02d}{t[2]:05d}'
            group.move(f'{gkey}', f'{new_gkey}')
            # print('changed:', group[f'{new_gkey}']['time'][()])
            # print('--')

    last_file = file

    if remove_flag:
        f.close()
        os.remove(file)
        print('removed:', file)
    else:
        start2, stop2 = util.get_data_file_times(file)
        print('file formatted:', file)
        print('before:', start1, stop1)
        print('after:', start2, stop2)
        print('=======================================================================================================')
