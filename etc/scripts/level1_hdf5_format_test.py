import h5py


#new = 'E:/icebear/level1/2020_07_06/ib3d_normal_prelate_bakker_01dB_1000ms_2020_07_06_17.h5'
#new = 'E:/icebear/code/ib3d_mobile_truck_bakker_12dB_1000ms_2020_08_21_20.h5'
new = 'E:/icebear/level1/2022_22_22/icebear_3d_01dB_1000ms_vis_2020_07_06_18_prelate_bakker.h5'


#fold = h5py.File(old, 'r')
fnew = h5py.File(new, 'r')

new_keys = fnew.keys()
for new_key in new_keys:
    try:
        print(new_key, fnew[f'{new_key}'][:])
    except:
        print(f'{new_key} BUSTED')

group = fnew['data']
gkeys = group.keys()
for gkey in gkeys:
    gg = group[f'{gkey}'].keys()
    #print(gg)
    for g in gg:
        try:
            popopopop = group[f'{gkey}'][f'{g}'][:]
            print(f'{g} PASSED')
        except:
            print(f'{g} BUSTED')
