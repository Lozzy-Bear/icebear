import numpy as np
import h5py
import matplotlib.pyplot as plt
from icebear import utils


if __name__ == '__main__':
    # Load the level 2 data file.
    filepath = '/beaver/backup/level1/'  # Enter file path to level 2 directory
    files = utils.get_all_data_files(filepath, '2020_03_31', '2020_03_31')  # Enter first sub directory and last
    # files = ['/beaver/backup/level1/2020_03_31/ib3d_normal_01dB_1000ms_2020_03_31_00_prelate_bakker.h5']
    for file in files[0:1]:
        f = h5py.File(file, 'r')
        keys = list(f['data'].keys())
        for key in keys:
            data = f['data'][key]
            if data['data_flag'][()]:
                spectra = np.array(data['spectra'][:, 0], dtype=np.complex64)
                spectra_corr = np.array(data['spectra_clutter_corr'][0], dtype=np.complex64)
                s = spectra - spectra_corr
                xspectra = np.array(data['xspectra'][:, :], dtype=np.complex64)
                xspectra_corr = np.array(data['xspectra_clutter_corr'][:], dtype=np.complex64)
                # x = xspectra - xspectra_corr
                # x = np.einsum('ij,j->j', xspectra, -1* xspectra_corr)
                x = xspectra - xspectra_corr
                print(key, x, xspectra, xspectra_corr)
                # print(key, spectra.shape, spectra_corr.shape, xspectra.shape, xspectra_corr.shape)
