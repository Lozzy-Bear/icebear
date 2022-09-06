from two_point_diff import clustering_chunk
import numpy as np
import h5py
import matplotlib.pyplot as plt


filepath = '/home/arl203/icebear/testing_temp_data/2021_02_19/ib3d_normal_swht_2021_02_19_prelate_bakker.h5'
data = h5py.File(filepath, 'r')

time = data['data']['time'][()]
lat = data['data']['latitude'][()]
lon = data['data']['longitude'][()]
beam = data['dev']['beam'][()]
tclust = data['dev']['temporal_cluster'][()]
sclust = data['dev']['spatial_cluster'][()]

threshold = 500  # Amount of value in a given grouping under which to ignore
dt = 15 * 60.0  # Time in seconds to group points in time by (ex: 15 minute chunks)

time_bins = np.linspace(time[0], time[-1], int(np.ceil((time[-1] - time[0])/dt)))
digitized_time = np.digitize(time, time_bins)
distance_bins = np.append(np.linspace(0, 100, 51), np.linspace(110, 300, 39))
distance_bins = np.append(distance_bins, np.linspace(350, 1000, 44))

bin_centers = [(distance_bins[i] + distance_bins[i + 1]) / 2. for i in range(len(distance_bins) - 1)]

for j in range(1, int(np.max(beam)) + 1, 1):  # beams 1, 2, 3
    for i in range(1, digitized_time[-1] + 1, 1):  # time slices 1 - 55
        idx = np.argwhere((digitized_time == i) & (beam == j))
        if len(lat[idx]) < threshold:
            continue
        p1 = np.squeeze(np.array([lat[idx], lon[idx]]).T)
        p2 = np.squeeze(np.array([lat[idx], lon[idx]]).T)
        h = clustering_chunk(p1, p2, bins=distance_bins)

        plt.figure()
        plt.step(bin_centers, h, where='mid', color='k')
        plt.xlabel("Separation distance bins [km]")
        plt.ylabel("Counts")
        plt.title(f"Beam: {j}, Time Chunk: {i}")
        plt.show()







