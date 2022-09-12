from two_point_diff import clustering_chunk, poisson_points
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse


def main(filepath, dt=900, threshold=500):
    data = h5py.File(filepath, 'r')
    time = data['data']['time'][()]
    lat = data['data']['latitude'][()]
    lon = data['data']['longitude'][()]
    beam = data['dev']['beam'][()]

    time_bins = np.linspace(time[0], time[-1], int(np.ceil((time[-1] - time[0]) / dt)))
    digitized_time = np.digitize(time, time_bins)
    distance_bins = np.append(np.arange(0, 100 + 2, 2), np.arange(105, 300 + 5, 5))
    distance_bins = np.append(distance_bins, np.arange(315, 1000 + 15, 15))

    for beam_num in range(1, int(np.max(beam)) + 1, 1):  # beams 1, 2, 3
        for time_slice in range(1, digitized_time[-1] + 1, 1):  # time slices 1 - 55
            idx = np.argwhere((digitized_time == time_slice) & (beam == beam_num))
            if len(lat[idx]) < threshold:
                continue
            p1 = np.hstack((lat[idx], lon[idx]))
            p2 = poisson_points(lat[idx], lon[idx])
            num_real = p1.shape[0]
            num_random = p2.shape[0]

            real_to_real = clustering_chunk(np.deg2rad(p1), np.deg2rad(p1),
                                            bins=distance_bins, r=105.0)
            random_to_random = clustering_chunk(np.deg2rad(p2), np.deg2rad(p2),
                                                bins=distance_bins, r=105.0)
            real_to_random = clustering_chunk(np.deg2rad(p1), np.deg2rad(p2),
                                              bins=distance_bins, r=105.0)

            dd = real_to_real / (num_real * (num_real - 1) * 0.5)
            rr = random_to_random / (num_random * (num_random - 1) * 0.5)
            dr = real_to_random / (num_real * num_random)
            xi = (dd - 2 * dr + rr) / rr

            bin_centers = [(distance_bins[i] + distance_bins[i + 1]) / 2. for i in range(len(distance_bins) - 1)]
            plt.figure(figsize=[12, 8])
            plt.suptitle(f"Beam: {beam_num}, Time Slice Number: {time_slice}")
            plt.subplot(121)
            plt.semilogx(bin_centers, xi)
            plt.xlabel("Log separation distance bins [km]")
            plt.ylabel("ξ")
            plt.subplot(122)
            plt.step(bin_centers, real_to_real, where='mid', color='b', label='DD')
            plt.step(bin_centers, random_to_random, where='mid', color='r', label='RR')
            plt.step(bin_centers, real_to_random, where='mid', color='k', label='DR')
            plt.xlabel("Separation distance bins [km]")
            plt.ylabel("Counts")
            plt.legend()
            plt.show()
    return


if __name__ == '__main__':

    files = ['/home/arl203/icebear/testing_temp_data/2021_02_19/ib3d_normal_swht_2021_02_19_prelate_bakker.h5']
    for file in files:
        main(file)
    # todo: output data -- start time, end time of chunk, beam, xi, dt, distance_bins, date
    # todo: process first 2021-02-22 then do everything


# print('mean point: ', np.mean(p2[:, 0]), np.mean(p2[:, 1]))
# plt.figure(1)
# plt.subplot(121)
# plt.scatter(p1[:, 1], p1[:, 0])
# plt.scatter(np.mean(p1[:, 1]), np.mean(p1[:, 0]))
# plt.ylabel('Latitude')
# plt.xlabel('Longitude')
# plt.xlim((-110, -100))
# plt.ylim((56, 62))
# plt.title('Real Data')
# plt.subplot(122)
# plt.scatter(p2[:, 1], p2[:, 0])
# plt.scatter(np.mean(p2[:, 1]), np.mean(p2[:, 0]))
# plt.ylabel('Latitude')
# plt.xlabel('Longitude')
# plt.title('Random Data')
# plt.xlim((-110, -100))
# plt.ylim((56, 62))
# plt.figure(2)
# plt.hist(p2[:, 1], bins=np.arange(-110.0, -100.0, 0.01))
# plt.figure(3)
# plt.hist(p2[:, 0], bins=np.arange(56.0, 62.0, 0.01))
# plt.show()


# plt.figure(figsize=[12, 8])
# plt.suptitle(f"Beam: {beam_num}, Time Slice Number: {time_slice}")
# plt.subplot(121)
# plt.semilogx(bin_centers, xi)
# plt.xlabel("Log separation distance bins [km]")
# plt.ylabel("ξ")
# plt.subplot(122)
# plt.step(bin_centers, real_to_real, where='mid', color='b', label='DD')
# plt.step(bin_centers, random_to_random, where='mid', color='r', label='RR')
# plt.step(bin_centers, real_to_random, where='mid', color='k', label='DR')
# plt.xlabel("Separation distance bins [km]")
# plt.ylabel("Counts")
# plt.legend()
# plt.show()