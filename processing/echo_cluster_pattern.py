import datetime
import glob
from two_point_diff import clustering_chunk, poisson_points
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats


def kernel_density_estimation(points):
    lat = points[:, 0]
    lon = points[:, 1]
    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lon_min = np.min(lon)
    lon_max = np.max(lon)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[lon_min:lon_max:100j, lat_min:lat_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([lon, lat])
    kernel = stats.gaussian_kde(values)
    z = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[lon_min, lon_max, lat_min, lat_max])
    # ax.plot(lon, lat, 'k.', markersize=2)
    ax.set_ylim([lat_min, lat_max])
    ax.set_xlim([lon_min, lon_max])
    plt.show()
    return


def do_calc(time, lat, lon, beam, distance_bins=None, dt=900, threshold=500):
    xo = np.array([])
    bn = np.array([])
    ti = np.array([])
    tf = np.array([])
    nd = np.array([])
    nr = np.array([])

    if distance_bins is None:
        distance_bins = np.append(np.arange(0, 100 + 2, 2), np.arange(110, 300 + 5, 5))
        distance_bins = np.append(distance_bins, np.arange(350, 1000 + 15, 15))

    time_bins = np.linspace(time[0], time[-1], int(np.ceil((time[-1] - time[0]) / dt)))
    digitized_time = np.digitize(time, time_bins)

    for beam_num in range(1, int(np.max(beam)) + 1, 1):  # beams 1, 2, 3
        for time_slice in range(1, digitized_time[-1] + 1, 1):
            idx = np.argwhere((digitized_time == time_slice) & (beam == beam_num))
            if len(lat[idx]) < threshold:
                print(f"\tskipped: less than {threshold} points in beam {beam_num}, slice {time_slice}")
                continue
            p1 = np.hstack((lat[idx], lon[idx]))
            p2 = poisson_points(lat[idx], lon[idx])
            p1 = np.unique(p1, axis=0)
            p2 = np.unique(p2, axis=0)
            num_real = p1.shape[0]
            num_random = p2.shape[0]

            print(f'\tpoints shape: {p1.shape} {p2.shape} for beam {beam_num}, slice {time_slice}')
            # kernel_density_estimation(p1)
            # kernel_density_estimation(p2)

            real_to_real = clustering_chunk(np.deg2rad(p1), np.deg2rad(p1), bins=distance_bins, r=105.0, mode='upper')
            random_to_random = clustering_chunk(np.deg2rad(p2), np.deg2rad(p2), bins=distance_bins, r=105.0, mode='upper')
            real_to_random = clustering_chunk(np.deg2rad(p1), np.deg2rad(p2), bins=distance_bins, r=105.0, mode='upper')

            dd = real_to_real / (num_real * (num_real - 1) * 0.5)
            rr = random_to_random / (num_random * (num_random - 1) * 0.5)
            dr = real_to_random / (num_real * num_random)
            xi = (dd - 2 * dr + rr) / rr - 1

            xo = np.append(xo, xi)
            bn = np.append(bn, np.ones(xi.shape, dtype=int) * beam_num)
            ti = np.append(ti, np.ones(xi.shape, dtype=float) * (time[0] + ((time_slice - 1) * dt)))
            tf = np.append(tf, np.ones(xi.shape, dtype=float) * (time[0] + ((time_slice - 1) * dt + dt - 1)))
            nd = np.append(nd, np.ones(xi.shape, dtype=float) * num_real)
            nr = np.append(nr, np.ones(xi.shape, dtype=float) * num_random)

            # bin_centers = [(distance_bins[i] + distance_bins[i + 1]) / 2. for i in range(len(distance_bins) - 1)]
            # plt.figure(figsize=[12, 8])
            # m1 = time[0] + (time_slice - 1) * dt
            # m2 = time[0] + (time_slice - 1) * dt + dt - 1
            # plt.suptitle(f"Beam: {beam_num}, Time: {m1 / (60 * 60)}  {m2 / (60 * 60)}")
            # plt.subplot(121)
            # plt.semilogx(bin_centers, xi)
            # plt.xlabel("Log separation distance bins [km]")
            # plt.ylabel("xi")
            # plt.subplot(122)
            # # plt.step(bin_centers, real_to_real, where='mid', color='b', label='DD')
            # # plt.step(bin_centers, random_to_random, where='mid', color='r', label='RR')
            # # plt.step(bin_centers, real_to_random, where='mid', color='k', label='DR')
            # plt.semilogx(bin_centers, dd, color='b', label='DD')
            # plt.semilogx(bin_centers, rr, color='r', label='RR')
            # plt.semilogx(bin_centers, dr, color='k', label='DR')
            # plt.xlabel("Separation distance bins [km]")
            # plt.ylabel("Counts")
            # plt.legend()
            # plt.show()

    return xo, bn, ti, tf, nd, nr


if __name__ == '__main__':
    # Load files to be processed
    files = glob.glob("/data/outness/*")
    files = sorted(files)
    print('files loaded:', files)

    # Config parameters
    outdir = f'/data/echo_cluster/'
    process_date = datetime.datetime.utcnow()
    process_date = np.array([process_date.year, process_date.month, process_date.day])
    dt = 6 * 60
    threshold = 10_000
    distance_bins = np.append(np.arange(0, 100 + 2, 2), np.arange(110, 300 + 5, 5))
    distance_bins = np.append(distance_bins, np.arange(350, 1000 + 15, 15))

    for file in files:
        print("file:", file)
        data = np.genfromtxt(file, dtype=float, delimiter=',')
        if data.shape[0] == 0:
            print("SKIPPING; empty file, shape =", data.shape)
            continue

        time = data[:, 0]
        mag_lon = data[:, 1]
        mag_lat = data[:, 2]
        beam = data[:, 3]
        time = (time - np.floor(time[0])) * 24 * 60 * 60
        print('read data shape:', time.shape, mag_lat.shape, mag_lon.shape, beam.shape)

        xi, b, ti, tf, nd, nr = do_calc(time, mag_lat, mag_lon, beam,
                                        distance_bins=distance_bins, dt=dt, threshold=threshold)

        # Write config data
        date = file.split('/')[-1]
        date = np.array([int(date[0:4]), int(date[4:6]), int(date[6:8])])
        outfile = outdir + file.split('/')[-1].split('.')[0] + '_cluster.h5'
        of = h5py.File(outfile, 'w')
        of.create_group('config')
        of.create_dataset('info/processed_date', data=process_date)
        of.create_dataset('info/time_spacing', data=dt)
        of.create_dataset('info/minimum_points', data=threshold)
        of.create_dataset('info/distance_bins', data=distance_bins)
        of.create_dataset(f'info/date', data=date)
        of.create_group('data')
        of.create_dataset('data/xi', data=xi)
        of.create_dataset('data/beam', data=b)
        of.create_dataset('data/time_start', data=ti)
        of.create_dataset('data/time_end', data=tf)
        of.create_dataset('data/num_real', data=nd)
        of.create_dataset('data/num_random', data=nr)
        of.close()
