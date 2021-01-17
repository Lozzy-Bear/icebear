import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
import multiprocessing as mp
import icebear
import icebear.utils as utils
from pathlib import Path


def _real_pre_integrate(theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
    return 2 * np.real(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) * 
                       np.exp(-(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) * 
                       np.exp(-2.0j * np.pi * ((u_in * np.sin(theta) * np.cos(phi)) + 
                                                (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(phi)))))


def _imag_pre_integrate(theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
    return 2 * np.imag(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) * 
                       np.exp(-(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) * 
                       np.exp(-2.0j * np.pi * ((u_in * np.sin(theta) * np.cos(phi)) + 
                                                (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(phi)))))


def _visibility_calculation(x, u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread, output):
    real_vis = dblquad(_real_pre_integrate, -np.pi / 2, np.pi / 2, lambda phi: -np.pi, lambda phi: np.pi,
                       args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
    imag_vis = dblquad(_imag_pre_integrate, -np.pi / 2, np.pi / 2, lambda phi: -np.pi, lambda phi: np.pi,
                       args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
    output.put((x, real_vis + imag_vis * 1.0j))


def simulate(config, azimuth, elevation, azimuth_spread, elevation_spread):
    """

    Parameters
    ----------
    config
    azimuth
    elevation
    azimuth_spread
    elevation_spread

    Returns
    -------

    """

    print('simulation start:')
    print("Number of processors: ", mp.cpu_count())
    print(f'\t-input azimuth {azimuth} deg x {azimuth_spread} deg')
    print(f'\t-input elevation {elevation} deg x {elevation_spread} deg')

    idx_length = len(azimuth)
    wavelength = 299792458 / config.center_freq
    u, v, w = utils.baselines(config.rx_ant_coords[0, :],
                              config.rx_ant_coords[1, :],
                              config.rx_ant_coords[2, :],
                              wavelength)
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    azimuth_spread = np.deg2rad(azimuth_spread)
    elevation_spread = np.deg2rad(elevation_spread)
    visibility_dist = np.zeros((int(len(u)/2), idx_length, idx_length, idx_length, idx_length), dtype=np.complex64)

    # Instantiate multi-core processing
    output = mp.Queue()
    pool = mp.Pool(mp.cpu_count() - 2)

    # Loop process to allow for multiple targets in an image. Typically only one target is used.
    for idx in range(idx_length):
        processes = [mp.Process(target=_visibility_calculation,
                                args=(x, u[x], v[x], w[x],
                                      azimuth[idx], azimuth_spread[idx],
                                      elevation[idx], elevation_spread[idx],
                                      output)) for x in range(int(len(u)/2))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Get process results from the output queue
        visibility_dist_temp = [output.get() for p in processes]

        visibility_dist_temp.sort()
        visibility_dist[:, idx, idx, idx, idx] = [r[1] for r in visibility_dist_temp]

        for p in processes:
            p.close()

    visibility = np.array(visibility_dist)
    for i in range(len(azimuth)):
        visibility[:, i, i, i, i] = visibility[:, i, i, i, i] / np.abs(visibility[0, i, i, i, i])
    visibility = np.sum(visibility_dist, axis=(1, 2, 3, 4))
    visibility = np.append(visibility, np.conjugate(visibility))
    coeffs = icebear.imaging.swht.unpackage_factors_hdf5(config.swht_coeffs, int(config.lmax))
    brightness = icebear.imaging.swht.swht_py(visibility, coeffs)

    # This section activates the experimental angular_frequency_beamforming()
    # for i in range(15, 95, 10):
    #     coeffs = icebear.imaging.swht.unpackage_factors_hdf5(config.swht_coeffs, i)
    #     brightness *= icebear.imaging.image.calculate_image(visibility, coeffs)

    brightness = icebear.imaging.image.brightness_cutoff(brightness, threshold=0.8)
    cx, cy, cx_spread, cy_spread, area = icebear.imaging.image.centroid_center(brightness)

    print(f'\t-result azimuth {cx} deg x {cx_spread} deg')
    print(f'\t-result elevation {cy} deg x {cy_spread} deg')
    print(f'\t-result area {area}')
    if len(azimuth) == 0:
        if np.allclose([azimuth, elevation, azimuth_spread, elevation_spread], [cx, cy, cx_spread, cy_spread], atol=5):
            print('\t-result matches input within error (10e-5)')

    return brightness


if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent / "dat/default.yml"
    config = utils.Config(str(path))

    # Change this to you swht_coeffs local save
    config.swht_coeffs = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85'
    config.lmax = 85

    brightness = simulate(config, np.array([20, -20, 0]), np.array([15, 15, 15]), np.array([3, 3, 3]), np.array([3, 3, 3]))
    plt.figure()
    plt.pcolormesh(brightness)
    plt.colorbar()
    plt.show()
