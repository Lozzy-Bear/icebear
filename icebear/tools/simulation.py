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


def simulate(config, azimuth, elevation, azimuth_extent, elevation_extent):
    """

    Parameters
    ----------
    config
    azimuth
    elevation
    azimuth_extent
    elevation_extent

    Returns
    -------

    """

    print('simulation start:')
    print("Number of processors: ", mp.cpu_count())
    print(f'\t-input azimuth {azimuth} deg x {azimuth_extent} deg')
    print(f'\t-input elevation {elevation} deg x {elevation_extent} deg')

    idx_length = len(azimuth)
    wavelength = 299792458 / config.center_freq
    u, v, w = utils.baselines(config.rx_ant_coords[0, :],
                              config.rx_ant_coords[1, :],
                              config.rx_ant_coords[2, :],
                              wavelength)
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    azimuth_extent = np.deg2rad(azimuth_extent)
    elevation_extent = np.deg2rad(elevation_extent)
    visibility_dist = np.zeros((int(len(u)/2), idx_length, idx_length, idx_length, idx_length), dtype=np.complex64)

    # Instantiate multi-core processing
    output = mp.Queue()
    pool = mp.Pool(mp.cpu_count() - 2)

    # Loop process to allow for multiple targets in an image. Typically only one target is used.
    for idx in range(idx_length):
        processes = [mp.Process(target=_visibility_calculation,
                                args=(x, u[x], v[x], w[x],
                                      azimuth[idx], azimuth_extent[idx],
                                      elevation[idx], elevation_extent[idx],
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
    coeffs = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, int(config.lmax))
    brightness = icebear.imaging.swht.swht_py(visibility, coeffs)

    # This section activates the experimental angular_frequency_beamforming()
    # for i in range(15, 95, 10):
    #     coeffs = icebear.imaging.swht.unpackage_factors_hdf5(config.swht_coeffs, i)
    #     brightness *= icebear.imaging.image.calculate_image(visibility, coeffs)

    brightness = icebear.imaging.swht.brightness_cutoff(brightness, threshold=0.0)
    cx, cy, cx_extent, cy_extent, area = icebear.imaging.swht.centroid_center(brightness)
    mx, my, _ = icebear.imaging.swht.max_center(brightness)

    mx = mx * config.resolution - config.fov[0, 0] + config.fov_center[0]
    my = my * config.resolution - config.fov[1, 0] + config.fov_center[1]
    cx = cx * config.resolution - config.fov[0, 0] + config.fov_center[0]
    cy = cy * config.resolution - config.fov[1, 0] + config.fov_center[1]
    cx_extent *= config.resolution
    cy_extent *= config.resolution
    area *= config.resolution ** 2
    print(f'\t-result azimuth {cx} deg x {cx_extent} deg -- max {mx}')
    print(f'\t-result elevation {cy} deg x {cy_extent} deg -- max {my}')
    print(f'\t-result area {area}')
    if len(azimuth) == 0:
        if np.allclose([azimuth, elevation, azimuth_extent, elevation_extent], [cx, cy, cx_extent, cy_extent], atol=5):
            print('\t-result matches input within error (10e-5)')

    return brightness


if __name__ == '__main__':
    path = Path(__file__).parent.parent.parent / "dat/default.yml"
    config = utils.Config(str(path))
    # Change this to you swht_coeffs local save

    coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85.h5'
    config.fov = np.array([[0, 360], [0, 180]])
    config.fov_center = np.array([90, 90])
    config.lmax = 85
    config.resolution = 1

    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_01_17_090az_045el_10res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_01_17_090az_045el_01res_218lmax.h5'
    # config.update_config(coeffs_file)
    # config.lmax = 85

    config.swht_coeffs = coeffs_file
    config.print_attrs()

    brightness = simulate(config, np.array([10]), np.array([10]), np.array([3]), np.array([3]))
    plt.figure()
    plt.pcolormesh(brightness)
    plt.colorbar()
    plt.show()
