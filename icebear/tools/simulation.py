import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
import multiprocessing as mp
import icebear
import icebear.utils as utils
from pathlib import Path
import cv2


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
    x = config.rx_ant_coords[0, :]
    y = config.rx_ant_coords[1, :]
    z = config.rx_ant_coords[2, :]
    err = 0.0
    x[7] += err
    y[7] += err
    z[7] += err

    u, v, w = utils.baselines(x, #config.rx_ant_coords[0, :],
                              y, #config.rx_ant_coords[1, :],
                              z, #config.rx_ant_coords[2, :],
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

        # for p in processes:
        #     p.close()

    visibility = np.array(visibility_dist)
    for i in range(len(azimuth)):
        visibility[:, i, i, i, i] = visibility[:, i, i, i, i] / np.abs(visibility[0, i, i, i, i])
    visibility = np.sum(visibility_dist, axis=(1, 2, 3, 4))
    visibility = np.append(np.conjugate(visibility), visibility)
    coeffs = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, int(config.lmax))
    coeffs2 = np.copy(coeffs)
    brightness = icebear.imaging.swht.swht_py(visibility, coeffs)

    # This section activates the experimental angular_frequency_beamforming()
    # for i in range(15, 95, 10):
    #     coeffs = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, i)
    #     brightness *= icebear.imaging.image.calculate_image(visibility, coeffs)

    brightness = icebear.imaging.swht.brightness_cutoff(brightness, threshold=0.9)
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

    intensity = icebear.imaging.swht.swht_py(visibility, coeffs2)
    intensity = icebear.imaging.swht.brightness_cutoff(intensity, threshold=0.0)

    return brightness, intensity


if __name__ == '__main__':
    # Pretty plot configuration.
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    path = Path(__file__).parent.parent.parent / "dat/default.yml"
    config = utils.Config(str(path))
    # Change this to your swht_coeffs local save

    # coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_01_26_360az_180el_1res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85.h5'
    # config.fov = np.array([[0, 360], [0, 180]])
    # config.fov_center = np.array([0, 0])
    # config.lmax = 85
    # config.resolution = 1

    coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_02_09_090az_045el_01res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_02_09_090az_045el_01res_85lmax.h5'
    config.update_config(coeffs_file)
    config.lmax = 85
    config.fov_center = np.array([270, 90])

    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_01_26_360az_180el_1res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_01_17_090az_045el_10res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2021_02_16_090az_045el_1res_85lmax.h5'
    # config.update_config(coeffs_file)
    # config.lmax = 85

    config.swht_coeffs = coeffs_file
    config.print_attrs()

    brightness, intensity = simulate(config, np.array([0]), np.array([10]), np.array([3]), np.array([3]))

    # with open('test.npy', 'wb') as f:
    #     np.save(f, brightness)
    #     np.save(f, intensity)

    # with open('test.npy', 'rb') as f:
    #     brightness = np.load(f)
    #     intensity = np.load(f)
    #
    plt.figure(figsize=[12, 5])
    # plt.pcolormesh(intensity)
    plt.pcolormesh(brightness)
    # plt.xticks(np.arange(0, 900+50, 50), np.arange(-45, 50, 5))
    # plt.yticks(np.arange(0, 450+50, 50), np.arange(0, 50, 5))
    plt.xticks(np.arange(0, 900+50, 50), np.arange(-45, 50, 5))
    plt.yticks(np.arange(0, 450+50, 50), np.arange(0, 50, 5))
    plt.grid()
    im = np.array(brightness * 255.0, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a, b, w, h = cv2.boundingRect(contours[0])
    x = contours[0][:, :, 0]
    x = np.append(x, x[0])
    y = contours[0][:, :, 1]
    y = np.append(y, y[0])
    moments = cv2.moments(contours[0])
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    index = np.unravel_index(np.argmax(brightness, axis=None), brightness.shape)
    # plt.plot(x, y, 'k', linewidth=3, label='Contour')
    # plt.plot([a, a+w, a+w, a, a], [b, b, b+h, b+h, b], 'r', linewidth=3, label='Bounding Rectangle')
    # plt.scatter(cx, cy, label='Centroid')
    plt.scatter(index[1], index[0], label='Maximum Brightness')
    plt.colorbar(label='Normalized Brightness')
    plt.title('')
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    #plt.xlim(300, 600)
    #plt.ylim(100, 300)
    plt.legend(loc='best')
    # plt.show()
