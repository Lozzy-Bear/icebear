import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
import multiprocessing as mp
from pathlib import Path
import time
import sys
import path as pth
ipath = pth.Path(__file__).abspath()
sys.path.append(ipath.parent.parent.parent)
import icebear
import icebear.utils as utils


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


def clean(dirty_image, dirty_beam, threshold_peak, iteration=10, damping=0.5):
    dirty_image = np.abs(dirty_image)
    for i in range(iteration):
        mx, my, _ = icebear.imaging.swht.max_center(dirty_image)
        max_peak = dirty_image[mx, my]
        dirty_image -= dirty_beam * max_peak * gamma
        if max_peak < threshold_peak:
            break
    brightness = np.convolve(dirty_image, ideal_beam)

    return brightness


def overlapIndices(a1, a2, shiftx, shifty):
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]

    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]

    return (int(a1xbeg), int(a1xend), int(a1ybeg), int(a1yend)), (int(a2xbeg), int(a2xend), int(a2ybeg), int(a2yend))


def hogbom(dirty, psf, thresh, gain=0.7, niter=1000):
    """
    Hogbom clean

    :param dirty: The dirty image, i.e., the image to be deconvolved

    :param psf: The point spread-function

    :param window: Regions where clean components are allowed. If
    True, thank all of the dirty image is assumed to be allowed for
    clean components

    :param gain: The "loop gain", i.e., the fraction of the brightest
    pixel that is removed in each iteration

    :param thresh: Cleaning stops when the maximum of the absolute
    deviation of the residual is less than this value

    :param niter: Maximum number of components to make if the
    threshold "thresh" is not hit
    """
    comps = np.zeros(dirty.shape)
    res = np.array(np.abs(dirty))
    for i in range(niter):
        mx, my = np.unravel_index(np.fabs(res).argmax(), res.shape)
        mval = res[mx, my] * gain
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf,
                                  mx - dirty.shape[0] / 2,
                                  my - dirty.shape[1] / 2)
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if np.fabs(res).max() < thresh:
            print('thresh met:', i)
            break
    return comps, res


def dcos2deg(dcosx):
    return 90 - np.rad2deg(np.arccos(dcosx))


def beamforming(x, k):
    """
    Inputs:
        x :   Coordinates of each antenna in lambdas.
              Dimension (3, nreceivers)
                [(x1, x2, ...); (y1, y2, ...); (z1, z2, ..)]
        k :   array-like (3, nx, ny). Beam direction [dcosx, dcosy, dcosz] represented as direction cosine
               "cos(\Theta_x), cos(\Theta_y), cos(\Theta_z)".
    Return:
        power:  array-like (nx, ny) from 0 to 1. Return the point spreadf function.
                Which is the signal power as a function of angle of arrival (beam direction),
                assuming a point-like target.
    """
    idx = np.where(~np.isfinite(k[2]))

    k[2, idx[0], idx[1]] = 0
    vk = np.exp(np.einsum('di,dxy->ixy', 2j * np.pi * x, k))  # * np.einsum('di,dxy->ixy',el,k)
    p = np.einsum('ixy,jxy->xy', vk, np.conjugate(vk))
    p = np.abs(p)

    return p / np.max(p)


def point_spread_function(ant_coordinates, l_max=0.4, m_max=0.4, nl=200, nm=200):
    """
    Inputs:
        ant_coordinates :   Coordinates of each antenna in lambdas.
                            Dimension (3, nreceivers)
                                [(x1, x2, ...); (y1, y2, ...); (z1, z2, ..)]
        l_max, m_max    :   Define the maximum direction cosine in the x and y dimension.
                            It accepts values from 0 to 1, where 0 = 0 degrees and 1 = 90 degrees.

        nl, nm          :    Number of grid points in the x and y dimension.
    Return:
        power:  array-like (nl, nm) from 0 to 1. Return the point spreadf function.
                Which is the signal power as a function of angle of arrival (beam direction),
                assuming a point-like target.
    """

    nr = len(ant_coordinates)

    # Define the antenna coordinates
    x = np.array([np.real(ant_coordinates), np.imag(ant_coordinates), np.zeros(nr)])

    # Define the x and y grid
    l = np.linspace(-l_max, l_max, nl)
    m = np.linspace(-m_max, m_max, nm)

    L, M = np.meshgrid(l, m, indexing='ij')
    k = np.array([L, M, np.sqrt(1.0 - L ** 2 - M ** 2)])

    # Estimate the point spread function
    psf = beamforming(x, k)
    l = dcos2deg(l)
    m = dcos2deg(m)

    return l, m, psf


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
            # p.close()

    visibility = np.array(visibility_dist)
    for i in range(len(azimuth)):
        visibility[:, i, i, i, i] = visibility[:, i, i, i, i] / np.abs(visibility[0, i, i, i, i])
    visibility = np.sum(visibility_dist, axis=(1, 2, 3, 4))
    visibility = np.append(np.conjugate(visibility), visibility)
    coeffs = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, int(config.lmax))
    coeffs2 = np.copy(coeffs)

    start_time = time.time()
    brightness = icebear.imaging.swht.swht_py(visibility, coeffs)
    print('time:', time.time() - start_time)

    # This section activates the experimental angular_frequency_beamforming()
    for i in range(15, 85, 10):
        coeffs = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, i)
        brightness *= icebear.imaging.swht.swht_py(visibility, coeffs)

    # Use the CLEAN mehthod instead of SSWHT
    # _, _, psf = point_spread_function((x + 1j*y), 1.0, 1.0, 450, 900)
    # comps, brightness = hogbom(brightness, psf, np.mean(np.abs(brightness)))

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
        if np.allclose([azimuth, elevation, azimuth_extent, elevation_extent], [cx, cy, cx_extent, cy_extent], atol=1):
            print('\t-result matches input within error (10e-1)')

    intensity = icebear.imaging.swht.swht_py(visibility, coeffs2)
    print(f'\t-max intentsity {np.max(intensity)} mean intensity {np.mean(intensity)}')
    intensity = icebear.imaging.swht.brightness_cutoff(intensity, threshold=0.0)

    return brightness, intensity, mx, my


if __name__ == '__main__':
    # Pretty plot configuration.
    from matplotlib import rc

    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # path = Path(__file__).parent.parent.parent / "dat/default.yml"
    path = ipath.parent.parent.parent + "/dat/default.yml"
    print(path)
    config = utils.Config(str(path))
    # Change this to your swht_coeffs local save

    # coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_01_26_360az_180el_1res_85lmax.h5'
    # coeffs_file = 'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85.h5'
    # config.fov = np.array([[0, 360], [0, 180]])
    # config.fov_center = np.array([0, 0])
    # config.lmax = 85
    # config.resolution = 1

    # 0.1 degree resolution normal fov coeffs
    coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5'
    # 1.0 degree resolution half sphere coeffs
    # coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_10_19_360az_090el_10res_85lmax.h5'
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

    # This code is used to check all possible angles of arrival
    # with open('f1deg.csv', 'a') as csv:
    #     print('Testing all possible angles of arrival at 0.1 deg resolution (-45 - 45 az, 0 - 45 el):')
    #     csv.write('x,y,cx,cy\n')
    #     # xx = np.arange(-45.0, 45.0, config.resolution)
    #     # yy = np.arange(0.0, 45.0, config.resolution)
    #     xx = np.arange(-30, 30+1, config.resolution)
    #     yy = np.arange(0, 20, config.resolution)
    #     for x in xx:
    #         for y in yy:
    #             if x < 30.0:
    #                 continue
    #             if x <= 30.0 and y < 0.0:
    #                 continue
    #             print(f'computing: x={x}, y={y}')
    #             _, _, cx, cy = simulate(config, np.array([x]), np.array([y]), np.array([3, 3]), np.array([3, 3]))
    #             csv.write(f'{x},{y},{cx},{cy}\n')
    # exit

    # This code checks target spreads
    # with open('elevation_extent_acc.csv', 'a') as csv:
    #     print('Testing all possible angles of arrival at 0.1 deg resolution (-45 - 45 az, 0 - 45 el):')
    #     csv.write('x,y,sx,sy,mx,my\n')
    #     spreads = np.arange(1.0, 5.0+0.1, 0.1)
    #     x = 0.0
    #     y = 10.0
    #     sx = 1.0
    #     for sy in spreads:
    #         print(f'computing: x={x}, y={y}')
    #         _, _, mx, my = simulate(config, np.array([x]), np.array([y]), np.array([sx]), np.array([sy]))
    #         csv.write(f'{x},{y},{sx},{sy},{mx},{my}\n')
    # exit()

    brightness, intensity, cx, cy = simulate(config, np.array([-10, 10]), np.array([10, 10]), np.array([3,3]), np.array([1,1]))

    plt.figure(figsize=[9, 8])
    plt.subplot(211)
    plt.pcolormesh(intensity, cmap='inferno', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Normalized Brightness')
    # Plot the contour, centroid, and bounding box for extent, area, and centroid locating
    # im = np.array(intensity * 255.0, dtype=np.uint8)
    # threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # a, b, w, h = cv2.boundingRect(contours[0])
    # x = contours[0][:, :, 0]
    # x = np.append(x, x[0])
    # y = contours[0][:, :, 1]
    # y = np.append(y, y[0])
    # moments = cv2.moments(contours[0])
    # cx = int(moments['m10'] / moments['m00'])
    # cy = int(moments['m01'] / moments['m00'])
    # plt.plot(x, y, 'k', linewidth=2, label='Contour')
    # plt.plot([a, a+w, a+w, a, a], [b, b, b+h, b+h, b], 'r', linewidth=3, label='Bounding Rectangle')
    # plt.scatter(cx, cy, label='Centroid')
    # Plot the maximum point
    index = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
    plt.scatter(index[1], index[0], label='Maximum Brightness', c='grey')
    print(index)
    # Set up plot labels and bars
    plt.xticks(np.arange(0, 900+50, 50), np.arange(-45, 50, 5))
    plt.yticks(np.arange(0, 450+50, 50), np.arange(0, 50, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.grid(linestyle='--')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.pcolormesh(brightness, cmap='inferno', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Normalized Brightness')
    # Plot the contour, centroid, and bounding box for extent, area, and centroid locating
    # im = np.array(brightness * 255.0, dtype=np.uint8)
    # threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # a, b, w, h = cv2.boundingRect(contours[0])
    # x = contours[0][:, :, 0]
    # x = np.append(x, x[0])
    # y = contours[0][:, :, 1]
    # y = np.append(y, y[0])
    # moments = cv2.moments(contours[0])
    # cx = int(moments['m10'] / moments['m00'])
    # cy = int(moments['m01'] / moments['m00'])
    # plt.plot(x, y, 'k', linewidth=2, label='Contour')
    # plt.plot([a, a+w, a+w, a, a], [b, b, b+h, b+h, b], 'r', linewidth=3, label='Bounding Rectangle')
    # plt.scatter(cx, cy, label='Centroid')
    # Plot the maximum point
    index = np.unravel_index(np.argmax(brightness, axis=None), brightness.shape)
    plt.scatter(index[1], index[0], label='Maximum Brightness', c='grey')
    print(index)
    # Set up plot labels and bars
    plt.xticks(np.arange(0, 900 + 50, 50), np.arange(-45, 50, 5))
    plt.yticks(np.arange(0, 450 + 50, 50), np.arange(0, 50, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.grid(linestyle='--')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
