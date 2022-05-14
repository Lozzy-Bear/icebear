import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
import multiprocessing as mp
import time
import sys
# import path as pth
# ipath = pth.Path(__file__).abspath()
# sys.path.append(ipath.parent.parent.parent)
import icebear
import icebear.utils as utils
import h5py
try:
    import cupy as cp
except:
    print('no cupy')


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
    visibility_dist = np.zeros((int(len(u)/2), idx_length), dtype=np.complex64)
    # visibility_dist = np.zeros((int(len(u)/2), idx_length, idx_length, idx_length, idx_length), dtype=np.complex64)

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
        visibility_dist[:, idx] = [r[1] for r in visibility_dist_temp]

        # for p in processes:
            # p.close()

    # visibility = np.array(visibility_dist)
    # for i in range(len(azimuth)):
    #     visibility[:, i, i, i, i] = visibility[:, i, i, i, i] / np.abs(visibility[0, i, i, i, i])
    # visibility = np.sum(visibility, axis=(1, 2, 3, 4))
    visibility = np.sum(visibility_dist, axis=1)
    visibility = np.append(np.conjugate(visibility), visibility)
    
    coeffs_np = np.zeros((450, 900, 92, 8), dtype=np.complex64)
    idx = 0
    for i in range(15, 95, 10):
        coeffs_np[:, :, :, idx] = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, i)
        idx += 1
    coeffs_cp = cp.asarray(coeffs_np)
    
    # Do processing the faster numpy way
    tf = 0
    for i in range(20):
        ts = time.time()
        visibility_np = np.tile(np.array(visibility, dtype=np.complex64), (8, 1))
        brightness = np.prod(np.einsum('ijkl,lk->ijl', coeffs_np, visibility_np), axis=2)
        brightness = np.abs(brightness / np.max(brightness))
        tf += time.time() - ts
    print('numpy time:', tf/20)
    
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

    # intensity = icebear.imaging.swht.swht_py(visibility, coeffs2)
    # print(f'\t-max intentsity {np.max(intensity)} mean intensity {np.mean(intensity)}')
    # intensity = icebear.imaging.swht.brightness_cutoff(intensity, threshold=0.0)

    # Doing processing with a new faster algorithm and comparing on cupy
    tf = 0
    for i in range(20):
        ts = time.time()
        visibility_cp = cp.tile(cp.array(visibility, dtype=cp.complex64), (8, 1))
        intensity = cp.prod(cp.einsum('ijkl,lk->ijl', coeffs_cp, visibility_cp), axis=2)
        intensity = cp.abs(intensity / cp.max(intensity))
        #intensity = cp.asnumpy(intensity)
        #intensity = icebear.imaging.swht.brightness_cutoff(intensity, threshold=0.0)
        tf += time.time() - ts
    print('cupy time:', tf/20)
    
    intensity = cp.asnumpy(intensity)
    print('compare the methods:', np.allclose(brightness, intensity), np.max(brightness - intensity))

    return brightness, intensity, mx, my


def simulate_vcz(config, azimuth, elevation, azimuth_extent, elevation_extent):
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

    u, v, w = utils.baselines(x,  # config.rx_ant_coords[0, :],
                              y,  # config.rx_ant_coords[1, :],
                              z,  # config.rx_ant_coords[2, :],
                              wavelength)

    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    azimuth_extent = np.deg2rad(azimuth_extent)
    elevation_extent = np.deg2rad(elevation_extent)

    visibility_dist = np.zeros((int(len(u) / 2), idx_length, idx_length, idx_length, idx_length), dtype=np.complex64)
    # Instantiate multi-core processing
    output = mp.Queue()
    pool = mp.Pool(mp.cpu_count() - 2)
    # Loop process to allow for multiple targets in an image. Typically only one target is used.
    for idx in range(idx_length):
        processes = [mp.Process(target=_visibility_calculation,
                                args=(x, u[x], v[x], w[x],
                                      azimuth[idx], azimuth_extent[idx],
                                      elevation[idx], elevation_extent[idx],
                                      output)) for x in range(int(len(u) / 2))]
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
    # visibility = np.sum(visibility, axis=(1, 2, 3, 4))
    visibility = np.sum(visibility_dist, axis=(1, 2, 3, 4))
    visibility = np.append(np.conjugate(visibility), visibility)

    # Visibilty data is now generated.
    start_time = time.time()
    l = np.arange(225, 315, 0.1)
    m = np.arange(90, 180, 0.1)
    l = np.cos(np.deg2rad(l))
    m = np.cos(np.deg2rad(m))
    ll, mm = np.meshgrid(l, m)
    brightness = np.zeros_like(ll, dtype=np.complex64)
    for cnt in range(92):
        brightness += visibility[cnt] * np.exp(1j * 2 * np.pi * (u[cnt]*ll + v[cnt]*mm))
    brightness *= 299792458.0 * wavelength ** 2 / (4 * (2 * np.pi) ** 3)
    print('time:', time.time() - start_time)

    brightness = icebear.imaging.swht.brightness_cutoff(brightness, threshold=0.0)
    intensity = np.copy(brightness)
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

    return brightness, intensity, mx, my


class Image:
    def __init__(self, targets_file, coeffs_file):
        self.targets_file = targets_file
        self.coeffs_file = coeffs_file
        self._load_targets()

        self.wavelength = 299792458 / 49.5e6
        antennas = np.array([[0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9],
                             [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.],
                             [0., 0.0895, 0.3474, 0.2181, 0.6834, -0.0587, -1.0668, -0.7540, -0.5266, -0.4087]])
        self.x = antennas[0, :]
        self.y = antennas[1, :]
        self.z = antennas[2, :]
        self.u = np.array([0])
        self.v = np.array([0])
        self.w = np.array([0])
        for i in range(len(self.x)):
            for j in range(i + 1, len(self.x)):
                self.u = np.append(self.u, (self.x[i] - self.x[j]) / self.wavelength)
                self.v = np.append(self.v, (self.y[i] - self.y[j]) / self.wavelength)
                self.w = np.append(self.w, (self.z[i] - self.z[j]) / self.wavelength)
        self.u = np.append(self.u, -1 * self.u)
        self.v = np.append(self.v, -1 * self.v)
        self.w = np.append(self.w, -1 * self.w)

        self.image_batch()

    def sswht(self, visibility):
        coeffs = icebear.imaging.swht.unpackage_coeffs(self.coeffs_file, 85)
        start_time = time.time()
        brightness = icebear.imaging.swht.swht_py(visibility, coeffs)
        # This section activates the experimental angular_frequency_beamforming()
        for i in range(15, 85, 10):
            coeffs = icebear.imaging.swht.unpackage_coeffs(self.coeffs_file, i)
            brightness *= icebear.imaging.swht.swht_py(visibility, coeffs)
        brightness = icebear.imaging.swht.brightness_cutoff(brightness, threshold=0.0)
        mx, my, _ = icebear.imaging.swht.max_center(brightness)
        tt = time.time() - start_time
        return mx, my, tt

    def vcz(self, visibility):
        start_time = time.time()
        l = np.arange(225, 315, 0.1)
        m = np.arange(90, 180, 0.1)
        l = np.cos(np.deg2rad(l))
        m = np.cos(np.deg2rad(m))
        ll, mm = np.meshgrid(l, m)
        brightness = np.zeros_like(ll, dtype=np.complex64)
        for cnt in range(92):
            brightness += visibility[cnt] * np.exp(1j * 2 * np.pi * (self.u[cnt] * ll + self.v[cnt] * mm))
        brightness *= 299792458.0 * self.wavelength ** 2 / (4 * (2 * np.pi) ** 3)
        brightness = icebear.imaging.swht.brightness_cutoff(brightness, threshold=0.0)
        mx, my, _ = icebear.imaging.swht.max_center(brightness)
        tt = time.time() - start_time
        return mx, my, tt

    def _load_targets(self):
        f = h5py.File(self.targets_file, 'r')
        self.targets = f['targets'][()]
        self.visibilities = f['visibilities'][()]
        return

    def image_batch(self):
        x1 = []
        y1 = []
        t1 = []
        x2 = []
        y2 = []
        t2 = []
        az = []
        el = []
        for t, v in zip(self.targets, self.visibilities):
            az.append(t[0, 0])
            el.append(t[0, 2])
            a, b, c = self.sswht(v)
            x1.append(a * 0.1 - 45.0)
            y1.append(b * 0.1)
            t1.append(c)
            a, b, c = self.vcz(v)
            x2.append(a * 0.1 - 45.0)
            y2.append(b * 0.1)
            t2.append(c)

        f = h5py.File('poop.h5', 'w')
        f.create_dataset('x1', data=np.asarray(x1))
        f.create_dataset('y1', data=np.asarray(y1))
        f.create_dataset('t1', data=np.asarray(t1))
        f.create_dataset('x2', data=np.asarray(x2))
        f.create_dataset('y2', data=np.asarray(y2))
        f.create_dataset('t2', data=np.asarray(t2))
        f.close()


def gpu_coeffs(config, rule):
    ts = time.time()
    k = np.zeros((450, 900, 92, len(rule)), dtype=np.complex64)
    idx = 0
    for i in rule:
        k[:, :, :, idx] = icebear.imaging.swht.unpackage_coeffs(config.swht_coeffs, i)
        idx += 1
    print(k.shape, k.nbytes/1e9, 'GB', time.time() - ts)
    return k


def gpu_visibility(visibilty, n):
    ts = time.time()
    v = np.tile(np.array(visibilty, dtype=np.complex64), (n, 1))
    # v = np.tile(np.array([visibilty], dtype=np.complex128).T, (1, n))
    print(v.shape, v.nbytes/1000, 'KB', time.time() - ts)
    return v

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

    # Image('C:/Users/TKOCl/PythonProjects/icebear/icebear/tools/simulated_data.h5',
    #       'C:/Users/TKOCl/PythonProjects/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5')
    # exit()

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
    # coeffs_file = '/home/radar/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5'
    coeffs_file = '/beaver/backup/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5'
    # coeffs_file = 'C:/Users/TKOCl/PythonProjects/icebear/swhtcoeffs_ib3d_2021_07_28_090az_045el_01res_85lmax.h5'
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
    # exit()

    brightness, intensity, cx, cy = simulate(config, np.array([-27]), np.array([5]), np.array([3]), np.array([3]))

    # Do VCZ basic processing
    # brightness, intensity, cx, cy = simulate_vcz(config, np.array([0]), np.array([85]), np.array([3]), np.array([3]))
    # plt.figure()
    # plt.pcolormesh(intensity, cmap='inferno', vmin=0.0, vmax=1.0)
    # plt.colorbar(label='Normalized Brightness')
    # plt.xlabel('Azimuth [deg]')
    # plt.ylabel('Elevation [deg]')
    # plt.grid(linestyle='--')
    # index = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
    # plt.scatter(index[1], index[0], label='Maximum Brightness', c='grey')
    # print(45.0 - index[1] * 0.1, 90.0 - index[0] * 0.1)
    # print(index)
    # plt.axis('equal')
    # plt.show()
    # exit()

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
