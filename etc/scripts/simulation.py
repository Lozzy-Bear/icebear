import matplotlib.pyplot as plt
import numpy as np
import math as mat
from scipy.integrate import dblquad
import multiprocessing as mp
import cv2
import icebear


# gaussian shape in image space
def gaussian_fit(x, peak, variance, mean):
    return peak * np.exp(-(x - mean) ** 2 / (2.0 * variance * variance))


# need to integrate this function over theta and phi, with u,v,w,theta_mean,theta_spread,phi_mean,phi_spread known
def real_image_pre_integration(theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
    return 2 * np.real(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) * np.exp(
        -(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) * np.exp(-2.0j * mat.pi * (
                (u_in * np.sin(theta) * np.cos(phi)) + (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(
            phi)))))


def imag_image_pre_integration(theta, phi, u_in, v_in, w_in, theta_mean, theta_spread, phi_mean, phi_spread):
    return 2 * np.imag(np.exp(-(theta - theta_mean) ** 2 / (2.0 * theta_spread * theta_spread)) * np.exp(
        -(phi - phi_mean) ** 2 / (2.0 * phi_spread * phi_spread)) * np.cos(phi) * np.exp(-2.0j * mat.pi * (
                (u_in * np.sin(theta) * np.cos(phi)) + (v_in * np.cos(theta) * np.cos(phi)) + (w_in * np.sin(
            phi)))))


def visibility_calculation(x, u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread, output):
    real_vis = dblquad(real_image_pre_integration, -mat.pi / 2, mat.pi / 2, lambda phi: -mat.pi, lambda phi: mat.pi,
                       args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
    imag_vis = dblquad(imag_image_pre_integration, -mat.pi / 2, mat.pi / 2, lambda phi: -mat.pi, lambda phi: mat.pi,
                       args=(u_in1, v_in1, w_in1, theta_mean, theta_spread, phi_mean, phi_spread))[0]
    output.put((x, real_vis + imag_vis * 1.0j))


if __name__ == '__main__':
    output = mp.Queue()
    print("Number of processors: ", mp.cpu_count())

    v_sol = 2.997E8
    center_freq = 49.5E6
    lambda_radar = v_sol / center_freq

    x_antenna_loc = [0., 15.10, 73.80, 24.2, 54.5, 54.5, 42.40, 54.5, 44.20, 96.9]
    y_antenna_loc = [0., 0., -99.90, 0., -94.50, -205.90, -177.2, 0., -27.30, 0.]
    z_antenna_loc = [0., 0.08952414692, 0.3473541846, 0.2181136458, 0.6834058436, -0.05865289508, -1.06679923,
                     -0.7540427261, -0.5265822222, -0.4087481019]
    num_xspectra = 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 + 1
    xspectra_x_diff = np.zeros((num_xspectra), dtype=np.float32)
    xspectra_y_diff = np.zeros((num_xspectra), dtype=np.float32)
    xspectra_z_diff = np.zeros((num_xspectra), dtype=np.float32)
    antenna_num_coh_index = 1
    for first_antenna in range(9):
        for second_antenna in range(first_antenna + 1, 10):
            xspectra_x_diff[antenna_num_coh_index] = x_antenna_loc[first_antenna] - x_antenna_loc[second_antenna]
            xspectra_y_diff[antenna_num_coh_index] = y_antenna_loc[first_antenna] - y_antenna_loc[second_antenna]
            xspectra_z_diff[antenna_num_coh_index] = z_antenna_loc[first_antenna] - z_antenna_loc[second_antenna]
            antenna_num_coh_index += 1
    azimuth_lags = 1000
    elevation_lags = 1000

    spreadx = 3
    spready = 3
    azi_rad_location_array = np.array([np.deg2rad(0)])
    azi_rad_extent_array = np.array([np.deg2rad(spreadx)])
    ele_rad_location_array = np.array([np.deg2rad(10)])
    ele_rad_extent_array = np.array([np.deg2rad(spready)])
    azi_rad_location_number = 0
    azi_rad_extent_number = 0
    ele_rad_location_number = 0
    ele_rad_extent_number = 0

    u = xspectra_x_diff / lambda_radar
    v = xspectra_y_diff / lambda_radar
    w = xspectra_z_diff / lambda_radar

    visibility_dist = np.zeros((46, 45, 10, 40, 4), dtype=np.complex64)

    print(azi_rad_location_array[azi_rad_location_number] * 180.0 / mat.pi,
          azi_rad_extent_array[azi_rad_extent_number] * 180.0 / mat.pi,
          ele_rad_location_array[ele_rad_location_number] * 180.0 / mat.pi,
          ele_rad_extent_array[ele_rad_extent_number] * 180.0 / mat.pi)

    pool = mp.Pool(mp.cpu_count() - 2)

    processes = [mp.Process(target=visibility_calculation, args=(
        x, u[x], v[x], w[x], azi_rad_location_array[azi_rad_location_number],
        azi_rad_extent_array[azi_rad_extent_number],
        ele_rad_location_array[ele_rad_location_number], ele_rad_extent_array[ele_rad_extent_number], output)) for x in
                 range(46)]

    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    visibility_dist_temp = [output.get() for p in processes]

    visibility_dist_temp.sort()
    visibility_dist[:, azi_rad_location_number, azi_rad_extent_number, ele_rad_location_number,
    ele_rad_extent_number] = [r[1] for r in visibility_dist_temp]

    for p in processes:
        p.close()

    visibility_dist = visibility_dist / np.abs(visibility_dist[0])

    # visibility_dist = np.ones_like(visibility_dist) + np.imag(visibility_dist) * 1.0j
    visibility_dist = np.real(visibility_dist) + np.imag(visibility_dist) * 1.0j

    V = np.concatenate((visibility_dist[:, azi_rad_location_number, azi_rad_extent_number, ele_rad_location_number,
                        ele_rad_extent_number], np.conjugate(visibility_dist[:, azi_rad_location_number,
                                                             azi_rad_extent_number, ele_rad_location_number,
                                                             ele_rad_extent_number])))

    factors = icebear.imaging.swht.unpackage_factors_hdf5(
        f'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85', 5)
    B = icebear.imaging.swht.swht_py(V, factors[:, :, 0:len(V)])

    for i in range(85, 95, 10):
        factors = icebear.imaging.swht.unpackage_factors_hdf5(
            f'X:/PythonProjects/icebear/swhtcoeffs_ib3d_2020-9-22_360-180-10-85', i)
        B *= icebear.imaging.swht.swht_py(V, factors[:, :, 0:len(V)])

    # B = swht(V, factors[:, :, 0:len(V)])
    B = np.abs(B / np.max(B))

    P = np.copy(B)
    P[P < 0.6] = 0.0
    im = np.array(P * 255.0, dtype=np.uint8)
    print(im.shape)
    threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ind = np.unravel_index(np.argmax(P, axis=None), P.shape)
    az = (ind[1])  # - np.ceil(P.shape[1]) / 2)
    el = (ind[0])  # - np.ceil(P.shape[0]) / 2)
    area = 0
    index = 0
    azs = 0
    els = 0
    for idx, c in enumerate(contours):
        temp_area = cv2.contourArea(c)
        if temp_area > area:
            area = temp_area
            index = idx
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            _, _, azs, els = cv2.boundingRect(c)

    print("ELAZ", (el, az), "CYCX", (cy, cx), "WH", (azs, els), "Area", area)
    if np.allclose(ind, [cy, cx], atol=5):
        print("\rCLOSE!")
    cv2.drawContours(im, contours[index], 0, (255, 255, 255), 3)

    plt.figure()
    plt.pcolormesh(B)
    plt.colorbar()

    plt.show()
