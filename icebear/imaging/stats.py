import numpy as np
import matplotlib.pyplot as plt
# import cv2
import csv


def stats(filename, time, rng, doppler, snr, intensity, threshold, resolution):
    """
    Given a range-doppler bin intensity plot and signal-to-noise ratio determine the
    centeroid (angle of arrival) of the intensity peak and its approximate area.
    """
    # time[5] = time[5,0:6]
    intensity = np.abs(intensity / np.max(intensity))
    intensity[intensity < threshold] = 0.0
    # plt.figure(1)
    # plt.pcolormesh(abs(intensity))
    # plt.colorbar()
    # im = np.array(intensity * 255, dtype = np.uint8)
    # threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ind = np.unravel_index(np.argmax(intensity, axis=None), intensity.shape)
    az = (ind[1] - np.floor(intensity.shape[1]) / 2) * resolution
    el = (ind[0] - np.floor(intensity.shape[0]) / 2) * resolution
    r = doppler * 1.5 / 2 - 200
    cosinelaw_alt = (-6378 + np.sqrt(6378 ** 2 + r ** 2 - 2 * 6378 * r * np.cos(np.deg2rad(90 + np.abs(el)))))
    # print("AZEL", az, el)
    # print("CONTOUR SHAPE: ", np.asarray(contours).shape)
    """
    for i in range(np.asarray(contours).shape[0]):
        M = cv2.moments(contours[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print("CXCY", cx, cy)
        if np.allclose(ind, [cy, cx], atol=10):
            cnt = contours[i]
    """
    cnt = 0.0  # contours[0]
    if cnt is None:
        print("ib_stats unable to resolve: ", time)
    else:
        # print("ib_stats saving data: ", time)
        area = 0.0  # cv2.contourArea(cnt)
        data_list = [time, rng, doppler, snr, az, el, area]
        print('Data stored:', data_list, cosinelaw_alt, "[km]")
        with open(filename, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(data_list)

    return None


def save_data(filename, data_list):
    print(data_list)
    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data_list)
    return None


def load_data(filename, time):
    rng = np.array([])
    doppler = np.array([])
    snr = np.array([])
    az = np.array([])
    el = np.array([])

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # print(row[0], time)
            if row[0] == time:
                rng = np.append(rng, int(row[1]))
                doppler = np.append(doppler, int(row[2]))
                snr = np.append(snr, float(row[3]))
                az = np.append(az, float(row[4]))
                el = np.append(el, float(row[5]))

    return rng, doppler, snr, az, el

