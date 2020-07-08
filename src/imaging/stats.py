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


def plot_stats(rng, doppler, snr, az, el):
    SNR = np.zeros((2000, 100))
    AZ = np.zeros_like(SNR)
    EL = np.zeros_like(SNR)
    print(rng)
    for x in range(len(rng)):
        SNR[int(doppler[x]), int(rng[x])] += snr[x]
        AZ[int(doppler[x]), int(rng[x])] += az[x]
        EL[int(doppler[x]), int(rng[x])] += np.abs(el[x])

    fig = plt.figure()
    fig.suptitle("ICEBEAR-3D 10 Antenna: 2019-10-25 8:27:23 UTC")

    ax1 = plt.subplot(131)
    plt.title("SNR")
    plt.pcolormesh(SNR)
    plt.xticks(np.arange(0, 100, 10), np.arange(-50, 50, 10) * 30.9)
    plt.yticks(np.arange(0, 2000, 100), np.arange(0, 2000, 100) * 0.75)
    plt.xlim(20, 80)
    plt.ylim(800, 1200)
    plt.colorbar()
    plt.grid(which='both')

    ax2 = plt.subplot(132, sharex=ax1)
    plt.title("Azimuth")
    plt.pcolormesh(AZ)
    plt.xticks(np.arange(0, 100, 10), np.arange(-50, 50, 10) * 30.9)
    plt.yticks(np.arange(0, 2000, 100), np.arange(0, 2000, 100) * 0.75)
    plt.xlim(20, 80)
    plt.ylim(800, 1200)
    plt.colorbar()
    plt.grid(which='both')

    ax3 = plt.subplot(133, sharex=ax1)
    plt.title("Elevation")
    plt.pcolormesh(EL, vmin=7.5, vmax=12.5)
    plt.xticks(np.arange(0, 100, 10), np.arange(-50, 50, 10) * 30.9)
    plt.yticks(np.arange(0, 2000, 100), np.arange(0, 2000, 100) * 0.75)
    plt.xlim(20, 80)
    plt.ylim(800, 1200)
    plt.colorbar()
    plt.grid(which='both')

    ax1.set_xlabel("Doppler [m/s]")
    ax1.set_ylabel("Estimated Range [km]")
    ax2.set_xlabel("Doppler [m/s]")
    ax2.set_ylabel("Estimated Range [km]")
    ax3.set_xlabel("Doppler [m/s]")
    ax3.set_ylabel("Estimated Range [km]")
    plt.show()

    return None

