import numpy as np
import csv
import matplotlib as mpl
import icebearModules as Imod
import scipy.interpolate as interp
import fringes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm


# Function Calls
def load_beam(array_type_str, A1, A2):
    """ Loads in the beam pattern for the given antenna pair """
    # theta = azimuth, gamma = elevation
    theta = np.zeros((361 * 181))
    gain = np.zeros_like(theta)
    gamma = np.zeros((361 * 181))

    if array_type_str == "3D":
        read = str(A1) + str(A2)
        load_file = '/home/icebear-cuda/galeschuk_research/noise/beam_pattern/Cushcraft_3D-A' + read + '.txt'
    else:
        read = str(int(A2 - A1))
        load_file = '/home/icebear-cuda/galeschuk_research/noise/beam_pattern/Cushcraft_linear-' + read + 'lam.txt'
    with open(load_file, newline='\n') as file:
        reader = csv.reader(file, delimiter='\t')
        count = 0
        for row in reader:
            theta[count] = float(row[0])
            gamma[count] = float(row[1])
            gain[count] = float(row[2])
            count += 1

    theta = np.arange(0, 361)
    gamma = np.arange(-90, 91)
    G = np.zeros((361, 181))
    for i in range(len(theta)):
        for j in range(len(gamma)):
            G[i, j] = gain[j * 361 + i]

    return G, theta, gamma


def offset(start, end, data_s, data_e, time):
    # Generate a trend line for given data with a gap between the data.
    fit_y = np.zeros(len(start) + len(end))
    fit_y[0:len(start)] = data_s[:]
    fit_y[len(start):len(end) + len(start)] = data_e[:]

    fit_x = np.zeros(len(start) + len(end))
    fit_x[0:len(start)] = np.arange(0, len(start))
    fit_x[len(start):len(end) + len(start)] = np.arange(len(time) - len(end), len(time))

    z = np.polyfit(fit_x, fit_y, 1)
    p = np.poly1d(z)
    y = p(np.arange(0, len(time)))

    return y


def best_fit(data, a_arr, b_arr, baseline):
    # Run a triple exponential fit over the data set
    # a_arr and b_arr are parameters of the fit, should be
    # adjusted to the data set
    smooth = np.zeros(len(data))
    beta = np.zeros(len(data))

    # Condition check to adjust for changes data trend
    if baseline < 5:
        a = a_arr[0]
        b = b_arr[0]
    else:
        if len(a_arr) > 1:
            a = a_arr[1]
            b = b_arr[1]
        else:
            a = a_arr[0]
            b = b_arr[0]

    # Generate the fit
    for i in range(0, len(smooth)):
        if i == 0:
            smooth[i] = data[i]
            beta[i] = data[i] - data[i]
        else:
            smooth[i] = a * data[i] + (1 - a) * (smooth[i - 1] + beta[i - 1])
            beta[i] = b * (smooth[i] - smooth[i - 1]) + (1 - b) * beta[i - 1]

    return smooth


def running_avg(data, window_size):
    # calculate a rolling or running average of a data set with
    # window size
    i = 0
    moving_average = np.copy(data)
    while i < len(data) - window_size + 1:
        this_window = data[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_average[i + int(window_size / 2)] = window_average
        i += 1

    return moving_average


def running_avg_xy(m, t, window_size):
    # calculate a rolling or running average of a data set with
    # window size
    i = 0
    data = m * np.exp(1j * t)  # x+iy
    real = np.real(data)
    imag = np.imag(data)
    while i < len(m) - window_size + 1:
        r_window = np.real(data[i: i + window_size])
        i_window = np.imag(data[i: i + window_size])
        r_window_average = sum(r_window) / window_size
        i_window_average = sum(i_window) / window_size
        real[i + int(window_size / 2)] = r_window_average
        imag[i + int(window_size / 2)] = i_window_average
        i += 1

    return real, imag


def phase_difference(A1, A2, x_pos, y_pos):
    x1 = x_pos[A1]
    y1 = y_pos[A1]
    x2 = x_pos[A2]
    y2 = y_pos[A2]

    # baseline distance
    d = -np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Negative d as the phase was inverted

    # Baseline Vector Angle Gamma (pnt)

    if (x1 - x2) == 0:
        if y2 > y1:
            pnt = 0
        else:
            pnt = np.pi
    elif x2 > x1:
        if (y2 - y1) == 0:
            pnt = np.pi / 2
        else:
            pnt = np.pi / 2 - np.arctan((y2 - y1) / (x2 - x1))
    else:
        if (y2 - y1) == 0:
            pnt = -np.pi / 2
        else:
            pnt = -np.pi / 2 - np.arctan((y2 - y1) / (x2 - x1))

    # pnt is the pointing direction of antenna baseline (arrow from A1 to A2)
    return d, pnt


def boundary(data, up_lim, dn_lim):
    n = np.zeros_like(data)
    for i in range(0, len(data)):
        while (data[i] >= np.pi) or (data[i] < -np.pi):
            if data[i] < -np.pi:
                data[i] += 2 * np.pi
                n[i] = n[i] - 1
            if data[i] >= np.pi:
                data[i] += -2 * np.pi
                n[i] = n[i] + 1
    return data


# Main Program

# Data loading parameters
array_type_str = "3D" # "3D" or "linear"
time_str = "2019-12-20_from_5_to_8"
source = '/ext_data3/icebear_3d_test/'
save_loc = '/home/icebear-cuda/galeschuk_research/noise_brian/plots/select_'
data_loc = '/home/icebear-cuda/galeschuk_research/noise_brian/data/'

# Calculation parameters
power_cutoff = 3
peak_determine = True
array_offset = 7  # in deg
beam_cutoff = 10
running_avg_window = 100

# Antenna Selection
array1 = np.array([2]) #0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,6,6,6,7,7,8])
array2 = np.array([7]) #1,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,3,4,5,6,7,8,9,4,5,6,7,8,9,5,6,7,8,9,6,7,8,9,7,8,9,8,9,9])

# Plotting options
plot_correction = False
goodness = True

plot_fringe_polar = True
plot_fringe_2D = True
plot_beam_fringe = True

plot_cygnus_angles = True
plot_beam_cygnus_3d = False
plot_cygnus_beam_2d = True
plot_cygnus_beam_3d = False

plot_power_phase = True
plot_phase_vs_expect = True
plot_power = False

plot_corr_phase_vs_expect = True
plot_power_vs_cygnus_beam = True
plot_measured_aoa_vs_theory = False

save_plot = False

# Plotting Params
mpl.rcParams['figure.figsize'] = [16.5, 14.0]
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['figure.titlesize'] = 25

# Antenna positions for determining baseline vector
# Note: These positions may not take into account the offset of the baseline
#   vectors from north. In this case, use the array_offset variable above
#   to rotate the pointing direction of the whole array
if array_type_str == "3D":
    # For ICEBEAR 3D
    x_pos = [0.0,15.10,73.80,24.2,54.5,54.5,42.40,54.5,44.20,96.9]
    y_pos = [0.0,0.0,-99.90,0.0,-94.50,-205.90,-177.2,0.0,-27.30,0.0]
    z_pos = []
else:
    # For ICEBEAR Classic (linear)
    x_pos = [0.0,6.0,12.0,18.0,24.0,30.0,36.0,42.0,48.0,54.0]
    y_pos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    z_pos = []

lam = 6.06

# Observation parameters for Cygnus A
RA  = 299.8681525
DEC = 40.73391555
LAT = 52.24319
LON = -106.450191

# Final product arrays
correction_array = np.zeros(len(array1))
variance_array = np.zeros(len(array1))
data_str_list = []
goodness_array = np.zeros((3,len(array1)))
original_array = np.zeros(len(array1))
quality_array = np.zeros(len(array1))

# Store information about current data set
statsF = open(time_str+"_Cygnus_A_stats.csv", "w")
statsF.write("Baseline,Power Width,Average Power,Max Power,Min Power,Phase Width,Average Phase,Phase Spread,Max Phase,Min Phase,Correction,Varriance,distance,gamma angle,Quality\n")

# go through each baseline
for ant in range(len(array1)):
    print("\n Beginning iteration for ICEBEAR {} Antenna pair {}-{}".format(array_type_str, array1[ant], array2[ant]))
    A1 = int(array1[ant])
    A2 = int(array2[ant])
    d = A2-A1

    # Read in data from desired time period. Must be already processed using noise_gen.py and saved in
    # /home/icebear-cuda/galeschuk_research/noise_brian/data/*.npy
    name_ext = ""
    read = str(A1) + str(A2)
    if array_type_str == "3D":
        dates = np.load(data_loc+time_str+"-x-spectrum_mag_phase_"+read+"dates.npy", allow_pickle=True)
        angle = np.load(data_loc+time_str+"-x-spectrum_mag_phase_"+read+"noise_array_phase.npy", allow_pickle=True)
        power = np.load(data_loc+time_str+"-x-spectrum_mag_phase_"+read+"noise_array.npy", allow_pickle=True)

    else:
        dates = np.load(data_loc+time_str+"-x-spectrum_mag_phase_12dates.npy")
        angle = np.load(data_loc+time_str+"-x-spectrum_mag_phase_"+read+name_ext+"noise_array_phase.npy")
        power = np.load(data_loc+time_str+"-x-spectrum_mag_phase_"+read+name_ext+"noise_array.npy")

    power = 10*np.log10(power)

    # Load beam pattern of antenna baseline
    print("Loading Beam Pattern ...")
    G, theta, gamma = load_beam(array_type_str, A1, A2)

    array_type_str = array_type_str + name_ext

    # Determine Cygnus A position at each time
    elevation, azimuth = Imod.star_track(RA, DEC, LAT, LON, dates)
    # recast azimuth and altitude
    AZ = 180 - (azimuth+90)
    ALT = 90-elevation
    # wrap azimuth taking into account the 7 degree offset
    for i in range(len(AZ)):
        if AZ[i]<0-array_offset:
            AZ[i] += 360
        if AZ[i]>360-array_offset:
            AZ[i] -= 360

    print("Determining Expected Phase Diff ...")

    # Calculate Beam Pattern Slice for Cygnus A
    cygnus_beam = np.zeros(len(dates))
    # wrap the beam pattern azimuth similarly
    theta = theta-array_offset
    for i in range(len(theta)):
        if theta[i]>360:
            theta[i]=theta[i]-360
#        elif theta[i]<=0:
#            theta[i]=theta[i]+360

    # sort the beam pattern by increasing theta (azimuth)
    ind = np.argsort(theta)
    theta = theta[ind]
    G = G[ind,:]

    # interpolate over the grid (theta, gamma) = (azimuth, elevation)
    # the data over the grid is G, the gain at each (theta, gamma)
    # we want to grab interpolations of the beam pattern at azimuth AZ and
    # altitude ALT, corresponding to the location of Cygnus A.

    # so the vector cygnus_beam holds the gain value of the beam pattern at the location of
    # Cygnus A at each point in time
    for i in range(0,len(dates)):
        cygnus_beam[i] = interp.interpn((theta,gamma),G,np.array([AZ[i],ALT[i]]),'linear')

    # Select data where Cygnus A is within the visible range of the ICEBEAR reciever
    # Note: Data set may already be small enough.
    time_delta = dates[-1]-dates[0]
    half_window = 720*5
    if time_delta.total_seconds() > 60*60*10:
        mid_time    = np.argwhere(elevation==np.min(elevation))
        dates       = dates[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        power       = power[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        angle       = angle[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        elevation   = elevation[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        ALT         = ALT[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        azimuth     = azimuth[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        AZ          = AZ[mid_time[0][0]-half_window:mid_time[0][0]+half_window]
        cygnus_beam = cygnus_beam[mid_time[0][0]-half_window:mid_time[0][0]+half_window]

    # Remove -INF values from power
    while np.min(power) < 0:
        for x in np.where(power==np.min(power)):
            power[x] = power[x-1]

    # Determine power peaks in data
    # First determine power offset from zero dB
    print("Evaluating Power ...")
    # grab the power from the first and last 360 points
    offset_s  = power[0:360]
    offset_st = dates[0:360]
    offset_e  = power[len(dates)-360:-1]
    offset_et = dates[len(dates)-360:-1]

    # Calculate the offset by making a trendline between the first and last 360 points
    power_offset = offset(offset_st, offset_et, offset_s, offset_e, dates)

    # Shift power to zero
    power_diff  = power-power_offset

    # Calculate running averages of the shifted power alone (power_fit) and the power and phase together
    # as a complex number (phase_fit, db_fit).
    power_fit = running_avg(power_diff,running_avg_window)
    real, imag = running_avg_xy(power,angle,running_avg_window)
    phase_fit = np.angle(real+1j*imag)
    db_fit = np.abs(real+1j*imag)

    # Store indices of signal with a db > limit
    peak_index = np.argwhere(power_fit>power_cutoff)
    peak_start = []
    peak_end = []
    for i in range(0,len(peak_index)):
        # first peak_index is by default the start of the first peak
        if i==0:
            peak_start.append(peak_index[i])
        # if the peak_indices aren't consecutive indices, then peak_index[i-1] is the end of the first peak
        # and peak_index[i] is the start of the second peak
        elif peak_index[i-1]+1 != peak_index[i]:
            peak_end.append(peak_index[i-1])
            peak_start.append(peak_index[i])
        # the last peak_index is by default the end of the last peak
        elif i==len(peak_index)-1:
            peak_end.append(peak_index[i])

    # Calculate expected phase measurement for baseline

    # First get the baseline angle with respect to North. An East-West baseline
    # will have an angle near 90 degrees.
    d, pnt = phase_difference(A1, A2, x_pos, y_pos)
    pnt = pnt + np.radians(array_offset) # xy positions do not take into account the 7 deg east of north pointing offset

    # Stellar Position Angles Alpha and Beta
    # 3D interferometry
    beta = azimuth*np.pi/180
    alpha = np.pi/2 - elevation*np.pi/180

    # Page 100 in Draven's thesis, this is the theoretical phase difference from 3D interferometry
    phi = 2*np.pi*d*(np.sin(pnt)*np.sin(beta)*np.sin(alpha) + np.cos(pnt)*np.cos(beta)*np.sin(alpha))/lam

    phi_min = np.argwhere(np.min(phi) == phi)[0][0]
    phi_max = np.argwhere(np.max(phi) == phi)[0][0]

    # Keep phi within -pi and pi
    phi = boundary(phi, np.pi, -np.pi)

    # Generate 2D fringe intensity
    xi = np.radians(np.linspace(-180, 180, 1000))
    fringe_beta = np.repeat(xi[:, np.newaxis], 1000, 1)
    xi = np.radians(np.linspace(-180, 180, 1000))
    fringe_alpha = np.repeat(xi[np.newaxis, :], 1000, 0)

    # Same expression from page 100 of Draven's thesis. Theoretical phase difference across all angles, not just the
    # angles corresponding to Cygnus A
    # todo: missing factor of 2?
    delta = d * (np.sin(pnt) * np.sin(fringe_beta) * np.cos(fringe_alpha) + np.cos(pnt) * np.cos(fringe_beta) * np.cos(
        fringe_alpha))
    I_2D = np.power(np.sin(delta * np.pi / lam), 2)

    # Interpolate to get Fringe at Cygnus Position
    cygnus_fringe = np.zeros(len(dates))
    for i in range(0, len(dates)):
        cygnus_fringe[i] = interp.interpn((np.degrees(fringe_alpha[0, :]), np.degrees(fringe_beta[:, 0])), I_2D,
                                          np.array([azimuth[i], elevation[i]]), "linear")

    # bore_sight will be where the azimuth is zero?
    # beam_peak is just where the maximum value of the beam is
    bore_sight = np.argwhere(np.min(np.abs(azimuth-7)) == np.abs(azimuth-7))[0][0]
    beam_peak = np.argwhere(np.max(cygnus_beam)==cygnus_beam)[0][0]

    # todo: what is going on here?
    point = 0
    peak_determine = True
    # Try to first match to slope changes (global min max), power must be above power cutoff
    if (phi_min > 1000) and (phi_min < 8 * 3600 - 1000) and (power_fit[phi_min] > power_cutoff):
        print("min phi corr")
        point = phi_min
    elif (phi_max > 1000) and (phi_max < 8 * 3600 - 1000) and (power_fit[phi_max] > power_cutoff):
        print("max phi corr")
        point = phi_max
    else:  # Otherwise, match to power peaks
        peak_determine = False
        print("Power Selection")

    # Phase correction using 30 minutes of data, 15 min (180 data points) either side of the peak lobe power
    print("Computing Phase Correction ...")
    index = np.where(np.max(cygnus_beam) == cygnus_beam)[0][0]
    phase_diff = angle[index - 180:index + 180]  # measured noise phase
    time = dates[index - 180:index + 180]
    phi_theo = phi[index - 180:index + 180]  # theoretical signal phase

    d_phi = phase_diff - phi_theo  # difference between measured and theoretical phase
    d_phi = boundary(d_phi, np.pi, -np.pi)  # wrap to within -pi to pi

    corr = np.sum(d_phi) / len(d_phi)  # average of differences across 30 minutes centred on peak gives the correction
    var = np.sqrt(np.sum((d_phi - corr) ** 2) / len(d_phi))

    # Store current iteration results
    correction_array[ant] = corr
    variance_array[ant] = var
    phase_corr = angle - corr

    # Generate stats file
    pwr_width = np.max(power[0:720]) - np.min(power[0:720])  # not including peaks, just background
    temp = np.copy(angle)
    for i in range(len(temp)):
        if temp[i]<0:
            temp[i] = temp[i]+2*np.pi
    check_wrap = np.average(angle[0:360]-angle[1:361])
    if check_wrap>np.radians(180): # (np.average(temp)>np.pi*0.9) and (np.average(temp)<np.pi*1.2):
        phs_width = np.max(temp[0:720]) - np.min(temp[0:720])
        phs_avg = np.average(temp)
        if phs_avg>np.pi:
            phs_avg = phs_avg-2*np.pi
        phs_max = np.max(phase_fit)-np.pi
        phs_min = np.min(phase_fit)-np.pi
    else:
        phs_width = np.max(angle[0:720]) - np.min(angle[0:720])
        phs_max = np.max(phase_fit)
        phs_min = np.min(phase_fit)
        phs_avg = np.average(angle)

    phs_spread = phs_max-phs_min

    # Determine if data is of good quality for corrections
    if (np.average(power) > 45):  # high average power means Cygnus signal gets lost
        data_str = "Bad"
    elif (phs_spread) < np.radians(100):  # not enough phase spread means calibration is difficult
        data_str = "Bad"
    else:
        data_str = "Good"
    data_str_list.append(data_str)

    # Record Stats of the run
    statsF.write(read + "," + str(pwr_width) + "," + str(np.average(db_fit)) + "," + str(np.max(db_fit)) + "," + str(
        np.min(db_fit)) + "," + str(np.degrees(phs_width)) + "," + str(np.degrees(phs_avg)) + "," + str(
        np.degrees(phs_spread)) + "," + str(np.degrees(phs_max)) + "," + str(np.degrees(phs_min)) + "," + str(
        np.degrees(corr)) + "," + str(np.degrees(var)) + "," + str(np.abs(d)) + "," + str(
        np.degrees(pnt)) + "," + data_str + "\n")

    # Goodness comparison uses reduced Coefficiant of Determination
    goodness_array[0, ant] = array1[ant]
    goodness_array[1, ant] = array2[ant]

    phase_corr = boundary(phase_corr,np.pi,-np.pi)

    goodness_array[2,ant] = 1 - np.sum((phase_corr-phi)**2)/(np.sum((phase_corr-np.mean(phase_corr))**2))
    original_array[ant] = 1 - np.sum((angle-phi)**2)/(np.sum((angle-np.mean(angle))**2))

    # Attempt to provide a scale of the phase range
    if (phs_spread) < np.radians(50):
        quality_array[ant] = 0
    elif (phs_spread) > np.radians(300):
        quality_array[ant] = 3
    elif (phs_spread) < np.radians(100):
        quality_array[ant] = 1
    else:
        quality_array[ant] = 2

    # This generates the fringe pattern along the plane of the baseline.
    fringe_theta, I, maxima, minima = fringes.fringe(array_type_str, lam, A1, A2)

    if np.abs(x_pos[A1] - x_pos[A2]) > np.abs(y_pos[A1] - y_pos[A2]):
        fringe_time = np.interp(azimuth, np.degrees(fringe_theta), I)  # interp.interpn(fringe_theta,I,azimuth,"linear")
    else:
        fringe_time = np.interp(elevation, np.degrees(fringe_theta),
                                I)  # interp.interpn(fringe_theta,I,90-elevation,"linear")

    # Plotting: To be done if option is selected
    # Save instead of show if option is selected

    # Plotting values for 3D plots
    ALT = ALT * np.pi / 180
    AZ = AZ * np.pi / 180

    R, P = np.meshgrid(gamma * np.pi / 180, theta * np.pi / 180)

    X, Y, Z = (G + 30) * np.sin(R) * np.cos(P), (G + 30) * np.sin(R) * np.sin(P), (G + 30) * np.cos(R)
    x, y, z = 50 * np.sin(ALT) * np.cos(AZ), 50 * np.sin(ALT) * np.sin(AZ), 50 * np.cos(ALT)
    cx, cy, cz = (cygnus_beam + 30) * np.sin(ALT) * np.cos(AZ), (cygnus_beam + 30) * np.sin(ALT) * np.sin(AZ), (
                cygnus_beam + 30) * np.cos(ALT)

    print("Generating Plots ...")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    if plot_correction == True:
        fig, axs = plt.subplots(2)
        axs[0].xaxis.set_major_locator(locator)
        axs[0].xaxis.set_major_formatter(formatter)
        axs[1].xaxis.set_major_locator(locator)
        axs[1].xaxis.set_major_formatter(formatter)

        index_list = [0]
        corr_step = [0]

        axs[0].plot(dates, power_fit)
        for i in range(len(corr_step)):
            axs[0].plot(dates[int(index_list[i])], power_fit[int(index_list[i])], marker='x', color='red')
        fig.suptitle("Noise Signal Power for Baseline {}-{}: Starting {}".format(A1, A2, dates[0]))
        axs[0].set(ylabel="Power (dB)")
        d_phi = angle - phi
        for i in range(len(d_phi)):
            if d_phi[i] > np.pi:
                d_phi[i] = d_phi[i] - 2 * np.pi
            elif d_phi[i] < -np.pi:
                d_phi[i] = d_phi[i] + 2 * np.pi

        axs[1].plot(dates, d_phi)
        axs[1].plot(dates, corr)
        for i in range(len(corr_step)):
            axs[1].plot(dates[int(index_list[i])], corr_step[i], marker='x', color='red')
        axs[1].set(ylabel="Phase diff (rad)", xlabel="Time (UTC)")

        plt.xlabel('Time (UTC)')

        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_phase_correction.png")
            plt.close()

    if plot_fringe_polar == True:
        lim1 = np.zeros_like(fringe_theta)
        lim2 = np.ones_like(fringe_theta)
        lim1[6667:13333] = 1  # +/- 30 deg
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar')
        if np.abs(x_pos[A1] - x_pos[A2]) > np.abs(y_pos[A1] - y_pos[A2]):
            ax.set_theta_zero_location("N")
            # ax.fill_between(fringe_theta, 0, 1.1, where=lim1==lim2, facecolor='green', alpha=0.3)
            ax.plot(np.radians(azimuth), np.ones_like(azimuth) * 1.1, marker='x')
            ax.plot(np.radians(azimuth), cygnus_beam / np.max(G))
            ax.plot(np.radians(azimuth), cygnus_fringe)
            pos_label = 'Azimuth'
        else:
            # ax.fill_between(fringe_theta, 0, 1.1, where=fringe_theta>70*np.pi/180, facecolor='green', alpha=0.3)
            ax.plot(np.radians(elevation), np.ones_like(azimuth) * 1.1, marker='x')
            ax.plot(np.radians(elevation), cygnus_beam / np.max(G))
            ax.plot(np.radians(elevation), cygnus_fringe)
            pos_label = 'Elevation'
        # ax.plot(fringe_theta, I)
        ax.set_title("Fringe pattern for Baseline {}-{}: d={}".format(A1, A2, d))

        plt.legend(['Cygnus A {}'.format(pos_label), 'Normalized Cygnus Beam Pattern', 'Fringe Pattern of Cygnus Path'])
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_linear_fringe.png")
            plt.close()

    if plot_fringe_2D == True:
        fig = plt.figure()
        plt.pcolormesh(np.degrees(fringe_beta), np.degrees(fringe_alpha), I_2D)
        plt.plot([-180, 180], [0, 0], color='black')
        plt.plot(azimuth, elevation, color='red')
        plt.legend(['Horizon', 'Cygnus A Path'])
        plt.title(
            "2D Fringe Pattern For ICEBEAR {}, Baseline{}-{}: d = {}, pnt = {}".format(array_type_str, A1, A2, d, pnt))
        plt.xlabel("Azimuth Angle (Degrees)")
        plt.ylabel("Elevation Angle (Degrees)")
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_2D_fringe.png")
            plt.close()

    if plot_beam_fringe == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, power_diff / np.max(power_diff), linestyle='None', marker='.')
        plt.plot(dates, cygnus_beam / np.max(G))
        plt.plot(dates, cygnus_fringe)
        # plt.plot(dates, cygnus_fringe*(cygnus_beam+30)/np.max(G+30))
        x = np.argwhere(elevation == np.min(elevation))
        plt.plot(dates[x], np.zeros(len(x)), marker='o')

        plt.title("Comparison of Radar Beam, Fringe Pattern, and Meaured Power: Baseline {}-{}".format(A1, A2))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Normalized Power")
        plt.legend(
            ['Power Variance around ~{}dB'.format(int(np.average(power_offset))), 'Normalized Beam Power (30dBi)',
             '2D Fringe Path'])
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_fringe_beam_power.png")
            plt.close()

    if plot_cygnus_angles == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, elevation)
        plt.plot(dates, azimuth)
        plt.xlabel('Time (UTC)')
        plt.ylabel("Degrees")
        plt.legend(("Elevation Angle of Cygnus A", "Azimuth Angle of Cygnus A"))
        plt.title("Cygnus A Location Angles")
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_cygnus_angles.png")
            plt.close()

    if plot_cygnus_beam_2d == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, cygnus_beam)
        plt.xlabel('Time (UTC)')
        plt.ylabel('Power (dBi)')
        plt.title("Beam Pattern For Cygnus A Path: Baseline " + str(A1) + "-" + str(A2))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_cygnus_beam_path.png")
            plt.close()

    if plot_beam_cygnus_3d == True:
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.set_zlim(np.min(G), np.max(G))

        norm = mpl.colors.Normalize(vmin=np.min(G), vmax=np.max(G))
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(norm(G)), alpha=0.4)
        m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        m.set_array([])
        plt.colorbar(m)
        ax.scatter(x, y, z, marker='x', color='green', alpha=0.1)
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_cygnus_vs_beam_pattern.png")
            plt.close()

    if plot_cygnus_beam_3d == True:
        ind = np.argwhere(cygnus_beam > beam_cutoff)
        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.set_zlim(np.min(G), np.max(G))

        ax.scatter(x[ind], y[ind], z[ind], marker='x', color='green', alpha=0.1)
        ax.scatter(cx, cy, cz, marker='.', color='blue')
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_cygnus_3d_beam_path.png")
            plt.close()

    if plot_power_phase == True:
        fig, axs = plt.subplots(2)
        axs[0].xaxis.set_major_locator(locator)
        axs[0].xaxis.set_major_formatter(formatter)
        axs[1].xaxis.set_major_locator(locator)
        axs[1].xaxis.set_major_formatter(formatter)

        axs[0].plot(dates, power, linestyle='None', marker='.')
        axs[0].plot(dates, db_fit)
        axs[0].plot(dates[bore_sight], power[bore_sight], markersize=10, marker='x', linestyle='None', color='red')
        # axs[0].plot(dates,power_fit+power_offset)
        fig.suptitle("Noise Power  {}-{} [{}]: d={}, pnt={}".format(A1, A2, dates[0], d, np.degrees(pnt)))
        axs[0].set(ylabel="Power (dB)")
        axs[0].legend(("Measured Power", "Running Average", "Bore-sight"))
        axs[1].plot(dates, angle * 180 / np.pi, linestyle='None', marker='.')
        axs[1].plot(dates, phase_fit * 180 / np.pi)
        axs[1].set(ylabel="Phase (Deg)", xlabel="Time (UTC)")
        axs[1].legend(("Measured Phase", "Running Average"))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_power_and_phase.png")
            plt.close()

    if plot_phase_vs_expect == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, phi * 180 / np.pi, color='green')
        plt.plot(dates, angle * 180 / np.pi, linestyle='None', marker='.', color='tab:blue')
        # if peak_determine==True:
        #    plt.fill_between(dates, np.min(phi*180/np.pi), np.max(phi*180/np.pi), where=power_diff>power_cutoff, facecolor='green', alpha=0.5)
        # else:
        #    plt.fill_between(dates, np.min(phi*180/np.pi), np.max(phi*180/np.pi), where=cygnus_beam>beam_cutoff, facecolor='green', alpha=0.5)
        plt.plot(dates[bore_sight], 0, markersize=10, marker='x', linestyle='None', color='red')
        plt.legend(("Theoretical Phase", "Measured Phase"))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Phase (deg)")
        plt.title("Theoretical and Measured Phase for Baseline {}-{}".format(A1, A2))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_theoretical_comparison.png")
            plt.close()

    if plot_corr_phase_vs_expect == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, (angle) * 180 / np.pi, linestyle='None', marker='.', color='tab:blue')
        plt.plot(dates, (phase_corr) * 180 / np.pi, linestyle='None', marker='.', color='tab:orange')
        plt.plot(dates, phi * 180 / np.pi, color='green')
        plt.plot(dates[bore_sight], 0, markersize=10, marker='x', linestyle='None', color='red')
        # plt.plot(dates[peak_index],0,marker='x')

        plt.legend(("Measured Phase", "Corrected Phase", "Theoretical Phase"))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Phase (deg)")
        plt.title("Theoretical and Corrected Phase for Baseline {}-{}".format(A1, A2))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_corrected_comparison.png")
            plt.close()

    if plot_power_vs_cygnus_beam == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, power_diff / np.max(power_diff), linestyle='None', marker='.')
        # if peak_determine==True:
        #    plt.fill_between(dates, np.min(cygnus_beam/np.max(G)), np.max(cygnus_beam/np.max(G)), where=power_diff>power_cutoff, facecolor='green', alpha=0.5)
        # else:
        #    plt.fill_between(dates, np.min(cygnus_beam/np.max(G)), np.max(cygnus_beam/np.max(G)), where=cygnus_beam>beam_cutoff, facecolor='green', alpha=0.5)
        plt.plot(dates, cygnus_beam / np.max(G))
        plt.plot(dates[bore_sight], 0, markersize=10, marker='x', linestyle='None', color='red')
        plt.plot(dates[beam_peak], 0, markersize=10, marker='x', linestyle='None', color='black')
        # plt.plot(dates[phi_min],0,markersize=10,marker='x',linestyle='None',color='green')
        plt.legend((
                   'Power Variance around ~{}dB'.format(int(np.average(power_offset))), 'Normalized Beam Power (30dBi)',
                   'Bore-sight', 'Beam Peak'))  # , 'Min Phi'))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Normalized Amplitude")
        plt.title("Beam Pattern and Noise for Baseline {}-{}: d = {}m".format(A1, A2, int(np.abs(d))))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_power_vs_beam_pattern.png")
            plt.close()

    if plot_power == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, power, linestyle='None', marker='.')
        # if peak_determine==True:
        #    plt.fill_between(dates, np.min(cygnus_beam/np.max(G)), np.max(cygnus_beam/np.max(G)), where=power_diff>power_cutoff, facecolor='green', alpha=0.5)
        # else:
        #    plt.fill_between(dates, np.min(cygnus_beam/np.max(G)), np.max(cygnus_beam/np.max(G)), where=cygnus_beam>beam_cutoff, facecolor='green', alpha=0.5)
        plt.plot(dates[bore_sight], np.median(power), markersize=10, marker='x', linestyle='None', color='red')
        plt.plot(dates[beam_peak], np.median(power), markersize=10, marker='x', linestyle='None', color='black')
        # plt.plot(dates[phi_min],np.median(power),markersize=10,marker='x',linestyle='None',color='green')
        plt.legend(('Noise Power', 'Bore-sight', 'Beam Peak'))  # , 'Min Phi'))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Power (dB)")
        plt.title("Noise Power for Baseline {}-{}: d = {}m".format(A1, A2, int(np.abs(d))))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_power.png")
            plt.close()

    if plot_measured_aoa_vs_theory == True:
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.plot(dates, aoa_original * 180 / np.pi)
        plt.plot(dates, aoa_corrected * 180 / np.pi)
        plt.plot(dates, aoa_theoretical * 180 / np.pi)
        plt.legend(("Measured Angle of Arival", "Corrected Angle of Arrival", "Theoretical Angle of Arrival"))
        plt.xlabel("Time (UTC)")
        plt.ylabel("Angle of Arrival (Deg)")
        plt.title("Angle of Arrivals for Baseline {}-{}".format(A1, A2))
        if save_plot == True:
            fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-A" + read + "_aoa_calculation.png")
            plt.close()

    print("End of Itteration")

if goodness == True:
    fig, ax = plt.subplots()

    print(goodness_array)

    Z = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            for ant in range(len(array1)):
                if (goodness_array[0, ant] == i) and (goodness_array[1, ant] == j):
                    Z[i, j] = goodness_array[2, ant]
                if (goodness_array[0, ant] == j) and (goodness_array[1, ant] == i):
                    Z[i, j] = goodness_array[2, ant]
                if i == j:
                    Z[i, j] = 1

    im = ax.imshow(Z, cmap='winter', vmax=1, vmin=-150)
    fig.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, int(Z[i, j]), ha="center", va="center", color="w")

    plt.xlabel("Antenna #")
    plt.ylabel("Antenna #")
    plt.title("Corrected Coefficient of Determination")
    # plt.colorbar()
    if save_plot == True:
        fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-corrected_fit.png")
        plt.close()

    fig, ax = plt.subplots()

    print(original_array)

    Y = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            for ant in range(len(array1)):
                if (goodness_array[0, ant] == i) and (goodness_array[1, ant] == j):
                    Y[i, j] = original_array[ant]
                if (goodness_array[0, ant] == j) and (goodness_array[1, ant] == i):
                    Y[i, j] = original_array[ant]
                if i == j:
                    Y[i, j] = 1

    im = ax.imshow(Y, cmap='winter', vmax=1, vmin=-150)
    fig.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, int(Y[i, j]), ha="center", va="center", color="w")

    plt.xlabel("Antenna #")
    plt.ylabel("Antenna #")
    plt.title("Measured Coefficient of Determination")
    # plt.colorbar()
    if save_plot == True:
        fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-measured_fit.png")
        plt.close()

    fig, ax = plt.subplots()

    X = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            for ant in range(len(array1)):
                if (goodness_array[0, ant] == i) and (goodness_array[1, ant] == j):
                    X[i, j] = Z[i, j] - Y[i, j]
                if (goodness_array[0, ant] == j) and (goodness_array[1, ant] == i):
                    X[i, j] = Z[i, j] - Y[i, j]
                if i == j:
                    X[i, j] = 0

    im = ax.imshow(X, cmap='winter', vmax=1, vmin=-150)
    fig.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, int(X[i, j]), ha="center", va="center", color="w")

    plt.xlabel("Antenna #")
    plt.ylabel("Antenna #")
    plt.title("Compared Coefficient of Determination")
    # plt.colorbar()
    if save_plot == True:
        fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-compare_fit.png")
        plt.close()

    fig, ax = plt.subplots()

    print(quality_array)

    Q = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            for ant in range(len(array1)):
                if (goodness_array[0, ant] == i) and (goodness_array[1, ant] == j):
                    Q[i, j] = quality_array[ant]
                if (goodness_array[0, ant] == j) and (goodness_array[1, ant] == i):
                    Q[i, j] = quality_array[ant]
                if i == j:
                    Q[i, j] = 0

    plt.imshow(Q, cmap='winter', vmax=3, vmin=0)
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            text = ax.text(i, j, int(Z[i, j]), ha="center", va="center", color="w")

    plt.xlabel("Antenna #")
    plt.ylabel("Antenna #")
    plt.title("Phase Quality")
    # plt.colorbar()
    if save_plot == True:
        fig.savefig(save_loc + "ICEBEAR-" + array_type_str + time_str + "-phase_quality.png")
        plt.close()

statsF.close()

if save_plot != True:
    plt.show()
    plt.close()

outF = open("corrections.txt", "w")
outF.write("Baseline" + "\t" + "Correction" + "\t" + "Variance" + "\t" + "Quality")
outF.write("\n")
for ant in range(len(array1)):
    # write line to output file
    outF.write(str(int(goodness_array[0, ant])) + str(int(goodness_array[1, ant])) + "\t" + str(
        correction_array[ant]) + "\t" + str(variance_array[ant]) + "\t" + str(data_str_list[ant]))
    outF.write("\n")
outF.close()

print(correction_array)
