#ICEBEAR 3D Test Script
#author: Devin Huyghebaert
#Date: Feb. 14, 2020

import sys
import string
import getopt
import matplotlib.cm
import os

#libraries
import numpy as np
import cmath as math
import math as mat
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import digital_rf
import time,calendar
from python_processing import ssmf as mf
import h5py
import pyfftw


#size of the plot created (in inches)
mpl.rcParams['figure.figsize'] = [15.0, 12.0]
mpl.rcParams['xtick.labelsize'] = 20.0
mpl.rcParams['ytick.labelsize'] = 20.0  

def abs2(x):
    return x.real**2 + x.imag**2

rx_lat = mat.radians(52.24)
rx_lon = mat.radians(253.55-360.00)
Earth_radius = 6378.1
tx_lat = mat.radians(50.893)
tx_lon = mat.radians(-109.403)

#antenna phase corrections in degrees
phase_corr = [0.0,-13.95,-6.345,-5.89,-3.14,16.86,10.2,-1.25,5.72,3.015]
#approximate average value of the antenna channel, can be used for rough calibration 
mag_corr = [6.708204,6.4031243,6.0827622,6.3245554,6.4031243,6.0827622,6.708204,6.0827622,5.830952,6.0]

#antenna location arrays
x_antenna_loc = [0.,15.10,73.80,24.2,54.5,54.5,42.40,54.5,44.20,96.9]
y_antenna_loc = [0.,0.,-99.90,0.,-94.50,-205.90,-177.2,0.,-27.30,0.]
z_antenna_loc = [0.,0.08952414692,0.3473541846,0.2181136458,0.6834058436,-0.05865289508,-1.06679923,-0.7540427261,-0.5265822222,-0.4087481019]

#calculate the different baselines
num_xspectra = 9+8+7+6+5+4+3+2+1+1

xspectra_x_diff = np.zeros((num_xspectra),dtype=np.float32)
xspectra_y_diff = np.zeros((num_xspectra),dtype=np.float32)
xspectra_z_diff = np.zeros((num_xspectra),dtype=np.float32)

antenna_num_coh_index = 1

for first_antenna in range(9):
	for second_antenna in range(first_antenna+1,10):
		xspectra_x_diff[antenna_num_coh_index] = x_antenna_loc[first_antenna]-x_antenna_loc[second_antenna]
		xspectra_y_diff[antenna_num_coh_index] = y_antenna_loc[first_antenna]-y_antenna_loc[second_antenna]
		xspectra_z_diff[antenna_num_coh_index] = z_antenna_loc[first_antenna]-z_antenna_loc[second_antenna]
		antenna_num_coh_index+=1

#create an array to store the complex values for the phase and magnitude corrections
complex_corr = np.zeros(10,dtype=complex)

#calculate complex numbers to amplitude and phase shift antenna data by
for x in range(10):
	complex_corr[x] = math.rect(1.0/mag_corr[x],math.pi*phase_corr[x]/180.0)

#do not concatenate arrays when printing
#np.set_printoptions(threshold='nan')

#sample rate of receiver
sample_rate=200000

#time of the first plot
year=2018
month=3
day=10
hour=2
hours=hour
minute=1
minutes=minute
second=0

#ranges
nrang = 2000

#number of averages for each plot
averages = 50

#number of plots to make
num_plots = 12

#spacing in seconds between each plot (from last sample used to the start of the next sample)
plot_spacing = 0.0

#array for storing the code to be analyzed
b_code = np.zeros((20000),dtype=np.float32)

#read in code to be tested
test_sig = scipy.fromfile(open("tx_codes/pseudo_random_code_test_8_lpf.txt"),dtype=scipy.complex64)

y=0

#sample code at 1/4 of the tx rate
for x in range(80000):
	if ((x+1)%4==0):
		if (test_sig[x]>0.0):
			b_code[y]=1.0
			y+=1
		else:
			b_code[y]=-1.0
			y+=1

codelen=20000
fdec=200
antenna_data = pyfftw.empty_aligned(codelen+nrang, dtype='complex64', n=16)

#generate array to store information for plotting
max_powers = np.zeros((averages),dtype=float)

testReadObj = digital_rf.DigitalRFReader(['/data/'])
channels = testReadObj.get_channels()
if len(channels) == 0:
    raise IOError("""Please run one of the example write scripts
        C: example_rf_write_hdf5, or Python: example_digital_rf_hdf5.py
        before running this example""")
print('found channels: %s' % (str(channels)))

#start_index, end_index = testReadObj.get_bounds('antenna0')
#cont_data_arr = testReadObj.get_continuous_blocks(start_index, end_index, 'antenna0')

print('done indexing channels')

#create the user defined number of plots
for plot_num in range(num_plots):
    time2=time.time()
    c_test_fft = np.zeros((int(20000/fdec),int(nrang)),dtype=np.float32)
    beam = np.zeros((9,int(20000/fdec),int(nrang)),dtype=np.float32)
    spectra = np.zeros((10,int(20000/fdec),int(nrang)),dtype=np.complex64)
    temp_spectra = np.zeros((10,int(20000/fdec),int(nrang)),dtype=np.complex64)
    spectra_temp = np.zeros((10,int(nrang),int(20000/fdec)),dtype=np.complex64)
    xspectra = np.zeros((int(num_xspectra),int(20000/fdec),int(nrang)),dtype=np.complex64)
    coherence_data = np.zeros((9,int(20000/fdec),int(nrang)),dtype=np.complex64)
    print(plot_num)
    #create the user defined number of averages
    for avg_num in range(averages):
	
        print(avg_num)

        seconds = second+(plot_num*(averages*0.1+plot_spacing))+(avg_num*0.1)

        #calculate the hour and minute from the seconds
        minutes = minute+int(seconds/60.0)
        seconds = seconds % 60.0
        hours = hour+int(minutes/60.0)
        minutes = minutes % 60.0

        time_tuple = (year,month,day,hours,minutes,seconds,-1,-1,0)

        #calculate the start sample
        start_sample = int((calendar.timegm(time_tuple))*sample_rate)-30
        #start_sample = 1510222210*sample_rate
        print(start_sample)

        #put in calculation for start sample here

        print('getting data')

        cput0=time.time()

        antenna_data = np.zeros((10,codelen+nrang),dtype=np.complex64)

        for antenna_num in range(10):
            try:	
                antenna_data[antenna_num,:] = (testReadObj.read_vector_c81d(start_sample, codelen+nrang, 'antenna%01d' %antenna_num))*complex_corr[antenna_num]
                #print np.median(np.abs(antenna_data[antenna_num,:]))
            except IOError:
                print('Read number %i went beyond existing data and raised an IOError' % (i))

        cput1=time.time()
        print("reading speed %1.2f" %((cput1-cput0)/0.1) )

        cput0=time.time()

        for antenna_num in range(10):
            temp_spectra[antenna_num,:,:]=np.transpose(abs2(mf.ssmf(antenna_data[antenna_num,:],b_code)))
            spectra[antenna_num,:,:]+=temp_spectra[antenna_num,:,:]
            temp_spectra[antenna_num,:,:]=np.sqrt(temp_spectra[antenna_num,:,:])		
            #xspectra[0,:,:]+=spectra[antenna_num,:,:]
            #xspectra[antenna_num,:,:]+=np.transpose((mf.ssmfx(antenna_data[3,:],antenna_data[antenna_num,:],b_code)))
            #print np.median(np.abs(spectra[antenna_num,:,:]))

        xspectra[0,:,:]+=1.0

        temp_ind = 0

        #for first_antenna in range(9):
        #    for second_antenna in range(first_antenna+1,10):
        #        xspectra[temp_ind,:,:]+=np.transpose((mf.ssmfx(antenna_data[first_antenna,:],antenna_data[second_antenna,:],b_code)))#/((temp_spectra[first_antenna,:,:])*(temp_spectra[second_antenna,:,:]))
        #        temp_ind+=1

    lam=6.0

    #xspectra = xspectra/averages

    print(np.max(np.abs(xspectra[1:45,:,:])))

    #calculate the fft frequencies
    fft_freq = np.fft.fftshift(np.fft.fftfreq(int(20000/fdec),fdec/200000.0))

    pwr = np.zeros((int(20000/fdec),nrang),dtype=np.float32)

    for antenna_num in range(10):
        #spectra[antenna_num,:,:]=spectra[antenna_num,:,:]-np.mean(spectra[antenna_num,0:100,:])
        #plt.imshow(np.fft.fftshift((np.abs(spectra[antenna_num,:,:].T)-np.median(np.abs(spectra[antenna_num,:,:].T)))/np.median(np.abs(spectra[antenna_num,:,:].T)),axes=1),origin='lower',vmax=3.0,vmin=-3.0,extent=[np.amin(fft_freq)-5.0,np.amax(fft_freq)+5.0,((-30.0)*1.5)/2.0,((nrang-30)*1.5)/2.0],aspect='auto',interpolation='none')
        #plt.colorbar()
        #plt.show()
        pwr+=np.abs(spectra[antenna_num,:,:]-np.mean(spectra[antenna_num,0:100,:]))

    plt.imshow(np.fft.fftshift((np.abs(pwr.T)-np.median(np.abs(pwr.T)))/np.median(np.abs(pwr.T)),axes=1),origin='lower',vmax=8.0,vmin=-3.0,extent=[np.amin(fft_freq)-5.0,np.amax(fft_freq)+5.0,((-30.0)*1.5)/2.0,((nrang-30)*1.5)/2.0],aspect='auto',interpolation='none')
    plt.colorbar()
    plt.show()

    print(np.max(pwr))

    noise=np.median(pwr)

    print(noise)

    snr = (pwr-noise)/noise

    print(np.median(np.abs(snr)))

    time0=time.time()

    #temp_indices = np.unravel_index(np.argmax(np.abs(xspectra[1:45,:,:])),xspectra[1:45,:,:].shape)
    temp_indices = np.unravel_index(np.argmax(np.abs(snr[:,:])),snr[:,:].shape)

    print(temp_indices)
    print(snr[temp_indices[0],temp_indices[1]])

    print(np.abs(xspectra[:,temp_indices[0],temp_indices[1]]))

    pwr=np.fft.fftshift(pwr,axes=0)
    snr=np.fft.fftshift(snr,axes=0)

    snr = 10*np.log10(snr)

    snr = np.ma.masked_where(snr < 1.0,snr)

    snr_2 = snr[:,400:nrang]


