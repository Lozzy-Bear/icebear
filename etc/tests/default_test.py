import matplotlib.pyplot as plt
import numpy as np
import math as mat
from scipy.integrate import dblquad
import multiprocessing as mp
import cv2
import h5py
import pickle


# gaussian shape in image space
def gaussian_fit(x,peak,variance,mean):
    return peak*np.exp(-(x-mean)**2/(2.0*variance*variance))


# need to integrate this function over theta and phi, with u,v,w,theta_mean,theta_spread,phi_mean,phi_spread known
def real_image_pre_integration(theta,phi,u_in,v_in,w_in,theta_mean,theta_spread,phi_mean,phi_spread):
    return 2*np.real(np.exp(-(theta-theta_mean)**2/(2.0*theta_spread*theta_spread))*np.exp(-(phi-phi_mean)**2/(2.0*phi_spread*phi_spread))*np.cos(phi)*np.exp(-2.0j*mat.pi*((u_in*np.sin(theta)*np.cos(phi))+(v_in*np.cos(theta)*np.cos(phi))+(w_in*np.sin(phi)))))#np.real(np.exp(-(theta-theta_mean)**2/(2.0*theta_spread*theta_spread))*np.exp(-(phi-(phi_mean-np.deg2rad(5)))**2/(2.0*phi_spread*phi_spread))*np.cos(phi)*np.exp(-2.0j*mat.pi*((u_in*np.sin(theta)*np.cos(phi))+(v_in*np.cos(theta)*np.cos(phi))+(w_in*np.sin(phi)))))


def imag_image_pre_integration(theta,phi,u_in,v_in,w_in,theta_mean,theta_spread,phi_mean,phi_spread):
    return 2*np.imag(np.exp(-(theta-theta_mean)**2/(2.0*theta_spread*theta_spread))*np.exp(-(phi-phi_mean)**2/(2.0*phi_spread*phi_spread))*np.cos(phi)*np.exp(-2.0j*mat.pi*((u_in*np.sin(theta)*np.cos(phi))+(v_in*np.cos(theta)*np.cos(phi))+(w_in*np.sin(phi)))))# np.imag(np.exp(-(theta-theta_mean)**2/(2.0*theta_spread*theta_spread))*np.exp(-(phi-(phi_mean-np.deg2rad(5)))**2/(2.0*phi_spread*phi_spread))*np.cos(phi)*np.exp(-2.0j*mat.pi*((u_in*np.sin(theta)*np.cos(phi))+(v_in*np.cos(theta)*np.cos(phi))+(w_in*np.sin(phi)))))


def visibility_calculation(x,u_in1,v_in1,w_in1,theta_mean,theta_spread,phi_mean,phi_spread, output):
    real_vis = dblquad(real_image_pre_integration, -mat.pi/2, mat.pi/2,lambda phi: -mat.pi, lambda phi: mat.pi,args=(u_in1,v_in1,w_in1,theta_mean,theta_spread,phi_mean,phi_spread))[0]
    imag_vis = dblquad(imag_image_pre_integration, -mat.pi/2, mat.pi/2,lambda phi: -mat.pi, lambda phi: mat.pi,args=(u_in1,v_in1,w_in1,theta_mean,theta_spread,phi_mean,phi_spread))[0]
    output.put((x, real_vis+imag_vis*1.0j))