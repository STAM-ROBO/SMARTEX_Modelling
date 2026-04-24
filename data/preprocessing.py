"""
Data preprocessing utilities: normalization, flattening, reshaping.
"""
import numpy as np
from scipy.signal import savgol_filter
import scipy
from scipy.signal import medfilt
import cv2 as cv2
from scipy.ndimage import uniform_filter1d
def calibrate_image(img, dark_ref, white_ref):
    pixels = img.reshape(-1, img.shape[-1])
    #print(dark_ref.shape,white_ref.shape,pixels.shape)
    pixels = (pixels - np.mean(dark_ref,0)) / (np.mean(white_ref,0) - np.mean(dark_ref,0))
    
    return pixels.reshape(img.shape)
    #return (img - np.mean(dark_ref,0)) / (np.mean(white_ref,0) - np.mean(dark_ref,0))
def calibrate_pixels(pixels,dark,white):
    
    return (pixels-np.mean(dark,0))/(np.mean(white,0)-np.mean(dark,0))
def snv(X, eps=1e-8):
    # X: (n_samples, n_bands)
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + eps
    return (X - mu) / sigma
def l2_normalize(X, eps=1e-12):
    norm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norm
def savgol_deriv(X, window_length=11, polyorder=2, deriv=1):
    # Applies along spectral axis (last dim)
    return savgol_filter(X, window_length=window_length, polyorder=polyorder,
                         deriv=deriv, axis=1)
def preprocess_spectra(cfg,hsi_data,dark,white):
    org_shape = hsi_data.shape
    #print('org_shape',org_shape)
    
    hsi_data = calibrate_image(hsi_data, dark, white)
    
    
    hsi_data_filtered = uniform_filter1d(
    hsi_data,
    size=3,
    axis=0,
    mode='reflect'
    )
    hsi_data = hsi_data_filtered.reshape(-1,242)
    hsi_data = snv(hsi_data)
    
    #if cfg.get("l2_after_snv", True):
        #hsi_data = l2_normalize(hsi_data)
    #if cfg.get("use_derivative", True):
    #hsi_data = savgol_deriv(hsi_data, window_length=3, polyorder=2, deriv=1)
    #hsi_data = medfilt(hsi_data, kernel_size=[1,5])
    #hsi_data = savgol_filter(hsi_data, window_length=5, polyorder=2, axis=-1)
    hsi_data = hsi_data.reshape(org_shape)
    return hsi_data