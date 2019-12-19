# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:06:28 2019

@author: svc_ccg
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt


def apply_gaussian_window(array, sigma_deg=15, center=None, pixelsPerDegree=5, output_dtype=np.uint8, clip=True, plot=False):

    height = array.shape[1]
    width = array.shape[2]
    
    if center is None:
        center = [int(round(d/2.)) for d in [height, width]]

    sigma_pix = sigma_deg*pixelsPerDegree
    
    gaussian_array = np.zeros((height, width))
    gaussian_array[center[0], center[1]] = 1

    gaussian_array = gaussian_filter(gaussian_array, sigma_pix, truncate=3)
    gaussian_array = gaussian_array/gaussian_array.max()
    
    if clip:
        gaussian_array = np.clip(gaussian_array, 0, 0.607)/0.607
    
    im_array_zero_centered = array.astype(np.int) - 127
    windowed_array = im_array_zero_centered*gaussian_array + 127
    windowed_array = windowed_array.astype(output_dtype)
    
    if plot:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(gaussian_array, cmap='gray')
        ax[1].imshow(windowed_array[0], cmap='gray')
        
    
    return windowed_array


def apply_linear_window(array, mask_start_radius=15, mask_end_radius=40, center=None, pixelsPerDegree=5, output_dtype=np.uint8, plot=False):
    
    height = array.shape[1]
    width = array.shape[2]
    
    mask_start_radius_pix = mask_start_radius * pixelsPerDegree
    mask_end_radius_pix = mask_end_radius * pixelsPerDegree
    
   
    if center is None:
        center = [int(round(d/2.)) for d in [height, width]]
    
    #Make array that is encodes distance from center position
    heightfromcenter = np.arange(height) - center[0]
    widthfromcenter = np.arange(width) - center[1]
    
    hh, ww = np.meshgrid(heightfromcenter, widthfromcenter)
    
    distfromcenter = (hh**2 + ww**2)**0.5
    if distfromcenter.shape[0] != height:
        distfromcenter = distfromcenter.T
    
    #Make mask by clipping and inverting distance array to get linear fall off between start and end radii
    mask_array = np.clip(distfromcenter, mask_start_radius_pix, mask_end_radius_pix)
    mask_array = mask_array - mask_start_radius_pix
    mask_array = 1 - mask_array/mask_array.max()
    
    im_array_zero_centered = array.astype(np.int) - 127
    windowed_array = im_array_zero_centered*mask_array + 127
    windowed_array = windowed_array.astype(output_dtype)

    if plot:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(mask_array, cmap='gray')
        ax[1].imshow(windowed_array[0], cmap='gray')
        
    
    return windowed_array
    