# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:06:28 2019

@author: svc_ccg
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_window(array, sigma_deg=15, center=None, pixelsPerDegree=5, output_dtype=np.uint8, clip=True):

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
    
    im_array_zero_centered = im_array.astype(np.int) - 127
    windowed_array = im_array_zero_centered*gaussian_array + 127
    windowed_array = windowed_array.astype(output_dtype)
    
    return windowed_array