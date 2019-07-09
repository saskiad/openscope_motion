# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:32:46 2019

@author: svc_ccg
"""
from matplotlib import pyplot as plt
import numpy as np


headings = ['forward', 'backward']
centers = [[0,0], [-60,-20]]
radius = [10]
speed = [10, 5, 2, 1]
dotNums = [20, 40, 80, 160]

lum = []
for dotNum in dotNums:
    d.dotNum=dotNum
    d.timePoints=60
    d.dotColors = [255]
    d.backgroundColor = 0
    d.makeStimulusArray()
    
    lum.append(np.mean([np.mean(im) for im in d.im_array[20:]]))

lum = np.array(lum)
plt.figure()
plt.plot(dotNums, lum/255., 'ko')
