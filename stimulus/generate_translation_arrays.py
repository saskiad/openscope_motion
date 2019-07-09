# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:32:46 2019

@author: svc_ccg
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize
import dotTranslationClass

d = dotTranslationClass.dotTranslation()

#FIND RELATIONSHIP BETWEEN DOTNUM AND SCREEN DENSITY
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

densityPerDot, pcov = scipy.optimize.curve_fit(lambda x,b: b*x, dotNums, lum/255.)
plt.plot(dotNums, np.array(dotNums)*densityPerDot[0])

#MAKE STIM ARRAYS
d = dotTranslationClass.dotTranslation()
headings = ['forward', 'backward']
centers = [[0,0], [-60,-20]]
radius = [10]
speeds = [10, 5, 2, 1]
dotDensities = [0.05, 0.1, 0.2]
dotNums = np.round(np.array(dotDensities)/densityPerDot[0]).astype(int)

for h in headings:
    for center in centers:
        for rad in radius:
            for speed in speeds:
                for dotNum in dotNums:
                    d.heading = h
                    d.centerPosDegrees=np.array(center)
                    d.radius = rad
                    d.speed = speed
                    d.dotNum = dotNum
                    saveFileName = h + '_center_' + str(center[0]) + str(center[1]) + '_radius_' + str(rad) + '_speed_' + str(speed) + '_dotNum_' + str(dotNum) + '.npz'
                    print(saveFileName)
                    d.makeStimulusArray(save=True, fileName=saveFileName)
                    

