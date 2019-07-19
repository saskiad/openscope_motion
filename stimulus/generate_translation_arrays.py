# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:32:46 2019

@author: svc_ccg
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize
import dotTranslationClass

#FIND RELATIONSHIP BETWEEN DOTNUM AND SCREEN DENSITY
d = dotTranslationClass.dotTranslation()
iterations = 10
dotNums = [20, 40, 80, 160]
densityPerDot_i = []
plt.figure()
for i in np.arange(iterations):
    lum = []
    for dotNum in dotNums:
        d.dotNum=dotNum
        d.timePoints=60
        d.dotColors = [255]
        d.backgroundColor = 0
        d.makeStimulusArray()
        
        lum.append(np.mean([np.mean(im) for im in d.im_array[20:]]))
    
    lum = np.array(lum)
    plt.plot(dotNums, lum/255., 'ko')
    
    densityPerDot, pcov = scipy.optimize.curve_fit(lambda x,b: b*x, dotNums, lum/255.)
    densityPerDot_i.append(densityPerDot[0])
    plt.plot(dotNums, np.array(dotNums)*densityPerDot[0])
densityPerDot = np.mean(densityPerDot_i)

#Determined before for scale factors 1/20 and 1/40
densityPerDot_20 = 0.0005227195853341336
densityPerDot_40 = 0.00012861953781512608


#MAKE STIM ARRAYS
d = dotTranslationClass.dotTranslation()
d.saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\translation"
headings = ['forward', 'backward']
centers = [[-50,-10]]
radius = [10]
speeds = [4, 2, 0.5, 0.1, 0.05]
dotDensities = [0.2, 0.4, 0.8]
scaleFactors = [1/20., 1/40.]


for h in headings:
    for center in centers:
        for rad in radius:
            for speed in speeds:
                for scale in scaleFactors:
                    for dotDensity in dotDensities:
                        d.heading = h
                        d.centerPosDegrees=np.array(center)
                        d.radius = rad
                        d.speed = speed
                        d.timePoints =  np.max([120, 60+(d.viewingDistance/speed)])
                        if scale == 0.05:
                            dotNum = np.round(dotDensity/densityPerDot_20).astype(int) 
                        else:
                            dotNum = np.round(dotDensity/densityPerDot_40).astype(int)
                        
                        d.dotNum = dotNum
                        d.scaleFactor = scale
                        d.findCenterDotMetrics()
                        saveFileName = (h + '_center_' + str(center[0]) + str(center[1]) + '_radius_' + str(rad) + '_speed_' + str(speed) 
                        + '_dotNum_' + str(dotNum) + '_speedAtCenter_' + str(int(d.dotSpeedInCenter)) + '_radiusAtCenter_' + str(int(d.dotRadiusInCenter)) 
                        + '_dotDensity_' + str(int(100*dotDensity)) + '_scaleFactor_' + str(int(scale*1000)) + '.npy')
                        
                        print(saveFileName)
                        d.makeStimulusArray(save=True, fileName=saveFileName, compressed=False, offset=-60)
                    

