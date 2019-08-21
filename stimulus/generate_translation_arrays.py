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
iterations = 2
dotNums = [20, 80, 320, 800, 1600, 5000]
densityPerDot_i = []
func = lambda x,a,b: b*x**(float(a))
plt.figure()
for i in np.arange(iterations):
    lum = []
    for dotNum in dotNums:
        d.dotNum=dotNum
        d.timePoints=500
        d.dotColors = [255]
        d.backgroundColor = 0
        d.makeStimulusArray(offset=-60)
        
        lum.append(np.mean([np.mean(im[200:400, 390:590]) for im in d.im_array]))
    
    lum = np.array(lum)
    plt.plot(lum/255., dotNums, 'ko')
    
    densityPerDot, pcov = scipy.optimize.curve_fit(func, lum/255., dotNums)
    densityPerDot_i.append(densityPerDot)
    plt.plot(lum/255., func(lum/255., *densityPerDot))

densityPerDot = np.mean(densityPerDot_i, axis=0)

#Determined before for scale factors 1/20 and 1/40
densityPerDot_20 = np.array([1.27355431e+00, 1.16888103e+04])
densityPerDot_40 = np.array([1.06916041e+00, 3.56860910e+04])


#MAKE STIM ARRAYS
d = dotTranslationClass.dotTranslation()
d.saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\translation"
headings = ['forward', 'backward']
centers = [[-50,-10]]
radius = [10]
speeds = [2.05, 1, 0.48, 0.24, 0.096, 0.05] #to make speeds at center [400,200,100,50,20,10]
dotDensities = [0.1, 0.2, 0.4]
scaleFactors = [1/20., 1/40.]

fig, ax = plt.subplots()
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
                        d.timePoints =  np.max([500, 60+3*(d.viewingDistance/speed)])
                        if scale == 0.05:
                            dotNum = np.round(func(dotDensity, *densityPerDot_20)).astype(int)
                        else:
                            dotNum = np.round(func(dotDensity, *densityPerDot_40)).astype(int)
                        
                        d.dotNum = dotNum
                        d.scaleFactor = scale
                        d.findCenterDotMetrics()
                        saveFileName = (h + '_center_' + str(center[0]) + str(center[1]) + '_radius_' + str(rad) + '_speed_' + str(speed) 
                        + '_dotNum_' + str(dotNum) + '_speedAtCenter_' + str(int(d.dotSpeedInCenter)) + '_radiusAtCenter_' + str(int(d.dotRadiusInCenter)) 
                        + '_dotDensity_' + str(int(100*dotDensity)) + '_scaleFactor_' + str(int(scale*1000)) + '.npy')
                        
                        print(saveFileName)
                        d.makeStimulusArray(save=True, fileName=saveFileName, compressed=False, offset=-60)
                        d.playMovie(ax=ax)
                    

