# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:39:07 2019

@author: svc_ccg
"""
import h5py, os
import pandas as pd

#Get parameters for the checkerboard trials
p = h5py.File(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\checkerboard\params.hdf5")

#Specify column names  and parameters to extract for dataframe
columnNames = ['backgroundDir', 'backgroundSpeed', 'numFrames', 'patchDir', 'patchPos', 'patchSize', 'patchSpeed']
paramsField = [u'trialBckgndDir',
 u'trialBckgndSpeed',
 u'trialNumFrames',
 u'trialPatchDir',
 u'trialPatchPos',
 u'trialPatchSize',
 u'trialPatchSpeed',
 u'trialStartFrame']


#Make checkerboard dataframe
cdf = pd.DataFrame(columns = columnNames)
for c, pf in zip(columnNames, paramsField):
    cdf[c] = p[pf][:]
    
#Save it
saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\checkerboard"
savePath = os.path.join(saveDir, 'checkerboardTrialParams.csv')

cdf.to_csv(savePath)