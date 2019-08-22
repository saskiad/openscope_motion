# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:09:46 2019

@author: svc_ccg
"""
import numpy as np
import pandas as pd
import os

p = pd.read_pickle(r"\\allen\programs\braintv\production\neuralcoding\prod57\specimen_883606638\ophys_session_923685768\923685768_469325_20190814_stim.pkl")
stim = p['stimuli']

dotstim1_framelist = np.array(pd.read_pickle(r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\stimulus\dot_stim_one_frame_list.pkl"))
dotstim2_framelist = np.array(pd.read_pickle(r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\stimulus\dot_stim_two_frame_list.pkl"))

movie_manifest = pd.read_pickle(r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\stimulus\dot_stim_movie_manifest.pkl")


def findTrialFrameStarts(framelist, movie_manifest, movieLength=60):
    ind = (framelist[:-1]==-1) & (framelist[1:]>-1)
    ind = np.insert(np.where(ind)[0]+1, 0, 0)
    
    framelistind = framelist[ind]
    movieInd = framelistind/movieLength
    
    return framelistind, movieInd

def parseMovieName(movieName, paramNames):
    parts = movieName.split('_')
    params = {a:[] for a in paramNames}
    if parts[0] == 'rotation':
        params['motion'] = 'rotation'
        params['direction'] = parts[1]
        params['centerLocation'] = np.nan
        params['approachSpeed'] = np.nan
        params['dotSpeed'] = parts[3]
        params['dotSize'] = parts[-1].split('.')[0]
        params['dotNum'] = parts[2]
    else:
        params['motion'] = 'translation'
        params['direction'] = parts[0]
        params['centerLocation'] = parts[2]
        params['approachSpeed'] = parts[6]
        params['dotSpeed'] = parts[10]
        params['dotSize'] = parts[12]
        params['dotNum'] = parts[8]
    
    return params
        
        
        

trialFrameStarts1, movieInd1 = findTrialFrameStarts(dotstim1_framelist, movie_manifest)
trialFrameStarts2, movieInd2 = findTrialFrameStarts(dotstim2_framelist, movie_manifest)

saveDir = r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\analysis"

columns = ['motion', 'direction', 'centerLocation', 'approachSpeed', 'dotSpeed', 'dotSize', 'dotNum']
rows=[]
for m in np.unique(movieInd1):
    movieName = os.path.basename(movie_manifest[m])
    paramdict = parseMovieName(movieName, columns)
    paramdict['movieFrameStart'] = m*60
    rows.append(paramdict) 
ddf1 = pd.DataFrame(rows)    
ddf1.to_csv(os.path.join(saveDir, 'dotstim1_param_table.csv'))

rows = []
for m in np.unique(movieInd2):
    movieName = os.path.basename(movie_manifest[m])
    paramdict = parseMovieName(movieName, columns)
    paramdict['movieFrameStart'] = m*60
    rows.append(paramdict) 
ddf2 = pd.DataFrame(rows)    
ddf2.to_csv(os.path.join(saveDir, 'dotstim2_param_table.csv'))    
    
    
