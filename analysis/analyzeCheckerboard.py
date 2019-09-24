# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:41:05 2019

@author: svc_ccg
"""

import numpy as np
import pandas as pd
import h5py, os, glob
from matplotlib import pyplot as plt
from numba import njit
import scipy.stats, scipy.interpolate


manifestFile = r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\analysis\manifest.csv"
mdf = pd.read_csv(manifestFile)

stim_table_dir = r"C:\Users\svc_ccg\Documents\GitHub\openscope_motion\analysis\stim tables"

#find checkerboard experiments
dff_paths = []
stim_table_paths = []
good_rows = []
for ind, row in mdf.iterrows():
    dff_path = glob.glob(os.path.join(os.path.join('/'+row.storage_directory, 'ophys_experiment*'), '*_dff.h5'))
    stim_table_path = glob.glob(os.path.join(stim_table_dir, str(row.ophys_session_id)+'*'))
    if len(dff_path)>0 and len(stim_table_path)>0:
        stim_table = pd.read_csv(stim_table_path[0])
        if 'patchSize' in stim_table.columns:
            dff_paths.append(dff_path[0])
            stim_table_paths.append(stim_table_path[0])
            good_rows.append(ind)

mdfcheck = mdf.iloc[good_rows]



#experimentID = '917456401'
#stim_table_file = r"C:\Users\svc_ccg\Desktop\Data\motionscope\pilot\917456401_stim_table.csv"
#cell_table_file = r"C:\Users\svc_ccg\Desktop\Data\motionscope\pilot\917456401_analysis.h5"

#stim_table = pd.read_csv(stim_table_file)
#cell_table = h5py.File(cell_table_file)
#
#f = h5py.File(r"\\allen\programs\braintv\production\neuralcoding\prod57\specimen_883606638\ophys_session_917456401\ophys_experiment_917784120\917784120_dff.h5")
#dff = f['data'][()]
#f.close()

patchSpeed = stim_table.patchSpeed.unique()
bckgndSpeed = stim_table.backgroundSpeed.unique()
patchDir = stim_table.patchDir.unique()
backgroundDir = stim_table.backgroundDir.unique()
patchSize = stim_table.patchSize.unique()
patchPos = stim_table.patchPos.unique()

def getSpeedInd(speed, direction):
    speedInd = np.where(patchSpeed==speed)[0][0]
    if direction == 0 or speedInd==0:
        speedInd += 2
    else:
        speedInd = abs(speedInd-2)
    return speedInd

def fillRedundant(mat):
    for mi in np.arange(mat.shape[2]):
        m = mat[:, :, mi]
        for x in np.arange(mat.shape[0]):
            m[x,x] = m[2,x]
    return mat  

def getExperimentDetails(mdf, expID):
    row = mdf.loc[mdf.ophys_session_id == expID]
    return row

def dictToHDF5(saveDict, filePath, fileOut=None, grp=None, overwrite=True):
    if fileOut is None:
        if os.path.isfile(filePath):
            if overwrite:
                os.remove(filePath)
            else:
                filePath = os.path.splitext(filePath)[0] + '_new' + os.path.splitext(filePath)[1]
        
        fileOut = h5py.File(filePath,'a')
        newFile = fileOut
    else:
        newFile = None
    if grp is None:    
        grp = fileOut['/']

    for key in saveDict:
        if key[0]=='_':
            continue
        elif type(saveDict[key]) is dict:
            dictToHDF5(saveDict[key],fileOut=fileOut,grp=grp.create_group(key))
        else:
            try:
                grp.create_dataset(key,data=saveDict[key],compression='gzip',compression_opts=1)
            except:
                try:
                    grp[key] = saveDict[key]
                except:
                    try:
                        grp.create_dataset(key,data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
                    except:
                        print('Could not save: ', key)                  
    if newFile is not None:
        newFile.close()
@njit    
def getShuffledTrialResponse(cellData, trialDur, num_iter=10000):
    means = np.zeros(num_iter)
    maxs = np.zeros(num_iter)
    for i in xrange(num_iter):
        start = np.random.randint(0, cellData.size-trialDur)
        strial = cellData[start:start+trialDur]
        means[i] = np.mean(strial)
        maxs[i] = np.max(strial)
    
    return np.mean(means), np.mean(maxs)

def makeArrayFromTrialResponseList(trialResponses):
    modeDuration = scipy.stats.mode([len(t) for t in trialResponses])[0][0]
    trialResponses_reshape = []
    for t in trialResponses:
        if t.size!=modeDuration:
            x = np.linspace(0, modeDuration, t.size)
            f = scipy.interpolate.interp1d(x, t)
            t = f(np.arange(modeDuration))
        
        trialResponses_reshape.append(t)
    
    return np.array(trialResponses_reshape)
    

saveDir = r"C:\Users\svc_ccg\Desktop\Data\motionscope\pilot"
for ind, (dff_path, stim_table_path) in enumerate(zip(dff_paths, stim_table_paths)):
    stim_table = pd.read_csv(stim_table_path)
    with h5py.File(dff_path) as f:
        dff = f['data'][()]
    
    expDetails = mdfcheck.iloc[ind, 1:].to_dict()
    print(expDetails)
    
    expDict = {'peakResp':[], 'meanResp':[], 'shuff_peakResp':[], 'shuff_meanResp':[], 'responseMat':[]}
    expDict.update(expDetails)
    peakRespPop = []
    meanRespPop = []
    
    maxTrialDuration = int(np.ceil(stim_table.duration.max()))
    for cell in np.arange(dff.shape[0]):
    #    plt.figure(cell)
        meanResp = np.full((2*len(patchSpeed)-1, 2*len(bckgndSpeed)-1, len(patchSize)), np.nan)
        peakResp = meanResp.copy()
        respMat = np.full((2*len(patchSpeed)-1, 2*len(bckgndSpeed)-1, len(patchSize), maxTrialDuration), np.nan)
        shuff_meanResp = meanResp.copy()
        shuff_peakResp = meanResp.copy()
        for trial in stim_table.trial_number.unique():
            tempdf = stim_table.loc[stim_table.trial_number==trial]
            psizeInd = np.where(patchSize==tempdf.patchSize.values[0])[0][0]
            pspeedInd = getSpeedInd(tempdf.patchSpeed.values[0], tempdf.patchDir.values[0])
            bspeedInd = getSpeedInd(tempdf.backgroundSpeed.values[0], tempdf.backgroundDir.values[0])
            
            means = []
            maxs = []
            trialResponses = []
            for ind, row in tempdf.iterrows():
                trialResponse = dff[cell, int(row.start):int(row.end)]
                meantrial = trialResponse.mean()
                maxtrial = trialResponse.max()
                
                trialResponses.append(trialResponse)
                means.append(meantrial)
                maxs.append(maxtrial)
            
            
            meanResp[pspeedInd, bspeedInd, psizeInd] = np.mean(means)
            peakResp[pspeedInd, bspeedInd, psizeInd] = np.mean(maxs)
            
            trialResponses = makeArrayFromTrialResponseList(trialResponses)
            respMat[pspeedInd, bspeedInd, psizeInd, :trialResponses.shape[-1]] = np.nanmean(trialResponses, axis=0)
            #SHUFFLE TO CORRECT FOR TRIAL DURATION DIFFERENCES
            #TODO: JUST GIVE ONE SHUFFLE VALUE FOR EACH UNIQUE TRIAL DURATION
            #TODO: FIGURE OUT HOW TO NORMALIZE RESPONSE BY THIS... MAYBE RETURN STD INSTEAD AND DIVIDE?
            trialDur = int(trialResponses.shape[-1])
            shuffmean, shuffmax = getShuffledTrialResponse(dff[cell], trialDur)
            shuff_meanResp[pspeedInd, bspeedInd, psizeInd] = shuffmean
            shuff_peakResp[pspeedInd, bspeedInd, psizeInd] = shuffmax
                
        
        for arr in [meanResp, peakResp, shuff_meanResp, shuff_peakResp]:
            arr = fillRedundant(arr)
            
        
        expDict['peakResp'].append(peakResp)
        expDict['meanResp'].append(meanResp)
        expDict['shuff_peakResp'].append(shuff_peakResp)
        expDict['shuff_meanResp'].append(shuff_meanResp)
        expDict['responseMat'].append(respMat)
        
            
    #    plt.imshow(np.nanmean(peakResp, axis=2))
        peakRespPop.append(np.nanmean(peakResp-shuff_peakResp, axis=2))
        meanRespPop.append(np.nanmean(meanResp-shuff_meanResp, axis=2))
    
    dictToHDF5(expDict, os.path.join(saveDir, str(expDetails['ophys_session_id']) + '_' + expDetails['targeted_structure']+'.hdf5'))
    fig,ax = plt.subplots(1,2)
    fig.suptitle(expDetails['targeted_structure'])
    for a, rp, title in zip(ax, [peakRespPop, meanRespPop], ['peak', 'mean']):
        a.imshow(np.nanmean(rp, axis=0), cmap='plasma')
        a.set_title(title)
    
                
                
                
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
