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
import matplotlib.gridspec as gridspec
import clust

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
    if mat.ndim>2:
        mat[:,:,1][np.isnan(mat[:,:,1])] = mat[:,:,0][np.isnan(mat[:,:,1])]
        for mi in np.arange(mat.shape[2]):
            m = mat[:, :, mi]
            for x in np.arange(mat.shape[0]):
                m[x,x] = m[2,x]
    else:
        for x in np.arange(mat.shape[0]):
            mat[x,x] = mat[2,x]
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
    
    return np.mean(means), np.mean(maxs), np.std(maxs)

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

    
roi_table_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\analysis\roi_tables"
saveDir = r"C:\Users\svc_ccg\Desktop\Data\motionscope\pilot"
failed = []
for ind, (dff_path, stim_table_path) in enumerate(zip(dff_paths, stim_table_paths)):
    stim_table = pd.read_csv(stim_table_path)
    with h5py.File(dff_path) as f:
        dff = f['data'][()]
    
    expDetails = mdfcheck.iloc[ind, 1:].to_dict()
    
    sessionID = expDetails['ophys_session_id']
    roi_table_file = glob.glob(os.path.join(roi_table_dir, '*' + str(sessionID) + '*'))
    if len(roi_table_file)==0:
        failed.append(sessionID)
        continue
    roi_table = pd.read_csv(roi_table_file[0])
    print(expDetails)
    
    metrics = ('meanResp', 'peakResp', 'shuff_meanResp', 'shuff_peakResp', 'stdPeakResp', 'shuff_stdPeakResp', 'responseMat')
    
    expDict = {a:[] for a in metrics}
    expDict.update(expDetails)
    peakRespPop = []
    meanRespPop = []
    endbuff = 30 #number of frames to take after the trial ends
    maxTrialDuration = int(np.ceil(stim_table.duration.max()))+endbuff
    for cell in np.arange(dff.shape[0]):
        if roi_table.iloc[cell]['valid']:
        #    plt.figure(cell)
            meanResp = np.full((2*len(patchSpeed)-1, 2*len(bckgndSpeed)-1, len(patchSize)), np.nan)
            peakResp = meanResp.copy()
    
            respMat = np.full((2*len(patchSpeed)-1, 2*len(bckgndSpeed)-1, len(patchSize), maxTrialDuration), np.nan)
            stdPeakResp = meanResp.copy()
            shuff_meanResp = meanResp.copy()
            shuff_peakResp = meanResp.copy()
            shuff_stdPeakResp = meanResp.copy()
    
            for trial in stim_table.trial_number.unique():
                tempdf = stim_table.loc[stim_table.trial_number==trial]
                psizeInd = np.where(patchSize==tempdf.patchSize.values[0])[0][0]
                pspeedInd = getSpeedInd(tempdf.patchSpeed.values[0], tempdf.patchDir.values[0])
                bspeedInd = getSpeedInd(tempdf.backgroundSpeed.values[0], tempdf.backgroundDir.values[0])
                
                means = []
                maxs = []
                trialResponses = []
                for ind, row in tempdf.iterrows():
                    trialResponse = dff[cell, int(row.start):int(row.end)+endbuff]
                    meantrial = trialResponse.mean()
                    maxtrial = trialResponse.max()
                    
                    trialResponses.append(trialResponse)
                    means.append(meantrial)
                    maxs.append(maxtrial)
                
                
                meanResp[pspeedInd, bspeedInd, psizeInd] = np.mean(means)
                peakResp[pspeedInd, bspeedInd, psizeInd] = np.mean(maxs)
                stdPeakResp[pspeedInd, bspeedInd, psizeInd] = np.std(maxs)
                
                trialResponses = makeArrayFromTrialResponseList(trialResponses)
                respMat[pspeedInd, bspeedInd, psizeInd, :trialResponses.shape[-1]] = np.nanmean(trialResponses, axis=0)
                #SHUFFLE TO CORRECT FOR TRIAL DURATION DIFFERENCES
                #TODO: JUST GIVE ONE SHUFFLE VALUE FOR EACH UNIQUE TRIAL DURATION
                #TODO: FIGURE OUT HOW TO NORMALIZE RESPONSE BY THIS... MAYBE RETURN STD INSTEAD AND DIVIDE?
    
                trialDur = int(row.end-row.start)
                shuffmean, shuffmax, shuffmaxstd = getShuffledTrialResponse(dff[cell], trialDur)
                shuff_meanResp[pspeedInd, bspeedInd, psizeInd] = shuffmean
                shuff_peakResp[pspeedInd, bspeedInd, psizeInd] = shuffmax
                shuff_stdPeakResp[pspeedInd, bspeedInd, psizeInd] = shuffmaxstd
                    
            
            for arr, name in zip([meanResp, peakResp, shuff_meanResp, shuff_peakResp, stdPeakResp, shuff_stdPeakResp], metrics):
                arr = fillRedundant(arr)
                expDict[name].append(arr)
            
            
            expDict['responseMat'].append(respMat)
                
    #        expDict['peakResp'].append(peakResp)
    #        expDict['meanResp'].append(meanResp)
    
            
                
        #    plt.imshow(np.nanmean(peakResp, axis=2))
            peakRespPop.append(np.nanmean(peakResp-shuff_peakResp, axis=2))
            meanRespPop.append(np.nanmean(meanResp-shuff_meanResp, axis=2))
    
    dictToHDF5(expDict, os.path.join(saveDir, str(expDetails['ophys_session_id']) + '_' + expDetails['targeted_structure']+'.hdf5'))
    fig,ax = plt.subplots(1,2)
    fig.suptitle(expDetails['targeted_structure'])
    for a, rp, title in zip(ax, [peakRespPop, meanRespPop], ['peak', 'mean']):
        a.imshow(np.nanmean(rp, axis=0), cmap='plasma')
        a.set_title(title)
    
                
                
              
            
#Load hdf5 files and aggregate by region
        
hdf5Dir = r"C:\Users\svc_ccg\Desktop\Data\motionscope\pilot"
hdf5Files = [hf for hf in os.listdir(hdf5Dir) if (os.path.isfile(os.path.join(hdf5Dir,hf))) and '60framebuff'  not in hf]

areas = ['VISam', 'VISl', 'VISpm', 'VISal']
regionDict = {a:{'respMat':[], 'expID': []} for a in areas}

for f in hdf5Files:
    expID = f.split('_')[0]
    area = f.split('_')[1].split('.')[0]
    area = area.split('6')[0]
    if area in areas:
        with h5py.File(os.path.join(hdf5Dir, f)) as h:
            numberCells = len(h['meanResp'])
            regionDict[area]['respMat'].extend(h['responseMat'][()])
            regionDict[area]['expID'].extend([expID]*numberCells)


def plotCellSummary(rmat, cellname = ''):
    gs = gridspec.GridSpec(3, 8)
    fig = plt.figure(cellname)
    
    bestsize = np.unravel_index(np.nanargmax(rmat), rmat.shape)[2]
    rbest = rmat[:, :, bestsize]
    
    ax0 = fig.add_subplot(gs[:, :3])
    ax0 = gridspec.GridSpecFromSubplotSpec(5, 5,
            subplot_spec=gs[:, :3], wspace=0.0, hspace=0.0)
    plotRespMat(rmat[:, :, 0], fig, ax0)
    
    ax1 = fig.add_subplot(gs[:, 3:6])
    plotRespMat(rmat[:, :, 1], ax1)
    
    ax2 = fig.add_subplot(gs[0, :2])
    ax2.plot(np.mean(rbest, axis=0))
    
    
def plotRespMat(rmat, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(5,5)

    if rmat.shape[2]==2:
        rmat = np.nanmean(rmat, 2)
    
    maxval = np.nanmax(rmat)
    minval = np.nanmin(rmat)
    for row in range(5):
        for col in range(5):
            a = fig.add_subplot(ax[row, col])
#            ax[row,col].plot(rmat[row,col], 'k')
#            ax[row,col].set_ylim([minval, maxval])
#            ax[row,col].set_xlim([0, rmat.shape[-1]])
            a.plot(rmat[row,col], 'k')
            a.set_ylim([minval, maxval])
            a.set_xlim([0, rmat.shape[-1]])

            if not (row==4 and col==0):
                a.set_axis_off()
            else:
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
# 


def getResponsive(rmat, stdthresh=5, hardthresh=0.3):
    baseline = rmat[2,2,0]
    baseline_std = np.nanstd(baseline)
    thresh = np.nanmean(baseline) + stdthresh*baseline_std
    if np.nanmax(rmat)> thresh and np.nanmax(rmat)>hardthresh:
        return True
    else:
        return False

def getPatchIndex(rmat):
    rmat = fillRedundant(rmat)
    patchmax = np.max(rmat[:, 2])
    bckmax = np.max(rmat[2,:])
    return (patchmax - bckmax)/(patchmax+bckmax)

def getOppIndex(rmat):
    q1 = rmat[:2,:2]
    q2 = rmat[:2, 3:]
    q3 = rmat[3:, :2]
    q4 = rmat[3:, 3:]
    
    opp = np.mean(q2 + q3)
    same = np.mean(q1 + q4)
    
    return (opp-same)/(opp+same)

def cumdist(v):
    v = np.sort(v)
    cd = np.array([np.sum(v<=val) for val in v])/float(len(v))
    
    return v, cd

pIs = []
oppIs = []
size_index = []
r_areas = []
r_mats = []
b_tuning = []
p_tuning = []
responsiveness = []
plot=False
for area in areas:
    rmats = np.array(regionDict[area]['respMat'])
    hasResp = [getResponsive(r) for r in rmats]
    print(area)
    print(str(np.sum(hasResp)) + ' responsive cells out of ' + str(len(hasResp)))
    responsiveness.append(np.sum(hasResp)/float(len(hasResp)))
    
    rmats = rmats[hasResp]
    r_areas.extend([area]*rmats.shape[0])
    
    rmats = np.array([fillRedundant(r) for r in rmats])
    pref_sizes = [np.unravel_index(np.nanargmax(r), r.shape)[2] for r in rmats]
    si = [(np.nanmax(r[:,:,0]) - np.nanmax(r[:,:,1]))/(np.nanmax(r[:,:,0]) + np.nanmax(r[:,:,1])) for r in rmats]
    size_index.append(si)
    
    r_meanoversize = np.array([np.nanmean(r,axis=2) for r in rmats])
    r_meanoversize_norm = np.array([r/np.nanmax(r) for r in r_meanoversize])
    r_bestsize = np.array([r[:,:,p] for r,p in zip(rmats, pref_sizes)])
    r_bestsize_norm = np.array([r/np.nanmax(r) for r in r_bestsize])
    r_maxovertime = np.nanmax(r_bestsize_norm, axis=3)
    r_mats.extend(r_maxovertime)
    
    pI = [getPatchIndex(r) for r in r_maxovertime]
    oppI = [getOppIndex(r) for r in r_maxovertime]
    b_tune = [np.max(r, axis=0) for r in r_maxovertime]
    p_tune = [np.max(r, axis=1) for r in r_maxovertime]
    print('median PI: ' + str(np.nanmedian(pI)) + ' std PI: ' + str(np.nanstd(pI)))
    
    pIs.extend(pI)
    oppIs.extend(oppI)
    b_tuning.extend(b_tune)
    p_tuning.extend(p_tune)
    regionMean = np.nanmean(r_maxovertime, axis=0)
    
    if plot:
        fig, ax = plt.subplots()
        fig.suptitle(area)
        im = ax.imshow(regionMean)
        plt.colorbar(im)
        
        fig, ax = plt.subplots()
        fig.suptitle(area)
    #    ax.hist(pref_sizes)
        ax.hist(si, np.arange(-1,1,0.2))

p_tuning = np.array(p_tuning)
b_tuning = np.array(b_tuning)
r_areas = np.array(r_areas)


##### plot responsiveness ############
fig, ax = plt.subplots()
ax.bar(np.arange(len(areas)), responsiveness, tick_label=areas)
ax.set_title('Percent cells responsive')


###### plot size index #########
for si, area in zip(size_index, areas):
    fig, ax = plt.subplots()
    ax.hist(si, np.arange(-1,1,0.2))
    ax.set_title(area + ' Size Index')

########Cluster rmats and plot area distributions#################
r_clust = np.array([r.flatten() for r in r_mats])
#cID, l = clust.kmeans(r_clust, 8)
#cID, l = clust.cluster(r_clust, 4, plot=True, nreps=100)
cIDh, l = clust.nestedPCAClust(r_clust, nSplit=3, varExplained=0.9)
cID = clust.getClustersFromHierarchy(cIDh)

for c in np.unique(cID):
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(8, 5)
    
    print(str(c) + ': ' + str(np.sum(cID==c)))
    mean_c = np.reshape(np.mean(r_clust[cID==c], axis=0), [5,5])
    
    axes[0].imshow(mean_c, cmap='plasma', interpolation='none')
    axes[0].set_title('cluster ' + str(c))
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xticklabels([-80, -20, 0, 20, 80])
    axes[0].set_yticks(np.arange(5))
    axes[0].set_yticklabels([-80, -20, 0, 20, 80])
    axes[0].set_xlabel('Background velocity (deg/s)')
    axes[0].set_ylabel('Patch velocity (deg/s)')
    
    percent_in_area = []
    for area in areas:
        a_in_c = r_areas[cID==c]
        p_in_c = np.sum(a_in_c == area)/float(np.sum(r_areas==area))
        percent_in_area.append(p_in_c)
        
    axes[1].bar(np.arange(len(areas)), percent_in_area, tick_label=areas)
    axes[1].set_ylabel('fraction cells in cluster')
    fig.tight_layout()

########    Background speed tuning     ##################### 
fig, ax = plt.subplots()
colors = ['b', 'm', 'g', 'k']
x = [-80, -20, 0, 20, 80]
for area,color in zip(areas, colors):    
    mean_btuning = np.mean(b_tuning[r_areas==area], axis=0)
    sem = np.std(b_tuning[r_areas==area], axis=0)/(np.sum(r_areas==area))**0.5
    ax.plot(x, mean_btuning, color)
    ax.fill_between(x, mean_btuning+sem, mean_btuning-sem, color=color, alpha=0.3) 
ax.set_title('Background Speed Tuning')
plt.legend(areas)

########    Patch speed tuning     ##################### 
fig, ax = plt.subplots()
colors = ['b', 'm', 'g', 'k']
x = [-80, -20, 0, 20, 80]
for area,color in zip(areas, colors):    
    mean_ptuning = np.mean(p_tuning[r_areas==area], axis=0)
    sem = np.std(p_tuning[r_areas==area], axis=0)/(np.sum(r_areas==area))**0.5
    ax.plot(x, mean_ptuning, color)
    ax.fill_between(x, mean_ptuning+sem, mean_ptuning-sem, color=color, alpha=0.3) 
ax.set_title('Patch Speed Tuning')
plt.legend(areas)
 

#########   Patch Index       ###########################   
pIs = np.array(pIs)
fig, ax = plt.subplots()
colors = ['b', 'm', 'g', 'k']
for area,color in zip(areas, colors):    
    pI = pIs[r_areas==area]
    xs, cd = cumdist(pI)
    ax.plot(xs, cd, color) 
ax.set_title('Patch Index')
plt.legend(areas)


########    Opponent motion index    ######################
oppIs = np.array(oppIs)
fig, ax = plt.subplots()
colors = ['b', 'm', 'g', 'k']
for area,color in zip(areas, colors):    
    oppI = oppIs[r_areas==area]
    xs, cd = cumdist(oppI)
    ax.plot(xs, cd, color) 
ax.set_title('Opponent Motion Index')
plt.legend(areas)



savePath = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\analysis"
for i in plt.get_fignums():
    f = plt.figure(i)
    a = f.get_axes()
    title = a[0].title.get_text()
    plt.savefig(os.path.join(savePath, title + '.png'), dpi=300)



