# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:24:20 2017

@author: saskiad
"""

import pandas as pd
import numpy as np
import os
import h5py
import cPickle as pickle
from sync import Dataset

def get_files(exptpath):
    for f in os.listdir(exptpath):
        if f.endswith('.pkl'):
            logpath = os.path.join(exptpath, f)
            print "Stimulus log:", f
        if f.endswith('sync.h5'):
            syncpath = os.path.join(exptpath, f)
            print "Sync file:", f
    return logpath, syncpath

def findlevels(inwave, threshold, window=0, direction='both'):
    temp = inwave - threshold
    if (direction.find("up")+1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0)>0)[0]
    elif (direction.find("down")+1):
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0)<0)[0]
    else:
        crossings = np.nonzero(np.ediff1d(np.sign(temp), to_begin=0))[0]
    
    zdiff = np.ediff1d(crossings)
    while(any(zdiff<window)):
        crossings = np.delete(crossings, (np.where(zdiff<window)[0][0]+1))
        zdiff = np.ediff1d(crossings)
    return crossings

def get_sync(syncpath):
#    head,tail = os.path.split(syncpath)    
    d = Dataset(syncpath)
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #microscope acquisition frames    
#    ophys_start = d.get_rising_edges('2p_acquiring')/sample_freq
    twop_vsync_fall = d.get_falling_edges('2p_vsync')/sample_freq
#    twop_vsync_fall = twop_vsync_fall[np.where(twop_vsync_fall > ophys_start)[0]]
    twop_diff = np.ediff1d(twop_vsync_fall)
    acquisition_rate = 1/np.mean(twop_diff)
    
#    stimulus frames
    stim_vsync_fall = d.get_falling_edges('stim_vsync')[1:]/sample_freq          #eliminating the DAQ pulse    
    stim_vsync_diff = np.ediff1d(stim_vsync_fall)
    dropped_frames = np.where(stim_vsync_diff>0.033)[0]
    dropped_frames = stim_vsync_fall[dropped_frames]
    long_frames = np.where(stim_vsync_diff>0.1)[0]
    long_frames = stim_vsync_fall[long_frames]
    print "Dropped frames: " + str(len(dropped_frames)) + " at " + str(dropped_frames)
    print "Long frames(>0.1 s): " + str(len(long_frames)) + " at " + str(long_frames) 
    
    try:
        #photodiode transitions
        photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq
    
        #test and correct for photodiode transition errors
        ptd_rise_diff = np.ediff1d(photodiode_rise)
        short = np.where(np.logical_and(ptd_rise_diff>0.1, ptd_rise_diff<0.3))[0]
        medium = np.where(np.logical_and(ptd_rise_diff>0.5, ptd_rise_diff<1.5))[0]
        for i in medium:
            if set(range(i-2,i)) <= set(short):
                ptd_start = i+1
#            elif set(range(i+1,i+3)) <= set(short):
#                ptd_end = i
        ptd_end = np.where(photodiode_rise>stim_vsync_fall.max())[0][0] - 1
    
        if ptd_start > 3:
            print "Photodiode events before stimulus start.  Deleted."
        
        ptd_errors = []
        while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
            error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
            print "Photodiode error detected. Number of frames:", len(error_frames)
            photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
            ptd_errors.append(photodiode_rise[error_frames[-1]])
            ptd_end-=1
            ptd_rise_diff = np.ediff1d(photodiode_rise)
            
        #calculate monitor delay
        first_pulse = ptd_start
        delay_rise = np.empty((ptd_end - ptd_start,1))    
        for i in range(ptd_end+1-ptd_start-1):     
            delay_rise[i] = photodiode_rise[i+first_pulse] - stim_vsync_fall[(i*120)+60]
        
        delay = np.mean(delay_rise[:-1])  
        delay_std = np.std(delay_rise[:-1])
        print "Delay:", round(delay, 4)
        print "Delay std:", round(delay_std, 4)
        if delay_std>0.005:
            print "Sync error needs to be fixed"
#            delay = 0.005   #this appears to be the delay on research rig
            delay = 0.0351
            print "Using assumed delay:", round(delay,4)
    except Exception as e:
        print e
        print "Process without photodiode signal"
        delay = 0.0351
        print "Assumed delay:", round(delay, 4)
            
    #adjust stimulus time with monitor delay
    stim_time = stim_vsync_fall + delay
    
    #convert stimulus frames into twop frames
    twop_frame = np.empty((len(stim_time),1))
    for i in range(len(stim_time)):
        crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        try:
            twop_frame[i] = crossings[0][0]
        except:
            twop_frame[i]=np.NaN
            if i > 100:
                print i
                print "Acquisition ends before stimulus"            
    return twop_frame, acquisition_rate
    

'''create sync table'''
def get_sync_table(logpath, twop_frames):        
    print "Loading stimulus log from:", logpath
    f = open(logpath, 'rb')
    data = pickle.load(f)
    f.close()
    if len(data['stimuli'])>1:
        num_trials = len(data['stimuli'])
        for trial in range(num_trials):
            temp = pd.DataFrame(columns=('start','end','trial'))
            starts = findlevels(data['stimuli'][trial]['frame_list'], -0.5, direction='up')
            ends = findlevels(data['stimuli'][trial]['frame_list'], -0.5, direction='down')
            ends = np.append(ends, [len(data['stimuli'][trial]['frame_list'])-1])
#            print len(ends), len(starts)
            if len(ends)>len(starts):
#                print len(ends), len(starts)
                starts = np.insert(starts, 0, 0)
            temp['start'] = starts
            temp['end'] = ends
            temp['trial'] = data['stimuli'][trial]['stim_path'].split('\\')[-1]
            if trial==0:
                stim_table = temp.copy()
            else:
                stim_table = stim_table.append(temp)
        stim_table.reset_index(inplace=True)
        sync_table = pd.DataFrame(np.column_stack((twop_frames[stim_table['start']], twop_frames[stim_table['end']])), columns=('start', 'end'))
        sync_table['trial'] = stim_table.trial
        sync_table['trial_number'] = np.NaN
        for index, row in sync_table.iterrows():
            sync_table['trial_number'].loc[index] = int(row.trial.split('_')[0][5:])
        
        checkerboard = pd.read_csv(r'/Users/saskiad/openscope_motion/analysis/checkerboardTrialParams.csv')
        sync_table = pd.merge(sync_table, checkerboard, on='trial_number')
    
    else:   
        stim_table = pd.DataFrame(columns=('start','end','start_frame'))
    
        frame_list  = np.array(data['stimuli'][0]['frame_list'])
        starts = findlevels(frame_list, -0.5, direction='up')
        starts = np.insert(starts, 0, 0)
        ends = starts+60
        # ends = findlevels(frame_list, 0, direction='down')
        stim_table['start'] = starts
        stim_table['end'] = ends
        stim_table['start_frame'] = frame_list[starts]
                 
        sync_table = pd.DataFrame(np.column_stack((twop_frames[stim_table['start']], twop_frames[stim_table['end']])), columns=('start', 'end'))
        sync_table['start_frame'] = stim_table.start_frame
        
        dots = pd.read_csv(r'/Users/saskiad/openscope_motion/analysis/dotstim1_param_table.csv')
        dots.rename(columns={'movieFrameStart':'start_frame'}, inplace=True)
        sync_table = pd.merge(sync_table, dots, on='start_frame')
    sync_table['duration'] = sync_table.end - sync_table.start           
    return sync_table



#exptpath = r'/Volumes/braintv/workgroups/nc-ophys/ImageData/Saskia/20170531_307744/NaturalScenesUP'            
#logpath, syncpath = get_files(exptpath)
for f in os.listdir(r'/Volumes/My Passport/Openscope Motion'):
    if f.endswith('.xlsx'):
        pass
    else:
        print f
        expt_path = os.path.join(r'/Volumes/My Passport/Openscope Motion', f)
        session_id = f.split('_')[-1]
        for f2 in os.listdir(expt_path):
            if f2.endswith('_stim.pkl'):
                logpath = os.path.join(expt_path, f2)
            if f2.endswith('_sync.h5'):
                syncpath = os.path.join(expt_path, f2)
        try:
            print logpath
            print syncpath    
            twop_frames, acquisition_rate = get_sync(syncpath)
            sync_table = get_sync_table(logpath, twop_frames)
            sync_table.to_csv(os.path.join(r'/Users/saskiad/Dropbox/Openscope Motion/stim_tables', session_id+'_stim_table.csv'))
            del(logpath)
            del(syncpath)
        except:
            pass
    
    

#logpath = r'/Volumes/My Passport/Openscope Motion/ophys_session_927781186/927781186_469326_20190820_stim.pkl'
#syncpath = r'/Volumes/My Passport/Openscope Motion/ophys_session_927781186/927781186_469326_20190820_sync.h5'
#
#twop_frames, acquisition_rate = get_sync(syncpath)           
#sync_table = get_sync_table(logpath, twop_frames)


