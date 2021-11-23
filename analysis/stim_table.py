# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:33:28 2019

@author: danielm
additions from saskiad Jun 7 2020
adapt to Motion stimuli by saskiad Nov 17 2021
"""
import os
import warnings

import numpy as np
import pandas as pd
import h5py


# Generic interface for creating stim tables. PREFERRED.
def create_stim_tables(
    exptpath,
    stimulus_names = ['drifting_gratings_grid', 'natural_movie_full', 
                      'natural_movie_one','checkerboard', 'dot'],
    verbose = True):
    """Create a stim table from data located in folder exptpath.

    Tries to extract a stim_table for each stim type in stimulus_names and
    continues if KeyErrors are produced.

    Inputs:
        exptpath (str)
            -- Path to directory in which to look for experiment-related files.
        stimulus_names (list of strs)
            -- Types of stimuli to try extracting.
        verbose (bool, default True)
            -- Print information about progress.

    Returns:
        Dict of DataFrames with information about start and end times of each
        stimulus presented in a given experiment.

    """
    data = load_stim(exptpath)
    twop_frames = load_alignment(exptpath)

    stim_table_funcs = {
        'drifting_gratings_grid': DGgrid_table,
        'natural_movie_full': NMfull_table,
        'natural_movie_one': NMone_table,
        'checkerboard': Checkerboard_table,
        'dot': Dot_table
    }
    stim_table = {}
    for stim_name in stimulus_names:
        try:
            stim_table[stim_name] = stim_table_funcs[stim_name](
                data, twop_frames
            )
        except KeyError:
            if verbose:
                print(
                    'Could not locate stimulus type {} in {}'.format(
                        stim_name, exptpath
                    )
                )
            continue

    return stim_table


def DGgrid_table(data, twop_frames, verbose = True):

    DG_idx = get_stimulus_index(data, 'drifting_gratings_grid.stim')

    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, DG_idx
    )

    if verbose:
        print 'Found {} of {} expected sweeps.'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    for attribute in ['TF', 'SF', 'Contrast', 'Ori', 'PosX', 'PosY']:
        stim_table[attribute] = get_attribute_by_sweep(
            data, DG_idx, attribute
        )[:len(stim_table)]

    return stim_table

def NMfull_table(data, twop_frames, verbose = True):
    
    NM_idx = get_stimulus_index(data, 'natural_movie_full.stim')
    
    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, NM_idx
    )
    if verbose:
        print 'Found {} of {} expected sweeps.'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    stim_table['Frame'] = np.array(
        data['stimuli'][NM_idx]['sweep_order'][:len(stim_table)]
    )

    return stim_table

def NMone_table(data, twop_frames, verbose = True):
    
    NM1_idx = get_stimulus_index(data, 'natural_movie_one.stim')
    
    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, NM1_idx
    )
    if verbose:
        print 'Found {} of {} expected sweeps.'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    stim_table['Frame'] = np.array(
        data['stimuli'][NM1_idx]['sweep_order'][:len(stim_table)]
    )

    return stim_table
    
def Checkerboard_table(data, twop_frames, verbose=True):
    CB_idx = get_stimulus_index_CBD(data, 'checkerboard_movie.npy')
    
    ###SET CHECKERBOARD STIM PATH HERE###
    f = h5py.File(r'/Users/saskiad/Code/openscope_motion/stimulus/final stimuli/checkerboard_params.hdf5', 'r')
    
    trials_keys = []
    for a in f.keys():
        if a.startswith("trial"):
            trials_keys.append(a)
    cb_table = pd.DataFrame()
    for a in trials_keys:
        cb_table[a[5:]] = f[a][()]      #remove "trial" from column names
    f.close()
    cb_table.rename(columns={'StartFrame':'Frame'}, inplace=True)
    cb_table['CB_index'] = range(len(cb_table))
    
    frame_list  = np.array(data['stimuli'][CB_idx]['frame_list'])
    starts = findlevels(frame_list, -0.5, direction='up')
    starts = np.insert(starts, 0, 0)             #first trial starts right at 0 

    
    stim_table_start = pd.DataFrame(
        np.column_stack((
            twop_frames[starts],
            frame_list[starts]
        )),
        columns=('Start', 'Frame')
    )
        
    stim_table = pd.merge(stim_table_start, cb_table, on='Frame')
    stim_table.sort_values(by='Start', inplace=True)

    return stim_table

def Dot_table(data, twop_frames, verbose=True):
    Dot_idx = get_stimulus_index_CBD(data, 'dot_stim_movie_int.npy')
    
    ###SET DOT STIM PATH HERE###
    dot_stim_path = r'/Users/saskiad/Code/openscope_motion/stimulus/final stimuli/dot_stim_table.xlsx'
    ds_table = pd.read_excel(dot_stim_path)
    
    frame_list  = np.array(data['stimuli'][Dot_idx]['frame_list'])
    starts = findlevels(frame_list, -0.5, direction='up')
    
    stim_table_start = pd.DataFrame(
        np.column_stack((
            twop_frames[starts],
            frame_list[starts]
        )),
        columns=('Start', 'Frame')
    )
    
    stim_table = pd.merge(stim_table_start, ds_table, on='Frame')
    stim_table.sort_values(by='Start', inplace=True)
    

    return stim_table

def get_stimulus_index(data, stim_name):
    """Return the index of stimulus in data.

    Returns the position of the first occurrence of stim_name in data. Raises a
    KeyError if a stimulus with a name containing stim_name is not found.

    Inputs:
        data (dict-like)
            -- Object in which to search for a named stimulus.
        stim_name (str)

    Returns:
        Index of stimulus stim_name in data.

    """
    for i_stim, stim_data in enumerate(data['stimuli']):
        if stim_name in stim_data['stim_path']:
            return i_stim

    raise KeyError('Stimulus with stim_name={} not found!'.format(stim_name))

def get_stimulus_index_CBD(data, movie_name):
    """Return the index of stimulus in data.

    Returns the position of the first occurrence of stim_name in data. Raises a
    KeyError if a stimulus with a name containing stim_name is not found. This is for stimuli that 
    (for some strange reason) don't have a stim name(!?!?)

    Inputs:
        data (dict-like)
            -- Object in which to search for a named stimulus.
        stim_name (str)

    Returns:
        Index of stimulus stim_name in data.

    """
    for i_stim, stim_data in enumerate(data['stimuli']):
        if movie_name in stim_data['movie_path']:
            return i_stim

    raise KeyError('Stimulus with stim_name={} not found!'.format(movie_name))



def get_display_sequence(data, stimulus_idx):

    display_sequence = np.array(
        data['stimuli'][stimulus_idx]['display_sequence']
    )
    pre_blank_sec = int(data['pre_blank_sec'])
    display_sequence += pre_blank_sec
    display_sequence *= int(data['fps'])  # in stimulus frames

    return display_sequence


def get_sweep_frames(data, stimulus_idx):

    sweep_frames = data['stimuli'][stimulus_idx]['sweep_frames']
    timing_table = pd.DataFrame(
        np.array(sweep_frames).astype(np.int),
        columns=('start', 'end')
    )
    timing_table['dif'] = timing_table['end']-timing_table['start']

    display_sequence = get_display_sequence(data, stimulus_idx)

    timing_table.start += display_sequence[0, 0]
    for seg in range(len(display_sequence)-1):
        for index, row in timing_table.iterrows():
            if row.start >= display_sequence[seg, 1]:
                timing_table.start[index] = (
                    timing_table.start[index]
                    - display_sequence[seg, 1]
                    + display_sequence[seg+1, 0]
                )
    timing_table.end = timing_table.start+timing_table.dif
    expected_sweeps = len(timing_table)
    timing_table = timing_table[timing_table.end <= display_sequence[-1, 1]]
    timing_table = timing_table[timing_table.start <= display_sequence[-1, 1]]
    actual_sweeps = len(timing_table)

    return timing_table, actual_sweeps, expected_sweeps


def get_attribute_by_sweep(data, stimulus_idx, attribute):

    attribute_idx = get_attribute_idx(data, stimulus_idx, attribute)

    sweep_order = data['stimuli'][stimulus_idx]['sweep_order']
    sweep_table = data['stimuli'][stimulus_idx]['sweep_table']

    num_sweeps = len(sweep_order)

    attribute_by_sweep = np.zeros((num_sweeps,))
    attribute_by_sweep[:] = np.NaN

    unique_conditions = np.unique(sweep_order)
    for i_condition, condition in enumerate(unique_conditions):
        sweeps_with_condition = np.argwhere(sweep_order == condition)[:, 0]

        if condition >= 0:  # blank sweep is -1
            try:
                attribute_by_sweep[sweeps_with_condition] = sweep_table[condition][attribute_idx]
            except:
                attribute_by_sweep[sweeps_with_condition] = sweep_table[condition][attribute_idx][0] 

    return attribute_by_sweep


def get_attribute_idx(data, stimulus_idx, attribute):
    """Return the index of attribute in data for the given stimulus.

    Returns the position of the first occurrence of attribute. Raises a
    KeyError if not found.
    """
    attribute_names = data['stimuli'][stimulus_idx]['dimnames']
    for attribute_idx, attribute_str in enumerate(attribute_names):
        if attribute_str == attribute:
            return attribute_idx

    raise KeyError('Attribute {} for stimulus_ids {} not found!'.format(
            attribute, stimulus_idx
        ))


def load_stim(exptpath, verbose = True):
    """Load stim.pkl file into a DataFrame.

    Inputs:
        exptpath (str)
            -- Directory in which to search for files with _stim.pkl suffix.
        verbose (bool)
            -- Print filename (if found).

    Returns:
        DataFrame with contents of stim pkl.

    """
    # Look for a file with the suffix '_stim.pkl'
    pklpath = None
    for f in os.listdir(exptpath):
        if f.endswith('_stim.pkl'):
            pklpath = os.path.join(exptpath, f)
            if verbose:
                print "Pkl file:", f

    if pklpath is None:
        raise IOError(
            'No files with the suffix _stim.pkl were found in {}'.format(
                exptpath
            )
        )

    return pd.read_pickle(pklpath)

def load_alignment(exptpath):
    for f in os.listdir(exptpath):
        if f.startswith('ophys_experiment'):
            ophys_path = os.path.join(exptpath, f)
    for f in os.listdir(ophys_path):
        if f.endswith('time_synchronization.h5'):
            temporal_alignment_file = os.path.join(ophys_path, f)           
    f = h5py.File(temporal_alignment_file, 'r')
    twop_frames = f['stimulus_alignment'].value
    f.close()
    return twop_frames


###FOR TESTING
#def load_alignment(exptpath):
#    for f in os.listdir(exptpath):
#        if f.endswith('time_synchronization.h5'):
#            temporal_alignment_file = os.path.join(exptpath, f)           
#    f = h5py.File(temporal_alignment_file, 'r')
#    twop_frames = f['stimulus_alignment'].value
#    f.close()
#    return twop_frames

def findlevels(inwave, threshold, window=0, direction='both'):
    '''Threshold crossing function'''
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

def print_summary(stim_table):
    """Print summary of generated stim_table.

    Print column names, number of 'unique' conditions per column (treating
    nans as equal), and average number of samples per condition.
    """
    print(
        '{:<20}{:>15}{:>15}\n'.format('Colname', 'No. conditions', 'Mean N/cond')
    )
    for colname in stim_table.columns:
        conditions, occurrences = np.unique(
            np.nan_to_num(stim_table[colname]), return_counts = True
        )
        print(
            '{:<20}{:>15}{:>15.1f}'.format(
                colname, len(conditions), np.mean(occurrences)
            )
        )


if __name__ == '__main__':
#    exptpath = r'\\allen\programs\braintv\production\neuralcoding\prod55\specimen_859061987\ophys_session_882666374\\'
    exptpath = r'//Users/saskiad/Documents/Data/Openscope_Motion/session_3'
    stim_table = create_stim_tables(exptpath)

