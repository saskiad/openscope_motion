# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:56:00 2020

@author: svc_ccg
"""

import os
import glob
import numpy as np
import h5py

CHECKERBOARD_DIR = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\checkerboard"
assert os.path.isdir(CHECKERBOARD_DIR)

paramfile = glob.glob(os.path.join(CHECKERBOARD_DIR, '*.hdf5'))[0]
params = h5py.File(paramfile)
checkerboard_movies = glob.glob(os.path.join(CHECKERBOARD_DIR, '*.npy'))

#order checkerboard_movies by their trial numbers
trial_number = [int(os.path.basename(c).split('_')[0][5:]) for c in checkerboard_movies]
trial_number_index = np.argsort(trial_number)
checkerboard_movies = [checkerboard_movies[t] for t in trial_number_index]

#Now make big concatenated array of all the trial conditions together
big_arr = []
for movie in checkerboard_movies:
    arr = np.load(movie)
    if arr.shape[1]!= 600 or arr.shape[2]!=960:
        raise ValueError('got unexpected shape: ', arr.shape)
    big_arr.append(arr)

big_movie = np.vstack(big_arr)
#if big_movie.shape != (10800, 600, 960):
#    raise ValueError('big movie has unexpected shape', big_movie.shape)

np.save(os.path.join(CHECKERBOARD_DIR, 'checkerboard_movie.npy'), big_movie)



##NOW BUILD THE FRAME LIST FOR THIS CONCATENATED MOVIE
def build_frame_list(n_conditions, condition_lengths, n_repeats, gray_frames):
    frame_list = []
    if n_conditions != len(condition_lengths):
        raise ValueError('number of conditions does not match number of condition lengths')
    for _ in range(n_repeats):
        conditions = range(n_conditions)
        random.shuffle(conditions)
        for condition in conditions:
            length = condition_lengths[condition]
            start = int(np.sum(condition_lengths[:condition]))
            frame_list += range(start, start+length)
            frame_list += [-1] * gray_frames
    
    return frame_list
    

frame_list = build_frame_list(len(checkerboard_movies), params['trialNumFrames'][()], 10, 60)

if len(frame_list) != 216000:
    raise ValueError('unexpected shape: ', len(frame_list))

with open(out_path, 'wb') as f:
    pickle.dump(frame_list, f)