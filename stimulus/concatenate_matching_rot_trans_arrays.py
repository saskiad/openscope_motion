# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:17:16 2020

@author: svc_ccg
"""
import os, glob
import numpy as np

rotDir = r"\\allen\programs\braintv\workgroups\nc-ophys\Saskia\openscope_motion\npyfiles_dec"
transDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\translation"

t_windowed = glob.glob(os.path.join(transDir, 'linear_windowed*'))
r_windowed = glob.glob(os.path.join(rotDir, 'linear_windowed_rotation_dir*'))

t_directions = []
t_speeds = []
for t in t_windowed:
    tb = os.path.basename(t)
    tb_split = tb.split('_')
    direction = tb_split[2]
    direction = 0 if direction == 'forward' else 180
    speed = int(tb_split[12])
    
    t_directions.append(direction)
    t_speeds.append(speed)
t_directions = np.array(t_directions)
t_speeds = np.array(t_speeds)


r_directions = []
r_speeds = []
for r in r_windowed:
    rb = os.path.basename(r)
    rb_split = rb.split('_')
    direction = int(rb_split[4])
    speed = int(rb_split[6])
    
    r_directions.append(direction)
    r_speeds.append(speed)
r_directions = np.array(r_directions)
r_speeds = np.array(r_speeds)

def find_nearest(array, val):
    diff = np.abs(array - val)
    return array[np.argmin(diff)]

r_matching_t = []
for t,s,d in zip(t_windowed, t_speeds, t_directions):
    matched_dir = d
    matched_speed = find_nearest(r_speeds, s)
    
    match = np.where((r_directions==matched_dir) & (r_speeds==matched_speed))[0][0]
    r_matching_t.append(r_windowed[match])
    
    
saveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\rot_trans_concatenated"
for t, r, s, d in zip(t_windowed, r_matching_t, t_speeds, t_directions):
    t_array = np.load(t)[:,:, 280:680]
    r_array = np.load(r)[:,:, 280:680]
    
    s = find_nearest(r_speeds,s)
    concat = np.concatenate((t_array, r_array), axis=2)
    np.save(os.path.join(saveDir, 'dir_' + str(d) + '_speed_' + str(s) + '_sidebyside.npy'), concat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    