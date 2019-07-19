# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:14:46 2019

@author: svc_ccg
"""
import os
import numpy as np
import scipy.io

matFileDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\checkerboard\matFiles"
fileList = [os.path.join(matFileDir, f) for f in os.listdir(matFileDir)]

npFileDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\checkerboard\npyFiles"
for f in fileList:
    n = scipy.io.loadmat(f)['fd']
    n = np.moveaxis(n, [0,1,2], [1,2,0])
    saveName = os.path.basename(f)[:-4] + '.npy'
    
    np.save(os.path.join(npFileDir, saveName), n)