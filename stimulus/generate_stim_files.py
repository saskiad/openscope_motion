# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:32:30 2019

@author: svc_ccg
"""

import os

arrayDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\translation"
fileList = os.listdir(arrayDir)
fileList = [f for f in fileList if os.path.isfile(os.path.join(arrayDir, f))]

stimSaveDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\translation\stimfiles"

for fname in fileList:
    saveName = os.path.join(stimSaveDir, fname[:-4]+'.stim')
    stimFile = open(saveName, 'w')
    stimFile.write('import os\n')
    stimFile.write('import shutil\n')
    stimFile.write('import numpy as np\n')
    stimFile.write('from camstim.core import ImageStimNumpyuByte, checkDirs\n')
    
    stimFile.write('moviesource = r\'' + os.path.join(arrayDir, fname) + '\'\n')
    
    stimFile.write('stimulus = MovieStim(movie_path=moviesource, window=window,frame_length=1.0/60.0,size=(1920, 1200),start_time=0.0,stop_time=None,flip_v=False,runs=1,)\n')
    stimFile.close()