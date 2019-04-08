# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 15:15:36 2019

@author: saskiad
"""


from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
from zro.proxy import DeviceProxy
import time
import datetime



# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=1,#0
                warp=Warp.Spherical,
                )

# paths to our stimuli
ms_path = 		r"C:\Users\Public\Desktop\pythondev\cam2p_scripts\tests\motion.stim" # 16 minutes

# load our stimuli
ms = Stimulus.from_file(ms_path, window)

# RF mapping / flashes
ms_ds = [(0,3600)]


ms.set_display_sequence(ms_ds)


# kwargs
#params = {
#    'syncpulse': True,
#    'syncpulseport': 1,
#    'syncpulselines': [4, 7],  # frame, start/stop
#    'trigger_delay_sec': 0.0,
#    'bgcolor': (-1,-1,-1),
#    'eyetracker': False,
#    'eyetrackerip': "W7DT12722",
#    'eyetrackerport': 1000,
#    'syncsqr': True,
#    'syncsqrloc': (0,0), 
#    'syncsqrfreq': 60,
#    'syncsqrsize': (100,100),
#    'showmouse': True
#}

params = {
    'syncsqrloc': (510,360),#added by DM
    'syncsqrsize': (50,140),#added by DM
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1, 2], #[5, 6],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[ms],
               pre_blank_sec=2, #TODO: 60
               post_blank_sec=2,
               params=params,
               )

# add in foraging so we can track wheel, potentially give rewards, etc
f = Foraging(window=window,
            auto_update=False,
            params=params,
            nidaq_tasks={'digital_input': ss.di,
                         'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")

# run it
ss.run()
