"""
Dot stimuli
"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
import numpy as np
import os
import cPickle as pickle

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',#'GammaCorrect30',
                screen=0,
                warp=Warp.Spherical,
#                warp=Warp.Disabled
                )

stimFileDir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\dot stimfiles"
movie_path_list = [os.path.join(stimFileDir,f) for f in os.listdir(stimFileDir) if os.path.isfile(os.path.join(stimFileDir,f))]

# set display sequences
with open(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motion stimuli\ds_dict_two.pkl", 'rb') as f:
    ds_dict = pickle.load(f)
    

movielist = []
for i, path in enumerate(movie_path_list):
    ds = ds_dict[str(i)]
    movie = Stimulus.from_file(path, window)
    movie.set_display_sequence(ds)
    movielist.append(movie)
    


params = {
    #`'syncsqr': True,
    'syncsqrloc': (900,560),
    'syncsqrsize': (100,140),
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1, 2],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=movielist,
               pre_blank_sec=0,
               post_blank_sec=0,
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
