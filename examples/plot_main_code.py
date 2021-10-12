# -*- coding: utf-8 -*-
"""
Title of the example
===========================

Short description
"""


import mne
from transfreq import functions
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_render import RenderingImShow


subj = '008'
ses = '01'

data_folder = '/media/mida/Volume/data_rest_task'
# define file paths
f_name_rest = op.join(data_folder, 'sub-'+subj,'ses-'+ses,'eeg','sub-'+subj+'_ses-'+ses+'_task-eyeclose_raw.fif')
f_name_task = op.join(data_folder, 'sub-'+subj,'ses-'+ses,'eeg','sub-'+subj+'_ses-'+ses+'_task-memory_raw.fif')

# load data
raw_rest = mne.io.read_raw_fif(f_name_rest)
raw_rest = raw_rest.pick_types(eeg = True, exclude=raw_rest.info['bads']+['TP9','TP10','FT9','FT10'])

raw_task = mne.io.read_raw_fif(f_name_task)
raw_task = raw_task.pick_types(eeg = True, exclude=raw_task.info['bads']+['TP9','TP10','FT9','FT10'])

# define good channels
tmp_idx = mne.pick_types(raw_rest.info, eeg=True, exclude='bads')
ch_names_rest = [raw_rest.ch_names[ch_idx] for ch_idx in tmp_idx]

tmp_idx = mne.pick_types(raw_task.info, eeg=True, exclude='bads')
ch_names_task = [raw_rest.ch_names[ch_idx] for ch_idx in tmp_idx]

# define time range 
tmin=0
tmax=min(raw_rest.times[-1],raw_task.times[-1])

# compute psds
n_fft = 512*2
bandwidth = 1
fmin = 2
fmax = 30

sfreq = raw_rest.info['sfreq']
n_per_seg = int(sfreq*2)

psds_rest, freqs = mne.time_frequency.psd_multitaper(raw_rest, fmin=fmin, fmax=fmax,tmin=tmin, tmax=tmax,bandwidth=bandwidth, adaptive=False,
                              low_bias=True, normalization='length', picks=ch_names_rest, proj=False, n_jobs=1, verbose=None)

psds_task, freqs = mne.time_frequency.psd_multitaper(raw_task, fmin=fmin, fmax=fmax,tmin=tmin, tmax=tmax,bandwidth=bandwidth, adaptive=False,
                              low_bias=True, normalization='length', picks=ch_names_task, proj=False, n_jobs=1, verbose=None)



# define channel positions
ch_locs_rest = np.zeros((len(ch_names_rest),3))

for ii in range(len(ch_names_rest)): 
    ch_locs_rest[ii,:] = raw_rest.info['chs'][ii]['loc'][:3]


# compute TFbox automatically
TFbox = functions.computeTF_auto(psds_rest, freqs, ch_names_rest, alpha_range = None, theta_range = None, method = 1, iterative=True)

###########################################################################
# plot transition frequency
fig2, ax = plt.subplots(2,2,figsize=(10,5))

functions.plot_TF(psds_rest, freqs, TFbox, showfig = True, ax = ax[0,0]);

###########################################################################
# plot clustering
functions.plot_clustering(TFbox, method = None, ax=ax[0,1]);   
functions.plot_chs(TFbox, ch_locs_rest, method = None, showfig=False, ax=ax[1,0]);
TF_klimesch = functions.compute_TF_klimesch(psds_rest, psds_task, freqs)
functions.plot_TF_klimesch(psds_rest,psds_task, freqs, TF_klimesch, showfig = False, ax = ax[1,1]);


###########################################################################
# plot channels on head surface

fig2.tight_layout()
fig2.show()








