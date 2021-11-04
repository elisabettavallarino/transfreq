# -*- coding: utf-8 -*-
"""
Authomatic computation of the transition frequency for one subject with transfreq and Klimesch's method
=======================================================================================================

Short description
"""


import mne
from transfreq import compute_TF_klimesch, computeTF_auto
from transfreq.viz import plot_TF, plot_TF_klimesch, plot_clustering, plot_chs
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


subj = '001'
ses = '01'

#data_folder = '/media/mida/Volume/data_rest_task'
data_folder = '../'
# define file paths
f_name_rest = op.join(data_folder, 'sub-'+subj,'ses-'+ses,'eeg','sub-'+subj+'_ses-'+ses+'_task-eyeclose_raw.fif')
f_name_task = op.join(data_folder, 'sub-'+subj,'ses-'+ses,'eeg','sub-'+subj+'_ses-'+ses+'_task-memory_raw.fif')

# load resting state data
raw_rest = mne.io.read_raw_fif(f_name_rest)
raw_rest = raw_rest.pick_types(eeg = True, exclude=raw_rest.info['bads']+['TP9','TP10','FT9','FT10'])

# load data redorded during task execution
raw_task = mne.io.read_raw_fif(f_name_task)
raw_task = raw_task.pick_types(eeg = True, exclude=raw_task.info['bads']+['TP9','TP10','FT9','FT10'])

# list of good channels
tmp_idx = mne.pick_types(raw_rest.info, eeg=True, exclude='bads')
ch_names_rest = [raw_rest.ch_names[ch_idx] for ch_idx in tmp_idx]

tmp_idx = mne.pick_types(raw_task.info, eeg=True, exclude='bads')
ch_names_task = [raw_rest.ch_names[ch_idx] for ch_idx in tmp_idx]

# define time range 
# since we are using the multitapers method to compute the spectra we need to
# have rest and task data of the same length in order to obtain the same frequecy 
# resolution, which is needed for the computation of the TF with Klimesch's method
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

###########################################################################
# compute TFbox automatically with the default method
TFbox = computeTF_auto(psds_rest, freqs, ch_names_rest)

###########################################################################
# plot TF, cluster and channel position

fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
plot_TF(psds_rest, freqs, TFbox, ax = ax1)
ax2 = fig.add_subplot(gs[0, 1])
plot_clustering(TFbox,ax = ax2)
ax3 = fig.add_subplot(gs[1, :])
plot_chs(TFbox, ch_locs_rest, ax = ax3)
fig.tight_layout()


###########################################################################
# compute TFbox with Klimesch's method
TF_klimesch = compute_TF_klimesch(psds_rest, psds_task, freqs)
 
###########################################################################
# plot transition frequency with klimesch's metod and transfreq to compare them

fig, ax = plt.subplots(1,2,figsize=(8,4))
plot_TF_klimesch(psds_rest, psds_task, freqs, TF_klimesch, ax = ax[0])
plot_TF(psds_rest, freqs, TFbox, ax = ax[1])
fig.tight_layout()
###########################################################################