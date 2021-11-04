# -*- coding: utf-8 -*-
"""
Manula computation of the transition frequency for one subject with transfreq 
=============================================================================

Short description
"""

import mne
from transfreq import create_cluster, computeTF_manual
from transfreq.viz import plot_psds, plot_coefficients, plot_chs, plot_clustering, plot_TF
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


subj = '001'
ses = '01'

data_folder = '../'
#data_folder = '/media/mida/Volume/data_rest_task'
# define file paths
f_name = op.join(data_folder, 'sub-'+subj,'ses-'+ses,'eeg','sub-'+subj+'_ses-'+ses+'_task-eyeclose_raw.fif')

# load resting state data
raw = mne.io.read_raw_fif(f_name)

# list of good channels
tmp_idx = mne.pick_types(raw.info, eeg=True, exclude='bads')
ch_names = [raw.ch_names[ch_idx] for ch_idx in tmp_idx]


# compute psds
n_fft = 512*2
bandwidth = 1
fmin = 2
fmax = 30

sfreq = raw.info['sfreq']
n_per_seg = int(sfreq*2)

psds, freqs = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax,bandwidth=bandwidth, adaptive=False,
                              low_bias=True, normalization='length', picks=ch_names, proj=False, n_jobs=1,
                              verbose=None)

# define channel positions
ch_locs = np.zeros((len(ch_names),3))

for ii in range(len(ch_names)): 
    ch_locs[ii,:] = raw.info['chs'][ii]['loc'][:3]



###########################################################################
# plot power spectrum to visually chose theta and alpha ranges
plot_psds(psds, freqs, average = True)

###########################################################################
# set theta and alpha ranges
alpha_range = [8,9.5]
theta_range = [6.5,7]

###########################################################################
# plot coefficients both 1d (ratio between alpha and theta coefficients) and 2d 
# alpha and theta coefficiant on the plane, and the power specrtum averaged over all channels
    
fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_coefficients(psds, freqs, ch_names, alpha_range=alpha_range,theta_range = theta_range,mode='1d',ax=ax1, order='sorted')
ax2 = fig.add_subplot(gs[1, 0])
plot_coefficients(psds, freqs, ch_names, alpha_range=alpha_range,theta_range = theta_range,mode='2d',ax=ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_psds(psds, freqs, average = True, ax = ax3)
fig.tight_layout()


###########################################################################
# chose channels from the figure to namually create a cluster

theta_chs_1d = ['C5','C3','T8','C1','C6']
alpha_chs_1d = ['P4','P2','CP2']

theta_chs_2d = ['C5','C3','T8']
alpha_chs_2d = ['P4','P2','Pz','CP2','POz','Fp2','PO4','P1']  


TFbox_1d = create_cluster(psds, freqs, ch_names, theta_chs_1d, alpha_chs_1d, theta_range, alpha_range, method='my_method_1d')
TFbox_2d = create_cluster(psds, freqs, ch_names, theta_chs_2d, alpha_chs_2d, theta_range, alpha_range, method='my_method_2d')

TFbox_1d = computeTF_manual(psds, freqs, TFbox_1d)
TFbox_2d = computeTF_manual(psds, freqs, TFbox_2d)


###########################################################################
# plots of clustering, channel positions and TFs with my_method_1d

fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_chs(TFbox_1d, ch_locs, mode='1d', ax = ax1)
ax2 = fig.add_subplot(gs[1, 0])
plot_clustering(TFbox_1d,mode = '1d', order = 'sorted',ax = ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_TF(psds, freqs, TFbox_1d, ax = ax3)
fig.tight_layout()


###########################################################################
# plots of clustering, channel positions and TFs with my_method_1d

fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_chs(TFbox_2d, ch_locs, mode='2d', ax = ax1)
ax2 = fig.add_subplot(gs[1, 0])
plot_clustering(TFbox_2d,mode = '2d', ax = ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_TF(psds, freqs, TFbox_2d, ax = ax3)
fig.tight_layout()
