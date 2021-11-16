# -*- coding: utf-8 -*-
"""
Manual computation of the transition frequency for one subject with transfreq 
=============================================================================

Short description
"""

import mne
from transfreq import compute_transfreq_manual
from transfreq.viz import (plot_psds, plot_coefficients, plot_channels,
                           plot_clusters, plot_transfreq)
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


subj = '001'
ses = '01'

#data_folder = '../'
data_folder = '/home/mida/Desktop/Betta/tmp_data_pathTF/preproc_files'
# define file paths
f_name = op.join(data_folder, 'sub'+subj,'ses'+ses,
                 'sub-'+subj+'_ses-'+ses+'_task-eyeclose_raw.fif')

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

psds, freqs = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax,
                                                bandwidth=bandwidth)

# define channel positions
ch_locs = np.zeros((psds.shape[0],3))

for ii in range(psds.shape[0]): 
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
# alpha and theta coefficiant on the plane, and the power specrtum averaged 
# over all channels
    
fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_coefficients(psds, freqs, ch_names=ch_names, alpha_range=alpha_range,
                  theta_range = theta_range,mode='1d',ax=ax1, order='sorted')
ax2 = fig.add_subplot(gs[1, 0])
plot_coefficients(psds, freqs, ch_names=ch_names, alpha_range=alpha_range,
                  theta_range = theta_range,mode='2d',ax=ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_psds(psds, freqs, average = True, ax = ax3)
fig.tight_layout()


###########################################################################
# chose channels from the figure to namually define the clusters and compute 
# the transition frequency

theta_chs_1d = ['C5','C3','T8','C1','C6']
alpha_chs_1d = ['P4','P2','CP2']

theta_chs_2d = ['C5','C3','T8']
alpha_chs_2d = ['P4','P2','Pz','CP2','POz','Fp2','PO4','P1']  



tfbox_1d = compute_transfreq_manual(psds, freqs, theta_chs_1d, alpha_chs_1d, 
                                    ch_names=ch_names, theta_range=theta_range,
                                    alpha_range=alpha_range, method='my_method_1d')
tfbox_2d = compute_transfreq_manual(psds, freqs, theta_chs_2d, alpha_chs_2d, 
                                    ch_names=ch_names, theta_range=theta_range, 
                                    alpha_range=alpha_range, method='my_method_2d')

###########################################################################
# plots of clustering, channel positions and tfs with my_method_1d

fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_channels(tfbox_1d, ch_locs, mode='1d', ax = ax1)
ax2 = fig.add_subplot(gs[1, 0])
plot_clusters(tfbox_1d,mode = '1d', order = 'sorted',ax = ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_transfreq(psds, freqs, tfbox_1d, ax = ax3)
fig.tight_layout()


###########################################################################
# plots of clustering, channel positions and tfs with my_method_1d

fig = plt.figure()
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
plot_channels(tfbox_2d, ch_locs, mode='2d', ax = ax1)
ax2 = fig.add_subplot(gs[1, 0])
plot_clusters(tfbox_2d,mode = '2d', ax = ax2)
ax3 = fig.add_subplot(gs[1, 1])
plot_transfreq(psds, freqs, tfbox_2d, ax = ax3)
fig.tight_layout()
