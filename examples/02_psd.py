#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 09:41:09 2021

@author: mida
"""

import mne
import os.path as op
import numpy as np
import pickle
import os.path
from os import makedirs


#################################################################################
# Define data paths

# local folder where to store the output of the analysis
data_path = '/home/mida/Desktop/Betta/tmp_data_pathTF'

#################################################################################
# All the subject in the dataset
subjs = ['001','002','003','004','005','006','007','008','009','010','011',
         '012','013','014','015','016','017','018','019','020','021','022',
         '023','024','025','026','027','028','029','030','031','032','033',
         '034','035','036','037','038','039','040','041','042','043','044',
         '045','046','047','048','049','050']
sess = ['01','02']

# Define some parameters for the psd computation
bandwidth = 1
fmin = 2
fmax = 30

# Compute psd of the preprocessed data
data_rest = {}
data_task = {}
for subj in subjs:
    if os.path.exists(op.join(data_path, 'preproc_files','sub'+subj)):
        data_rest[subj] = {}
        data_task[subj] = {}
        for ses in sess:
            if os.path.exists(op.join(data_path, 'preproc_files','sub'+subj,
                                      'ses'+str(ses))):
                data_rest[subj][ses] = {}
                data_task[subj][ses] = {}
    
                # file paths
                f_name_rest = op.join(data_path,'preproc_files','sub'+subj,
                                      'ses'+str(ses),'sub-'+subj+'_ses-'+
                                      ses+'_task-eyeclose_raw.fif')
                f_name_task = op.join(data_path,'preproc_files','sub'+subj,
                                      'ses'+str(ses),'sub-'+subj+'_ses-'+
                                      ses+'_task-memory_raw.fif')
        
                # load data - rest
                raw_rest = mne.io.read_raw_fif(f_name_rest)
                raw_rest = raw_rest.pick_types(eeg = True, 
                                               exclude=raw_rest.info['bads']+
                                               ['TP9','TP10','FT9','FT10'])
    
                # load data - task
                raw_task = mne.io.read_raw_fif(f_name_task)
                raw_task = raw_task.pick_types(eeg = True, 
                                               exclude=raw_task.info['bads']+
                                               ['TP9','TP10','FT9','FT10'])
    
                # set time interval so that raw_rest and raw_task have the same
                # duration
                tmin=0
                tmax=min(raw_rest.times[-1],raw_task.times[-1])
    
                # save channel names (only for resting data)
                ch_idxs = mne.pick_types(raw_rest.info, eeg=True, exclude='bads')
                data_rest[subj][ses]['ch_names'] = [raw_rest.ch_names[ch_idx] 
                                                    for ch_idx in ch_idxs]
    
                # save channel locations
                data_rest[subj][ses]['ch_locs'] = np.zeros((len(data_rest[subj][ses]['ch_names']),3))
                for ii in range(len(data_rest[subj][ses]['ch_names'])): 
                    data_rest[subj][ses]['ch_locs'][ii,:] = raw_rest.info['chs'][ii]['loc'][:3]
    
                # compute power spectrum with multitapers - rest
                data_rest[subj][ses]['psds'], data_rest[subj][ses]['freqs'] = mne.time_frequency.psd_multitaper(
                    raw_rest,fmin=fmin, fmax=fmax,tmin=tmin,tmax=tmax,
                    bandwidth=bandwidth,adaptive=False,low_bias=True,
                    normalization='length', picks=data_rest[subj][ses]['ch_names'],
                    proj=False, n_jobs=1, verbose=None)
    
                # compute power spectrum with multitapers - task
                ch_idxs = mne.pick_types(raw_task.info, eeg=True, exclude='bads')
                ch_names = [raw_task.ch_names[ch_idx] for ch_idx in ch_idxs]
    
                data_task[subj][ses]['psds'], data_task[subj][ses]['freqs'] = mne.time_frequency.psd_multitaper(
                    raw_task, fmin=fmin, fmax=fmax,tmin=tmin, tmax=tmax,
                    bandwidth=bandwidth, adaptive=False,low_bias=True,
                    normalization='length', picks=ch_names, proj=False,
                    n_jobs=1, verbose=None)
    

if not op.isdir(op.join(data_path,'psds')):
        makedirs(op.join(data_path,'psds'))    
data_rest_file = open(op.join(data_path,'psds','data_rest.pkl'), "wb")
data_task_file = open(op.join(data_path,'psds','data_task.pkl'), "wb")
pickle.dump(data_rest, data_rest_file)
data_rest_file.close()    
pickle.dump(data_task, data_task_file)
data_task_file.close()
