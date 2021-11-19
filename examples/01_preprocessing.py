#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:34:15 2021

@author: mida
"""


import preprocessing_info as prep_info
import numpy as np

#################################################################################
# Define data paths

# lacal folder where the data have been downloaded
data_root = '/media/mida/Volume/data_rest_task' 
# local folder where to store the output of the analysis
data_path = '/home/mida/Desktop/Betta/tmp_data_pathTF'
datatype = 'eeg'
task = 'Rest'
suffix = 'eeg'

#################################################################################
# Subjects included in the paper analysis 

subj_per_session = {'01':['001','002','003','004','006','007','008','009',
                          '010','011','012','013','014','015','016','017',
                          '019','020','021','023','024','025','026','027',
                          '028','030','031','032','033','034','035','037',
                          '038','039','040','041','042','043','044','045',
                          '046','047','048','049','050'],
                    '02':['001','002','004','006','007','008','009','010',
                          '011','012','013','014','015','016','017','018',
                          '019','020','021','023','024','025','026','027']}

#################################################################################
# Preprocess data
dropped_epo_total = np.zeros((len(subj_per_session['01'])+len(subj_per_session['02']),2),dtype=int)
cc=0
for session in subj_per_session.keys():
    for subject in subj_per_session[session]:  
        dropped_epo_total[cc,:] = prep_info.preprocessing(data_root, datatype, subject, session, task, suffix, data_path)
        cc+=1