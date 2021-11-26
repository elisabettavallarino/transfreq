# -*- coding: utf-8 -*-
"""
Part 2: Preprocessing
=====================
Drop bad subjects and preprocess the data
"""

import preprocessing_info as prep_info
import os

#################################################################################
# Define data paths

# Local folder where the data have been downloaded
cwd = os.getcwd()
data_root = os.path.join(cwd, 'transfreq_data')
if not os.path.exists(data_root):
    raise Exception('Cannot find data in {}. Please check.'.format(data_root))

# Local folder where to store the output of the analysis
data_path = os.path.join(cwd, 'transfreq_data_preproc')
datatype = 'eeg'
task = 'Rest'
suffix = 'eeg'

if os.path.exists(data_path):
    raise Exception('Folder {} already exists. Please check'.format(
        data_path))

#################################################################################
# Subjects included in analysis

subj_per_session = {'01': ['001', '002', '003', '004', '006', '007', '008', '009',
                           '010', '011', '012', '013', '014', '015', '016', '017',
                           '019', '020', '021', '023', '024', '025', '026', '027',
                           '028', '030', '031', '032', '033', '034', '035', '037',
                           '038', '039', '040', '041', '042', '043', '044', '045',
                           '046', '047', '048', '049', '050'],
                    '02': ['001', '002', '004', '006', '007', '008', '009', '010',
                           '011', '012', '013', '014', '015', '016', '017', '018',
                           '019', '020', '021', '023', '024', '025', '026', '027']}

#################################################################################
# Preprocess data
for session in subj_per_session.keys():
    for subject in subj_per_session[session]:  
        prep_info.preprocessing(data_root, datatype, subject,
                                session, task, suffix, data_path)
