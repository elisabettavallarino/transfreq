# -*- coding: utf-8 -*-
"""
Part 4: Compute transition frequency
====================================
Compute the theta-to-alpha transition frequency with transfreq and with the
classical Klimesch's method.
"""

from transfreq import compute_transfreq, compute_transfreq_klimesch
import os.path as op
import os
import pickle


###############################################################################
# Define data paths

# Local folder where to store the output of the analysis
cwd = os.getcwd()
data_path = os.path.join(cwd, 'transfreq_data_preproc')
psds_path = op.join(data_path, 'psds')

if not os.path.exists(psds_path):
    raise Exception(
        'Cannot find data in {}. Please compute power spectra first.'.format(
            psds_path))

with open(op.join(data_path, 'psds', 'data_rest.pkl'), 'rb') as f_rest:
    data_rest = pickle.load(f_rest)
    
with open(op.join(data_path, 'psds', 'data_task.pkl'), 'rb') as f_task:
    data_task = pickle.load(f_task)

###############################################################################
# Transition frequency with transfreq
methods = [1, 2, 3, 4]
for subj in data_rest.keys():
    for ses in data_rest[subj].keys():
        psds = data_rest[subj][ses]['psds']
        freqs = data_rest[subj][ses]['freqs']
        ch_names = data_rest[subj][ses]['ch_names'] 
        data_rest[subj][ses]['tfbox'] = {}
        for meth in methods:
            data_rest[subj][ses]['tfbox'][meth] = \
                compute_transfreq(psds, freqs, ch_names, alpha_range=None,
                                  theta_range=None, method=meth, iterative=True)

###############################################################################
# Transition frequency with Klimesch's method
for subj in data_rest.keys():
    for ses in data_rest[subj].keys():
        psds_rest = data_rest[subj][ses]['psds']
        psds_task = data_task[subj][ses]['psds']
        freqs = data_rest[subj][ses]['freqs']
        data_task[subj][ses]['tf_klimesch'] = \
            compute_transfreq_klimesch(psds_rest, psds_task, freqs)

###############################################################################
# Save data (overwrite existing files)
data_rest_file = open(op.join(data_path, 'psds', 'data_rest.pkl'), "wb")
data_task_file = open(op.join(data_path, 'psds', 'data_task.pkl'), "wb")
pickle.dump(data_rest, data_rest_file)
data_rest_file.close()    
pickle.dump(data_task, data_task_file)
data_task_file.close()
