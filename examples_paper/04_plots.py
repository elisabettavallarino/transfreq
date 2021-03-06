# -*- coding: utf-8 -*-
"""
Part 5: Plot results
====================
Visualize and compare the transition frequency estimated with transfreq
and with Klimesch's method
"""

from transfreq.viz import (plot_channels, plot_transfreq_klimesch, 
                           plot_transfreq_minimum, plot_transfreq)
import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

###############################################################################
# Define data paths

# Local folder where to store the output of the analysis
cwd = os.getcwd()
data_path = os.path.join(cwd, 'transfreq_data_preproc')
psds_path = op.join(data_path, 'psds')

if not os.path.exists(psds_path):
    raise Exception(
        'Cannot find data in {}. Please compute power spectra and TF first.'.format(
            psds_path))

with open(op.join(data_path, 'psds', 'data_rest.pkl'), 'rb') as f_rest:
    data_rest = pickle.load(f_rest)
    
with open(op.join(data_path, 'psds', 'data_task.pkl'), 'rb') as f_task:
    data_task = pickle.load(f_task)

###############################################################################
# Subjects for which klimesch's method does not provide satisfying results

bad_subjs = {'01': ['003', '013', '020', '023', '027', '031', '037', '042',
                    '047'],
             '02': ['002', '007', '009', '011', '013', '015', '018', '019',
                    '020', '027']}


###############################################################################
# Results for an illustrative case (subject 001, session 01)

subj = '001'
ses = '01'
ch_locs = data_rest[subj][ses]['ch_locs']
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tf_klimesch = data_task[subj][ses]['tf_klimesch']
tf_minimum = data_rest[subj][ses]['tf_minimum']

# Clustering methods
meths = [1, 2, 3, 4]

# Plots of the channels in G_theta and G_alpha over the topomaps
fig = plt.figure(constrained_layout=True, figsize=(15, 10))
subfigs = fig.subfigures(2, 2, wspace=0.07)
subfigs = subfigs.ravel()
for i_meth, meth in enumerate(meths):
    tfbox = data_rest[subj][ses]['tfbox'][meth]
    plot_channels(tfbox, ch_locs, subfig=subfigs[i_meth])

# Plots of transition frequencies with Klimesch's method and tranfsreq
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
ax = ax.ravel()
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch, ax=ax[0])
plot_transfreq_minimum(psds_rest, freqs, tf_minimum, ax=ax[1])
for i_meth, meth in enumerate(meths):
    tfbox = data_rest[subj][ses]['tfbox'][meth]
    plot_transfreq(psds_rest, freqs, tfbox, ax=ax[i_meth+2])
fig.tight_layout()    


###############################################################################
# Boxplots

# Some properties for boxplots visualisation
linewidth = 3
leg_fs = 14
lab_fs = 18
ticks_fs = 15
ylim = (-3,5)
width = 0.5
boxprops = dict(linewidth=2)
flierprops = dict(marker='o', markerfacecolor='k', markersize=10,
                  markeredgecolor='none')
medianprops = dict(linewidth=2, color='orange')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
whiskerprops = dict(linewidth=2)
capprops = dict(linewidth=2)
meths = [1, 2, 3, 4]
meths_label = ['1D T', '1D MS', '2D KM', '2D aKM', 'minimum']
col = ['red', 'orange', 'olive', 'purple', 'brown', 'cyan']

total_subjs = [str(ses) for subj in data_rest.keys() for ses in data_rest[subj].keys()]
total_bad_subjs = [str(subj) for ses in bad_subjs.keys() for subj in bad_subjs[ses]]
err_tf = np.zeros((len(total_subjs)-len(total_bad_subjs), len(meths)+1))

i_sub = 0
for subj in data_rest.keys():
    for ses in data_rest[subj].keys():
        if subj not in bad_subjs[ses]:
            for i_meth, meth in enumerate(data_rest[subj][ses]['tfbox'].keys()):
                err_tf[i_sub, i_meth] = data_task[subj][ses]['tf_klimesch']-data_rest[subj][ses]['tfbox'][meth]['tf']
            err_tf[i_sub, -1] = data_task[subj][ses]['tf_klimesch']-data_rest[subj][ses]['tf_minimum']
            i_sub = i_sub+1
            

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=4)
ax1.boxplot(err_tf[:,:4], meanline=True, widths = width, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
            meanprops=meanpointprops, whiskerprops=whiskerprops, capprops=capprops)
ax1.set_ylabel(r'$\Delta_{\rm{TF}} \coloneq {\rm{TF_{Klimesch}}}-{\rm{TF}}_{transfreq}$ [Hz]', fontsize=lab_fs)
ax1.set_xticks(list(np.arange(1, len(meths)+1)))
ax1.set_xticklabels(meths_label[:4], rotation=45, fontsize=ticks_fs)
ax1.tick_params(axis='both', which='minor', labelsize=ticks_fs)
ax1.grid()
ax1.set_ylim(ylim)

ax2 = plt.subplot2grid((1, 7), (0, 5), colspan=2)
ax2.boxplot(err_tf[:,-1], meanline=True, widths = width, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
            meanprops=meanpointprops, whiskerprops=whiskerprops, capprops=capprops)
ax2.set_ylabel(r'$\Delta_{\rm{TF}} \coloneq {\rm{TF_{Klimesch}}}-{\rm{TF}}_{\rm{minimum}}$ [Hz]', fontsize=lab_fs)
ax2.set_xticks([1])
ax2.set_xticklabels([meths_label[-1]], rotation=45, fontsize=ticks_fs)
ax2.tick_params(axis='both', which='minor', labelsize=ticks_fs)
ax2.grid()
ax2.set_ylim(ylim)

fig.supxlabel('Method', fontsize=lab_fs)
plt.tight_layout()

###############################################################################
# Improvements of transfeq over Klimesch's method

# subject 013, session 01
subj = '013'
ses = '01'
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tfbox = data_rest[subj][ses]['tfbox'][4]
tf_klimesch = data_task[subj][ses]['tf_klimesch']

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch, ax=ax[0])
plot_transfreq(psds_rest, freqs, tfbox, ax=ax[1])
fig.tight_layout()    


# subject 037, session 01
subj = '037'
ses = '01'
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tfbox = data_rest[subj][ses]['tfbox'][4]
tf_klimesch = data_task[subj][ses]['tf_klimesch']

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch, ax=ax[0])
plot_transfreq(psds_rest, freqs, tfbox, ax=ax[1])
ax[0].lines.remove(fig.axes[0].lines[2])
ax[0].legend()
fig.tight_layout()    

plt.show()
