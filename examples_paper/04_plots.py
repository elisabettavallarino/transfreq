# -*- coding: utf-8 -*-
"""
Part 4
======
Description
"""


from transfreq.viz import plot_channels, plot_transfreq_klimesch, plot_transfreq
import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt




###############################################################################
# Define data paths

# local folder where to store the output of the analysis
data_path = '/home/mida/Desktop/Betta/tmp_data_pathTF'

with open(op.join(data_path,'psds','data_rest.pkl'), 'rb') as f_rest:
    data_rest = pickle.load(f_rest)
    
with open(op.join(data_path,'psds','data_task.pkl'), 'rb') as f_task:
    data_task = pickle.load(f_task)
    
    
###############################################################################
# subjects for which klimesch's method does not provide satisfiing results

bad_subjs = {'01':['003','013','020','023','027','031','037','042','047'],
             '02':['002','007','009','011','013','015','018','019','020','022',
                   '027']}


###############################################################################
# Results for an illustrative case (subject 001, session 01)

subj = '001'
ses = '01'
ch_locs = data_rest[subj][ses]['ch_locs']
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tf_klimesch = data_task[subj][ses]['tf_klimesch']
# clustering 


# channle position
meths = [1,2,3,4]

fig, ax = plt.subplots(2,2,figsize=(10,5))
ax = ax.ravel()
for i_meth, meth in enumerate(meths):
    tfbox = data_rest[subj][ses]['tfbox'][meth]
    plot_channels(tfbox, ch_locs,ax=ax[i_meth])

# transition frequencies with Klimesch's method

fig, ax = plt.subplots(3,2,figsize=(10,10))
ax = ax.ravel()
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch,ax=ax[0])
ax[1].set_visible(False)
for i_meth, meth in enumerate(meths):
    tfbox = data_rest[subj][ses]['tfbox'][meth]
    plot_transfreq(psds_rest, freqs, tfbox,ax=ax[i_meth+2])
fig.tight_layout()    



###############################################################################
# boxplots

linewidth = 2
leg_fs = 12
lab_fs=14
ticks_fs=10
boxprops = dict(linewidth=2)
flierprops = dict(marker='o', markerfacecolor='k', markersize=10,
                  markeredgecolor='none')
medianprops = dict(linewidth=2)
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
whiskerprops = dict(linewidth = 2)
capprops = dict(linewidth = 2)
meths = [1,2,3,4]
meths_label = [str(meth) for meth in meths]
col = ['red','orange','olive','purple','brown','cyan']

total_subjs = [str(ses) for subj in data_rest.keys() for ses in data_rest[subj].keys()]
total_bad_subjs = [str(subj) for ses in bad_subjs.keys() for subj in bad_subjs[ses]]
err_tf = np.zeros((len(total_subjs)-len(total_bad_subjs), len(meths)))

i_sub = 0
for subj in data_rest.keys():
    for ses in data_rest[subj].keys():
        if subj not in bad_subjs[ses]:
            for i_meth, meth in enumerate(data_rest[subj][ses]['tfbox'].keys()):
                err_tf[i_sub,i_meth] = data_task[subj][ses]['tf_klimesch']-data_rest[subj][ses]['tfbox'][meth]['tf']
            i_sub = i_sub+1
fig = plt.figure()
plt.boxplot(err_tf, meanline=True, boxprops=boxprops,flierprops=flierprops,medianprops=medianprops,
            meanprops=meanpointprops,whiskerprops=whiskerprops,capprops=capprops)
plt.ylabel(r'$\Delta_{TF} \coloneq TF_{Klimesch}-TF_{TransFreq}$ [Hz]',fontsize = lab_fs)
plt.xlabel('Clustering method',fontsize = lab_fs)
plt.xticks(np.arange(1,len(meths)+1),labels=meths_label, rotation=0,fontsize=ticks_fs)
plt.yticks(fontsize=ticks_fs)
plt.grid()
plt.tight_layout()


###############################################################################
# improvements of transfeq over the Klimesch's method

# subject 013, session 01
subj = '013'
ses = '01'
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tfbox = data_rest[subj][ses]['tfbox'][4]
tf_klimesch = data_task[subj][ses]['tf_klimesch']


fig, ax = plt.subplots(1,2, figsize=(8,4))
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch,ax=ax[0])
plot_transfreq(psds_rest, freqs, tfbox,ax=ax[1])
fig.tight_layout()    

# subject 037, session 01
subj = '037'
ses = '01'
psds_rest = data_rest[subj][ses]['psds']
psds_task = data_task[subj][ses]['psds']
freqs = data_rest[subj][ses]['freqs']
tfbox = data_rest[subj][ses]['tfbox'][4]
tf_klimesch = data_task[subj][ses]['tf_klimesch']


fig, ax = plt.subplots(1,2, figsize=(8,4))
plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf_klimesch,ax=ax[0])
plot_transfreq(psds_rest, freqs, tfbox,ax=ax[1])
ax[0].lines.remove(fig.axes[0].lines[2])
ax[0].legend()
fig.tight_layout()    















