# -*- coding: utf-8 -*-
"""
Preprocessing routine
=====================
Preprocessing includes:\n
  * Selection of proper time-interval
  * Band pass filtering in [2, 50]Hz
  * Mark bad segments
  * Bad channels interpolation
  * Independent component analysis
  * Additional automatic removal of bad segments
"""

import mne
import pandas as pd
import os.path as op
from os import makedirs
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from autoreject import get_rejection_threshold


def preprocessing(data_root, datatype, subject, session, task, suffix, data_path):
    
    bids_path = BIDSPath(subject=subject, task=task, suffix=suffix, session=session,
                         datatype=datatype, root=data_root)
    elec_path = BIDSPath(subject=subject, task=task, suffix='electrodes', session=session,
                         datatype=datatype, root=data_root)

    # Load data for one subject and remove unused channels
    extra_params = {'preload': True}
    raw = read_raw_bids(bids_path=bids_path, extra_params=extra_params, verbose=False)
    raw.drop_channels(['TP9', 'TP10', 'FT9', 'FT10', 'X', 'Y', 'Z'])
    elec_df = pd.read_csv(elec_path, sep='\t', header=0, index_col=None)
    ch_names = elec_df['name'].tolist()[:-3]
    ch_coords = (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).dot(
        (elec_df[['x', 'y', 'z']].to_numpy(dtype=float)[:-3, :]*10**(-3)).T)).T
    ch_pos = dict(zip(ch_names, ch_coords))

    # Define EEG channel montage
    montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
    raw.set_montage(montage)
    ch_type = {}
    for ch_name in raw.info['ch_names']:
        if ch_name == 'VEOG':
            ch_type[ch_name] = 'eog'
        elif ch_name == 'X':
            ch_type[ch_name] = 'misc'
        elif ch_name == 'Y':
            ch_type[ch_name] = 'misc'
        elif ch_name == 'Z':
            ch_type[ch_name] = 'misc'
        else:
            ch_type[ch_name] = 'eeg'

    raw.set_channel_types(ch_type)

    # Extract time-intervals corresponding to eye-closed (EC) and task conditions
    tmin_EC = None
    tmin_task1 = None
    for ann in raw.annotations:
        if ('Eyes Closed' in ann['description']) & (tmin_EC is None):
            tmin_EC = ann['onset']
        elif ('Eyes Closed' in ann['description']) & (tmin_EC is not None):
            tmax_EC = ann['onset']
        elif ('Tone' in ann['description']) & (tmin_task1 is None):
            tmin_task1 = ann['onset']
            tmax_task1 = tmin_task1
        elif ('Tone' in ann['description']) & (tmin_task1 is not None):
            if ann['onset'] - tmax_task1 < 6:
                tmax_task1 = ann['onset']

    # Manual selection for two subjects
    if subject == '033':
        tmin_EC = 402
        tmax_EC = 460
    if subject == '019':
        tmin_EC = 355
        tmax_EC = 414

    raw_EC = raw.copy().crop(tmin=tmin_EC, tmax=tmax_EC)
    raw_task = raw.copy().crop(tmin=tmin_task1, tmax=tmax_task1)

    # Band-pass filter between 2 and 50 hz
    raw_EC.filter(2, 50, method='fir', fir_design='firwin', filter_length='auto',
                  fir_window='hamming', picks='all')
    raw_task.filter(2, 50, method='fir', fir_design='firwin', filter_length='auto',
                    fir_window='hamming', picks='all')

    raw_EC = raw_EC.pick_types(eeg=True, exclude=raw_EC.info['bads'])
    raw_task = raw_task.pick_types(eeg=True, exclude=raw_task.info['bads'])

    # Visually inspect data and mark bad channels
    duration = 1
    overlap = 0
    eve_EC = mne.make_fixed_length_events(raw_EC, id=1, start=0, stop=None, duration=duration,
                                          first_samp=False, overlap=overlap)
    eve_EC[:, 0] = raw_EC.first_samp + eve_EC[:, 0]
    eve_task = mne.make_fixed_length_events(raw_task, id=1, start=0, stop=None, duration=duration,
                                            first_samp=False, overlap=overlap)
    eve_task[:, 0] = raw_task.first_samp + eve_task[:, 0]
    epo_EC = mne.Epochs(raw_EC, eve_EC, preload=True, baseline=None, tmin=0, tmax=duration,
                        proj=False, reject=None, flat=None, detrend=None, reject_by_annotation=False)
    epo_task = mne.Epochs(raw_task, eve_task, preload=True, baseline=None, tmin=0,
                          tmax=duration, proj=False, reject=None, flat=None,
                          detrend=None, reject_by_annotation=True)

    if subject in drop_idxs['ses'+session]['EC'].keys():
        drop_idx = drop_idxs['ses'+session]['EC'][subject]
    else:
        drop_idx = []
    epo_EC.drop(drop_idx)
    data_epo_EC = epo_EC.get_data()    
    data_raw_EC = np.zeros((data_epo_EC.shape[1], data_epo_EC.shape[0]*data_epo_EC.shape[2]))
    for i_ch in range(data_epo_EC.shape[1]):
        data_raw_EC[i_ch, :] = np.reshape(data_epo_EC[:, i_ch, :].squeeze(),
                                          (1, data_epo_EC.shape[0]*data_epo_EC.shape[2]))
    raw_EC = mne.io.RawArray(data_raw_EC, raw_EC.info)

    if subject in drop_idxs['ses'+session]['task'].keys():
        drop_idx = drop_idxs['ses'+session]['task'][subject]
    else:
        drop_idx = []
    epo_task.drop(drop_idx)
    data_epo_task = epo_task.get_data()
    data_raw_task = np.zeros((data_epo_task.shape[1], data_epo_task.shape[0]*data_epo_task.shape[2]))
    for i_ch in range(data_epo_task.shape[1]):
        data_raw_task[i_ch, :] = np.reshape(data_epo_task[:, i_ch, :].squeeze(),
                                            (1, data_epo_task.shape[0]*data_epo_task.shape[2]))
    raw_task = mne.io.RawArray(data_raw_task, raw_task.info)

    # Manually mark bad channels
    raw_EC.info['bads'] = bad_chs['ses'+session]['EC'][subject]
    raw_task.info['bads'] = bad_chs['ses'+session]['task'][subject]
    raw_EC.annotations.delete(np.arange(len(raw_EC.annotations)))
    raw_task.annotations.delete(np.arange(len(raw_task.annotations)))
        
    # Interpolate bad channels
    raw_EC.interpolate_bads()
    raw_task.interpolate_bads()
    
    # Rereferencing
    raw_EC.set_eeg_reference(ref_channels='average', ch_type='eeg')
    raw_task.set_eeg_reference(ref_channels='average', ch_type='eeg')
    
    # Independent component analysis (ICA) of EC
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard', random_state=42)
    picks = mne.pick_types(raw_EC.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')
    ica.fit(raw_EC, picks=picks)
    ica.exclude = bad_ICA_EC['ses'+session][subject]
    ica.apply(raw_EC)
    
    # ICA of task
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard', random_state=42)
    picks = mne.pick_types(raw_task.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')
    ica.fit(raw_task, picks=picks)
    ica.exclude = bad_ICA_task['ses'+session][subject]
    ica.apply(raw_task)
    
    # Automatically reject bad epochs
    duration = 1
    overlap = 0
    eve_EC = mne.make_fixed_length_events(raw_EC, id=1, start=0, stop=None, duration=duration,
                                          first_samp=False, overlap=overlap)
    eve_EC[:, 0] = raw_EC.first_samp + eve_EC[:, 0]
    eve_task = mne.make_fixed_length_events(raw_task, id=1, start=0, stop=None, duration=duration,
                                            first_samp=False, overlap=overlap)
    eve_task[:, 0] = raw_task.first_samp + eve_task[:, 0]
    epo_EC = mne.Epochs(raw_EC, eve_EC, preload=True, baseline=None, tmin=0, tmax=duration,
                        proj=False, reject=None, flat=None, detrend=None, reject_by_annotation=False)
    epo_task = mne.Epochs(raw_task, eve_task, preload=True, baseline=None, tmin=0,
                          tmax=duration, proj=False, reject=None, flat=None, detrend=None,
                          reject_by_annotation=True)
    
    reject_EC = get_rejection_threshold(epo_EC, ch_types='eeg')
    epo_EC_clean = epo_EC.drop_bad(reject=reject_EC)
    data_EC_clean = epo_EC_clean.get_data()
    
    reject_task = get_rejection_threshold(epo_task, ch_types='eeg')
    epo_task_clean = epo_task.drop_bad(reject=reject_task)
    data_task_clean = epo_task_clean.get_data()

    data_raw_EC = np.zeros((data_EC_clean.shape[1], data_EC_clean.shape[0]*data_EC_clean.shape[2]))
    for i_ch in range(data_EC_clean.shape[1]):
        data_raw_EC[i_ch, :] = np.reshape(data_EC_clean[:, i_ch, :].squeeze(),
                                          (1, data_EC_clean.shape[0]*data_EC_clean.shape[2]))
    data_raw_task = np.zeros((data_task_clean.shape[1], data_task_clean.shape[0]*data_task_clean.shape[2]))
    for i_ch in range(data_task_clean.shape[1]):
        data_raw_task[i_ch, :] = np.reshape(data_task_clean[:, i_ch, :].squeeze(),
                                            (1, data_task_clean.shape[0]*data_task_clean.shape[2]))
    
    raw_EC = mne.io.RawArray(data_raw_EC, raw_EC.info)
    raw_task = mne.io.RawArray(data_raw_task, raw_task.info)
    
    # Saving EC and task preprocessed data
    f_path = op.join(data_path, 'preproc_files', 'sub'+subject, 'ses'+session)
    if not op.isdir(f_path):
        makedirs(f_path)
    
    f_name_EC = op.join(f_path, 'sub-'+subject+'_ses-'+session+'_task-eyeclose_raw.fif')
    f_name_task = op.join(f_path, 'sub-'+subject+'_ses-'+session+'_task-memory_raw.fif')
    
    raw_EC.save(f_name_EC, overwrite=True)
    raw_task.save(f_name_task, overwrite=True)

    return


drop_idxs = {'ses01': {'EC': {'003': [39],
                              '031': [0],
                              '032': [0, 1, 2, 32, 33, 56],
                              '036': [2, 3, 4],
                              '038': [33, 34],
                              '041': [0, 1],
                              '046': [21, 24]},
                       'task': {'031': [0, 1, 2, 3, 24],
                                '034': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 95, 96, 97,
                                        98, 99, 100, 101],
                                '036': [0, 1],
                                '038': [21, 22, 60, 61, 78, 79, 111, 112],
                                '041': [0],
                                '046': [6, 7, 8, 9, 10, 21, 22, 27, 28, 32, 45, 51, 58, 59,
                                        60, 62, 63, 70, 71, 72,
                                        73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                                        87, 88, 89, 90, 92, 94,
                                        95, 96, 97, 108, 114, 115],
                                '047': [0, 1, 2, 3, 4, 5, 6, 88, 89, 90, 91, 92, 93, 94, 95,
                                        96, 97, 98, 99, 100,
                                        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                                        112, 113, 114, 115]}},
             'ses02': {'EC': {},
                       'task': {'013': [47, 48, 49],
                                '016': [23, 24, 27, 28, 33, 34, 35, 59, 60, 61]}}
             }

bad_chs = {'ses02': {'EC': {'001': [],
                            '002': [],
                            '004': ['T8', 'Cz'],
                            '006': ['AF7'],
                            '007': [],
                            '008': ['CP2'],
                            '009': ['CP2', 'CP1'],
                            '010': ['CP2'],
                            '011': [],
                            '012': [],
                            '013': ['CP2', 'CP1'],
                            '014': ['Cz'],
                            '015': [],
                            '016': ['Pz', 'POz'],
                            '017': [],
                            '018': [],
                            '019': [],
                            '020': ['CP2', 'CP1'],
                            '021': [],
                            '022': [],
                            '023': ['CP2', 'CP1'],
                            '024': ['T8', 'CP2', 'CP1'],
                            '025': [],
                            '026': [],
                            '027': ['Cz']},
                     'task': {'001': ['T8', 'FT7', 'F4'],
                              '002': [],
                              '004': ['T8', 'Cz'],
                              '005': [],
                              '006': [],
                              '007': [],
                              '008': ['CP2'],
                              '009': ['CP2', 'CP1'],
                              '010': ['CP2'],
                              '011': [],
                              '012': [],
                              '013': ['CP2', 'CP1'],
                              '014': ['Cz'],
                              '015': [],
                              '016': ['Pz', 'POz'],
                              '017': [],
                              '018': [],
                              '019': [],
                              '020': ['CP2', 'CP1'],
                              '021': [],
                              '022': [],
                              '023': ['CP2', 'CP1'],
                              '024': ['T8', 'CP2', 'CP1'],
                              '025': [],
                              '026': [],
                              '027': ['Cz']}},
           'ses01': {'EC': {'001': ['TP7', 'FT8'],
                            '002': ['P7', 'TP7'],
                            '003': [],
                            '004': ['FCz', 'Cz'],
                            '005': [],
                            '006': ['CP2', 'CP1'],
                            '007': ['FT7', 'T7'],
                            '008': [],
                            '009': [],
                            '010': [],
                            '011': [],
                            '012': [],
                            '013': ['F2'],
                            '014': ['AF8', 'F6', 'F8'],
                            '015': [],
                            '016': ['Cz'],
                            '017': [],
                            '018': [],
                            '019': [],
                            '020': [],
                            '021': ['F1', 'Cz'],
                            '022': ['F1'],
                            '023': [],
                            '024': ['T8', 'CP1', 'CP2'],
                            '025': [],
                            '026': ['T7'],
                            '027': [],
                            '028': [],
                            '029': [],
                            '030': ['CP1'],
                            '031': [],
                            '032': [],
                            '033': [],
                            '034': ['T8', 'T7'],
                            '035': ['F1'],
                            '036': ['F8'],
                            '037': ['CP5', 'CP6'],
                            '038': [],
                            '039': [],
                            '040': ['Cz', 'C2', 'FCz'],
                            '041': [],
                            '042': [],
                            '043': [],
                            '044': [],
                            '045': [],
                            '046': [],
                            '047': [],
                            '048': ['P6'],
                            '049': [],
                            '050': []},
                     'task': {'001': ['TP7', 'FT8', 'T8'],
                              '002': ['P7', 'TP7'],
                              '003': [],
                              '004': ['FCz', 'Cz', 'T7'],
                              '005': [],
                              '006': ['CP2', 'CP1'],
                              '007': ['FT7', 'T7'],
                              '008': [],
                              '009': [],
                              '010': [],
                              '011': [],
                              '012': [],
                              '013': [],
                              '014': [],
                              '015': [],
                              '016': ['Cz'],
                              '017': [],
                              '018': [],
                              '019': [],
                              '020': ['F5'],
                              '021': ['F1', 'Cz'],
                              '022': ['F1'],
                              '023': [],
                              '024': ['CP1', 'CP2'],
                              '025': [],
                              '026': [],
                              '027': [],
                              '028': [],
                              '029': [],
                              '030': ['CP1', 'FC6', 'FT8'],
                              '031': [],
                              '032': ['AF7'],
                              '033': [],
                              '034': ['T8', 'T7'],
                              '035': ['F1'],
                              '036': [],
                              '037': ['CP6'],
                              '038': [],
                              '039': [],
                              '040': ['Cz', 'C2', 'FCz'],
                              '041': [],
                              '042': [],
                              '043': [],
                              '044': [],
                              '045': [],
                              '046': [],
                              '047': [],
                              '048': ['P6'],
                              '049': [],
                              '050': []}}}


bad_ICA_EC = {'ses01': {'001': [2, 5, 6, 7, 9, 19, 20, 21],
                        '002': [1, 4, 10, 12, 14, 16, 20, 21, 24, 28, 30, 32, 33, 37, 40],
                        '003': [0, 16, 24, 25, 27, 36, 37],
                        '004': [0, 9, 14, 16, 17, 19, 28, 29],
                        '005': [],
                        '006': [15, 16, 18, 19, 21, 26, 27, 28, 29, 31, 32],
                        '007': [3, 4, 15, 16, 19, 21, 24, 30],
                        '008': [7],
                        '009': [0, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27],
                        '010': [0, 11, 13, 16, 17, 20, 21, 23, 25],
                        '011': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                        '012': [26, 27, 28, 33, 34, 35, 37, 38, 38, 40],
                        '013': [1, 2, 8, 13, 18, 19, 24, 25, 33],
                        '014': [3, 5, 17, 21, 27, 28, 31, 33, 34],
                        '015': [16, 17, 18, 19, 20, 23, 25, 26],
                        '016': [13, 14, 18, 24],
                        '017': [6, 8, 9, 11, 12, 13, 14, 15, 16, 17],
                        '018': [],
                        '019': [11, 16, 17, 18, 20, 22, 23, 24, 25, 30],
                        '020': [17, 20, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33],
                        '021': [3, 10, 11, 14, 15, 17],
                        '022': [9, 15, 16, 21, 28],
                        '023': [4, 5, 6, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                        '024': [4, 6, 12, 14, 15, 16, 19, 21, 22, 23, 29],
                        '025': [13, 21, 22, 24],
                        '026': [0, 7],
                        '027': [12, 15, 17, 25],
                        '028': [0, 2, 28],
                        '029': [0, 4, 6, 8, 13, 16, 26, 31],
                        '030': [0, 1, 2, 3, 5, 6, 10, 11, 14, 15, 16, 17, 21, 23, 30, 31],
                        '031': [4, 15, 18, 28, 31, 32, 33],
                        '032': [0, 16, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 34],
                        '033': [0, 1, 4, 11, 26, 32, 34, 37],
                        '034': [3, 15, 28],
                        '035': [7, 10],
                        '036': [0, 1, 2, 3, 4, 16, 25, 33, 36, 37],
                        '037': [0, 1, 3, 4, 5, 6, 10, 12, 20, 30, 43],
                        '038': [0, 1, 6, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                24, 26, 27, 28],
                        '039': [4, 10, 18, 19],
                        '040': [0, 4, 9, 15, 19, 29],
                        '041': [6, 9, 10, 12, 15, 16],
                        '042': [1, 3, 21, 25, 31, 37],
                        '043': [0, 6, 35],
                        '044': [0, 4, 35],
                        '045': [0, 1, 3, 4, 11, 12, 14, 15, 16, 17],
                        '046': [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 16, 18, 20, 22, 23],
                        '047': [4, 15, 16, 17, 24, 25, 26, 29, 30, 31, 32, 34],
                        '048': [0, 1, 2, 3, 18, 27, 28, 35, 36, 37, 38, 39],
                        '049': [0, 1],
                        '050': [0, 2, 14, 15, 31, 34, 36]},
              'ses02': {'001': [5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 23, 24, 25, 26, 31,
                                34],
                        '002': [0, 20, 22, 24, 35, 38, 39],
                        '004': [14, 19, 20, 21, 22],
                        '006': [0, 5, 16, 18, 21, 22, 23],
                        '007': [4, 7, 14, 16],
                        '008': [2, 10],
                        '009': [0, 5, 6, 12, 14, 21, 23, 24, 25, 26],
                        '010': [0, 8, 12, 13, 17, 19, 21, 23, 24, 25, 26],
                        '011': [10, 12, 17, 18, 19, 24, 25, 27, 32, 34, 41],
                        '012': [0, 2, 31, 33, 34, 35],
                        '013': [19, 20, 21, 26],
                        '014': [0, 5, 18, 20, 23, 24, 26, 28, 29, 33],
                        '015': [1, 14, 18, 20, 21, 23, 24, 25, 26, 28],
                        '016': [14, 15, 18, 19, 24, 25],
                        '017': [8, 9, 11, 14, 15, 18, 19, 22, 26, 27],
                        '018': [6, 10, 21, 22, 25, 27, 28, 38],
                        '019': [0, 7, 15, 18, 19, 20, 21, 23, 25, 26, 27],
                        '020': [1, 5, 13, 17, 18, 23, 24, 25, 26, 27, 30, 31, 33],
                        '021': [12, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26],
                        '022': [0, 2, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28],
                        '023': [13, 14, 16, 21, 23, 24, 25, 26, 27, 28],
                        '024': [7, 10, 20, 21, 22, 23, 24, 27],
                        '025': [10, 11, 13, 17, 18, 20, 21, 22, 23],
                        '026': [5, 12, 14, 19, 22, 26],
                        '027': [13, 14, 19, 20, 21]},
              }


bad_ICA_task = {'ses01': {'001': [0, 2, 3, 4, 9, 12, 20, 27, 28],
                          '002': [1, 2, 5, 6, 27, 34, 38, 40, 41, 42],
                          '003': [0, 1, 11, 12, 14, 15],
                          '004': [3, 11, 12, 13, 20, 23, 25, 26, 27],
                          '005': [],
                          '006': [0, 12, 13, 14, 16, 17, 18, 20, 21, 22],
                          '007': [2, 5, 11, 12, 13, 17, 18, 19, 21],
                          '008': [0, 5, 8, 26, 27, 31],
                          '009': [0, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                          '010': [0, 5, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
                          '011': [0, 8, 9, 11, 13, 15, 16, 20, 21, 22, 25, 26, 28, 29],
                          '012': [0, 11, 12, 16, 18, 20, 22, 23, 24, 25, 29, 30],
                          '013': [0, 4, 6, 11, 13, 14, 15, 16, 17, 24, 25, 29, 32],
                          '014': [0, 5, 12, 19, 20, 22, 23, 24, 26, 30, 31, 33],
                          '015': [0, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23],
                          '016': [0, 9, 11, 12, 16, 17, 19, 23],
                          '017': [0, 6, 9, 10, 11, 13, 14, 15, 18, 21, 22, 24, 25, 26, 27, 28],
                          '018': [],
                          '019': [0, 3, 9, 13, 14, 15, 18, 19, 23, 24, 25, 26],
                          '020': [0, 5, 9, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 24, 25, 27,
                                  28, 29, 30, 31, 33],
                          '021': [0, 7, 8, 10, 11, 12, 15, 16],
                          '022': [0, 2, 4, 6, 7, 8, 11, 12, 15, 17, 23, 25],
                          '023': [0, 4, 5, 6, 7, 8, 10, 11, 12, 13],
                          '024': [0, 1, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16],
                          '025': [20, 26],
                          '026': [0],
                          '027': [0, 12, 13, 23, 28, 30],
                          '028': [0, 1, 2, 6, 7, 11, 13, 14, 15, 16, 18, 20, 23, 26],
                          '029': [0, 1, 4, 5, 9, 10, 12, 14, 15],
                          '030': [0, 1, 2, 5, 7, 9, 10, 11, 12, 13, 16, 19, 20, 22, 23, 24, 25],
                          '031': [0, 11, 12, 17, 20],
                          '032': [0, 1, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 27],
                          '033': [0, 1, 5, 6, 7, 8, 9, 10, 12, 14, 16],
                          '034': [0, 1, 4, 5, 6, 11, 13, 14, 18, 19, 20, 21, 22, 23, 28],
                          '035': [0, 7],
                          '036': [0, 1, 2, 3, 4, 8, 30, 31, 32],
                          '037': [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17],
                          '038': [0, 1, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                          '039': [0, 1, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                          '040': [0, 1, 2, 7, 11, 12, 13, 15, 16, 17, 18, 22, 25, 26, 27],
                          '041': [0, 7, 11, 12, 15, 16, 19, 22, 25, 26, 27],
                          '042': [0, 4, 6, 11, 16, 20, 21, 22, 29],
                          '043': [0, 1, 5, 7, 9],
                          '044': [0, 1, 4, 12, 14, 19],
                          '045': [0, 2, 3, 7, 9, 12, 14, 19],
                          '046': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18,
                                  19, 20, 22, 23, 25, 26, 30],
                          '047': [0, 1, 4, 8, 13, 36, 39],
                          '048': [0, 2, 3, 7, 8, 10, 12, 18, 19, 20, 21, 22, 23, 24, 26, 30, 31],
                          '049': [0, 1, 2, 3, 10, 12],
                          '050': [0, 1, 2, 3, 8, 12, 14, 15]},
                'ses02': {'001': [0, 1, 2, 3, 5, 7, 11, 12, 13, 14, 15, 19, 20, 21, 22, 25, 26],
                          '002': [0, 4, 6, 11, 12, 13, 20, 21, 24, 25, 28, 33, 35, 37, 38],
                          '004': [1, 11, 13, 14, 21, 23],
                          '006': [0, 5, 10, 13, 14, 17, 18, 19, 23, 24],
                          '007': [0, 1, 3, 8, 9, 11, 25, 31, 36],
                          '008': [0, 4, 13, 22],
                          '009': [0, 10, 12, 14, 15, 16, 17, 18],
                          '010': [0, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22],
                          '011': [3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 21, 23, 26],
                          '012': [1, 6, 15, 19, 21, 23, 25, 31, 36],
                          '013': [2, 3, 5, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23, 28, 29, 30],
                          '014': [0, 1, 2, 11, 13, 14, 17, 19, 20, 21, 22, 24, 25],
                          '015': [0, 6, 9, 13, 14, 15, 16, 17, 18],
                          '016': [0, 2, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19],
                          '017': [0, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 20, 21, 24, 26],
                          '018': [2, 17, 26, 29, 30, 33, 35, 36, 37, 39, 40],
                          '019': [8, 12, 15, 16, 18, 21, 22, 23, 25, 26],
                          '020': [0, 7, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32],
                          '021': [0, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16],
                          '022': [0, 1, 2, 4, 5, 7, 9, 10, 13, 14, 15, 17, 18, 20, 21, 22, 23],
                          '023': [0, 2, 3, 9, 10, 11, 12],
                          '024': [0, 1, 3, 9, 10, 11, 12, 13],
                          '025': [0, 10, 14, 16, 17, 18, 19],
                          '026': [0, 3, 5, 6, 17],
                          '027': [1, 4, 14, 16, 17, 18, 19, 20, 21, 22, 23]},

                }
