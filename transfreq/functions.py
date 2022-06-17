# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:41:37 2021

@author: elisabetta
"""
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans, MeanShift

# extended names of the clustering methods 
meth_names = ['1D thresholding', '1D mean-shift', '2D k-means', '2D adjusted k-means']


def _compute_cluster(psds, freqs, ch_names= None, alpha_range=None, theta_range=None, method=4):
    """
    Creates a cluster databese and compute the cluster
    
    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix 
        freqs: array, shape (N_freqs,)
            frequencies at which the psds is computed
        ch_names: None | list of strings (default None)
            names of the channels (must be ordered as they are in psds)
        theta_range: tuple | list | array | None (default None)
            theta range to compute alpha coefficients (eg: (5,7), [5,7], np.array([5,7])).
            If None it is set automatically
        alpha_range: tuple | list | array | None (default None)
            alpha range to compute theta coefficients (eg: (9,11), [9,11], np.array([9,11])).
            If None it is set automatically
        method: 1, 2, 3, 4 (default 4)
            clustering method

    Returns:
        tfbox: dictionary
            Dictionary containing:\n
                * the method used to compute the cluster
                * the cluster: pandas dataframe (rows: alpha coefficients,theta coefficients,
                  clustering labels; columns:channel names)
                * the transition frequency (tf)

    """
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0], 1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    ap = freqs[ap]

    # define indices of the psds within the alpha and theta ranges
    if alpha_range is None:
        f_alpha_idx = np.where((freqs >= ap-1) & (freqs <= ap+1))[0]
    elif len(alpha_range) != 2:
        raise ValueError("len(alpha_range) must be 2")
    elif alpha_range[0] < freqs[0] or alpha_range[-1] > freqs[-1]:
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0] > alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
    
    if theta_range is None:
        if ap-1 > 7:
            f_theta_idx = np.where((freqs >= 5) & (freqs <= 7))[0]
        else:
            f_theta_idx = np.where((freqs >= ap-3) & (freqs < ap-1))[0]
    elif len(theta_range) != 2:
        raise ValueError("len(theta_range) must be 2")
    elif theta_range[0] < freqs[0] or theta_range[-1] > freqs[-1]:
        raise ValueError("theta_range must be inside the interval [freqs[0], freqs[-1]]")
    elif theta_range[0] > theta_range[-1]:
        raise ValueError("theta_range[-1] must be greater than theta_range[0]")
    else:
        f_theta_idx = np.where((freqs >= theta_range[0]) & (freqs <= theta_range[1]))[0]
    
    if ch_names is None: ch_names = ch_names = ['ch'+'0'*(len(str(psds.shape[0]))-len(str(ch_idx+1)))+str(ch_idx+1) for ch_idx in range(psds.shape[0])]
    
    # compute alpha and theta coefficients
    alpha_coef = psds[:, f_alpha_idx].mean(axis=1)
    theta_coef = psds[:, f_theta_idx].mean(axis=1)

    # difine the labels associated to the cluster 
    labels = np.ones(len(ch_names), dtype=int)*2

    if method == 1:
        n_ch = 4
        ratio_coef = alpha_coef/theta_coef
        theta_idx = np.where(ratio_coef <= np.sort(ratio_coef)[n_ch-1])[0]
        alpha_idx = np.where(ratio_coef >= np.sort(ratio_coef)[-n_ch])[0]
        labels[theta_idx] = 0
        labels[alpha_idx] = 1
        
    elif method == 2:
        ratio_coef = alpha_coef/theta_coef
        kmeans1d = MeanShift(bandwidth=None).fit(ratio_coef.reshape((-1, 1)))
        
        lab_count = 2
        for label in range(max(kmeans1d.labels_)+1):
            if kmeans1d.labels_[np.argsort(ratio_coef)[0]] == label:
                theta_idx = np.where(kmeans1d.labels_ == label)[0]
                labels[theta_idx] = 0
            elif kmeans1d.labels_[np.argsort(ratio_coef)[-1]] == label:
                alpha_idx = np.where(kmeans1d.labels_ == label)[0]
                labels[alpha_idx] = 1
            else:
                tmp_idx = np.where(kmeans1d.labels_ == label)[0]
                labels[tmp_idx] = lab_count
                lab_count += 1

    elif method == 3:
        coef2d = np.zeros((len(alpha_coef), 2))
        coef2d[:, 0] = alpha_coef
        coef2d[:, 1] = theta_coef

        # fitting the fuzzy-c-means
        kmeans2d = KMeans(n_clusters=2, random_state=0).fit(coef2d)

        if kmeans2d.cluster_centers_[0, 0] > kmeans2d.cluster_centers_[1, 0]:
            alpha_label = 0
            theta_label = 1
        else:
            alpha_label = 1 
            theta_label = 0

        alpha_idx = np.where(kmeans2d.predict(coef2d) == alpha_label)[0]
        theta_idx = np.where(kmeans2d.predict(coef2d) == theta_label)[0]
        labels[theta_idx] = 0
        labels[alpha_idx] = 1
        
    elif method == 4:
        coef2d = np.zeros((len(alpha_coef), 2))
        coef2d[:, 0] = alpha_coef
        coef2d[:, 1] = theta_coef

        # fitting the fuzzy-c-means
        kmeans2d = KMeans(n_clusters=2, random_state=0).fit(coef2d)

        if kmeans2d.cluster_centers_[0, 0] > kmeans2d.cluster_centers_[1, 0]:
            alpha_center = kmeans2d.cluster_centers_[0, :]
            theta_center = kmeans2d.cluster_centers_[1, :]
        else:
            alpha_center = kmeans2d.cluster_centers_[1, :]
            theta_center = kmeans2d.cluster_centers_[0, :]

        coeff_ang = -1/((alpha_center[1]-theta_center[1])/(alpha_center[0]-theta_center[0]))
        if coeff_ang > 0:
            alpha_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii] < coeff_ang*(alpha_coef[ii]-alpha_center[0]) + alpha_center[1]]
            theta_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii] > coeff_ang*(alpha_coef[ii]-theta_center[0])+theta_center[1]]
        else:
            alpha_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii] > coeff_ang*(alpha_coef[ii]-alpha_center[0]) + alpha_center[1]]
            theta_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii] < coeff_ang*(alpha_coef[ii]-theta_center[0]) + theta_center[1]]
            
        labels[theta_idx] = 0
        labels[alpha_idx] = 1

    else:
        raise ValueError("Non valid method input. Supported values are 1, 2, 3, 4 ")

    cluster = pd.DataFrame(index=['alpha_coef', 'theta_coef', 'labels'], columns=ch_names)
    cluster.loc['alpha_coef'] = alpha_coef
    cluster.loc['theta_coef'] = theta_coef
    cluster.loc['labels'] = labels
    
    tfbox = {'cluster': cluster, 'method': method, 'tf': None}
    
    return tfbox



   
def compute_transfreq(psds, freqs, ch_names=None, theta_range=None, alpha_range=None, method=4, iterative=True):
    """
    Automatically compute transition frequency

    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix 
        freqs: array, shape (N_freqs,)
            frequencies at which the psds is computed
        ch_names: None | list of strings (default None)
            name of the channels (must be ordered as they are in psds)
        theta_range: tuple | list | array | None (default None)
            theta range to compute alpha coefficients (eg: (5,7), [5,7], np.array([5,7])).
            If None it is set automatically
        alpha_range: tuple | list | array | None (default None)
            alpha range to compute theta coefficients (eg: (9,11), [9,11], np.array([9,11])).
            If None it is set automatically
        method: 1, 2, 3, 4 (default 4)
            clustering method 
        iterative: bool (default True)
            Whether to use the iterative method (default) or not 
            
    Returns:
        tfbox: dictionary
            Dictionary containing:\n
                * the method used to compute the cluster
                * the cluster: pandas dataframe (rows: alpha coefficients,theta coefficients,
                  clustering labels; columns:channel names)
                * the transition frequency (tf)

    """
    if ch_names is None: ch_names = ch_names = ['ch'+'0'*(len(str(psds.shape[0]))-len(str(ch_idx+1)))+str(ch_idx+1) for ch_idx in range(psds.shape[0])]
    
    if not isinstance(iterative, bool):
        raise ValueError("iterative must be a boolean")
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0], 1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    ap = freqs[ap]
        
    # initialize needed quantities
    err = np.inf
    toll = max(freqs[1]-freqs[0], 0.1)
    n_iter = 0
    max_iter = 20
    tf_new = 0

    # while loop for computation of tf
    while abs(err) > toll and n_iter < max_iter:
        tfbox = _compute_cluster(psds, freqs, ch_names,
                                 alpha_range=alpha_range, theta_range=theta_range, method=method)
        cluster = tfbox['cluster']

        tf_old = tf_new
        theta_coef = cluster.loc['theta_coef'].values
        alpha_coef = cluster.loc['alpha_coef'].values
        theta_idx = np.where(cluster.loc['labels'] == 0)[0]
        alpha_idx = np.where(cluster.loc['labels'] == 1)[0]
        
        theta_psds = (psds[theta_idx, :]*(theta_coef[theta_idx]/theta_coef[theta_idx].sum()).reshape(-1, 1)).sum(axis=0)
        alpha_psds = (psds[alpha_idx, :]*(alpha_coef[alpha_idx]/alpha_coef[alpha_idx].sum()).reshape(-1, 1)).sum(axis=0)
        
        tf_new = 5
        f_in_idx = np.where((freqs >= 5) & (freqs <= ap-0.5))[0]
        psds_diff = alpha_psds - theta_psds
        for f in np.flip(f_in_idx)[:-1]:
            if psds_diff[f]*psds_diff[f-1] < 0:
                if (abs(psds_diff[f]) < abs(psds_diff[f-1])) & (freqs[f] >= 5):
                    tf_new = freqs[f]
                else:
                    tf_new = freqs[f-1]
                break

        n_iter = n_iter + 1
        if tf_new == 5 and n_iter == 20:
            tf_new = tf_old

        # compute the error (if iterative is False the error is set to zero to esc the loop)
        if iterative is True:
            err = tf_new - tf_old
        elif iterative is False:
            err = 0

        tfbox['tf'] = tf_new
        tfbox['cluster'] = cluster
        tfbox['n_iter'] = n_iter
        
     
        if ap-1 > tf_new:
            alpha_range = [ap-1, ap+1]
        else:
            alpha_range = [tf_new, ap+1]
        theta_range = [max(tf_new-3, freqs[0]), tf_new-1]

    return tfbox
        

    
def compute_transfreq_manual(psds, freqs, theta_chs, alpha_chs, ch_names=None,
                   theta_range=None, alpha_range=None, method='user_method'):
    """
    Compute transition frequency given a customezed cluster 

    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix
        freqs: array, shape (N_freqs,)
            frequencies at which the psds is computed
        theta_chs: tuple | list of integers or string
            indeces or names of the theta channels in the cluster
        alpha_chs: tuple | list of integers or string
            indices or names of the theta channels in the cluster
        ch_names: None | list of strings (default None)
            names of the channels (must be ordered as they are in psds)
        theta_range: tuple | list | array | None (default None)
            theta range to compute alpha coefficients (eg: (5,7), [5,7], np.array([5,7])).
            If None it is set automatically
        alpha_range: tuple | list | array | None (default None)
            alpha range to compute theta coefficients (eg: (9,11), [9,11], np.array([9,11])).
            If None it is set automatically
        method: str (default 'user_method')
            name the user wants to assign to the customized cluster
            
    Returns:
        tfbox: dictionary
            Dictionary containing:\n
                * the method used to compute the cluster
                * the cluster: pandas dataframe (rows: alpha coefficients,theta coefficients,
                  clustering labels; columns:channel names)
                * the transition frequency (tf)

    """
    
    if len((set(theta_chs)).intersection(set(alpha_chs))) != 0:
        raise ValueError("theta_chs and alpha_chs must not have common elements")
        
    if (ch_names is None and type(theta_chs[0]) is not int):
        raise ValueError("if ch_names is None theta_chs must be a tuple or a list of integers, corresponding to the theta channel indices")
    if (ch_names is None and type(alpha_chs[0]) is not int):
        raise ValueError("if ch_names is None alpha_chs must be a tuple or a list of integers, corresponding to the alpha channel indices")
    
    if ch_names is None: ch_names = ['ch'+'0'*(len(str(psds.shape[0]))-len(str(ch_idx+1)))+str(ch_idx+1) for ch_idx in range(psds.shape[0])]
    
    if type(theta_chs[0]) is int: theta_chs = [ch_names[ii-1] for ii in theta_chs]
    if type(alpha_chs[0]) is int: alpha_chs = [ch_names[ii-1] for ii in alpha_chs]
    
    
    
    theta_idx = [ch_names.index(ch) for ch in theta_chs]
    alpha_idx = [ch_names.index(ch) for ch in alpha_chs]
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0], 1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    ap = freqs[ap]

    # define indices of the psds within the alpha and theta ranges
    if alpha_range is None:
        f_alpha_idx = np.where((freqs >= ap-1) & (freqs <= ap+1))[0]
    elif len(alpha_range) != 2:
        raise ValueError("len(alpha_range) must be 2")
    elif alpha_range[0] < freqs[0] or alpha_range[-1] > freqs[-1]:
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0] > alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
    
    if theta_range is None:
        if ap-1 > 7:
            f_theta_idx = np.where((freqs >= 5) & (freqs <= 7))[0]
        else:
            f_theta_idx = np.where((freqs >= ap-3) & (freqs < ap-1))[0]
    elif len(theta_range) != 2:
        raise ValueError("len(theta_range) must be 2")
    elif theta_range[0] < freqs[0] or theta_range[-1] > freqs[-1]:
        raise ValueError("theta_range must be inside the interval [freqs[0], freqs[-1]]")
    elif theta_range[0] > theta_range[-1]:
        raise ValueError("theta_range[-1] must be greater than theta_range[0]")
    else:
        f_theta_idx = np.where((freqs >= theta_range[0]) & (freqs <= theta_range[1]))[0]


    
    
    alpha_coef = psds[:, f_alpha_idx].mean(axis=1)
    theta_coef = psds[:, f_theta_idx].mean(axis=1)
    labels = np.ones(len(ch_names), dtype=int)*2
    labels[theta_idx] = 0
    labels[alpha_idx] = 1
    
    cluster = pd.DataFrame(index=['alpha_coef', 'theta_coef', 'labels'], columns=ch_names)
    cluster.loc['alpha_coef'] = alpha_coef
    cluster.loc['theta_coef'] = theta_coef
    cluster.loc['labels'] = labels


    theta_psds = (psds[theta_idx, :]*(theta_coef[theta_idx]/theta_coef[theta_idx].sum()).reshape(-1, 1)).sum(axis=0)
    alpha_psds = (psds[alpha_idx, :]*(alpha_coef[alpha_idx]/alpha_coef[alpha_idx].sum()).reshape(-1, 1)).sum(axis=0)

    tf_new = 5
    f_in_idx = np.where((freqs >= 5) & (freqs <= ap-0.5))[0]
    psds_diff = alpha_psds - theta_psds
    for f in np.flip(f_in_idx)[:-1]:
        if psds_diff[f]*psds_diff[f-1] < 0:
            if (abs(psds_diff[f]) < abs(psds_diff[f-1])) & (freqs[f] >= 5):
                tf_new = freqs[f]
            else:
                tf_new = freqs[f-1]
            break
    

    tfbox = {'cluster': cluster, 'method': method, 'tf': tf_new}

    return tfbox



def compute_transfreq_klimesch(psds_rest, psds_task, freqs):
    
    """
    Compute transition frequency with Klimesch's method
    
    Parameters:
        psds_rest: array, shape (N_sensors, N_freqs)
            power spectral matrix of the resting state data
        psds_task: array, shape (N_sensors, N_freqs)
            power spectral matrix of the data recorded dunring a task execution
        freqs: array, shape (N_freqs,)
            frequencies at which the psds_rest and psds_task are computed
        
    Returns:
        tf: scalar
    """
    
    # normalise power spectra
    psds_rest = psds_rest/psds_rest.sum(axis=1).reshape((psds_rest.shape[0], 1))
    psds_task = psds_task/psds_task.sum(axis=1).reshape((psds_task.shape[0], 1))
       
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap_idx = f_in_idx[0] + np.argmax(psds_rest.mean(axis=0)[f_in_idx])
    ap_f = freqs[ap_idx]

    # find transition frequency with Klimesch
    tf = 5
    f_in_idx = np.where((freqs >= 5) & (freqs <= ap_f-0.5))[0]
    psds_diff = psds_rest.mean(axis=0)-psds_task.mean(axis=0)
    for f in np.flip(f_in_idx)[:-1]:
        if psds_diff[f]*psds_diff[f-1] < 0:
            if (abs(psds_diff[f]) < abs(psds_diff[f-1])) & (freqs[f] >= 5):
                tf = freqs[f]
            else:
                tf = freqs[f-1]
            break
    return tf


def compute_transfreq_minimum(psds_rest, freqs):
    
    """
    Compute transition frequency with minimum method
    
    Parameters:
        psds_rest: array, shape (N_sensors, N_freqs)
            power spectral matrix of the resting state data
        freqs: array, shape (N_freqs,)
            frequencies at which the psds_rest and psds_task are computed
        
    Returns:
        tf: scalar
    """
    
    # normalise power spectra
    psds_rest = psds_rest/psds_rest.sum(axis=1).reshape((psds_rest.shape[0], 1))
       
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap_idx = f_in_idx[0] + np.argmax(psds_rest.mean(axis=0)[f_in_idx])
    ap_f = freqs[ap_idx]
    
    # tf freqs range
    idx_freqs = np.where((freqs >= 2) & (freqs <= ap_f))[0]
    tf_minimum_idx = idx_freqs[0] + np.argmin(psds_rest.mean(axis=0)[idx_freqs])
    tf_minimum = freqs[tf_minimum_idx]
    
    return tf_minimum