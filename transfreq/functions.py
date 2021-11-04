# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:41:37 2021

@author: elisabetta
"""
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans, MeanShift

# extended names of the clustering methods 
meth_names = ['1D thresholding', '1D Mean-Shift', '2D k-means', '2D adjusted k-means']


def _compute_cluster(psds, freqs, ch_names, alpha_range = None, theta_range = None, method = 4):
    """
    Creates a cluster databese and compute the cluster
    
    Parameters:
        psds: array, shape (N_sources, N_freqs)
            power spectral matrix 
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        ch_names: list of strings
            names of the channels (must be ordered as they are in psds)
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
        TFbox: dictionary
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; columns: 
                   channel names)
                -) the transition freqency (TF) (set to None if not computed yet)
    """
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
     
    
    # define indices of the psds within the alpha and theta ranges
    if alpha_range is None:
        f_alpha_idx = np.where((freqs>=AP-1) & (freqs<=AP+1))[0]
    elif len(alpha_range)!=2:
        raise ValueError("len(alpha_range) must be 2")
    elif (alpha_range[0]<freqs[0] or alpha_range[-1]>freqs[-1]):
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0]>alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs>=alpha_range[0]) & (freqs<=alpha_range[1]))[0]
    
    if theta_range is None:
        if AP-1>7: f_theta_idx = np.where((freqs>=5)  & (freqs<=7) )[0]
        else: f_theta_idx = np.where((freqs>=AP-3)  & (freqs<AP-1) )[0]
    elif len(theta_range)!=2:
        raise ValueError("len(theta_range) must be 2")
    elif (theta_range[0]<freqs[0] or theta_range[-1]>freqs[-1]):
        raise ValueError("theta_range must be inside the interval [freqs[0], freqs[-1]]")
    elif theta_range[0]>theta_range[-1]:
        raise ValueError("theta_range[-1] must be greater than theta_range[0]")
    else:
        f_theta_idx = np.where((freqs>=theta_range[0]) & (freqs<=theta_range[1]))[0]
    
    # compute alpha and theta coefficients
    alpha_coef = psds[:,f_alpha_idx].mean(axis=1)
    theta_coef = psds[:,f_theta_idx].mean(axis=1)
    
    
    # difine the labels associated to the cluster 
    labels = np.ones(len(ch_names), dtype=int)*2
        
    if method == 1:
        n_ch = 4
        ratio_coef = alpha_coef/theta_coef
        theta_idx = np.where(ratio_coef<=np.sort(ratio_coef)[n_ch-1])[0]
        alpha_idx = np.where(ratio_coef>=np.sort(ratio_coef)[-n_ch])[0]
        labels[theta_idx] = 0
        labels[alpha_idx] = 1
        
    elif method == 2:
        ratio_coef = alpha_coef/theta_coef
        
        kmeans1d =  MeanShift(bandwidth=None).fit(ratio_coef.reshape((-1,1)))
        
        lab_count = 2
        for label in range(max(kmeans1d.labels_)+1):
            if kmeans1d.labels_[np.argsort(ratio_coef)[0]]==label:
                theta_idx = np.where(kmeans1d.labels_==label)[0]
                labels[theta_idx] = 0
            elif kmeans1d.labels_[np.argsort(ratio_coef)[-1]]==label:
                alpha_idx = np.where(kmeans1d.labels_==label)[0]
                labels[alpha_idx] = 1
            else:
                tmp_idx = np.where(kmeans1d.labels_==label)[0]
                labels[tmp_idx] = lab_count
                lab_count += 1
            
    elif method ==3:
        coef2d = np.zeros((len(alpha_coef),2))
        coef2d[:,0] = alpha_coef
        coef2d[:,1] = theta_coef

        # fitting the fuzzy-c-means
        kmeans2d = KMeans(n_clusters=2, random_state=0).fit(coef2d)

        if kmeans2d.cluster_centers_[0,0]> kmeans2d.cluster_centers_[1,0]:
            alpha_label = 0 
            theta_label = 1 
        else:
            alpha_label = 1 
            theta_label = 0
            
        alpha_idx = np.where(kmeans2d.predict(coef2d)==alpha_label)[0]
        theta_idx = np.where(kmeans2d.predict(coef2d)==theta_label)[0]
        labels[theta_idx] = 0
        labels[alpha_idx] = 1
        
    elif method == 4:
        coef2d = np.zeros((len(alpha_coef),2))
        coef2d[:,0] = alpha_coef
        coef2d[:,1] = theta_coef

        # fitting the fuzzy-c-means
        kmeans2d = KMeans(n_clusters=2, random_state=0).fit(coef2d)

        if kmeans2d.cluster_centers_[0,0]> kmeans2d.cluster_centers_[1,0]:
            alpha_center = kmeans2d.cluster_centers_[0,:]
            theta_center = kmeans2d.cluster_centers_[1,:] 
        else:
            alpha_center = kmeans2d.cluster_centers_[1,:]
            theta_center = kmeans2d.cluster_centers_[0,:] 


        coeff_ang =-1/( (alpha_center[1]-theta_center[1])/(alpha_center[0]-theta_center[0]) )
        if coeff_ang >0:
            alpha_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii]<coeff_ang*(alpha_coef[ii]-alpha_center[0])+alpha_center[1]]
            theta_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii]>coeff_ang*(alpha_coef[ii]-theta_center[0])+theta_center[1]]
        else:
            alpha_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii]>coeff_ang*(alpha_coef[ii]-alpha_center[0])+alpha_center[1]]
            theta_idx = [ii for ii in range(len(alpha_coef)) if theta_coef[ii]<coeff_ang*(alpha_coef[ii]-theta_center[0])+theta_center[1]]
            
        labels[theta_idx] = 0
        labels[alpha_idx] = 1
        
        
    else:
        raise ValueError("Non valid method input. Supported values are 1, 2, 3, 4 ")
    
        
    cluster = pd.DataFrame(index=['alpha_coef','theta_coef','labels'], columns=ch_names)
    cluster.loc['alpha_coef']=alpha_coef
    cluster.loc['theta_coef']=theta_coef
    cluster.loc['labels']=labels
    
    TFbox = {'cluster':cluster, 'method':method, 'TF':None}
    
    return TFbox
    

    
def create_cluster(psds, freqs, ch_names, theta_chs, alpha_chs, theta_range=None, alpha_range=None, method='user_method'):
    
    """
    Manually creates a TFbox
    
    Parameters:
        psds: array, shape (N_sources, N_freqs)
            power spectral matrix
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        ch_names: list of strings
            names of the channels (must be ordered as they are in psds)
        theta_chs: tuple | list 
            names of the theta channels in the cluster
        alpha_chs: tuple | list
            names of the theta channels in the cluster
        theta_range: tuple | list | array | None (default None)
            theta range to compute alpha coefficients (eg: (5,7), [5,7], np.array([5,7])).
            If None it is set automatically
        alpha_range: tuple | list | array | None (default None)
            alpha range to compute theta coefficients (eg: (9,11), [9,11], np.array([9,11])).
            If None it is set automatically
        method: str (default 'user_method')
            name the user wants to assign to the customized cluster
            
    Returns:
        TFbox: dictionary
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; olumns: 
                   channel names)
                -) the transition freqency (TF) (set to None if not computed yet)
    """
    
    if len((set(theta_chs)).intersection(set(alpha_chs)))!=0:
        raise ValueError("theta_chs and alpha_chs must not have common elements")
    
    theta_idx = [ch_names.index(ch) for ch in theta_chs]
    alpha_idx = [ch_names.index(ch) for ch in alpha_chs]
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
    
        
    # define indices of the psds within the alpha and theta ranges
    if alpha_range is None:
        f_alpha_idx = np.where((freqs>=AP-1) & (freqs<=AP+1))[0]
    elif len(alpha_range)!=2:
        raise ValueError("len(alpha_range) must be 2")
    elif (alpha_range[0]<freqs[0] or alpha_range[-1]>freqs[-1]):
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0]>alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs>=alpha_range[0]) & (freqs<=alpha_range[1]))[0]
    
    if theta_range is None:
        if AP-1>7: f_theta_idx = np.where((freqs>=5)  & (freqs<=7) )[0]
        else: f_theta_idx = np.where((freqs>=AP-3)  & (freqs<AP-1) )[0]
    elif len(theta_range)!=2:
        raise ValueError("len(theta_range) must be 2")
    elif (theta_range[0]<freqs[0] or theta_range[-1]>freqs[-1]):
        raise ValueError("theta_range must be inside the interval [freqs[0], freqs[-1]]")
    elif theta_range[0]>theta_range[-1]:
        raise ValueError("theta_range[-1] must be greater than theta_range[0]")
    else:
        f_theta_idx = np.where((freqs>=theta_range[0]) & (freqs<=theta_range[1]))[0]
    

    
    alpha_coef = psds[:,f_alpha_idx].mean(axis=1)
    theta_coef = psds[:,f_theta_idx].mean(axis=1)
    labels = np.ones(len(ch_names), dtype=int)*2
    labels[theta_idx] = 0
    labels[alpha_idx] = 1
    
    cluster = pd.DataFrame(index=['alpha_coef','theta_coef','labels'], columns=ch_names)
    cluster.loc['alpha_coef']=alpha_coef
    cluster.loc['theta_coef']=theta_coef
    cluster.loc['labels']=labels
    
    TFbox = {'cluster':cluster, 'method':method, 'TF': None}
    
    return TFbox


def computeTF_manual(psds, freqs, TFbox):
    """
    Compute transition frequency given a customized cluster
    
    Parameters:
        psds: array
            power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        TFbox: dictionary
            output of create_cluster
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; olumns: 
                   channel names)
                -) the transition freqency (TF) 
            
            
    Returns:
        TFbox: dictionary
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; olumns: 
                   channel names)
                -) the transition freqency (TF) 
    """
    cluster = TFbox['cluster']
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
    
    theta_coef = cluster.loc['theta_coef'].values
    alpha_coef = cluster.loc['alpha_coef'].values
    theta_idx = np.where(cluster.loc['labels']==0)[0]
    alpha_idx = np.where(cluster.loc['labels']==1)[0]
    
    theta_psds = (psds[theta_idx,:]*(theta_coef[theta_idx]/theta_coef[theta_idx].sum()).reshape(-1,1)).sum(axis=0)
    alpha_psds = (psds[alpha_idx,:]*(alpha_coef[alpha_idx]/alpha_coef[alpha_idx].sum()).reshape(-1,1)).sum(axis=0)
    
    TF_new = 5
    f_in_idx = np.where((freqs>=5)&(freqs<=AP-0.5))[0]
    psds_diff =alpha_psds-theta_psds
    for f in np.flip(f_in_idx)[:-1]:
        if psds_diff[f]*psds_diff[f-1]<0: 
            if (abs(psds_diff[f])<abs(psds_diff[f-1])) & (freqs[f]>=5): TF_new = freqs[f]
            else: TF_new = freqs[f-1]
            break
    TFbox['TF']=TF_new
    
    return TFbox

   
def computeTF_auto(psds, freqs, ch_names, theta_range = None, alpha_range = None, method = 4, iterative=True):
    
    """
    Automatically compute transition frequency
    
    Parameters:
        psds: array, shape (N_sources, N_freqs)
            power spectral matrix 
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        ch_names: list of strings
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
        TFbox: dictionary
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; olumns: 
                   channel names)
                -) the transition freqency (TF) 
    """
    
    if not isinstance(iterative,bool):
        raise ValueError("iterative must be a boolean")
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
        
    # initialize needed quantities
    err = np.inf
    toll = max(freqs[1]-freqs[0], 0.1)
    n_iter = 0
    max_iter = 20
    TF_new = 0
    
    
    # while loop for computation of TF
    while (abs(err)>toll and n_iter<max_iter):
        TFbox = _compute_cluster(psds, freqs, ch_names, alpha_range = alpha_range, theta_range = theta_range, method = method)
        cluster = TFbox['cluster']
        
        TF_old = TF_new
        
        theta_coef = cluster.loc['theta_coef'].values
        alpha_coef = cluster.loc['alpha_coef'].values
        theta_idx = np.where(cluster.loc['labels']==0)[0]
        alpha_idx = np.where(cluster.loc['labels']==1)[0]
        
        theta_psds = (psds[theta_idx,:]*(theta_coef[theta_idx]/theta_coef[theta_idx].sum()).reshape(-1,1)).sum(axis=0)
        alpha_psds = (psds[alpha_idx,:]*(alpha_coef[alpha_idx]/alpha_coef[alpha_idx].sum()).reshape(-1,1)).sum(axis=0)
        
        TF_new = 5
        f_in_idx = np.where((freqs>=5)&(freqs<=AP-0.5))[0]
        psds_diff =alpha_psds-theta_psds
        for f in np.flip(f_in_idx)[:-1]:
            if psds_diff[f]*psds_diff[f-1]<0: 
                if (abs(psds_diff[f])<abs(psds_diff[f-1])) & (freqs[f]>=5): TF_new = freqs[f]
                else: TF_new = freqs[f-1]
                break
        
        n_iter = n_iter +1
        if (TF_new==5 and n_iter==20): 
            TF_new = TF_old
        
        # compute the error (if iterative is False the error is set to zero to esc the loop)
        if iterative == True:
            err = TF_new - TF_old
        elif iterative == False:
            err = 0
     
        TFbox['TF'] = TF_new
        TFbox['cluster'] = cluster   
     
        if AP-1> TF_new:
            alpha_range = [AP-1,AP+1]
        else:
            alpha_range = [TF_new,AP+1]
        theta_range = [max(TF_new-3,freqs[0]),TF_new-1]
        
        
    return TFbox
        

def compute_TF_klimesch(psds_rest, psds_task, freqs):
    
    """
    Compute transition frequency with Klimesch's method
    
    Parameters:
        psds_rest: array, shape (N_sources, N_freqs)
            power spectral matrix of the resting state data
        psds_task: array, shape (N_sources, N_freqs)
            power spectral matrix of the data recorded dunring a task execution
        freqs: array, shape (N_freqs,)
            frequncies at which the psds_rest and psds_task are computed
        
    Returns:
        TF: scalar
             
    """
    
    # normalise power spectra
    psds_rest = psds_rest/psds_rest.sum(axis=1).reshape((psds_rest.shape[0],1))
    psds_task = psds_task/psds_task.sum(axis=1).reshape((psds_task.shape[0],1))
       
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP_idx = f_in_idx[0] + np.argmax(psds_rest.mean(axis=0)[f_in_idx])
    AP_f = freqs[AP_idx]

    # find transition frequency with Klimesch
    TF=5
    f_in_idx = np.where((freqs>=5)&(freqs<=AP_f-0.5))[0]
    psds_diff = psds_rest.mean(axis=0)-psds_task.mean(axis=0)
    for f in np.flip(f_in_idx)[:-1]:
        if psds_diff[f]*psds_diff[f-1]<0:
            if (abs(psds_diff[f])<abs(psds_diff[f-1])) & (freqs[f]>=5): TF = freqs[f]; 
            else: TF = freqs[f-1];
            break
    return TF
