# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:41:37 2021

@author: elisabetta
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from visbrain.objects import ColorbarObj, SceneObj
from .TopoObj_mod import TopoObj
import warnings
import cv2

    


#colors_orig = ['C3','C9','C2','C1','C4','C5']
colors_orig = ['C0','C1','#aef956','C3','C4','C5']
meth_names = ['1D thresholding', '1D Mean-Shift', '2D k-means', '2D adjusted k-means']

def _compute_cluster(psds, freqs, ch_names, alpha_range = None, theta_range = None, method = 4):
    """
    Creates a cluster databese
    
    Parameters:
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        ch_names: list of strings
            name of the channels
        theta_range: tuple | list | array
            theta range to compute alpha coefficients (eg: [5,7])
        alpha_range: tuple | list | array
            alpha range to compute theta coefficients (eg: [9,11])
        method: 1, 2, 3, 4
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
    

    
def create_cluster(psds, freqs, ch_names, theta_range, alpha_range, theta_idx, alpha_idx, method):
    
    """
    Manually creates a TFbox
    
    Parameters:
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        ch_names: list of strings
            name of the channels
        theta_range: tuple | list | array
            theta range to compute alpha coefficients (eg: [5,7])
        alpha_range: tuple | list | array
            alpha range to compute theta coefficients (eg: [9,11])
        theta_idx: tuple | list | array
            indices of the theta channels in the cluster
        alpha_idx: tuple | list | array
            indices of the theta channels in the cluster
        method: str
            the name the user wants to assign to the customized cluster
            
    Returns:
        TFbox: dictionary
            Dictionary containing:
                -) the method used to compute the cluster
                -) the cluster: pandas dataframe (rows: alpha 
                   coefficients, theta coefficients, clustering labels; olumns: 
                   channel names)
                -) the transition freqency (TF) (set to None if not computed yet)
    """
    
    if len((set(alpha_idx)).intersection(set(theta_idx)))!=0:
        raise ValueError("theta_idx and alpha_idx must not have common elements")
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
        
    if len(alpha_range)!=2:
        raise ValueError("len(alpha_range) must be 2")
    elif (alpha_range[0]<freqs[0] or alpha_range[-1]>freqs[-1]):
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0]>alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs>=alpha_range[0]) & (freqs<=alpha_range[1]))[0]
    
    if len(theta_range)!=2:
        raise ValueError("len(theta_range) must be 2")
    elif (theta_range[0]<freqs[0] or theta_range[-1]>freqs[-1]):
        raise ValueError("theta_range must be inside the interval [freqs[0], freqs[-1]]")
    elif theta_range[0]>theta_range[-1]:
        raise ValueError("theta_range[-1] must be greater than theta_range[0]")
    else:
        f_theta_idx = np.where((freqs>=theta_range[0]) & (freqs<=theta_range[1]))[0]
    
    if not isinstance(method, str):
        warnings.warn("method is not a string, it has been trasformed to string")
        method = str(method)
    
    alpha_coef = psds[:,f_alpha_idx].mean(axis=1)
    theta_coef = psds[:,f_theta_idx].mean(axis=1)
    labels = np.ones(len(ch_names), dtype=int)*2
    labels[theta_idx] = 0
    labels[alpha_idx] = 1
    
    cluster = pd.DataFrame(index=['alpha_coef','theta_coef','labels'], columns=ch_names)
    cluster.loc['alpha_coef']=alpha_coef
    cluster.loc['theta_coef']=theta_coef
    cluster.loc['labels']=labels
    
    TFbox = {'cluster':cluster, 'method':method, 'TF':None}
    
    return TFbox


def computeTF_manual(psds, freqs, TFbox):
    """
    Compute transition frequency given a customized cluster
    
    Parameters:
        psds: array
            power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        ch_names: list of strings
            names of the channels
            
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
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        ch_names: list of strings
            name of the channels
        theta_range: tuple | list | array
            theta range to compute theta coefficients
        alpha_range: tuple | list | array
            alpha range to compute alpha coefficients
        method: 1, 2, 3, 4
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
        
    
    err = np.inf
    toll = max(freqs[1]-freqs[0], 0.1)
    n_iter = 0
    max_iter = 20
    TF_new = 0
    
    
    
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
    psds_rest = psds_rest/psds_rest.sum(axis=1).reshape((psds_rest.shape[0],1))
    psds_task = psds_task/psds_task.sum(axis=1).reshape((psds_task.shape[0],1))
       
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP_idx = f_in_idx[0] + np.argmax(psds_rest.mean(axis=0)[f_in_idx])
    AP_f = freqs[AP_idx]

    # find transition frequency with Klimesch
    TF = 5
    f_in_idx = np.where((freqs>=5)&(freqs<=AP_f-0.5))[0]
    psds_diff = psds_rest.mean(axis=0)-psds_task.mean(axis=0)
    for f in np.flip(f_in_idx)[:-1]:
        if psds_diff[f]*psds_diff[f-1]<0:
            if (abs(psds_diff[f])<abs(psds_diff[f-1])) & (freqs[f]>=5): TF = freqs[f]; 
            else: TF = freqs[f-1];
            break
    return TF

              
def plot_clustering(TFbox, method = None, showfig = True, ax = None):
    
    """
    Plot clustering 
    
    Parameters:
        cluster: dictionary
            
        method: None, 1, 2, 3, 4
            method to be used for plotting. if None (default) the method contained in TFbox is taken
        
    """
    cluster = TFbox['cluster']
    if (method is None  and TFbox['method'] in [1, 2, 3, 4]):
        method = TFbox['method']
    elif (method is None  and TFbox['method'] not in [1, 2, 3, 4]):
        raise ValueError(" method in cluster not in [1, 2, 3, 4], proviade a method as input")
    elif (method is not None and method not in [1, 2, 3, 4]):
        raise ValueError(" Non valid input method. Valid valuer are: None, 1, 2, 3, 4")
    
    theta_coef = cluster.loc['theta_coef'].values
    alpha_coef = cluster.loc['alpha_coef'].values
    labels = cluster.loc['labels'].values
    
    ch_names = cluster.columns.tolist()
    colors = [colors_orig[label] for label in labels]
    if method in [1, 2]:
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,4))
        else: 
            fig = ax.get_figure()
        ax.scatter(np.arange(len(labels)), alpha_coef/theta_coef, c=colors)
        ax.grid()
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(ch_names, rotation = 'vertical')
        ax.set_xlabel(r'$\alpha/\theta$ coefficients')
        ax.set_ylabel(r'$\theta$ coefficients')
        ax.set_xlim(-1, len(labels))
        ax.set_ylim(0,max(alpha_coef/theta_coef)+(max(alpha_coef/theta_coef)-min(alpha_coef/theta_coef))/10)
        ax.scatter(-1,-1,color=colors_orig[0], label=r'$G_{\theta}$')
        ax.scatter(-1,-1,color=colors_orig[1], label=r'$G_{\alpha}$')
        ax.set_title('Method '+ str(method)+': '+meth_names[method])
        ax.legend()
        
        
    
    elif method in [3, 4]:
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else: 
            fig = ax.get_figure()
        
        ax.scatter(alpha_coef, theta_coef, c=colors)
        for i, txt in enumerate(ch_names):
            ax.annotate(txt, (alpha_coef[i], theta_coef[i]))
        ax.grid()
        ax.set_xlabel(r'$\alpha$ coefficients')
        ax.set_ylabel(r'$\theta$ coefficients')
        ax.set_xlim(min(alpha_coef)-(max(alpha_coef)-min(alpha_coef))/10,max(alpha_coef)+(max(alpha_coef)-min(alpha_coef))/10)
        ax.set_ylim(min(theta_coef)-(max(theta_coef)-min(theta_coef))/10,max(theta_coef)+(max(theta_coef)-min(theta_coef))/10)
        
        ax.scatter(-1,-1,color=colors_orig[0], label=r'$G_{\theta}$')
        ax.scatter(-1,-1,color=colors_orig[1], label=r'$G_{\alpha}$')
        ax.set_title('Method '+ str(method)+': '+meth_names[method])
        ax.legend()
        
        
    if ax is None: fig.tight_layout()
    
    if showfig == False:
        plt.close(fig)
    
    return fig
          

def plot_coefficients(psds, freqs, ch_names, theta_range = None, alpha_range = None, showfig=True):
    
    
    """
    Plot coefficients
    
    Parameters:
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        ch_names: list of strings
            name of the channels
        theta_range: tuple | list | array
            theta range to compute alpha coefficients (eg: [5,7])
        alpha_range: tuple | list | array
            alpha range to compute theta coefficients (eg: [9,11])
    
    """
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    # individual alpha peak
    f_in_idx = np.where((freqs>=7)&(freqs<=13))[0]
    AP = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    AP = freqs[AP]
    
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

    
    fig = plt.figure(figsize=(15,4))
    
    ax = plt.subplot2grid((1,5), (0, 0), colspan=3,rowspan=1)
    ax.scatter(np.arange(len(ch_names)), alpha_coef/theta_coef)
    ax.grid()
    ax.set_xticks(np.arange(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation = 'vertical')
    ax.set_xlabel(r'channels')
    ax.set_ylabel(r'$\alpha/\theta$ coefficients')
    ax.set_xlim(-1, len(ch_names))
    ax.set_ylim(0,max(alpha_coef/theta_coef)+(max(alpha_coef/theta_coef)-min(alpha_coef/theta_coef))/10)
    
    ax = plt.subplot2grid((1,5), (0, 3), colspan=2,rowspan=1)
    ax.scatter(alpha_coef, theta_coef)
    for i, txt in enumerate(ch_names):
            ax.annotate(txt, (alpha_coef[i], theta_coef[i]))
    ax.grid()
    
    ax.set_xlabel(r'$\alpha$ coefficients')
    ax.set_ylabel(r'$\theta$ coefficients')
    ax.set_xlim(min(alpha_coef)-(max(alpha_coef)-min(alpha_coef))/10,max(alpha_coef)+(max(alpha_coef)-min(alpha_coef))/10)
    ax.set_ylim(min(theta_coef)-(max(theta_coef)-min(theta_coef))/10,max(theta_coef)+(max(theta_coef)-min(theta_coef))/10)
        
    fig.tight_layout()
    
    if showfig == False:
        plt.close(fig)
        
    return fig
        
        
def plot_TF(psds, freqs, TFbox, showfig = True, ax = None):
    """
    Plot transition frequency
    
    Parameters:
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        TFbox
    
    """
    
    
        
    if TFbox['TF'] is None:
       raise ValueError("Cannot plot TF because its value is None. Please compute TF before using this function") 
    
    # Normalize power spectrum 
    psds = psds/psds.sum(axis=1).reshape((psds.shape[0],1))
    
    theta_coef = TFbox['cluster'].loc['theta_coef'].values
    alpha_coef = TFbox['cluster'].loc['alpha_coef'].values
    labels = TFbox['cluster'].loc['labels'].values
    method = TFbox['method']
    theta_idx = np.where(labels == 0)[0]
    alpha_idx = np.where(labels == 1)[0]
    
    theta_psds = (psds[theta_idx,:]*(theta_coef[theta_idx]/theta_coef[theta_idx].sum()).reshape(-1,1)).sum(axis=0)
    alpha_psds = (psds[alpha_idx,:]*(alpha_coef[alpha_idx]/alpha_coef[alpha_idx].sum()).reshape(-1,1)).sum(axis=0)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else: 
        fig = ax.get_figure()
        
    ax.plot(freqs, theta_psds, c=colors_orig[0], label = r'$G_{\theta}$')
    ax.plot(freqs, alpha_psds, c=colors_orig[1], label = r'$G_{\alpha}$')
    ax.axvline(TFbox['TF'], c='k', label = 'TF = '+str(np.round(TFbox['TF'],decimals=2))+' Hz')
    ax.grid()
    ax.set_title('Method '+ str(method)+': '+meth_names[method])
    ax.legend()
    ax.set_xlim(min(freqs),20)
    ax.set_xlabel('frequency [Hz]')
    if ax is None: fig.tight_layout()
    
    if showfig == False:
        plt.close(fig)
    
    return fig
        
        
    
        
def plot_chs(TFbox, ch_locs, method = None, showfig = True, ax = None):
    
    """
    Plot clustered channels oh head surface
    
    Parameters:
        TFbox: 
        ch_locs: array
            channels locations (unit of measure: mm), shape: number of channels x 3
        method: None, 1, 2, 3, 4
    
    """
    
    
    if (method is None  and TFbox['method'] in [1, 2, 3, 4]):
        method = TFbox['method']
    elif (method is None  and TFbox['method'] not in [1, 2, 3, 4]):
        raise ValueError(" method in cluster not in [1, 2, 3, 4], proviade a method as input")
    elif (method is not None and method not in [1, 2, 3, 4]):
        raise ValueError(" Non valid input method. Valid valuer are: None, 1, 2, 3, 4")
    
    
    theta_coef = TFbox['cluster'].loc['theta_coef'].values
    alpha_coef = TFbox['cluster'].loc['alpha_coef'].values
    labels = TFbox['cluster'].loc['labels'].values
    ch_names = TFbox['cluster'].columns.tolist()
    theta_idx = np.where(labels == 0)[0]
    alpha_idx = np.where(labels == 1)[0]
    
    kw_top = dict(margin=0.2, chan_offset=(0., 0.1, 0.), chan_size=12, line_color='black', line_width = 5)
    kw_cbar = dict(cbtxtsz=16, txtsz=14., width=.3, txtcolor='black', cbtxtsh=1.8,
                   rect=(0., -2., 1., 4.), border=True)
        
    
    
    sc = SceneObj(bgcolor='white', size=(1600, 1000),verbose=False)
    
    if method in [1,2]:
        # Theta coefficient
        
        radius_1 = theta_coef[theta_idx]
        # Create the topoplot and the object :
        t_obj_1 = TopoObj('topo', alpha_coef/theta_coef, channels=ch_names,  xyz=ch_locs*1000, ch_idx = theta_idx,radius = radius_1,
                          clim=(min(alpha_coef/theta_coef), max(alpha_coef/theta_coef)), chan_mark_color='red' , **kw_top)
        cb_obj_1 = ColorbarObj(t_obj_1, cblabel=r'alpha/theta coefficients', **kw_cbar)
        # Add the topoplot and the colorbar to the scene :
        sc.add_to_subplot(t_obj_1, row=0, col=0, title_color='black', width_max=600,height_max=800,
                          title='Method '+ str(method)+': '+meth_names[method-1]+'\n \n                            G theta',
                          title_size=24.0)
        sc.add_to_subplot(cb_obj_1, row=0, col=1, width_max=100,height_max=800)
        
     
        # Alpha coefficient
        radius_2 = alpha_coef[alpha_idx]
        t_obj_2 = TopoObj('topo', alpha_coef/theta_coef, channels=ch_names, xyz=ch_locs*1000,ch_idx = alpha_idx, radius = radius_2,
                          clim=(min(alpha_coef/theta_coef), max(alpha_coef/theta_coef)), chan_mark_color='red' ,**kw_top)
        cb_obj_2 = ColorbarObj(t_obj_2, cblabel=r'alpha/theta coefficients', **kw_cbar)
        # Add the topoplot and the colorbar to the scene :
        sc.add_to_subplot(t_obj_2, row=0, col=2, title_color='black', width_max=600,height_max=800,
                          title='\n \n                            G alpha',
                          title_size=24.0)
        sc.add_to_subplot(cb_obj_2, row=0, col=3, width_max=100,height_max=800)
         
       
    
    elif method in [3,4]:
        # Theta coefficient
    
        # Create the topoplot and the object :
        t_obj_1 = TopoObj('topo', theta_coef, channels=ch_names,  xyz=ch_locs*1000, ch_idx = theta_idx,
                          clim=(min(theta_coef), max(theta_coef)), chan_mark_color='red' , **kw_top)
    
        cb_obj_1 = ColorbarObj(t_obj_1, cblabel='Theta coefficients', **kw_cbar)
       
        # Add the topoplot and the colorbar to the scene :
        sc.add_to_subplot(t_obj_1, row=0, col=0, title_color='black', width_max=600,height_max=800,
                          title='Method '+ str(method)+': '+meth_names[method-1]+'\n \n                            G theta',
                          title_size=24.0)
       
        sc.add_to_subplot(cb_obj_1, row=0, col=1, width_max=100,height_max=800)
        
        # Alpha coefficient
        
        t_obj_2 = TopoObj('topo', alpha_coef, channels=ch_names, xyz=ch_locs*1000,ch_idx = alpha_idx,
                          clim=(min(alpha_coef), max(alpha_coef)), chan_mark_color='red' ,**kw_top)
        
        cb_obj_2 = ColorbarObj(t_obj_2, cblabel='Alpha coefficients', **kw_cbar)
        
        # Add the topoplot and the colorbar to the scene :
        sc.add_to_subplot(t_obj_2, row=0, col=2, title_color='black', width_max=600,height_max=800,
                          title='\n \n                            G alpha',
                          title_size=24.0)
        
        sc.add_to_subplot(cb_obj_2, row=0, col=3, width_max=100,height_max=800)
        
        
    
    
    image = sc.render()
       
    i_left = 0
    i_right = image.shape[1]-1
    i_top = 0
    i_bottom = image.shape[0]-1
    
    while np.array_equal(image[:,i_left,:],image[:,0,:]):
        i_left += 1
        
    while np.array_equal(image[:,i_right,:],image[:,-1,:]):
        i_right -= 1
        
    while np.array_equal(image[i_top,:,:],image[0,:,:]):
        i_top += 1
        
    while np.array_equal(image[i_bottom,:,:],image[-1,:,:]):
        i_bottom -= 1
        
    crop_sc = sc.render()[i_top-10:i_bottom+10,i_left-10:i_right+10]
        
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else: 
        fig = ax.get_figure()
        
    ax.imshow(crop_sc, interpolation = 'none')
    #ax.set_visible(False)
    ax.axis('off')
    
    if showfig == False:
        plt.close(fig)       
        
    return fig
        
       
        
        
def plot_TF_klimesch(psds_rest,psds_task, freqs, TF, showfig = True, ax = None):
    """
    Plot transition frequency
    
    Parameters:
        psds: array
            cross power spectral matrix (N_sources x N_freqs)
        freqs: array
            frequncies at which the psds is computed
        TFbox
    
    """
    # Normalize power spectrum 
    psds_rest = psds_rest/psds_rest.sum(axis=1).reshape((psds_rest.shape[0],1))
    psds_task = psds_task/psds_task.sum(axis=1).reshape((psds_task.shape[0],1))
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else: 
        fig = ax.get_figure()
        
    ax.plot(freqs, psds_task.mean(axis=0), c=colors_orig[0], label = r'$G_{task}$')
    ax.plot(freqs, psds_rest.mean(axis=0), c=colors_orig[1], label = r'$G_{rest}$')
    ax.axvline(TF, c='k', label = 'TF = '+str(np.round(TF,decimals=2))+' Hz')
    ax.grid()
    ax.set_title('Klimesch''s method')
    ax.legend()
    ax.set_xlim(min(freqs),20)
    ax.set_xlabel('frequency [Hz]')
    if ax is None: fig.tight_layout()
    
    if showfig == False:
        plt.close(fig)
    
    return fig
       
        
        
