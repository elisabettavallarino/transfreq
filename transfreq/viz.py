import numpy as np
import matplotlib.pyplot as plt
import warnings
from visbrain.objects import ColorbarObj, SceneObj
from .TopoObj_mod import TopoObj

# Define colors for plots
colors_orig = ['C0', 'C1', '#aef956', 'C3', 'C4', 'C5']
# Extended names of the clustering methods
meth_names = ['1D thresholding', '1D mean-shift', '2D k-means', '2D adjusted k-means']


def plot_clusters(tfbox, mode=None, order='standard', showfig=True, ax=None):
    """
    Plot clusters

    Parameters:
        tfbox: dictionary
            output of either create_cluster, compute_transfreq_manual or compute_transfreq
        mode: None, '1d', '2d' (default None)
            mode to be used for plotting. If '1d' the ratio between alpha and
            theta coefficients will be plotted. If '2d' alpha and theta coefficients
            will be plotted on the plane. If None (default) and the method contained
            in tfbox is 1, 2, 3 or 4 the mode will be set automatically.
        order: 'standard' or 'sorted' (default 'standard')
            Way to plot the coefficients (only when mode='1d' or the methods
            contained in the tfbox is 1 or 2, otherwise is not considered)
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing the clustering
    """

    if (mode == '2d' or (mode is None and tfbox['method'] in [3, 4])) and order == 'sorted':
        warnings.warn(
            "order can only be 'standard' if mode is '2d' or mode is None and method in tfbox is 3 or 4. Automatically switched to 'standard'")
        order = 'standard'

    cluster = tfbox['cluster']

    if mode is None and tfbox['method'] not in [1, 2, 3, 4]:
        raise ValueError(" provide a mode, either '1d' or '2d' ")
    method = tfbox['method']

    theta_coef = cluster.loc['theta_coef'].values
    alpha_coef = cluster.loc['alpha_coef'].values
    labels = cluster.loc['labels'].values

    ch_names = cluster.columns.tolist()

    if (mode == '1d') or ((mode is None) and (method in [1, 2])):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        else:
            fig = ax.get_figure()

        if order == 'standard':
            colors = [colors_orig[label] for label in labels]
            ax.scatter(np.arange(len(labels)), alpha_coef / theta_coef, c=colors)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(ch_names, rotation='vertical')
        if order == 'sorted':
            sort_idx = np.argsort(alpha_coef / theta_coef)
            colors = [colors_orig[labels[idx]] for idx in sort_idx]
            ax.scatter(np.arange(len(ch_names)), (alpha_coef / theta_coef)[sort_idx], c=colors)
            ax.set_xticks(np.arange(len(ch_names)))
            ax.set_xticklabels([ch_names[idx] for idx in sort_idx], rotation='vertical')

        ax.grid()

        ax.set_xlabel(r'channel')
        ax.set_ylabel(r'$\alpha/\theta$ coefficient')
        ax.set_xlim(-1, len(labels))
        ax.set_ylim(0,
                    max(alpha_coef / theta_coef) + (max(alpha_coef / theta_coef) - min(alpha_coef / theta_coef)) / 10)
        ax.scatter(-1, -1, color=colors_orig[0], label=r'$G_{\theta}$')
        ax.scatter(-1, -1, color=colors_orig[1], label=r'$G_{\alpha}$')
        if method in [1, 2, 3, 4]:
            ax.set_title('Method ' + str(method) + ': ' + meth_names[method - 1])
        else:
            ax.set_title('Method ' + str(method))
        ax.legend()

    elif (mode == '2d') or ((mode is None) and (method in [3, 4])):
        colors = [colors_orig[label] for label in labels]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        else:
            fig = ax.get_figure()

        ax.scatter(alpha_coef, theta_coef, c=colors)
        for i, txt in enumerate(ch_names):
            ax.annotate(txt, (alpha_coef[i], theta_coef[i]))
        ax.grid()
        ax.set_xlabel(r'$\alpha$ coefficient')
        ax.set_ylabel(r'$\theta$ coefficient')
        ax.set_xlim(min(alpha_coef) - (max(alpha_coef) - min(alpha_coef)) / 10,
                    max(alpha_coef) + (max(alpha_coef) - min(alpha_coef)) / 10)
        ax.set_ylim(min(theta_coef) - (max(theta_coef) - min(theta_coef)) / 10,
                    max(theta_coef) + (max(theta_coef) - min(theta_coef)) / 10)

        ax.scatter(-1, -1, color=colors_orig[0], label=r'$G_{\theta}$')
        ax.scatter(-1, -1, color=colors_orig[1], label=r'$G_{\alpha}$')
        if method in [1, 2, 3, 4]:
            ax.set_title('Method ' + str(method) + ': ' + meth_names[method - 1])
        else:
            ax.set_title('Method ' + str(method))
        ax.legend()

    if ax is None:
        fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig


def plot_coefficients(psds, freqs, ch_names, theta_range=None, alpha_range=None, mode='2d', order='standard',
                      showfig=True, ax=None):
    """
    Plot alpha and theta coefficients

    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix
        freqs: array, shape (N_freqs,)
            frequencies at which the psds is computed
        ch_names: None | list of strings (default None)
            name of the channels (must be ordered as they are in psds)
        theta_range: tuple | list | array (default (5,7))
            theta range to compute alpha coefficients (eg: (5,7), [5,7], np.array([5,7]))
        alpha_range: tuple | list | array (default (ap-1,ap+1))
            alpha range to compute theta coefficients (eg: (9,11), [9,11], np.array([9,11]))
        mode: '1d' or '2d' (default '2d')
            If '1d' the ratio between the alpha tha theta coefficients will be shown, if
            '2d' alpha and theta coefficients ar displayed on the plane
        order: 'standard' or 'sorted' (default 'standard')
            Way to plot the coefficients (only when mode='1d', otherwise is not considered)
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing the coefficients
    """
    if mode == '2d' and order == 'sorted':
        warnings.warn("if mode is '2d', order can oly be 'standard'. Automatically switched to 'standard'")
        order = 'standard'
    # Normalize power spectrum
    psds = psds / psds.sum(axis=1).reshape((psds.shape[0], 1))
    
    # define channel names if not given as input 
    if ch_names is None: ch_names = ['ch'+'0'*(len(str(psds.shape[0]))-len(str(ch_idx+1)))+str(ch_idx+1) for ch_idx in range(psds.shape[0])]
    
    
    # individual alpha peak
    f_in_idx = np.where((freqs >= 7) & (freqs <= 13))[0]
    ap = f_in_idx[0] + np.argmax(psds.mean(axis=0)[f_in_idx])
    ap = freqs[ap]

    if alpha_range is None:
        f_alpha_idx = np.where((freqs >= ap - 1) & (freqs <= ap + 1))[0]
    elif len(alpha_range) != 2:
        raise ValueError("len(alpha_range) must be 2")
    elif alpha_range[0] < freqs[0] or alpha_range[-1] > freqs[-1]:
        raise ValueError("alpha_range must be inside the interval [freqs[0], freqs[-1]]")
    elif alpha_range[0] > alpha_range[-1]:
        raise ValueError("alpha_range[-1] must be greater than alpha_range[0]")
    else:
        f_alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]

    if theta_range is None:
        if ap - 1 > 7:
            f_theta_idx = np.where((freqs >= 5) & (freqs <= 7))[0]
        else:
            f_theta_idx = np.where((freqs >= ap - 3) & (freqs < ap - 1))[0]
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

    # generate figure
    if mode == '1d':
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        else:
            fig = ax.get_figure()

        if order == 'standard':
            ax.scatter(np.arange(len(ch_names)), alpha_coef / theta_coef)

            ax.set_xticks(np.arange(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation='vertical')
        if order == 'sorted':
            sort_idx = np.argsort(alpha_coef / theta_coef)
            ax.scatter(np.arange(len(ch_names)), (alpha_coef / theta_coef)[sort_idx])

            ax.set_xticks(np.arange(len(ch_names)))
            ax.set_xticklabels([ch_names[idx] for idx in sort_idx], rotation='vertical')

        ax.grid()
        ax.set_xlabel(r'channel')
        ax.set_ylabel(r'$\alpha/\theta$ coefficient')
        ax.set_xlim(-1, len(ch_names))
        ax.set_ylim(0,
                    max(alpha_coef / theta_coef) + (max(alpha_coef / theta_coef) - min(alpha_coef / theta_coef)) / 10)

    if mode == '2d':
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        else:
            fig = ax.get_figure()

        ax.scatter(alpha_coef, theta_coef)
        for i, txt in enumerate(ch_names):
            ax.annotate(txt, (alpha_coef[i], theta_coef[i]))
        ax.grid()

        ax.set_xlabel(r'$\alpha$ coefficient')
        ax.set_ylabel(r'$\theta$ coefficient')
        ax.set_xlim(min(alpha_coef) - (max(alpha_coef) - min(alpha_coef)) / 10,
                    max(alpha_coef) + (max(alpha_coef) - min(alpha_coef)) / 10)
        ax.set_ylim(min(theta_coef) - (max(theta_coef) - min(theta_coef)) / 10,
                    max(theta_coef) + (max(theta_coef) - min(theta_coef)) / 10)

    fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig


def plot_channels(tfbox, ch_locs, mode=None, showfig=True, subfig=None):
    """
    Plot clustered channels on head topomap

    Parameters:
        tfbox: dictionary
            output of either create_cluster, compute_transfreq_manual or compute_transfreq
        ch_locs: array, shape (N_channels, 3)
            channels locations (unit of measure: mm),
        mode: None, '1d', '2d' (default None)
            mode to be used for plotting. If '1d' the ratio between alpha and
            theta coefficients will be plotted. If '2d' alpha and theta coefficients
            will be plotted on the plane. If None (default) and the method contained
            in tfbox is 1, 2, 3 of 4 the mode will be set automatically.
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing the clustered channels on head topomap
    """

    if mode is None and tfbox['method'] not in [1, 2, 3, 4]:
        raise ValueError(" provide a mode, either '1d' or '2d' ")
    
        
    method = tfbox['method']

    theta_coef = tfbox['cluster'].loc['theta_coef'].values
    alpha_coef = tfbox['cluster'].loc['alpha_coef'].values
    labels = tfbox['cluster'].loc['labels'].values
    ch_names = tfbox['cluster'].columns.tolist()
    theta_idx = np.where(labels == 0)[0]
    alpha_idx = np.where(labels == 1)[0]

    kw_top = dict(margin=0.2, chan_offset=(0., 0.1, 0.), chan_size=22, line_color='black', line_width=5)
    
    sc_alpha = SceneObj(bgcolor='white', size=(1000, 1000), verbose=False)
    sc_theta = SceneObj(bgcolor='white', size=(1000, 1000), verbose=False)
    
    if (mode == '1d') or ((mode is None) and (method in [1, 2])):
        # Theta coefficient

        radius_1 = theta_coef[theta_idx]
        # Create the topoplot and the object :
        t_obj_1 = TopoObj('topo', alpha_coef / theta_coef, channels=ch_names, xyz=ch_locs * 1000, ch_idx=theta_idx,
                          radius=radius_1,
                          clim=(min(alpha_coef / theta_coef), max(alpha_coef / theta_coef)), chan_mark_color='red',
                          **kw_top)
        # Add the topoplot and the colorbar to the scene :
        sc_theta.add_to_subplot(t_obj_1, row=0, col=0, title_color='black', width_max=600, height_max=800)
        
        # Alpha coefficient
        radius_2 = alpha_coef[alpha_idx]
        t_obj_2 = TopoObj('topo', alpha_coef / theta_coef, channels=ch_names, xyz=ch_locs * 1000, ch_idx=alpha_idx,
                          radius=radius_2,
                          clim=(min(alpha_coef / theta_coef), max(alpha_coef / theta_coef)), chan_mark_color='red',
                          **kw_top)
        # Add the topoplot and the colorbar to the scene :
        sc_alpha.add_to_subplot(t_obj_2, row=0, col=0, title_color='black', width_max=600, height_max=800)
    
    elif (mode == '2d') or ((mode is None) and (method in [3, 4])):
        # Theta coefficient

        # Create the topoplot and the object :
        t_obj_1 = TopoObj('topo', theta_coef, channels=ch_names, xyz=ch_locs * 1000, ch_idx=theta_idx,
                          clim=(min(theta_coef), max(theta_coef)), chan_mark_color='red', **kw_top)

        # Add the topoplot and the colorbar to the scene :
        sc_theta.add_to_subplot(t_obj_1, row=0, col=0, title_color='black', width_max=600, height_max=800)

        # Alpha coefficient

        t_obj_2 = TopoObj('topo', alpha_coef, channels=ch_names, xyz=ch_locs * 1000, ch_idx=alpha_idx,
                          clim=(min(alpha_coef), max(alpha_coef)), chan_mark_color='red', **kw_top)

        
        # Add the topoplot and the colorbar to the scene :
        sc_alpha.add_to_subplot(t_obj_2, row=0, col=0, title_color='black', width_max=600, height_max=800)

        

    
    # crop sc_theta image
    image_theta = sc_theta.render()
    i_left = 0
    i_right = image_theta.shape[1] - 1
    i_top = 0
    i_bottom = image_theta.shape[0] - 1
    while np.array_equal(image_theta[:, i_left, :], image_theta[:, 0, :]):
        i_left += 1
    while np.array_equal(image_theta[:, i_right, :], image_theta[:, -1, :]):
        i_right -= 1
    while np.array_equal(image_theta[i_top, :, :], image_theta[0, :, :]):
        i_top += 1
    while np.array_equal(image_theta[i_bottom, :, :], image_theta[-1, :, :]):
        i_bottom -= 1
    crop_sc_theta = sc_theta.render()[i_top - 10:i_bottom + 10, i_left - 10:i_right + 10]

    # crop sc_alpha image
    image_alpha = sc_alpha.render()
    i_left = 0
    i_right = image_alpha.shape[1] - 1
    i_top = 0
    i_bottom = image_alpha.shape[0] - 1
    while np.array_equal(image_alpha[:, i_left, :], image_alpha[:, 0, :]):
        i_left += 1
    while np.array_equal(image_alpha[:, i_right, :], image_alpha[:, -1, :]):
        i_right -= 1
    while np.array_equal(image_alpha[i_top, :, :], image_alpha[0, :, :]):
        i_top += 1
    while np.array_equal(image_alpha[i_bottom, :, :], image_alpha[-1, :, :]):
        i_bottom -= 1
    crop_sc_alpha = sc_alpha.render()[i_top - 10:i_bottom + 10, i_left - 10:i_right + 10]

    if method in [1, 2, 3, 4]:
        title = 'Method ' + str(method) + ': ' + meth_names[method - 1]
    else:
        title = 'Method ' + str(method)

    if (mode == '1d') or ((mode is None) and (method in [1, 2])):
        cb_label_theta = r'$\alpha/\theta$ coefficient'
        cb_label_alpha = r'$\alpha/\theta$ coefficient'
        vmin_theta = min(alpha_coef / theta_coef)
        vmax_theta = max(alpha_coef / theta_coef)
        vmin_alpha = min(alpha_coef / theta_coef)
        vmax_alpha = max(alpha_coef / theta_coef)
    elif (mode == '2d') or ((mode is None) and (method in [3, 4])):
        cb_label_theta = r'$\theta$ coefficient'
        cb_label_alpha = r'$\alpha$ coefficient'
        vmin_theta = min(theta_coef)
        vmax_theta = max(theta_coef)
        vmin_alpha = min(alpha_coef)
        vmax_alpha = max(alpha_coef)
        
    titlesize = 18
    suptitlesize = 18
    labelsize = 12
    if subfig is None:
        subfig, ax = plt.subplots(1, 2, figsize=(8, 4),constrained_layout=True)
    else:
        ax = subfig.subplots(1, 2)
        
        
        
    
    theta_plot = ax[0].imshow(crop_sc_theta,vmin=vmin_theta, vmax=vmax_theta, interpolation='none')
    cb_theta = subfig.colorbar(theta_plot, ax=ax[0],shrink=0.7)
    cb_theta.set_ticks([vmin_theta,vmax_theta])
    cb_theta.set_label(cb_label_theta,fontsize=labelsize)
    ax[0].set_title(r'$G_{\theta}$',fontsize=titlesize)
    alpha_plot = ax[1].imshow(crop_sc_alpha,vmin=vmin_alpha, vmax=vmax_alpha, interpolation='none')
    cb_alpha = subfig.colorbar(alpha_plot, ax=ax[1],shrink=0.7)
    cb_alpha.set_ticks([vmin_alpha,vmax_alpha])
    cb_alpha.set_label(cb_label_alpha,fontsize=labelsize)
    ax[1].set_title(r'$G_{\alpha}$',fontsize=titlesize)
    # ax.set_visible(False)
    ax[0].axis('off')
    ax[1].axis('off')
    subfig.suptitle(title,fontsize=suptitlesize)
    if showfig is False:
        plt.close(subfig)

    return subfig

def plot_transfreq(psds, freqs, tfbox, showfig=True, ax=None):
    """
    Plot alpha and theta power spectra profiles and the transition frequency

    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        tfbox: dictionary
            output of either create_cluster, compute_transfreq_manual or compute_transfreq
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing alpha and theta power spectra profiles and the transition frequency

    """

    if tfbox['tf'] is None:
        raise ValueError("Cannot plot tf because its value is None. Please compute tf before using this function")

        # Normalize power spectrum
    psds = psds / psds.sum(axis=1).reshape((psds.shape[0], 1))

    theta_coef = tfbox['cluster'].loc['theta_coef'].values
    alpha_coef = tfbox['cluster'].loc['alpha_coef'].values
    labels = tfbox['cluster'].loc['labels'].values
    method = tfbox['method']
    theta_idx = np.where(labels == 0)[0]
    alpha_idx = np.where(labels == 1)[0]

    theta_psds = (psds[theta_idx, :] * (theta_coef[theta_idx] / theta_coef[theta_idx].sum()).reshape(-1, 1)).sum(axis=0)
    alpha_psds = (psds[alpha_idx, :] * (alpha_coef[alpha_idx] / alpha_coef[alpha_idx].sum()).reshape(-1, 1)).sum(axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    ax.plot(freqs, theta_psds, c=colors_orig[0], label=r'$S_{\theta}$')
    ax.plot(freqs, alpha_psds, c=colors_orig[1], label=r'$S_{\alpha}$')
    ax.axvline(tfbox['tf'], c='k', label='TF=' + str(np.round(tfbox['tf'], decimals=2)) + ' Hz')
    ax.grid()
    if method in [1, 2, 3, 4]:
        ax.set_title('Method ' + str(method) + ': ' + meth_names[method - 1])
    else:
        ax.set_title('Method ' + str(method))
    ax.legend()
    ax.set_xlim(min(freqs), 20)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectrum (normalised)')
    if ax is None:
        fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig


def plot_transfreq_klimesch(psds_rest, psds_task, freqs, tf, showfig=True, ax=None):
    """
    Plot rest and task power spectra profiles and the transition frequency computed
    with Klimesch's method

    Parameters:
        psds_rest: array, shape (N_sensors, N_freqs)
            power spectral matrix of the resting state data
        psds_task: array, shape (N_sensors, N_freqs)
            power spectral matrix of the data recorded dunring a task execution
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        tf: scalar
            the tf computed with Klimesch's method (output of compute_transfreq_klimesch)
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing rest and task power spectra profiles and the
            transition frequencycomputed with Klimesch's method
    """

    # Normalize power spectrum
    psds_rest = psds_rest / psds_rest.sum(axis=1).reshape((psds_rest.shape[0], 1))
    psds_task = psds_task / psds_task.sum(axis=1).reshape((psds_task.shape[0], 1))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    ax.plot(freqs, psds_task.mean(axis=0), c=colors_orig[0], label=r'$S_{task}$')
    ax.plot(freqs, psds_rest.mean(axis=0), c=colors_orig[1], label=r'$S_{rest}$')
    ax.axvline(tf, c='k', label='TF=' + str(np.round(tf, decimals=2)) + ' Hz')
    ax.grid()
    ax.set_title("Klimesch's method")
    ax.legend()
    ax.set_xlim(min(freqs), 20)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectrum (normalised)')
    if ax is None:
        fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig


def plot_transfreq_minimum(psds_rest, freqs, tf, showfig=True, ax=None):
    """
    Plot rest power spectrum profile and the transition frequency computed with
    minimum method

    Parameters:
        psds_rest: array, shape (N_sensors, N_freqs)
            power spectral matrix of the resting state data
        psds_task: array, shape (N_sensors, N_freqs)
            power spectral matrix of the data recorded dunring a task execution
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        tf: scalar
            the tf computed with Klimesch's method (output of compute_transfreq_klimesch)
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing rest and task power spectra profiles and the
            transition frequencycomputed with Klimesch's method
    """

    # Normalize power spectrum
    psds_rest = psds_rest / psds_rest.sum(axis=1).reshape((psds_rest.shape[0], 1))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    ax.plot(freqs, psds_rest.mean(axis=0), c=colors_orig[1], label=r'$S_{rest}$')
    ax.axvline(tf, c='k', label='TF=' + str(np.round(tf, decimals=2)) + ' Hz')
    ax.grid()
    ax.set_title("Minimum method")
    ax.legend()
    ax.set_xlim(min(freqs), 20)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectrum (normalised)')
    if ax is None:
        fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig


def plot_psds(psds, freqs, average=True, showfig=True, ax=None):
    """
    Plot normalised power spectrum

    Parameters:
        psds: array, shape (N_sensors, N_freqs)
            power spectral matrix
        freqs: array, shape (N_freqs,)
            frequncies at which the psds is computed
        average: bool (default True)
            if True the power spectrum is averaged over the channels. If False
            the power spectrum of each channel is dislayed
        showfig: bool (default True)
            if True figure will be showed, if False figure will not be showed
        ax: instance of Axes | None
            Axes to plot into. If None, axes will be created.

    Returns:
        fig: instance of Figure
            Figure representing rest and task power spectra profiles and the
            transition frequencycomputed with Klimesch's method

    """

    # Normalize power spectrum
    psds = psds / psds.sum(axis=1).reshape((psds.shape[0], 1))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if average is True:
        ax.plot(freqs, psds.mean(axis=0))
    if average is False:
        ax.plot(freqs, psds.T)
    ax.grid()
    ax.set_title("power spectrum")
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectrum (normalised)')
    if ax is None:
        fig.tight_layout()

    if showfig is False:
        plt.close(fig)

    return fig
