# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:41:36 2021

@author: mcman
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.close('all')

name = 'LightMagFP0000001'
in_name = name+'_cluster_r.h5'

n = 256
tof_range = [0, 1e6]
rt = [251800, 252200]

with h5py.File(in_name, mode='r') as f:
    tdc_time = f['tdc_time'][()]
    tdc_type = f['tdc_type'][()]
    x = f['Cluster']['x'][()]
    y = f['Cluster']['y'][()]
    tot = f['Cluster']['tot'][()]
    toa = f['Cluster']['toa'][()]

    pulse_times = tdc_time[()][np.where(tdc_type == 1)]
    pulse_corr = np.searchsorted(pulse_times, toa[()])
    t_e = 1e-3*(toa[()]-pulse_times[pulse_corr-1])

    tof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
    tof_corr = np.searchsorted(pulse_times, tof_times)
    t_i = 1e-3*(tof_times-pulse_times[tof_corr-1])

    tof_index = np.where(np.logical_and(
        t_i > tof_range[0], t_i < tof_range[1]))[0]

    xs = [[]]*len(tof_index)
    ys = [[]]*len(tof_index)
    ts = [[]]*len(tof_index)

    for [j, i] in enumerate(tof_corr[tof_index]):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')
        if idxl < idxr:
            xs[j] = x[idxl:idxr]
            ys[j] = y[idxl:idxr]
            ts[j] = t_e[idxl:idxr]

    xs = np.array([a for ll in xs for a in ll])
    ys = np.array([a for ll in ys for a in ll])
    ts = np.array([a for ll in ts for a in ll])

    index = np.where(np.logical_and(ts > rt[0], ts < rt[1]))[0]
    xs = xs[index]
    ys = ys[index]
    ts = ts[index]

    print(len(tof_index))
    print(len(xs))

    plt.figure(1)
    plt.hist(t_i[tof_index], bins=100)

    plt.figure(2)
    plt.subplot(223)
    plt.hist2d(ys, xs, bins=n, range=[[0, 256], [0, 256]])
    plt.xlabel("y")
    plt.ylabel("x")

    plt.subplot(224)
    plt.hist2d(ts, xs, bins=n, range=[rt, [0, 256]])
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(221)
    plt.hist2d(ys, ts, bins=n, range=[[0, 256], rt])
    plt.xlabel("y")
    plt.ylabel("t")
