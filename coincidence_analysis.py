# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:41:36 2021

@author: mcman
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as SM
mpl.rc('image', cmap='jet')
plt.close('all')

# 'Ar_P_L_G_DVT000001'  # 'ar_Mike000000'  # "Ar_S_000001"  #
name = 'xe_2tdc_horiz_10_000000'  # "9_3 data\\N_P_L_G000001"
in_name = name+'_cluster.h5'

n = 128
tof_range = [4720, 4752]
rt = [4325, 4425]

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
    tofs = [[]]*len(tof_index)

    for [j, i] in enumerate(tof_corr[tof_index]):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')
        if idxl < idxr:
            xs[j] = x[idxl:idxr]
            ys[j] = y[idxl:idxr]
            ts[j] = t_e[idxl:idxr]
            tofs[j] = np.array([t_i[tof_index][j]]*(idxr-idxl))

    xs = np.array([a for ll in xs for a in ll])
    ys = np.array([a for ll in ys for a in ll])
    ts = np.array([a for ll in ts for a in ll])
    tofs = np.array([a for ll in tofs for a in ll])

    index = np.where(np.logical_and(ts > rt[0], ts < rt[1]))[0]
    xs = xs[index]
    ys = ys[index]
    ts = ts[index]
    tofs = tofs[index]

    plt.figure(1)
    plt.hist(t_i[tof_index], range=tof_range, bins=200)

    plt.figure(2)
    plt.subplot(223)
    plt.hist2d(ys, xs, bins=n, range=[[0, 256], [0, 256]])
    plt.xlabel("y")
    plt.ylabel("x")

    print(tof_range)

    plt.subplot(224)
    plt.hist2d(tofs, xs, bins=n, range=[tof_range, [0, 256]])
    plt.xlabel("t")
    plt.ylabel("x")

    plt.subplot(221)
    plt.hist2d(ys, tofs, bins=n, range=[[0, 256], tof_range])
    plt.xlabel("y")
    plt.ylabel("t")
    plt.ylabel("t")


plt.figure(5)
ax = plt.axes(projection='3d')
h, edges = np.histogramdd((xs, ys, tofs), bins=n, range=[
                          [0, 256], [0, 256], tof_range])
xs, ys, zs = np.meshgrid(edges[0][:-1]+edges[0][1]/2,
                         edges[1][:-1]+edges[1][1]/2, edges[2][:-1]+edges[2][1]/2)

xs, ys, zs, h = (xs.flatten(), ys.flatten(), zs.flatten(), h.flatten())

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("ToF (ns)")
index = np.where(h > 1)

cm = SM().to_rgba(h[index])
#cm[:, 3] = h[index]/max(h[index])


ax.scatter3D(xs[index], ys[index], zs[index], color=cm, s=h[index])
# b=plt.hist(y_s[index], bins=n,range=[0,256])#, range=rtc)
