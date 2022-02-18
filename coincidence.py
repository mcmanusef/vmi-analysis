# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:29:11 2022

@author: mcman
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as SM
mpl.rc('image', cmap='jet')
plt.close('all')

name = 'kr000000'
in_name = name+'_cluster.h5'

t0 = 0 * 252.2
tof_range = [0, 1000]
etof_range = [0, 1000]

t0 = t0*1000
tof_range = np.array(tof_range)*1000
etof_range = np.array(etof_range)*1000
# %% Loading Data
with h5py.File(in_name, mode='r') as f:
    x = f['Cluster']['x'][()]
    y = f['Cluster']['y'][()]
    pulse_corr = f['Cluster']['pulse_corr'][()]

    t_tof = f['t_tof'][()]
    t_tof = t_tof-t0
    tof_corr = f['tof_corr'][()]

    t_etof = f['t_etof'][()]
    t_etof = t_etof-t0
    etof_corr = f['etof_corr'][()]


# %% e-TOF Coincidence

etof_index = np.where(np.logical_and(
    t_etof > etof_range[0], t_etof < etof_range[1]))[0]
xint = [[]]*len(etof_index)
yint = [[]]*len(etof_index)
tint = [[]]*len(etof_index)

for [j, i] in enumerate(etof_corr[etof_index]):
    idxr = np.searchsorted(pulse_corr, i, side='right')
    idxl = np.searchsorted(pulse_corr, i, side='left')

    if idxl < idxr:
        xint[j] = x[idxl:idxr]
        yint[j] = y[idxl:idxr]
        tint[j] = np.array([t_etof[etof_index][j]]*(idxr-idxl))


xint = np.array([a for ll in xint for a in ll])
yint = np.array([a for ll in yint for a in ll])
tint = np.array([a for ll in tint for a in ll])

# %% i-TOF Coincidence

tof_index = np.where(np.logical_and(
    t_tof > tof_range[0], t_tof < tof_range[1]))[0]

xs = [[]]*len(tof_index)
ys = [[]]*len(tof_index)
ts = [[]]*len(tof_index)
tofs = [[]]*len(tof_index)

for [j, i] in enumerate(tof_corr[tof_index]):
    idxr = np.searchsorted(pulse_corr, i, side='right')
    idxl = np.searchsorted(pulse_corr, i, side='left')

    if idxl < idxr:
        xs[j] = xint[idxl:idxr]
        ys[j] = yint[idxl:idxr]
        ts[j] = tint[idxl:idxr]
        tofs[j] = np.array([t_tof[tof_index][j]]*(idxr-idxl))

xs = np.array([a for ll in xs for a in ll])
ys = np.array([a for ll in ys for a in ll])
ts = np.array([a for ll in ts for a in ll])
tofs = np.array([a for ll in tofs for a in ll])

# %% Plotting
plt.figure(1)
plt.hist(tofs, bins=300, range=tof_range)

plt.figure(2)
plt.hist(ts, bins=300, range=etof_range)
