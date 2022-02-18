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
from datetime import datetime

mpl.rc('image', cmap='jet')
plt.close('all')

name = 'kr000002'
in_name = name+'_cluster.h5'

t0 = 252.2  # in us
tof_range = [15, 17]  # in us
etof_range = [0, 50]  # in ns

t0 = t0*1000
tof_range = np.array(tof_range)*1000
etof_range = np.array(etof_range)
# %% Loading Data

print('Loading Data:', datetime.now().strftime("%H:%M:%S"))

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
print('Starting e-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))

etof_index = np.where(np.logical_and(
    t_etof > etof_range[0], t_etof < etof_range[1]))[0]
xint = [[]]*len(etof_index)
yint = [[]]*len(etof_index)
tint = [[]]*len(etof_index)

for [j, i] in enumerate(etof_corr[etof_index]):
    if j % 10000 == 0:
        print("    {num:.2f}%".format(num=100*j/len(etof_index)))
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

print('Starting i-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))
tof_index = np.where(np.logical_and(
    t_tof > tof_range[0], t_tof < tof_range[1]))[0]

xs = [[]]*len(tof_index)
ys = [[]]*len(tof_index)
ts = [[]]*len(tof_index)
tofs = [[]]*len(tof_index)

for [j, i] in enumerate(tof_corr[tof_index]):
    if j % 10000 == 0:
        print("    {num:.2f}%".format(num=100*j/len(tof_index)))
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
print('Plotting:', datetime.now().strftime("%H:%M:%S"))
plt.figure(1)
plt.hist(tofs, bins=300, range=tof_range)

plt.figure(2)
plt.hist(ts, bins=min(int(np.diff(etof_range)*4), 300), range=etof_range)


plt.figure(5)
ax = plt.axes(projection='3d')
h, edges = np.histogramdd((xs, ys, ts), bins=256, range=[
                          [0, 256], [0, 256], etof_range])
xc, yc, zc = np.meshgrid(edges[0][:-1]+edges[0][1]/2,
                         edges[1][:-1]+edges[1][1]/2, edges[2][:-1]+edges[2][1]/2)

xc, yc, zc, h = (xc.flatten(), yc.flatten(), zc.flatten(), h.flatten())

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("ToF (ns)")
index = np.where(h > 5)

cm = SM().to_rgba(h[index])


ax.scatter3D(xc[index], yc[index], zc[index], color=cm, s=h[index]*3)
