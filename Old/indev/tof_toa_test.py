# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:55:29 2023

@author: mcman
"""
import functools

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.stats import pearsonr
matplotlib.use('Qt5Agg')


def smear(array, amount=0.26):
    noise = np.random.rand(len(array))*amount
    return array+noise


def create_numpy_array_from_lists(values, indices, length):
    arr = np.zeros(length)  # if max(indices) is 5, array should be of size 6 (0-5)
    arr[indices] = values
    return arr


# Open the HDF5 file
with h5py.File(r"C:\Users\mcman\Code\VMI\indev\test.cv3", 'r') as f:
    # Get the data
    x, y = f['x'][()], f['y'][()]
    t = smear(f['t'][()], amount=0)
    t_corr = f['cluster_corr'][()]
    etof = smear(f['t_etof'][()])
    tof_corr = f['etof_corr'][()]

length=max(max(tof_corr),max(t_corr))+1

# x=list(create_numpy_array_from_lists(x, t_corr))
# y=list(create_numpy_array_from_lists(y, t_corr))
#%%
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
plt.tight_layout()
ax[1, 0].axis('off')

rtof = [520, 540]
rtoa = [270, 310]
xc,yc=(128,128)
width=20

tof = create_numpy_array_from_lists(etof, tof_corr, length=length)
toa = create_numpy_array_from_lists(t, t_corr,length=length)

x=create_numpy_array_from_lists(x, t_corr,length=length)-xc
y=create_numpy_array_from_lists(y, t_corr,length=length)-yc



idx=functools.reduce(np.logical_and,
    [rtof[0]<tof,rtof[1]>tof,rtoa[0]<toa,rtoa[1]>toa,x**2+y**2<width**2]
)

ftof,ftoa=tof[idx],toa[idx]


ax[0, 1].hist2d(ftof, ftoa, bins=100)

plt.xlabel("ToF (ns)")
plt.sca(ax[0, 0])
plt.ylabel("ToA (ns)")
plt.tight_layout()
for rtof in [(i, i+2) for i in range(520, 540, 2)]:

    idx=functools.reduce(np.logical_and,[rtof[0]<tof,rtof[1]>tof,rtoa[0]<toa,rtoa[1]>toa,x**2+y**2<width**2])
    ftof,ftoa=tof[idx],toa[idx]

    ax[0, 0].hist(ftoa, bins=(rtoa[1]-rtoa[0])*60, range=rtoa, orientation='horizontal')
    ax[1, 1].hist(ftof, bins=(rtof[1]-rtof[0])*10, range=rtof)
