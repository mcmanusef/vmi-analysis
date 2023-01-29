# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:55:29 2023

@author: mcman
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def smear(array, amount=0.26):
    noise = np.random.rand(len(array))*amount
    return array+noise


def create_numpy_array_from_lists(values, indices):
    arr = np.zeros(max(indices) + 1)  # if max(indices) is 5, array should be of size 6 (0-5)
    arr[indices] = values
    return arr


# Open the HDF5 file
with h5py.File('s.cv3', 'r') as f:
    # Get the data
    x, y = f['x'][()], f['y'][()]
    t = smear(f['t'][()]/1e3, amount=1.6)
    t_corr = f['cluster_corr'][()]
    etof = smear(f['t_etof'][()]/1e3)
    tof_corr = f['etof_corr'][()]


# f1 = plt.figure()
# a = plt.hist2d(x, smear(t, amount=1.6), bins=256, range=[[0, 256], [280, 340]])
# # Plot the histogram
# f, (a1, a2) = plt.subplots(1, 2)
# a1.hist(smear(t, amount=1.6), bins=100, range=[280, 340])
# a2.hist(smear(etof), bins=100, range=[500, 560])
# # plt.show()

tof = list(create_numpy_array_from_lists(etof, tof_corr))
toa = list(create_numpy_array_from_lists(t, t_corr))
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
plt.tight_layout()
ax[1, 0].axis('off')

rtof = [520, 540]
rtoa = [300, 330]
ftof, ftoa, __ = zip(*list(filter(lambda x: rtof[0] < x[0] < rtof[1] and rtoa[0] < x[1] < rtoa[1] and not (194 < x[2] < 204), zip(tof, toa, x))))
ax[0, 1].hist2d(ftof, ftoa, bins=100)
# ax[0, 0].hist(ftoa, bins=(rtoa[1]-rtoa[0])*60, range=rtoa, orientation='horizontal', histtype='step', color='k')
# ax[1, 1].hist(ftof, bins=(rtof[1]-rtof[0])*10, range=rtof)

plt.xlabel("ToF (ns)")
plt.sca(ax[0, 0])
plt.ylabel("ToA (ns)")
plt.tight_layout()

# ax[0, 0].hist(ftoa, bins=(rtoa[1]-rtoa[0])*10, orientation='horizontal')
# ax[1, 1].hist(ftof, bins=(rtof[1]-rtof[0])*10)

for rtof in [(i, i+2) for i in range(520, 540, 2)]:

    ftof, ftoa, __ = zip(*list(filter(lambda x: rtof[0] < x[0] < rtof[1] and rtoa[0] < x[1] < rtoa[1] and not (194 < x[2] < 204), zip(tof, toa, x))))

    ax[0, 0].hist(ftoa, bins=(rtoa[1]-rtoa[0])*60, range=rtoa, orientation='horizontal')
    ax[1, 1].hist(ftof, bins=(rtof[1]-rtof[0])*10, range=rtof)
