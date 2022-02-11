# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:35:54 2021

@author: mcman
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.close('all')


files = ["ar000000_cluster.h5"]
names = ["Data"]


offset = 252000
tof_range = [0, 16000]

plt.figure(0)

for i, fname in enumerate(files):
    with h5py.File(fname, mode='r') as f:
        tdc_time = f['tdc_time'][()]
        tdc_type = f['tdc_type'][()]
        pulse_times = tdc_time[()][np.where(tdc_type == 1)]
        print(len(pulse_times))
        tof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_i = 1e-3*(tof_times-pulse_times[tof_corr-1])-offset+4000*i

    plt.hist(t_i, bins=300, range=tof_range, label=names[i], density=False)
    print(len(t_i[np.where(t_i < 4000)]))


plt.title("Combined TOF Spectrum")
plt.xlabel("TOF (ns)")
plt.ylabel("Counts")
plt.legend()
plt.ylabel("Counts")
plt.legend()
