# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:00:13 2022

@author: mcman
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.close(fig=1)

files = ["air000001.h5"]
names = ["Air1"]

offset = 252000
tof_range = [0, 4000]

plt.figure(1)

for i, fname in enumerate(files):
    with h5py.File(fname, mode='r') as f:
        tdc_time = f['tdc_time'][()]
        tdc_type = f['tdc_type'][()]
        times = tdc_time[()][np.where(tdc_type == 1)]
        lengths = np.diff(tdc_time)[np.where(tdc_type == 1)]
        pulse_times = times[np.where(lengths > 1e6)]
        tof_times = times[np.where(lengths < 1e6)]
        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_i = 1e-3*(tof_times-pulse_times[tof_corr-1])-offset
        print(len(pulse_times))
        print(len(tof_times))

    plt.hist(t_i, bins=300, range=tof_range, label=names[i], density=False)


plt.title("Combined TOF Spectrum")
plt.xlabel("TOF (ns)")
plt.ylabel("Counts")
plt.legend()
plt.ylabel("Counts")
plt.legend()
