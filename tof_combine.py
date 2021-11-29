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

# ['Kr_P_L_G000000', 'Kr_P_L_G000001']
files = ['Ar_P_L_G_05kV000001', 'ar_Mike000000', 'Ar_P_L_G_DVT2000000',
         'Ar_P_L_G_25VT000000', 'Ar_P_L_G_3kV000000']
names = ['0.5 kV (F)', '1 kV (Last Week)', '2.0 kV (T)', '2.5kV (T)',
         '3.0kV (F)']  # ['Friday', 'Thursday']

files = ['9_3 data\\'+i+'_cluster.h5' for i in files]

offset = 252000
tof_range = [0, 10000]

plt.figure()

for i, fname in enumerate(files):
    with h5py.File(fname, mode='r') as f:
        tdc_time = f['tdc_time'][()]
        tdc_type = f['tdc_type'][()]
        pulse_times = tdc_time[()][np.where(tdc_type == 1)]
        tof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_i = 1e-3*(tof_times-pulse_times[tof_corr-1])-offset

    plt.hist(t_i, bins=300, range=tof_range, label=names[i], density=True)

plt.title("Combined TOF Spectrum")
plt.xlabel("TOF (ns)")
plt.ylabel("Counts")
plt.legend()
