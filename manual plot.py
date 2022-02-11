# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:44:55 2022

@author: mcman
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rc('image', cmap='jet')
plt.close(fig=1)

files = ["ar0_cluster.h5", "bg000001_cluster.h5"]
names = ["Argon", "Background"]

offset = 000
tof_range = [0, 16000]

plt.figure(1)

for i, fname in enumerate(files):
    with h5py.File(fname, mode='r') as f:
        t_i = f["t_tof"][()]
        pulses = f["pulse_times"][()]
    plt.hist(t_i, bins=1000, range=tof_range,
             label=names[i])  # , weights=1/len(pulses)*np.ones_like(t_i))

    print(names[i], ':')
    hits = len(np.where(np.logical_and(
        t_i > tof_range[0], t_i < tof_range[1]))[0])
    print(hits, "TOF Hits")
    print(len(t_i), "Total")
    print(len(pulses), "Pulses")
    print(hits/len(pulses), "Hits Per Shot")
    print("")

plt.title("Ion TOF Spectrum")
plt.xlabel("TOF (ns)")
plt.ylabel("Counts")
plt.legend()
