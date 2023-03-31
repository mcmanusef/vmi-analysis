# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:27:57 2023

@author: mcman
"""
# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

# %%

data_raw = {}
with h5py.File('ar001.h5') as f:
    for k in f.keys():
        data_raw[k] = np.array(list(f[k][()]))

data_clustered = {}
with h5py.File('xe.cv3') as f:
    for k in f.keys():
        data_clustered[k] = np.array(list(f[k][()]))

# %%
plt.close(fig='all')
print(data_raw.keys())
# print(data_clustered.keys())

# rp=data_raw['tdc_time'][data_raw['tdc_type']==1]/1e3
# fp=data_raw['tdc_time'][data_raw['tdc_type']==2]/1e3

# if fp[0]<rp[0]:
#     fp=fp[1:]
#     rp=rp[:-1]

# pulses= rp[np.where(fp-rp>1000)]

# plt.figure(1)
# plt.hist(fp-rp, bins=100)#, range=[0,1e4])

rising = data_raw['tdc_time'][data_raw['tdc_type'] == 3]/1e3
falling = data_raw['tdc_time'][data_raw['tdc_type'] == 4]/1e3

if falling[0] < rising[0]:
    falling = falling[1:]
while len(rising) > len(falling):
    rising = rising[:-1]

# pulse_index=np.searchsorted(pulses, rising, side='right')-1

# p_num, counts= tuple(map(np.array,zip(*[(k, len(list(g))) for k,g in it.groupby(pulse_index)])))
# # hits=[np.sum(np.where(pulse_index==x)) for x in multi_hits]

# multi_hits=p_num[np.where(counts>1)]
# multi_hit_counts=counts[np.where(counts>1)]

e_diff = np.diff(rising)
e_len = (falling-rising)[:-1]

plt.figure()
to_show = 100
plt.eventplot(rising[:to_show])
# _=plt.hist(np.diff(data_raw['tdc_time']/1000), bins=1000, range=[0,1e6])

plt.figure("e_diff")
_ = plt.hist(e_diff, bins=1000, range=[0, 2e6])

plt.figure("e_len")
_ = plt.hist(e_len, bins=1000, range=[0, 100])

# plt.figure()
# plt.hist2d(e_diff, e_len, bins=300, range=[[0,2000],[0,50]])
# plt.colorbar()

plt.figure("etof")
def smear(x): return x+np.random.random_sample(size=x.shape)*0.26


_ = plt.hist(smear(data_clustered['t_etof']/1000), bins=1000, range=[400, 800])

# # plt.figure("itof")
# # _=plt.hist(data_clustered['t_tof']/1000,bins=1000, range=[0,3e4])
