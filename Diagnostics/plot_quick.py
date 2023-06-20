import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cv3_analysis
from minor_utils.tof_calibration import get_calibration

from cv3_analysis import load_cv3

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")

wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230601"
# wdir=r'C:\Users\mcman\Code\VMI\Data'
name="kr001_p"
file=r"C:\Users\mcman\Code\VMI\Data\clust_v3\xe005_e.cv3"
# file=os.path.join(wdir,fr"{name}.h5")
#
#
# data = {}
# with h5py.File(file) as f:
#     for k in f.keys():
#         data[k] = f[k][()]
#
# plt.figure("TDC Times")
# plt.plot(data['tdc_time'])
#
# plt.figure("ToA Times")
# plt.plot(data['toa'][::1000])
#
# plt.figure("TDC1 Pulse Lengths")
# if not data['tdc_type'][-1]==1:
#     lens=np.diff(data['tdc_time'])[np.argwhere(data['tdc_type']==1)]
# else:
#     lens=np.diff(data['tdc_time'])[np.argwhere(data['tdc_type']==1)[:-1]]
# plt.hist(lens,bins=500)
#
# plt.figure("TDC2 Pulse Lengths")
# plt.hist(np.diff(data['tdc_time'])[np.argwhere(data['tdc_type']==3)],bins=1000)
#
# plt.figure("Pulse Time Differences")
# pulse_times = (data['tdc_time'][np.argwhere(data['tdc_type'] == 1)][np.argwhere(lens > 1e6)[:, 0]]/1000).flatten()
# diffs=np.diff(pulse_times)
# plt.hist(np.diff(pulse_times), bins=300, range=(1e6,1e6+1e2))
# plt.yscale('log')
#
#
# plt.figure('Raw x-y')
# plt.hist2d(data['x'],data['y'],weights=data['tot'],bins=256, range=((0,256),(0,256)))
#%%
rx=ry=(1,256)
rt=(0,1e6)


data2={}
# file=os.path.join(wdir, fr"{name}.cv3")
with h5py.File(file) as f:
    # for k in f.keys():
    #     data2[k] = f[k][()]
    etof=f['t_etof'][()]/1000
    smeared_etof=np.asarray(list(map(cv3_analysis.smear,etof)))
    plt.figure("e-tof")
    plt.hist(etof, bins=1000,range=(748_430,748_430+0.26*1000/5))
#%%
# plt.figure('Clustered x-y')
# plt.hist2d(data2['x'],data2['y'],bins=256, range=((0,256),(0,256)))

# pulse_selection = data2['tof_corr'][np.argwhere(np.logical_and(data2['t_tof']>.764e9,data2['t_tof']<.766e9))]
# index=np.argwhere([x in pulse_selection for x in data2['cluster_corr']]).flatten()
# xf,yf,tf=data2['x'][index],data2['y'][index],data2['t'][index]/1000

# plt.figure('Coincidence x-y')
# plt.hist2d(xf,yf,bins=256, range=((0,256),(0,256)),norm='log')
#
#
center=(125,127,528.75)
xf, yf, tf = load_cv3(file,raw=True,center=center)
#%%
rx=ry=(-128,128)
rt=(-10,10)
# rx=ry=(0,256)
# rt=(520,540)

x,y,t= map(np.asarray,zip(*[(a,b,c) for a,b,c in zip(xf,yf,tf) if rx[0]<a<rx[1] and ry[0]<b<ry[1] and rt[0]<c<rt[1]]))



#%%

plt.figure("x-y")
plt.hist2d(x,y,bins=512, range=(rx,ry),norm=matplotlib.colors.PowerNorm(0.5))
plt.figure("x-t")
plt.hist2d(x,t,bins=256, range=(rx,rt))
plt.figure("y-t")
plt.hist2d(y,t,bins=256,range=(ry,rt))

#%%
with h5py.File(os.path.join(wdir, fr"{name}.cv3")) as f:
    itof=f["t_tof"][()]/1000
    pulses=max(f['tof_corr'])

plt.figure("i-tof")
plt.hist((itof),bins=500,range=(750e3, 780e3),weights=np.ones_like(itof)*5/pulses)
plt.xlabel("m/q")
plt.ylabel("Counts (per shot per unit m/q)")

#%%
plt.figure()
plt.hist(data2['y'], bins=4096)
#%%
