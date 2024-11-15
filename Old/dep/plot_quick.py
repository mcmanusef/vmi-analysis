import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import cv3_analysis
from minor_utils.tof_calibration import get_calibration

from cv3_analysis import load_cv3

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")

wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230601"
# wdir=r'C:\Users\mcman\Code\VMI\Data'
name="kr001_p"
file=r"C:\Users\mcman\Code\VMI\indev\test.cv3"
# file=r"C:\Users\mcman\Code\VMI\Data\75test1.cv3"
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
rt=(1e6,1e6-1e3)


data2={}
# file=os.path.join(wdir, fr"{name}.cv3")
with h5py.File(file) as f:
    for k in f.keys():
        data2[k] = f[k][()]
    etof=f['t_etof'][()]#/1000
    smeared_etof=np.asarray(list(map(cv3_analysis.smear,etof)))
    plt.figure("e-tof")
    plt.hist(smeared_etof, bins=1000,range=(0,1e6))
#%%
# plt.figure('Clustered x-y')
# # plt.hist2d(data2['x'],data2['y'],bins=256, range=((0,256),(0,256)))
#
# pulse_selection = data2['tof_corr'][np.argwhere(np.logical_and(data2['t_tof']>757500,data2['t_tof']<759000))]
# index=np.argwhere([x in pulse_selection for x in data2['cluster_corr']]).flatten()
# xf,yf,tf=data2['x'][index],data2['y'][index],data2['t'][index]/1000
#
# #%%
# plt.figure('Coincidence x-y')
# plt.hist2d(xf,yf,bins=256, range=((0,256),(0,256)), vmax=30)
#
# plt.figure('Coincidence x-t')
# plt.hist2d(xf,tf,bins=256, range=((0,256),(748.290,748.320)))
#
#%%
rx=ry=(0,256)
rt=(748290,748320)
rt=(748425,748495)
rt=(440,470)
center=(125,127,748460)
xf, yf, tf = load_cv3(file,raw=True,center=(0,0,0),width=1000000000000,smearing=0.00026)
plt.figure("x-y full")
plt.hist2d(xf,yf, bins=256, range=(rx,ry),vmax=100)
plt.figure("etof")
plt.hist(tf, bins=128, range=rt)
tf=tf*1000
# plt.figure("tf")
# plt.hist(tf, bins=1000,range=(0,1000))
#%%
# rx=ry=(0,256)
# rt=(280,320)
# rt=(520,540)
# x,y,t= map(np.asarray,zip(*[(a,b,c) for a,b,c in zip(xf,yf,tf) if True or rx[0]<a<rx[1] and ry[0]<b<ry[1] and rt[0]<c<rt[1]]))


# xf,yf,tf=data2['x'],data2['y'],data2['t']
x,y,t= map(np.asarray,zip(*[(a,b,c) for a,b,c in zip(xf,yf,tf) if True or rx[0]<a<rx[1] and ry[0]<b<ry[1] and rt[0]<c<rt[1]]))


#%%

plt.figure("x-y")
plt.hist2d(x,y,bins=512, range=(rx,ry))
plt.colorbar()
plt.figure("x-t")
plt.hist2d(x,t,bins=512, range=(rx,rt))
plt.figure("y-t")
plt.hist2d(y,t,bins=512,range=(ry,rt))

#%%
# file=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230705\hv.cv3"

with h5py.File(file) as f:
    itof=f["t_tof"][()]
    pulses=max(f['tof_corr'])
def cal(x): return (2.1691*x/1000-1623.6797)**2

plt.figure("i-tof")
plt.hist(get_calibration()(itof),bins=2000,range=(0,100),weights=np.ones_like(itof)*20/pulses)
plt.xlabel("m/q")
plt.ylabel("Counts (per shot per unit m/q)")

plt.figure("i-tof uc")
plt.hist((itof),bins=2000,range=(0,40e3),weights=np.ones_like(itof)*20/pulses)
#%%
# plt.figure()
# plt.hist(data2['t'], range=(0,1e6), bins=2048)



#%%
mlab.figure()
hist3d=np.histogramdd((x,y,t), range=(rx,ry,rt), bins=64)[0]
print(np.max(hist3d))
mlab.contour3d(hist3d**0.66, contours=10, transparent=True)
mlab.show()
#%%
