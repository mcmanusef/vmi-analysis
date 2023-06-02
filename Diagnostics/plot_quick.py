import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from minor_utils.tof_calibration import get_calibration

from cv3_analysis import load_cv3

matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")

wdir=r"C:\Users\mcman\Code\VMI\Data"
name="kr002_s"
file=os.path.join(wdir,fr"{name}.h5")


data = {}
with h5py.File(file) as f:
    for k in f.keys():
        data[k] = f[k][()]

plt.figure("TDC Times")
plt.plot(data['tdc_time'])

plt.figure("ToA Times")
plt.plot(data['toa'][::1000])

asdjklfn
plt.figure("TDC1 Pulse Lengths")
lens=np.diff(data['tdc_time'])[np.argwhere(data['tdc_type']==1)]
plt.hist(lens,bins=500)

plt.figure("TDC2 Pulse Lengths")
plt.hist(np.diff(data['tdc_time'])[np.argwhere(data['tdc_type']==3)],bins=1000)

plt.figure("Pulse Time Differences")
pulse_times = (data['tdc_time'][np.argwhere(data['tdc_type'] == 1)][np.argwhere(lens > 1e6)[:, 0]]/1000).flatten()
diffs=np.diff(pulse_times)
plt.hist(np.diff(pulse_times), bins=300, range=(1e6,1e6+1e2))
plt.yscale('log')


plt.figure('Raw x-y')
plt.hist2d(data['x'],data['y'],weights=data['tot'],bins=256, range=((0,256),(0,256)))
#%%
rx=ry=(0,256)
rt=(0,1e6)
rt=(7.4845e5,7.48565e5)

data2={}
with h5py.File(os.path.join(wdir, fr"{name}.cv3")) as f:
    for k in f.keys():
        data2[k] = f[k][()]
plt.figure('Clustered x-y')
plt.hist2d(data2['x'],data2['y'],bins=256, range=((0,256),(0,256)))

pulse_selection = data2['tof_corr'][np.argwhere(np.logical_and(data2['t_tof']>.764e9,data2['t_tof']<.766e9))]
index=np.argwhere([x in pulse_selection for x in data2['cluster_corr']]).flatten()
xf,yf,tf=data2['x'][index],data2['y'][index],data2['t'][index]

plt.figure('Coincidence x-y')
# plt.hist2d(xf,yf,bins=256, range=((0,256),(0,256)),norm='log')
#
#
# xa, ya, ta = load_cv3(os.path.join(wdir, fr"{name}.cv3"),raw=True,center=(0,0,0))

x,y,t= map(np.asarray,zip(*[(a,b,c) for a,b,c in zip(xf,yf,tf) if rx[0]<a<rx[1] and ry[0]<b<ry[1] and rt[0]<c<rt[1]]))

plt.figure("e-tof")
plt.hist(t, bins=1000,range=rt)


plt.figure("x-y")
plt.hist2d(x,y,bins=256, range=(rx,ry))
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
