# -*- coding: utf-8 -*-
"""
Created ozn Mon Jul 19 11:13:03 2021

@author: mcman
"""

import h5py
import matplotlib as mpl
mpl.rc('image', cmap='jet')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable as SM
plt.close('all')

n=256
rt=[251920,252120]
rtot=[0,50]

if 1:
    rtc=rt-np.average(rt)
else:
    rtc=[-20,20]

name='LightMagFP000001'#'LnGM_FP000000'#'Data2000000'#
in_name=name+'_cluster_r.h5'#'ar004'#'xe_2tdc_vert_8_000000'

use_cluster=1
with h5py.File(in_name,mode='r')as f:
    if use_cluster==0:
        fh5=f
    else:
        fh5=f['Cluster']
    toa=fh5['toa'][()]
    t=fh5['t'][()]
    x=fh5['x'][()]
    y=fh5['y'][()]
    
    tot=fh5['tot'][()]
    
plt.figure(-1)
thist=plt.hist(t,bins=n,range=rt)

if 0:
    maxi=10000
    x=x[0:maxi]
    y=y[0:maxi]
    t=t[0:maxi]
    tot=tot[0:maxi]

index=np.where(np.logical_and(t>rt[0], t<rt[1]))[0]

x_s=x[index]
y_s=y[index]
t_s=t[index]
tot_s=tot[index]

plt.figure(1)
plt.subplot(223)
plt.hist2d(y_s,x_s,bins=n,range=[[0,256],[0,256]])
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
plt.hist2d(t_s,x_s,bins=n,range=[rt,[0,256]])
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(221)
plt.hist2d(y_s,t_s,bins=n,range=[[0,256],rt])
plt.xlabel("y")
plt.ylabel("t")



totspace=np.linspace(rtot[0],rtot[1], num=n)
dtot=totspace[1]-totspace[0]
plt.figure(2)
a=plt.hist2d(t_s,tot_s,bins=n,range=[rt,rtot])
plt.xlabel("t")
plt.ylabel("tot")

t=np.linspace(rt[0],rt[1], num=n)
means=[np.average(t,weights=i**2) if sum(i)>0 else 0 for i in a[0].transpose()]
plt.plot(means,totspace)

t_correction=np.zeros_like(tot[index])
for i in range(n):
    corr=np.where(np.logical_and(tot[index]>=totspace[i],tot[index]<totspace[i]+dtot))
    t_correction[corr]=means[i]

tfix=t_s-t_correction

    
index=np.where(np.logical_and(t_s-t_correction>rtc[0], t_s-t_correction<rtc[1]))    

plt.figure(3)
a=plt.hist2d(tfix,tot_s,bins=n,range=[rtc,rtot],weights=tot_s)
plt.xlabel("t")
plt.ylabel("tot")


plt.figure(4)
plt.subplot(223)
plt.hist2d(y_s[index],x_s[index],bins=n,range=[[0,256],[0,256]])
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
plt.hist2d(tfix[index],x_s[index],bins=n,range=[rtc,[0,256]])
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(221)
a=plt.hist2d(y_s[index],tfix[index],bins=n,range=[[0,256],rtc])
plt.xlabel("y")
plt.ylabel("t")

# means=[np.average(a[2][:-1],weights=i**2) if sum(i)>0 else 0 for i in a[0]]
# plt.plot(a[1][:-1],means)


plt.figure(5)
ax=plt.axes(projection='3d')
h, edges=np.histogramdd((x_s[index],y_s[index],tfix[index]),bins=n,range=[[0,256],[0,256],rtc])
xs,ys,zs=np.meshgrid(edges[0][:-1]+edges[0][1]/2,edges[1][:-1]+edges[1][1]/2,edges[2][:-1]+edges[2][1]/2)

xs,ys,zs,h=(xs.flatten(),ys.flatten(),zs.flatten(),h.flatten())

index=np.where(h>5)

cm=SM().to_rgba(h[index])
cm[:,3]=h[index]/max(h[index])


ax.scatter3D(xs[index],ys[index],zs[index],color=cm,s=h[index]*2)
#b=plt.hist(y_s[index], bins=n,range=[0,256])#, range=rtc)