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

n=128
rtoa=[144,400]
rtot=[0,100]

if 0:
    rtoac=rtoa-np.average(rtoa)
else:
    rtoac=[-20,20]

name='arLightin2.3-7000002'#'ar004'#'xe_2tdc_vert_8_000000'#
in_name=name+'_cluster_r.h5'

use_cluster=1
with h5py.File(in_name,mode='r')as f:
    if use_cluster==0:
        fh5=f
        toa=f['t'][()]
    else:
        fh5=f['Cluster']
        toa=f['Cluster']['toa'][()]
    x=fh5['x'][()]
    y=fh5['y'][()]
    
    tot=fh5['tot'][()]
    
plt.figure(-1)
toahist=plt.hist(toa,bins=n,range=rtoa)

if 0:
    maxi=10000
    x=x[0:maxi]
    y=y[0:maxi]
    toa=toa[0:maxi]
    tot=tot[0:maxi]

index=np.where(np.logical_and(toa>rtoa[0], toa<rtoa[1]))[0]

x_s=x[index]
y_s=y[index]
t_s=toa[index]
tot_s=tot[index]

plt.figure(1)
plt.subplot(223)
plt.hist2d(y_s,x_s,bins=256,range=[[0,256],[0,256]])
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
plt.hist2d(t_s,x_s,bins=n,range=[rtoa,[0,256]])
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(221)
plt.hist2d(y_s,t_s,bins=n,range=[[0,256],rtoa])
plt.xlabel("y")
plt.ylabel("t")



totspace=np.linspace(rtot[0],rtot[1], num=n)
dtot=totspace[1]-totspace[0]
plt.figure(2)
a=plt.hist2d(t_s,tot_s,bins=n,range=[rtoa,rtot])
plt.xlabel("toa")
plt.ylabel("tot")

t=np.linspace(rtoa[0],rtoa[1], num=n)
means=[np.average(t,weights=i**2) if sum(i)>0 else 0 for i in a[0].transpose()]
plt.plot(means,totspace)

toa_correction=np.zeros_like(tot[index])
for i in range(n):
    corr=np.where(np.logical_and(tot[index]>=totspace[i],tot[index]<totspace[i]+dtot))
    toa_correction[corr]=means[i]

tfix=t_s-toa_correction

    
index=np.where(np.logical_and(t_s-toa_correction>rtoac[0], t_s-toa_correction<rtoac[1]))    

plt.figure(3)
a=plt.hist2d(tfix,tot_s,bins=n,range=[rtoac,rtot],weights=tot_s)
plt.xlabel("toa")
plt.ylabel("tot")


plt.figure(4)
plt.subplot(223)
plt.hist2d(y_s[index],x_s[index],bins=n,range=[[0,256],[0,256]],vmax=150)
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
plt.hist2d(tfix[index],x_s[index],bins=n,range=[rtoac,[0,256]],vmax=200)
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(221)
a=plt.hist2d(y_s[index],tfix[index],bins=n,range=[[0,256],rtoac],vmax=200)
plt.xlabel("y")
plt.ylabel("t")

means=[np.average(a[2][:-1],weights=i**2) if sum(i)>0 else 0 for i in a[0]]
plt.plot(a[1][:-1],means)


plt.figure(5)
ax=plt.axes(projection='3d')
h, edges=np.histogramdd((x_s[index],y_s[index],tfix[index]),bins=n,range=[[0,256],[0,256],rtoac])
xs,ys,zs=np.meshgrid(edges[0][:-1]+edges[0][1]/2,edges[1][:-1]+edges[1][1]/2,edges[2][:-1]+edges[2][1]/2)

xs,ys,zs,h=(xs.flatten(),ys.flatten(),zs.flatten(),h.flatten())

index=np.where(h>3)

cm=SM().to_rgba(h[index])
cm[:,3]=h[index]/max(h[index])


ax.scatter3D(xs[index],ys[index],zs[index],color=cm,s=h[index]*2)
#b=plt.hist(y_s[index], bins=n,range=[0,256])#, range=rtoac)