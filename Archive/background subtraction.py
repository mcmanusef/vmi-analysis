# -*- coding: utf-8 -*-

import itertools
import platform
import argparse
import time
import glob
import os
import sys
import h5py
from matplotlib import cm,colors
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import tpx3_analysis.standard_layouts as SL
import tpx3_analysis.threaded_sort as TS
from  tpx3_analysis import Converter
import sklearn.cluster as cluster
import gc
sys.path.append('C:\\Users\\mcman\\Documents\\VMI\\timepix3_analysis')
plt.close('all')
gc.collect()


dname='ar006.h5'
bname='bg006.h5' #'xe_2tdc_vert_8_000000'#

max_to_plot=-1

with h5py.File(dname, mode='r') as fh5:
    
    pulse_times=fh5['tdc_time'][()][np.where(fh5['tdc_type'][()]==1)]
    pulse_corr=np.searchsorted(pulse_times,fh5['toa'][()])
    time_after=1e-3*(fh5['toa'][()]-pulse_times[pulse_corr-1])

    x_s=fh5['x'][()][0:max_to_plot]
    y_s=fh5['y'][()][0:max_to_plot]
    t_s=time_after[0:max_to_plot]#+tot[0:max_to_plot]
    
    zmin=0
    zmax=1000
    in_window=np.where(np.logical_and(t_s>zmin,t_s<zmax))
    
    x_s=x_s[in_window]
    y_s=y_s[in_window]
    t_s=t_s[in_window]
    pulses=pulse_corr[0:max_to_plot][in_window]
    del pulse_corr
    tot_s=fh5['tot'][()][0:max_to_plot][in_window]

    
    gc.collect()

z_s=((t_s-zmin)*256/(zmax-zmin)).astype(int)

plt.figure(1)
a1=plt.hist2d(x_s,y_s,bins=256,range=[[0,256],[0,256]],weights=tot_s)
plt.colorbar()

# plt.figure(2)
# plt.hist2d(t_s,y_s,bins=256,range=[[0,1000],[0,256]])#,weights=tot_s)

# plt.figure(3)
# plt.hist2d(t_s,x_s,bins=256,range=[[0,1000],[0,256]])#,weights=tot_s)


with h5py.File(bname, mode='r') as fh5:
    #frames = fh5['frame_number'][()]
    #tdc_time=fh5['tdc_time'][()] # tdc times are stored in ps
    #tdc_type = fh5['tdc_type'][()]
    #x = fh5['x'][()]
    #y = fh5['y'][()]
    #tot = fh5['tot'][()]
    #toa = fh5['toa'][()]
    #cluster_idx = fh5['cluster_index'][()]
    
    
    
    pulse_times=fh5['tdc_time'][()][np.where(fh5['tdc_type'][()]==1)]
    pulse_corr=np.searchsorted(pulse_times,fh5['toa'][()])
    time_after=1e-3*(fh5['toa'][()]-pulse_times[pulse_corr-1])
#plt.scatter(x[0:max_to_plot],time_after[0:max_to_plot])

    # plt.figure(1)
    # plt.hist(time_after,bins=100, range=[0,10000])
    
    x_s=fh5['x'][()][0:max_to_plot]
    y_s=fh5['y'][()][0:max_to_plot]
    t_s=time_after[0:max_to_plot]#+tot[0:max_to_plot]
    
    zmin=0
    zmax=1000
    in_window=np.where(np.logical_and(t_s>zmin,t_s<zmax))
    
    x_s=x_s[in_window]
    y_s=y_s[in_window]
    t_s=t_s[in_window]
    pulses=pulse_corr[0:max_to_plot][in_window]
    del pulse_corr
    tot_s=fh5['tot'][()][0:max_to_plot][in_window]
    
    gc.collect()
    
plt.figure(4)
a2=plt.hist2d(x_s,y_s,bins=256,range=[[0,256],[0,256]],weights=tot_s)
plt.colorbar()

# plt.figure(5)
# plt.hist2d(t_s,y_s,bins=256,range=[[0,1000],[0,256]])#,weights=tot_s)

# plt.figure(6)
# plt.hist2d(t_s,x_s,bins=256,range=[[0,1000],[0,256]])#,weights=tot_s)
    
# plt.figure(4)
# plt.scatter(x_s,y_s,c=cind)
    
# ax=plt.axes(projection='3d')
# ax.scatter3D(x_s,y_s,t_s,c=cind)
# #ax.set_zlim3d(400, 500)
# ax.set_ylim3d(0, 256)
# ax.set_xlim3d(0, 256)

# counts=np.zeros((3,256,256))
# index_xy=256*x_s+y_s
# index_xz=256*x_s+z_s
# index_yz=256*y_s+z_s    
# index=index_xy
# for i in range(256):
#     for j in range(256):
#         if j==0:
#             print(i)
#         a=np.where(index_xy==256*i+j)[0]
#         b=np.where(index_xz==256*i+j)[0]
#         c=np.where(index_yz==256*i+j)[0]
#         counts[0,i,j]=len(a)
#         counts[1,i,j]=len(b)
#         counts[2,i,j]=len(c)


   

    
dbscan=cluster.DBSCAN(eps=1, min_samples=4)

clst_idx=np.array([])
current_index=0
temp=0
#~1000 laser shots per minute (based on stony brook data 1m points=31k laser shots)
clustering = False
if clustering:
    for i in range(max(pulses)+1): 
        print(i, "/", max(pulses))
        idx=np.where(pulses==i)
        temp=temp+len(idx[0])
        if len(idx[0])>0:
            db = dbscan.fit(np.column_stack((x_s[idx],y_s[idx])))
            clst=db.labels_
            clst2=np.where(clst==-1,clst,clst+current_index)
            clst_idx=np.append(clst_idx,clst2)
            current_index=current_index+max(clst)+1
            
    plt.figure(5)
    idx=np.where(np.logical_and(clst_idx<500,clst_idx>=-1))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_s[idx],y_s[idx],t_s[idx],c=clst_idx[idx])
#    ax.set_zlim3d(400, 500)
#    ax.set_ylim3d(0, 256)
#    ax.set_xlim3d(0, 256)

#    plt.scatter(x_s[idx],y_s[idx],c=clst_idx[idx])
    
#     xm=np.zeros(int(max(clst_idx)+1))
#     ym=np.zeros(int(max(clst_idx)+1))
#     tm=np.zeros(int(max(clst_idx)+1))
    
#     for i in range(int(max(clst_idx)+1)):
#         idx=np.where(clst_idx==i)
#         xm[i]=np.average(x_s[idx], weights=tot_s[idx])
#         ym[i]=np.average(y_s[idx], weights=tot_s[idx])
#         tm[i]=np.average(z_s[idx], weights=tot_s[idx])
    
# plt.figure(2)
# plt.subplot(221)
# plt.imshow(counts[0])
# plt.xlabel("y")
# plt.ylabel("x")
# #plt.scatter(ym,xm,c='r')

# plt.subplot(222)
# plt.imshow(counts[1])
# plt.xlabel("t")
# plt.ylabel("x")
# #plt.scatter(tm,xm,c='r')

# plt.subplot(223)
# plt.imshow(counts[2].transpose())
# plt.xlabel("y")
# plt.ylabel("t")
# #plt.scatter(ym,tm,c='r')k