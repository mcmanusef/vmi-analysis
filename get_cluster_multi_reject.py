#! /bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:35:22 2021

@author: mcman
"""
import h5py
import numpy as np
import sklearn.cluster as cluster
#import sys
from multiprocessing import Pool
from datetime import datetime
#from multiprocessing import set_start_method
from numba import njit
import argparse
parser = argparse.ArgumentParser(prog='get_cluster_multi_reject', description="Clusters data with rejection based on pulse time differences")
parser.add_argument('--g', dest='groupsize', type=int, default=int(1e9), help="Clusters this many pulses at a time")
parser.add_argument('--start', dest='start', type=float, default=0, help="Start of rejection window in ms")
parser.add_argument('--end', dest='end', type=float, default=1e9, help="End of rejection window in ms")
parser.add_argument('--reverse',action='store_true',help="Reject all data within window")
parser.add_argument('filename')
args = parser.parse_args()



def get_clusters(data):
    dbscan=cluster.DBSCAN(eps=1, min_samples=4)
    x=data[0]
    y=data[1]
    db = dbscan.fit(np.column_stack((x,y)))
    clst=db.labels_
    return clst
    
#@njit
def cluster_analysis(filename,groupsize=int(1e9),start=1+25e-6,end=1+30e-6,reverse=False):
    in_name=filename+'.h5'
    out_name=filename+'_cluster_r.h5'
    
    print('Loading Data:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(in_name, mode='r') as fh5:
        tdc_time=fh5['tdc_time'][()] # tdc times are stored in ps
        tdc_type = fh5['tdc_type'][()]
        x = fh5['x'][()]
        y = fh5['y'][()]
        tot = fh5['tot'][()]
        toa = fh5['toa'][()]
        
    toa=np.where(np.logical_and(x>=194,x<204),toa-25000,toa)
    pulse_times=tdc_time[np.where(tdc_type==1)]
    #print(np.diff(pulse_times))
    to_keep=np.where(np.logical_xor(reverse,np.logical_and(1e9*start<=np.diff(pulse_times),np.diff(pulse_times)<=1e9*end)))[0]
    pulses=np.searchsorted(pulse_times,toa)
    time_after=1e-3*(toa-pulse_times[pulses-1])
    
    #print('Rejecting Data:', datetime.now().strftime("%H:%M:%S"))
    print(len(to_keep),'/',len(pulse_times))
    data=[]
    
    print('Formatting and Rejecting Data:', datetime.now().strftime("%H:%M:%S"))
    num=max(pulses)+1
    #Format data into correct format
    for i in to_keep: 
        idxr=np.searchsorted(pulses,i,side='right')
        idxl=np.searchsorted(pulses,i,side='left')
        if idxl<idxr:
            #print(i,idxr-idxl)
            data.append([x[idxl:idxr],y[idxl:idxr]])
    #print(data)
    dlength=sum([len(i[0]) for i in data])
    print('Clustering:', datetime.now().strftime("%H:%M:%S"))    
    #Gather data
    clusters=list(np.zeros(len(data)))
    for i in range(int(len(data)/groupsize)+1):
        temp=min((i+1)*groupsize,len(data))
        #print(i*groupsize,temp)
        with Pool(16) as p:
            to_add=p.map(get_clusters,data[i*groupsize:temp])
            clusters[i*groupsize:temp]=list(to_add)
    current_index=0
    place=0
    
    clust=np.zeros(dlength, dtype=int)
    pulse_index=np.zeros(len(data),dtype=int)
        
    print('Collecting Cluster Indicies:', datetime.now().strftime("%H:%M:%S"))
    #Collect all data into one array
    i=0
    for c in clusters:
        temp=max(c)
        c=np.where(c==-1,c,c+current_index)
        clust[place:place+len(c)]=c
        pulse_index[i]=place
        place=place+len(c)
        current_index=current_index+temp+1
        i=i+1
        
    num=int(max(clust)+1)
    xm=np.zeros(num)
    ym=np.zeros(num)
    tm=np.zeros(num)
    totm=np.zeros(num)
    tota=np.zeros(num)
    cur_pulse=0
    print('Averaging Across Clusters:', datetime.now().strftime("%H:%M:%S"))
    #Find weighted average of x,y,tot,toa
    i=0
    while i<num and cur_pulse<len(pulse_index):
        #print(i,'/',num)
        if cur_pulse==0:
            idx=np.where(clust[:pulse_index[cur_pulse]]==i)
        else:
            idx=pulse_index[cur_pulse-1]+np.where(clust[pulse_index[cur_pulse-1]:pulse_index[cur_pulse]]==i)
        #print(idx[0])
        if len(idx[0])==0:
            cur_pulse=cur_pulse+1
        else:
            xm[i]=np.average(x[idx], weights=tot[idx])
            ym[i]=np.average(y[idx], weights=tot[idx])
            tm[i]=np.average(time_after[idx], weights=tot[idx])
            #totm[i]=max(tot[idx])
            tota[i]=np.average(tot[idx], weights=tot[idx])
            i=i+1
        
    print(len(xm))
    print('Saving:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(out_name,'w') as f:
        f.create_dataset('x',data=x)
        f.create_dataset('y',data=y)
        f.create_dataset('t',data=time_after)
        f.create_dataset('toa',data=toa)
        f.create_dataset('tot',data=tot)
        f.create_dataset('tdc_time',data=tdc_time)
        f.create_dataset('tdc_type',data=tdc_type)
        f.create_dataset('cluster_index',data=clust)
        
        g=f.create_group('Cluster')
        g.create_dataset('x',data=xm)
        g.create_dataset('y',data=ym)
        g.create_dataset('toa',data=tm)
        g.create_dataset('tot',data=tota)
        

if __name__ == '__main__':
    filename=args.filename
    #filename='ar001'
    if filename[-3:]=='.h5':
        filename=filename[0:-3]
    
    cluster_analysis(filename, groupsize=args.groupsize,start=args.start,end=args.end,reverse=args.reverse)