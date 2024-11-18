import functools
import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Old import cv3_analysis
from cluster_v3 import iter_dataset

def iter_min(indexes, curr, iters):
    min_index=np.argmin(indexes)
    out=list(curr)
    out[min_index]=next(iters[min_index],None)
    return out

def correlate_tof_coincidence(cluster_data, etof_data, itof_data):
    cluster = next(cluster_data, None)
    etof = next(etof_data, None)
    itof = next(itof_data, None)
    while not (cluster is None or etof is None or itof is None):
        pulses=[cluster[0], etof[0], itof[0]]
        if pulses[0]==pulses[1] and pulses[1]==pulses[2]:
            yield cluster[1][0], cluster[1][1], etof[1], itof[1]
        cluster,etof,itof=iter_min(pulses, (cluster,etof,itof), (cluster_data, etof_data, itof_data))


def in_ranges(vals, ranges):
    return all(map(lambda x,r: r[0]<x<r[1], vals,ranges))
matplotlib.rc('image', cmap='jet')
matplotlib.use('Qt5Agg')
plt.close("all")

wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20230601"
name="kr003_e.cv3"
file=os.path.join(wdir,name)
center=(128,128,748444)
rx=[0,256]
ry=[0,256]
rz=(748_430_000,748_490_000)
ri=(.764e9,.766e9)
file_dict={}
with h5py.File(file) as f:
    for k in f.keys():
         file_dict[k] = iter_dataset(f, k)

    cluster_data = zip(file_dict['cluster_corr'], zip(file_dict['x'], file_dict['y'], file_dict['t']))
    etof_data=zip(file_dict['etof_corr'], file_dict['t_etof'])
    itof_data=zip(file_dict['tof_corr'], file_dict['t_tof'])

    correlated_data=correlate_tof_coincidence(cluster_data,etof_data,itof_data)

    coincidence_data=((x, y, cv3_analysis.smear(t / 1000)) for (x, y, t, itof) in correlated_data if in_ranges((x, y, t, itof), (rx, ry, rz, ri)))

    data=map(functools.partial(cv3_analysis.rotate_coords, theta=1.1),
             map(functools.partial(cv3_analysis.centering, center=center),
                 filter(cv3_analysis.in_good_pixels, coincidence_data)))
    x,y,z= tuple(map(np.asarray,zip(*data)))




#%%
rx=[-128,128]
ry=[-128,128]
rz=[-10,20]

x2,y2,z2= map(np.asarray,zip(*[(a,b,c) for a,b,c in zip(x,y,z) if rx[0]<a<rx[1] and ry[0]<b<ry[1] and rz[0]<c<rz[1]]))

#%%
plt.figure()
plt.hist2d(x2,y2,range=[[-128,128],ry],bins=1024,norm=matplotlib.colors.PowerNorm(0.5))
plt.figure()
plt.hist2d(x2,z2,range=[rx,rz],bins=1024,norm=matplotlib.colors.PowerNorm(0.5))
plt.figure()
plt.hist2d(z2,y2,range=[rz,ry],bins=1024,norm=matplotlib.colors.PowerNorm(0.5))

#%%
