# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:16:36 2021

@author: mcman
"""

import h5py
import numpy as np
import sklearn.cluster as cluster
from multiprocessing import Pool
from datetime import datetime
import argparse
import os

# %% Initializing
parser = argparse.ArgumentParser(prog='get_cluster',
                                 description="Clusters data with rejection based on pulse time differences")

parser.add_argument('--g', dest='groupsize',
                    type=int, default=int(1e9),
                    help="Clusters this many pulses at a time")

parser.add_argument('--start', dest='start',
                    type=float, default=0,
                    help="Start of rejection window in ms")

parser.add_argument('--t', dest='t', type=int,
                    default=0, help="time in ns at which the laser is in the chamber")

parser.add_argument('--end', dest='end',
                    type=float, default=1e9,
                    help="End of rejection window in ms")

parser.add_argument('--combined', action='store_true',
                    help="Use TDC1 for both TOF and Laser Timing pulses")

parser.add_argument('--reverse', action='store_true',
                    help="Reject all data within window")

parser.add_argument('--out', dest='output',
                    default='notset', help="The output HDF5 file")

parser.add_argument('filename')
args = parser.parse_args()


def get_clusters(data):
    """
    Run DBSCAN Clustering on data.

    Parameters
    ----------
    data : (int[], int[])
        Tuple consisting of x and y coordinates of points

    Returns
    -------
    clst : int[]
        Cluster indicies for the points in data

    """
    dbscan = cluster.DBSCAN(eps=1, min_samples=4)
    x = data[0]
    y = data[1]
    db = dbscan.fit(np.column_stack((x, y)))
    clst = db.labels_
    return clst


if __name__ == '__main__':
    # %% Loading Data
    filename = args.filename

    if filename[-3:] == '.h5':
        filename = filename[0:-3]

    in_name = filename+'.h5'
    if args.output == 'notset':
        out_name = filename+'_cluster.h5'
    else:
        out_name = args.output

    print('Loading Data:', datetime.now().strftime("%H:%M:%S"))

    with h5py.File(in_name, mode='r') as fh5:
        tdc_time = fh5['tdc_time'][()]
        tdc_type = fh5['tdc_type'][()]
        x = fh5['x'][()]
        y = fh5['y'][()]
        tot = fh5['tot'][()]
        toa = fh5['toa'][()]

    toa = np.where(np.logical_and(x >= 194, x < 204), toa-25000, toa)
    if not args.combined:
        pulse_times = tdc_time[np.where(tdc_type == 1)]
        tof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_tof = 1e-3*(tof_times-pulse_times[tof_corr-1])-args.t

    else:
        times = tdc_time[()][np.where(tdc_type == 1)]
        lengths = np.diff(tdc_time)[np.where(tdc_type == 1)]
        pulse_times = times[np.where(lengths > 1e6)]
        tof_times = times[np.where(lengths < 1e6)]
        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_tof = 1e-3*(tof_times-pulse_times[tof_corr-1])-args.t
        etof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
        etof_corr = np.searchsorted(pulse_times, etof_times)
        t_etof = 1e-3*(etof_times-pulse_times[etof_corr-1])-args.t

    to_keep = np.where(np.logical_xor(args.reverse, np.logical_and(
        1e9*args.start <= np.diff(pulse_times),
        np.diff(pulse_times) <= 1e9*args.end)))[0]

    pulses = np.searchsorted(pulse_times, toa)
    time_after = 1e-3*(toa-pulse_times[pulses-1])-args.t

    # %% Formatting Data
    print('Formatting and Rejecting Data:',
          datetime.now().strftime("%H:%M:%S"))
    data = []
    num = max(pulses)+1

    # Format data into correct format
    for i in to_keep:
        idxr = np.searchsorted(pulses, i, side='right')
        idxl = np.searchsorted(pulses, i, side='left')
        if idxl < idxr:
            data.append([x[idxl:idxr], y[idxl:idxr]])

    # %% Clustering Data

    print('Clustering:',
          datetime.now().strftime("%H:%M:%S"))
    # Gather data
    clusters = list(np.zeros(len(data)))
    for i in range(int(len(data)/args.groupsize)+1):
        temp = min((i+1)*args.groupsize, len(data))
        with Pool(os.cpu_count()) as p:
            to_add = p.map(get_clusters, data[i*args.groupsize:temp])
            clusters[i*args.groupsize:temp] = list(to_add)

    # %% Collecting Clustered Data
    dlength = sum([len(i[0]) for i in data])
    current_index = 0
    place = 0

    clust = np.zeros(dlength, dtype=int)
    pulse_index = np.zeros(len(data), dtype=int)

    print('Collecting Cluster Indicies:',
          datetime.now().strftime("%H:%M:%S"))
    # Collect all data into one array
    for (i, c) in enumerate(clusters):
        temp = max(c)
        c = np.where(c == -1, c, c+current_index)
        clust[place:place+len(c)] = c
        pulse_index[i] = place
        place = place+len(c)
        current_index = current_index+temp+1

    # %% Averaging Across Clusters
    num = int(max(clust)+1)
    xm = np.zeros(num)
    ym = np.zeros(num)
    tm = np.zeros(num)
    toam = np.zeros(num)
    totm = np.zeros(num)
    cur_pulse = 0

    print('Averaging Across Clusters:',
          datetime.now().strftime("%H:%M:%S"))

    # Find weighted average of x,y,tot,toa
    i = 0
    while i < num and cur_pulse < len(pulse_index):
        if cur_pulse == 0:
            idx = np.where(clust[:pulse_index[cur_pulse]] == i)
        else:
            idx = pulse_index[cur_pulse-1] + np.where(
                clust[pulse_index[cur_pulse-1]:pulse_index[cur_pulse]] == i)

        if len(idx[0]) == 0:
            cur_pulse = cur_pulse+1
        else:
            xm[i] = np.average(x[idx], weights=tot[idx])
            ym[i] = np.average(y[idx], weights=tot[idx])
            tm[i] = np.average(time_after[idx], weights=tot[idx])
            toam[i] = np.average(toa[idx], weights=tot[idx])
            totm[i] = np.average(tot[idx], weights=tot[idx])
            i = i+1

    print(len(xm))

    # %% Saving Data to Output File
    print('Saving:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(out_name, 'w') as f:
        f.create_dataset('pulse_times', data=pulse_times)
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('t', data=time_after)
        f.create_dataset('t_tof', data=t_tof)
        if args.combined:
            f.create_dataset('t_etof', data=t_etof)
        f.create_dataset('toa', data=toa)
        f.create_dataset('tot', data=tot)
        f.create_dataset('tdc_time', data=tdc_time)
        f.create_dataset('tdc_type', data=tdc_type)
        f.create_dataset('cluster_index', data=clust)

        g = f.create_group('Cluster')
        g.create_dataset('x', data=xm)
        g.create_dataset('y', data=ym)
        g.create_dataset('t', data=tm)
        g.create_dataset('toa', data=toam)
        g.create_dataset('tot', data=totm)
