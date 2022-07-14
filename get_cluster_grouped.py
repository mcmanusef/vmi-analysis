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
import numba
from numba import njit
# %% Initializing


def h5append(dataset, newdata):
    """
    Append data to the end of a h5 dataset

    Parameters
    ----------
    dataset : h5py Dataset
        The dataset to append data to.
    newdata : Array
        The data to be appended.

    Returns
    -------
    None.

    """
    dataset.resize((dataset.shape[0] + len(newdata)), axis=0)
    dataset[-len(newdata):] = newdata


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

parser.add_argument('--cutoff', dest='cutoff',
                    type=float, default=500,
                    help="Time cutoff between itof and laser pulses in ns")

parser.add_argument('--uncombined', action='store_true',
                    help="Use TDC1 for both TOF and Laser Timing pulses")

parser.add_argument('--single', action='store_true',
                    help="Do not use multithreading to preserve RAM")

parser.add_argument('--reverse', action='store_true',
                    help="Reject all data within window")

parser.add_argument('--out', dest='output',
                    default='notset', help="The output HDF5 file")

parser.add_argument('--max', dest='max',
                    type=int, default=10000,
                    help="Max pixel events per laser shot")

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


@njit
def __reformat__(clusters, dlength, dilen, current_index):
    place = 0
    clust = np.zeros((dlength,), dtype=numba.int64)
    pulse_index = np.zeros((dilen,), dtype=numba.int64)
    for (j, c) in enumerate(clusters):
        temp = max(c)
        c = np.where(c == -1, c, c+current_index)
        clust[place:place+len(c)] = c
        pulse_index[j] = place
        place = place+len(c)
        current_index = current_index+temp+1

    return clust, current_index, pulse_index


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
        tdc_time = fh5['tdc_time']
        tdc_type = fh5['tdc_type']
        x = fh5['x']
        y = fh5['y']
        tot = fh5['tot']
        toa = fh5['toa']
        toa = np.where(np.logical_and(x[()] >= 194, x[()] < 204), toa[()]-25000, toa[()])
        with h5py.File(out_name, mode='w') as f:
            f.create_dataset('x', data=x[()], chunks=(1024,))
            f.create_dataset('y', data=y[()], chunks=(1024,))
            f.create_dataset('toa', data=toa, chunks=(1024,))
            f.create_dataset('tot', data=tot[()], chunks=(1024,))
            f.create_dataset('tdc_time', data=tdc_time[()], chunks=(1024,))
            f.create_dataset('tdc_type', data=tdc_type[()], chunks=(1024,))

    with h5py.File(out_name, mode='r+') as f:

        tdc_time = f['tdc_time']
        tdc_type = f['tdc_type']
        x = f['x']
        y = f['y']
        tot = f['tot']
        toa = f['toa']

    # %% Preliminary Calculations
        tdc1 = np.where(tdc_type[()] == 1)
        times = tdc_time[()][tdc1]

        if not tdc_type[-1] == 1:
            lengths = np.diff(tdc_time)[tdc1]
            pulse_times = times[np.where(lengths > 1e3*args.cutoff)]
            tof_times = times[np.where(lengths < 1e3*args.cutoff)]
        else:
            lengths = np.diff(tdc_time)[tdc1[:-1]]
            pulse_times = times[np.where(lengths > 1e3*args.cutoff)[:-1]]
            tof_times = times[np.where(lengths < 1e3*args.cutoff)[:-1]]

        del lengths, times, tdc1

        tof_corr = np.searchsorted(pulse_times, tof_times)
        t_tof = 1e-3*(tof_times-pulse_times[tof_corr-1])-args.t

        f.create_dataset('t_tof', data=t_tof, chunks=(1024,))
        f.create_dataset('tof_corr', data=tof_corr, chunks=(1024,))
        del t_tof, tof_corr, tof_times

        etof_times = tdc_time[()][np.where(tdc_type[()] == 3)]
        etof_corr = np.searchsorted(pulse_times, etof_times)
        t_etof = 1e-3*(etof_times-pulse_times[etof_corr-1])-args.t

        f.create_dataset('t_etof', data=t_etof, chunks=(1024,))
        f.create_dataset('etof_corr', data=etof_corr, chunks=(1024,))
        del t_etof, etof_corr, etof_times

        pulses = np.searchsorted(pulse_times, toa[()])
        while not max(pulses) == pulses[-1]:
            pulses = pulses[0:-1]
        t = 1e-3*(toa[()]-pulse_times[pulses-1])-args.t

        f.create_dataset('pulse_times', data=pulse_times, chunks=(1024,))
        f.create_dataset('t', data=t, chunks=(1024,))
        f.create_dataset('pulse_corr', data=pulses, chunks=(1024,))
        del t, pulse_times

    # %% Formatting Data
        print('Formatting and Rejecting Data:',
              datetime.now().strftime("%H:%M:%S"))
        data = []
        num = max(pulses)+1

        pixelhits = 0
        loss = 0
        # Format data into correct format
        for i in range(num):
            idxr = np.searchsorted(pulses, i, side='right')
            idxl = np.searchsorted(pulses, i, side='left')
            if idxl < idxr:
                if idxr-idxl > args.max:
                    loss = loss+(idxr-idxl)/len(x)
                    print("    Warning, Discarded pulse number {pulse}, due to having {num} pixel events ({perc:.2}%)".format(
                        pulse=i+1, num=idxr-idxl, perc=(idxr-idxl)/len(x)*100))
                else:
                    data.append([x[idxl:idxr], y[idxl:idxr]])
                    pixelhits = pixelhits+idxr-idxl

        print("    Total Loss = ", loss*100)
        print("    Total Data = {p}/{x} ({d:.2}%)".format(p=pixelhits,
                                                          x=len(x), d=pixelhits/len(x)*100))
        del loss, pulses, pixelhits, num

    # %% Preparing for Loop
        f.create_dataset('cluster_index', data=[0], chunks=(1024,), maxshape=(None,))
        g = f.create_group('Cluster')
        g.create_dataset('x', data=[0], chunks=(1024,), maxshape=(None,))
        g.create_dataset('y', data=[0], chunks=(1024,), maxshape=(None,))
        g.create_dataset('t', data=[0], chunks=(1024,), maxshape=(None,))
        g.create_dataset('toa', data=[0], chunks=(1024,), maxshape=(None,))
        g.create_dataset('tot', data=[0], chunks=(1024,), maxshape=(None,))
        g.create_dataset('pulse_corr', data=[0], chunks=(1024,), maxshape=(None,))

        current_index = new_index = cur_idx = 0

    # %% Processing Groups
        imax = int(len(data)/args.groupsize)+1
        start = datetime.now()
        for i in range(imax):
            print("Group "+str(i+1)+"/"+str(imax))
            print("\t"+datetime.now().strftime("%H:%M:%S"))

            # %%% Clustering

            threads = os.cpu_count() if not args.single else 1
            temp = min((i+1)*args.groupsize, len(data))
            di = data[i*args.groupsize:temp]
            clusters = list(np.zeros(len(di)))

            if not args.single:
                with Pool(threads) as p:
                    to_add = p.map(get_clusters, di)
                    clusters = list(to_add)
            else:
                to_add = map(get_clusters, di)
                clusters = list(to_add)

            # %%% Reformatting
            current_index = new_index
            dlength = sum([len(j[0]) for j in di])
            clust, new_index, pulse_index = __reformat__(
                clusters, dlength, len(di), current_index)
            # print("Clusters:", clust)
            # print("pulse_index:", pulse_index)

            del clusters, di

            # %%% Averaging Across Clusters
            num = int(max(clust)+1-current_index)
            t = f['t']
            max_pulse = len(pulse_index)
            xm = np.zeros(num)
            ym = np.zeros(num)
            tm = np.zeros(num)
            toam = np.zeros(num)
            totm = np.zeros(num)
            cur_pulse = 0

            # Find weighted average of x,y,tot,toa
            j = 0
            while j < num and cur_pulse < max_pulse:
                if cur_pulse == 0:
                    idx = np.where(clust[:pulse_index[cur_pulse]] == j+current_index)[0] + cur_idx
                else:
                    idx = pulse_index[cur_pulse-1] + np.where(
                        clust[pulse_index[cur_pulse-1]:pulse_index[cur_pulse]] == j+current_index)[0] + cur_idx

                if len(idx) == 0:
                    cur_pulse = cur_pulse+1
                else:
                    xm[j] = np.average(x[idx], weights=tot[idx])
                    ym[j] = np.average(y[idx], weights=tot[idx])
                    tm[j] = np.average(t[idx], weights=tot[idx])
                    toam[j] = np.average(toa[idx], weights=tot[idx])
                    totm[j] = np.average(tot[idx], weights=tot[idx])
                    j = j+1

            pulse_corr = np.searchsorted(f['pulse_times'], toam)

            cur_idx += dlength
            # %%% Saving Data to Output File
            h5append(f['cluster_index'], clust)
            g = f['Cluster']
            h5append(g['x'], xm)
            h5append(g['y'], ym)
            h5append(g['t'], tm)
            h5append(g['toa'], toam)
            h5append(g['tot'], totm)
            h5append(g['pulse_corr'], pulse_corr)

            now = datetime.now()
            print("\tEstimated Finish: ", (start+imax/(i+1)*(now-start)).strftime("%m/%d/%Y, %H:%M:%S"))
