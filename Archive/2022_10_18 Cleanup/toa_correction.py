# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:22:23 2022

@author: mcman
"""
import h5py
import numpy as np
import sklearn.cluster as cluster
from multiprocessing import Pool
from datetime import datetime
import argparse
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pdf")

# %% Initializing
parser = argparse.ArgumentParser(prog='toa_correction',
                                 description="Performs time rejection and toa correction of clustered data")

parser.add_argument('--start', dest='start',
                    type=float, default=-1e9,
                    help="Start of rejection window in ns")

parser.add_argument('--end', dest='end',
                    type=float, default=1e9,
                    help="End of rejection window in ns")

parser.add_argument('--m', dest='manual', action='store_true',
                    help="Manually define rejection window")

parser.add_argument('--range', dest='range', type=float, default=2000,
                    help="width of rejection windoow in ns")

parser.add_argument('--out', dest='output', default='notset',
                    help="The output HDF5 file")

parser.add_argument('filename')
args = parser.parse_args()

if __name__ == '__main__':
    # %% Loading Data
    print('Loading Data:', datetime.now().strftime("%H:%M:%S"))
    filename = args.filename

    if filename[-3:] == '.h5':
        filename = filename[0:-3]

    in_name = filename+'.h5'
    if args.output == 'notset':
        out_name = filename[:-8]+'_corrected.h5'
    else:
        out_name = args.output

    with h5py.File(args.filename, mode='r') as fh5:
        x = fh5['Cluster']['x'][()]
        y = fh5['Cluster']['y'][()]
        tot = fh5['Cluster']['tot'][()]
        time_after = fh5['Cluster']['t'][()]

    fsize = (10, 10)
    window = [0, 0, 1, 0.95]
    rtot = [0, 200]
    n = 256

    # %% Time Cropping Data
    print('Time Cropping:', datetime.now().strftime("%H:%M:%S"))

    if not args.manual:
        toa_hist, toa_bins = np.histogram(
            time_after, bins=10000, range=[0, 1e6])
        toa_peak = toa_bins[np.where(toa_hist == max(toa_hist))][0]

        toa_range = args.range

        toa_int = [toa_peak-toa_range/2, toa_peak+toa_range/2]
    else:
        toa_int = [args.start, args.end]

    index = np.where(np.logical_and(
        time_after > toa_int[0], time_after < toa_int[1]))

    xs = x[index]
    ys = y[index]
    ts = time_after[index]
    tots = tot[index]

    # %% ToA Correction
    print('Starting ToA Correction:', datetime.now().strftime("%H:%M:%S"))

    a = plt.hist2d(ts, tots, bins=256, range=[toa_int, rtot])

    totspace = np.linspace(rtot[0], rtot[1], num=n)
    dtot = totspace[1]-totspace[0]
    toa_correction = np.zeros_like(tots)
    t = np.linspace(toa_int[0], toa_int[1], num=n)
    means = [np.average(t, weights=i**2) if sum(i) >
             0 else 0 for i in a[0].transpose()]

    for i in range(n):
        corr = np.where(np.logical_and(
            tots >= totspace[i], tots < totspace[i]+dtot))
        toa_correction[corr] = means[i]

    tfix = ts-toa_correction

    # %% Saving
    print('Saving:', datetime.now().strftime("%H:%M:%S"))
    copyfile(in_name, out_name)
    with h5py.File(out_name, mode='r+') as fh5:
        g = fh5['Cluster']
        del g['x'], g['y'], g['t'], g['tot']
        g.create_dataset('x', data=xs)
        g.create_dataset('y', data=ys)
        g.create_dataset('t', data=tfix)
        g.create_dataset('tot', data=tots)
