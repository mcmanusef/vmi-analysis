# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:41:02 2021

@author: mcman
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.backends.backend_pdf
import matplotlib as mpl
from datetime import datetime
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 12
plt.close('all')

parser = argparse.ArgumentParser(
    prog='Analyze Data', description="Analyze a clustered dataset and save the data to a pdf")
parser.add_argument('--out', dest='output',
                    default="output.pdf", help="Output Filename")
parser.add_argument('--t', dest='t', type=int,
                    default=0, help="time in ns at which the laser is in the chamber")
parser.add_argument('--noclust', action='store_true',
                    help="Do not analyze any clustered data.")
parser.add_argument('filename')
args = parser.parse_args()

with matplotlib.backends.backend_pdf.PdfPages(args.output) as pdf:
    # %% Loading Data
    print('Loading Unclustered Data:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(args.filename, mode='r') as fh5:
        tdc_time = fh5['tdc_time'][()]
        tdc_type = fh5['tdc_type'][()]
        x = fh5['x'][()]
        y = fh5['y'][()]
        time_after = fh5['t'][()]
        tot = fh5['tot'][()]
        toa = fh5['toa'][()]
        t_tof = fh5['t_tof'][()]
        pulse_times = fh5['pulse_times'][()]

    fsize = (10, 10)
    window = [0, 0, 1, 0.95]
    rtot = [0, 200]
    n = 256

    # %% Page 1: Unprocessed Data
    plt.figure(figsize=fsize)
    plt.suptitle('Unprocessed')
    plt.subplot(221)
    plt.hist(np.diff(pulse_times)*1e-6, bins=300, range=[0, 1100])
    plt.title("Pulse Times")
    plt.xlabel('Pulse Time Differences (us)')
    plt.ylabel('Counts')
    plt.tight_layout(rect=window)

    plt.subplot(222)
    plt.hist2d(y, x, bins=256, range=[[0, 256], [0, 256]])
    plt.title("Full VMI Image")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout(rect=window)

    plt.subplot(223)
    plt.hist(t_tof, bins=100, range=[0, 1e6])
    tof_hist, tof_bins = np.histogram(t_tof, bins=1000, range=[0, 1e6])
    plt.title("Ion TOF Spectrum")
    plt.xlabel("TOF (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    plt.subplot(224)
    plt.hist(time_after, bins=100, range=[0, 1e6])
    toa_hist, toa_bins = np.histogram(time_after, bins=10000, range=[0, 1e6])
    plt.title("Electron TOA Spectrum")
    plt.xlabel("TOA (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    pdf.savefig()

    # %% Time Cropping Data
    print('Time Cropping:', datetime.now().strftime("%H:%M:%S"))

    toa_peak = toa_bins[np.where(toa_hist == max(toa_hist))][0]
    tof_peak = tof_bins[np.where(tof_hist == max(tof_hist))][0]

    toa_range = 200
    tof_range = 16000

    toa_int = [toa_peak-toa_range/2, toa_peak+toa_range/2]
    tof_int = [tof_peak-tof_range/2, tof_peak+tof_range/2]

    index = np.where(np.logical_and(
        time_after > toa_int[0], time_after < toa_int[1]))
    xs = x[index]
    ys = y[index]
    ts = time_after[index]
    tots = tot[index]

    # %% Page 2: Time Cropped Spectra
    plt.figure(figsize=fsize)
    plt.suptitle('Time Cropped')

    plt.subplot(211)
    plt.hist(t_tof, bins=500, range=tof_int)
    plt.title("Ion TOF Spectrum")
    plt.xlabel("TOF (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    plt.subplot(212)
    plt.hist(time_after, bins=100, range=toa_int)
    plt.title("Electron TOA Spectrum")
    plt.xlabel("TOA (ns)")
    plt.ylabel("Count")
    plt.tight_layout(rect=window)

    pdf.savefig()

    # %% Page 3: Time Cropped VMI
    plt.figure(figsize=fsize)
    plt.suptitle('VMI Time Cropped')

    plt.subplot(223)
    plt.hist2d(ys, xs, bins=256, range=[[0, 256], [0, 256]])
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout(rect=window)

    plt.subplot(224)
    plt.hist2d(ts, xs, bins=256, range=[toa_int, [0, 256]])
    plt.xlabel("t (ns)")
    plt.ylabel("x")
    plt.tight_layout(rect=window)

    plt.subplot(221)
    plt.hist2d(ys, ts, bins=256, range=[[0, 256], toa_int])
    plt.xlabel("y")
    plt.ylabel("t (ns)")
    plt.tight_layout(rect=window)

    plt.subplot(222)
    plt.title("ToA vs ToT")
    plt.hist2d(ts, tots, bins=256, range=[toa_int, rtot])
    plt.xlabel("ToA (ns)")
    plt.ylabel("ToT (ns)")
    plt.tight_layout(rect=window)

    pdf.savefig()

    if not args.noclust:

        # %% Loading Clustered Data
        print('Loading Clustered Data:', datetime.now().strftime("%H:%M:%S"))
        with h5py.File(args.filename, mode='r') as fh5:
            x = fh5['Cluster']['x'][()]
            y = fh5['Cluster']['y'][()]
            tot = fh5['Cluster']['tot'][()]
            t = fh5['Cluster']['t'][()]-args.t

        # %%Page 4: Clustered VMI
        plt.figure(figsize=fsize)
        plt.suptitle('Clustered VMI')

        index = np.where(np.logical_and(t > toa_int[0], t < toa_int[1]))
        xs = x[index]
        ys = y[index]
        ts = t[index]
        tots = tot[index]

        plt.subplot(223)
        plt.hist2d(ys, xs, bins=256, range=[[0, 256], [0, 256]])
        plt.xlabel("y")
        plt.ylabel("x")
        plt.tight_layout(rect=window)

        plt.subplot(224)
        plt.hist2d(ts, xs, bins=256, range=[toa_int, [0, 256]])
        plt.xlabel("t (ns)")
        plt.ylabel("x")
        plt.tight_layout(rect=window)

        plt.subplot(221)
        plt.hist2d(ys, ts, bins=256, range=[[0, 256], toa_int])
        plt.xlabel("y")
        plt.ylabel("t (ns)")
        plt.tight_layout(rect=window)

        plt.subplot(222)
        plt.title("ToA vs ToT")
        a = plt.hist2d(t, tot, bins=256, range=[toa_int, rtot])
        plt.xlabel("ToA (ns)")
        plt.ylabel("ToT (ns)")
        plt.tight_layout(rect=window)

        pdf.savefig()

        # %% ToA Correction
        print('Starting ToA Correction:', datetime.now().strftime("%H:%M:%S"))

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

        # %% Page 5: ToA Corrected VMI
        plt.figure(figsize=fsize)
        plt.suptitle('ToA Corrected VMI')

        plt.subplot(223)
        plt.hist2d(ys, xs, bins=256, range=[[0, 256], [0, 256]])
        plt.xlabel("y")
        plt.ylabel("x")
        plt.tight_layout(rect=window)

        plt.subplot(224)
        plt.hist2d(tfix, xs, bins=256, range=[
                   [-toa_range/2, toa_range/2], [0, 256]])
        plt.xlabel("t (ns)")
        plt.ylabel("x")
        plt.tight_layout(rect=window)

        plt.subplot(221)
        plt.hist2d(ys, tfix, bins=256, range=[
                   [0, 256], [-toa_range/2, toa_range/2]])
        plt.xlabel("y")
        plt.ylabel("t (ns)")
        plt.tight_layout(rect=window)

        plt.subplot(222)
        plt.title("ToA vs ToT")
        plt.hist2d(tfix, tots, bins=256, range=[
                   [-toa_range/2, toa_range/2], rtot])
        plt.xlabel("ToA (ns)")
        plt.ylabel("ToT (ns)")
        plt.tight_layout(rect=window)

        pdf.savefig()
