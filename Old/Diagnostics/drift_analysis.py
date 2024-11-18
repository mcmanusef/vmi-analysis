import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from Old.coincidence_v4 import load_file
matplotlib.use('Qt5Agg')

sort_key=lambda x: int(x.split('_')[1]) if x.endswith('.cv4') else 0
for file in sorted(os.listdir(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208"), key=sort_key):
    if file.endswith(".cv4"):
        plt.figure(1)
        x,y,t,etof,itof=load_file(os.path.join(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208",file))
        r=(749000,749100)
        etof=etof+0.26*np.random.random_sample(len(etof))
        etof=etof[np.argwhere(np.logical_and(r[0]<etof,etof<r[1])).flatten()]
        mean=np.mean(etof)
        x_axis=np.linspace(0,1,len(etof))
        # plt.scatter(x_axis,etof, s=0.1, c='k')
        kernel_size=10000
        gaussian_kernel=scipy.signal.windows.gaussian(kernel_size*5, std=kernel_size)
        gaussian_kernel=gaussian_kernel/np.sum(gaussian_kernel)
        rolling_average=scipy.signal.fftconvolve(etof-mean, gaussian_kernel, mode='same',)+mean
        # rolling_average=scipy.ndimage.uniform_filter1d(etof, kernel_size)
        plt.plot(x_axis, rolling_average, lw=1, label=f'{file}')
        plt.suptitle('e-ToF Drift')
        plt.title('Gaussian Filter (σ=10000)')
        plt.xlabel('Time (relative to start of file, arb. units)')
        plt.ylabel('e-ToF (ns)')
        plt.ylim(749050,749060)
        plt.legend()
        # plt.figure(file)
        # plt.scatter(x_axis, etof-rolling_average, s=.1, label=f'{file}')
        plt.figure(f"{file} hist")
        plt.hist(etof, bins=1000, histtype='step', color='k')
        plt.twiny(plt.gca())
        plt.hist(etof-rolling_average, bins=1000, histtype='step', color='r')

#
# x,y,t,etof,itof=load_file(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20240208\o2_04_a.cv4")
# r=(749000,749100)
# etof=etof+0.26*np.random.random_sample(len(etof))
# etof=etof[np.argwhere(np.logical_and(r[0]<etof,etof<r[1])).flatten()]
# plt.scatter(range(len(etof)),etof, s=0.1, c='k')
# kernel_size=1000
# rolling_average=scipy.ndimage.gaussian_filter1d(etof, kernel_size)
# plt.plot(rolling_average, c='r', lw=1.5, label=f'Gaussian Filter (σ={kernel_size})')
# plt.ylim(749000,749100)