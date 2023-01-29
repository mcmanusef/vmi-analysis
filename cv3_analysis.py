# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:08:56 2023

@author: mcman
"""

import h5py
import numpy as np
import functools
import itertools
import random
import matplotlib.pyplot as plt
from numba import njit, vectorize


def correlate_tof(data_iter, tof_data):
    dc = next(data_iter, None)
    tc = next(tof_data, None)
    while not (dc is None or tc is None):
        if dc[0] > tc[0]:
            tc = next(tof_data, None)
        elif dc[0] < tc[0]:
            dc = next(data_iter, None)
        else:
            yield (dc[0], tuple(list(dc[1])+[tc[1]]))
            tc = next(tof_data, None)


@njit(cache=True)
def dist(x, y, x0, y0):
    return np.sqrt((x-x0)**2+(y-y0)**2)


@njit(cache=True)
def in_good_pixels(coords):
    x, y, t = coords
    conditions = np.array([dist(x, y, 195, 234) < 1.5,
                           dist(x, y, 203, 185) < 1.5,
                           dist(x, y, 253, 110) < 1.5,
                           dist(x, y,  23, 255) < 1.5,
                           dist(x, y, 204, 187) < 1.5,
                           dist(x, y,  98, 163) < 1.5])
    return not np.any(conditions)


@vectorize
def P_xy(x):
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


@vectorize
def P_z(t):
    Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
          (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0))
    # Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)  # old
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


@njit(cache=True)
def smear(x, amount=0.26):
    return x+np.random.rand()*amount


@njit
def momentum_conversion(coords):
    x, y, t = coords
    return P_xy(x), P_xy(y), P_z(t)


@njit
def centering(x, center=(128, 128, 528.5)):
    return (x[0]-center[0], x[1]-center[1], x[2]-center[2])


@njit(cache=True)
def rotate_coords(coords, theta=-1, phi=0):
    x, y, z = coords
    xp, yp, zp = x*np.cos(theta)+y*np.sin(theta), y*np.cos(theta)-x*np.sin(theta), z
    return zp * np.cos(phi)+yp*np.sin(phi), yp*np.cos(phi)-zp*np.sin(phi), xp


def partition(list_in, n):
    indices = list(range(len(list_in)))
    random.shuffle(indices)
    index_partitions = [sorted(indices[i::n]) for i in range(n)]
    return [[list_in[i] for i in index_partition]
            for index_partition in index_partitions]


def load_cv3(file, pol=0, width=0.05):
    data = {}
    with h5py.File(file) as f:
        for k in f.keys():
            data[k] = list(f[k][()])

    __, coords = tuple(zip(*list(correlate_tof(
        zip(data["cluster_corr"], zip(data["x"], data["y"])),
        zip(data["etof_corr"], map(smear, list(np.array(data["t_etof"])/1000)))))))

    px, py, pz = map(np.array, zip(*list(
        filter(lambda x: abs(x[2]) < width,
               map(functools.partial(rotate_coords, phi=pol),
                   map(momentum_conversion,
                       map(centering,
                           filter(in_good_pixels, coords))))))))
    return px, py, pz
# # %%
#     parts = partition(list(range(len(px))), 10)
#     plt.figure(1)
#     hists = []
#     for i in range(9):
#         plt.subplot(331+i)
#         plt.hist2d(px[parts[i]], py[parts[i]], bins=100, range=[[-1, 1], [-1, 1]], cmap='jet')
#     # plt.figure(2)
#     plt.hist2d(px, py, bins=100, range=[[-1, 1], [-1, 1]], cmap='jet')
