# -*- coding: utf-8 -*-
"""
Plots a reconstruction for Mike
@author: mcman
"""


import logging
import os
from functools import partial

import ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.interpolate as spi
import scipy.io
from matplotlib.cm import ScalarMappable as SM
from scipy import signal
from skimage import measure


def axes(lower, upper, center=(0, 0, 0)):
    xx = yy = zz = np.arange(lower, upper, 0.1)
    xc = np.zeros_like(xx)+center[0]
    yc = np.zeros_like(xx)+center[1]
    zc = np.zeros_like(xx)+center[2]
    mlab.plot3d(xc, yy+yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+upper+0.05, center[1], center[2], "y", scale=0.05)
    mlab.plot3d(xc, yc, zz+zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0], center[1]+upper, center[2], "z", scale=0.05)
    mlab.plot3d(xx+zc, yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+0.025, center[1], center[2]+upper, "x", scale=0.05)


pi = np.pi
plt.close('all')
# mlab.close(all=True)

source = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20221209\reconstruction_compare.mat"
data = scipy.io.loadmat(source)
data2 = scipy.io.loadmat(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new\xe011_e.mat")


def plot3d(data, axis):
    mlab.figure(size=(1000, 1000))
    out = data
    x3, y3, z3 = np.meshgrid(*axis)
    width = np.max(x3)
    nbins = len(out[0, 0])

    minbin = 1
    numbins = 20
    numbins = min(numbins, int(out.max())-minbin)
    cm = SM(cmap='jet').to_rgba(np.array(range(numbins))**0.7)
    cm[:, 3] = (np.array(range(numbins))/numbins)**0.3

    for i in range(numbins):
        iso_val = i*(int(out.max())-minbin)/numbins+minbin
        verts, faces, _, _ = measure.marching_cubes(
            out, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))
        mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                             faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])

    axes(-width*.7, width*.7)


plot3d(data['reconstruct_hist']*100, (data["reconstruct_ax"], data["reconstruct_ax"], data["reconstruct_ax"]))
plot3d(data['raw_hist']*100, (data["raw_x"], data["reconstruct_ax"], data["reconstruct_ax"]))
plot3d(data2['hist'], (data2['xv'], data2['yv'], data2['zv']))
