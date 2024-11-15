# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:36:11 2022

@author: mcman
"""


import os
import numpy as np
import scipy.signal as signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
source = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new"


def blur3d(array, sigma, width):
    x = np.arange(-width, width+1, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width+1, 1)
    z = np.arange(-width, width+1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return signal.convolve(array, np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)))[width:-width, width:-width, width:-width]


for d in os.listdir(source):
    if d[-4:] == '.mat' and d == 'xe005_e.mat':
        print(d)
        data = loadmat(os.path.join(source, d))
        plt.figure(d)
        xv = data['xv'][0]
        plt.imshow(np.sum(blur3d(data['hist'], 0.5, 5)[:, (np.abs(data['xv']) < 0.1)[0], :], 1),
                   extent=[min(xv), max(xv), min(xv), max(xv)])
        plt.title(d[:-4])
        plt.xlabel("Major Axis")
        plt.ylabel("Minor Axis")
        plt.savefig(d[:-4]+".png")
