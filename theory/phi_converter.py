# -*- coding: utf-8 -*-
"""
Converts a file from phi's cartesian format to a mat file
@author: mcman
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import savemat


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def getmeans(edges):
    return np.asarray([(a+b)/2 for a, b in pairwise(edges)])


for i in [4, 5, 6]:
    file = f"cartesian-ellip0{i}-pz0-tot-momspe-2d-sw.dat"
    data = [[float(x) for x in l.split()] for l in open(file, 'r') if l.strip()]
    px, py, sig, *_ = map(list, zip(*data))

    size = len(set(px))

    # plt.hist2d(px,py,bins=size,weights=sig, cmap='jet')

    out_dict = {}

    hist, xe, ye = np.histogram2d(py, px, bins=size, weights=sig)

    out_dict["hist"] = np.reshape(hist, (size, 1, size))**1
    out_dict["xv"] = getmeans(xe)
    out_dict["yv"] = getmeans(ye)
    out_dict["zv"] = np.array(0)

    savemat(f"theory_0{i}.mat", out_dict)
