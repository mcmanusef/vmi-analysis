# -*- coding: utf-8 -*-
"""
Converts a file from phi's cartesian format to a mat file
@author: mcman
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import savemat
import scipy


def getmeans(edges):
    return np.asarray([(a+b)/2 for a, b in itertools.pairwise(edges)])


wdir=r"C:\Users\mcman\Code\VMI"
files = [None]

els = [1]
for file, el in zip(files, els):
    print(el)
    px, py, *sigs = map(list,
                        zip(*[[float(x) for x in l.split()] for l in open(os.path.join(wdir,file), 'r') if l.strip()]))

    size = 2048  # len(set(x**2+y**2 for x, y in zip(px, py)))
    xv = np.linspace(min(px), max(px), size)
    yv = np.linspace(min(py), max(py), size)

    for i, sig in enumerate(sigs):
        if i != 3:
            continue
        data = scipy.interpolate.griddata(
            (np.asarray(px), np.asarray(py)),
            sig,
            tuple(map(np.ravel, np.meshgrid(xv, yv))),
            fill_value=0).reshape(size, size)

        savemat(os.path.join(wdir, rf"theory_0{el}_{i}.mat"), {
            "hist": np.reshape(data, (size, 1, size))**1,
            "xv": xv,
            "yv": yv,
            "zv": np.array(0)})
