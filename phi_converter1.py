# -*- coding: utf-8 -*-
"""
Converts a file from phi's cartesian format to a mat file
@author: mcman
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import savemat
import scipy


def getmeans(edges):
    return np.asarray([(a+b)/2 for a, b in itertools.pairwise(edges)])


files = ["ellip01_slices_momentum.dat",
         "ellip03_slices_momentum.dat",
         ]

els = [1, 3, 4, 5, 6]
for file, el in zip(files, els):
    print(el)
    px, py, sig, *_ = map(list, zip(*[[float(x) for x in l.split()] for l in open(file, 'r') if l.strip()]))

    size = 2048  # len(set(x**2+y**2 for x, y in zip(px, py)))
    xv = np.linspace(min(px), max(px), size)
    yv = np.linspace(min(py), max(py), size)

    data = scipy.interpolate.griddata(
        (np.asarray(px), np.asarray(py)),
        sig,
        tuple(map(np.ravel, np.meshgrid(xv, yv))),
        fill_value=0).reshape(size, size)

    savemat(f"theory_0{el}.mat", {
        "hist": np.reshape(data, (size, 1, size))**1,
        "xv": xv,
        "yv": yv,
        "zv": np.array(0)})
