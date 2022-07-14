# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:38:30 2022

@author: mcman
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit


def g2d(x, xc, yc, w, a):
    return a*np.exp(-((x[0]-xc)**2+(x[1]-yc)**2)/w**2)


def p2d(x, xc, yc, w, a):
    return ((x[0]-xc)**4+(x[1]-yc)**4)*w + a


def g1d(x, xc, w, a):
    return a*np.exp(-(x-xc)**2/w**2)


def p1d(x, xc, w, a):
    return a+(x-xc)**4*w


num = 1000

distance = np.zeros(num)
count = np.zeros(num)
uncertainty = np.zeros(num)
dist2 = np.zeros(num)

with h5py.File("Data\\xe007_p_cluster.h5", mode='r') as f:
    for i in range(num):
        where = np.where(f["cluster_index"][()] == i)
        if len(where[0]) < 10:
            continue
        count[i] = len(where[0])
        # plt.figure(i+1)
        tot = f['tot'][where]
        toa = f['toa'][where]
        coords = (f['x'][where], f['y'][where])
        max_ind = np.where(tot == max(tot))[0]
        xslice = np.where(coords[1] == coords[1][max_ind])

        xrange = np.linspace(0, 255, 256)
        yrange = np.linspace(0, 255, 256)

        x, y = np.meshgrid(xrange, yrange)

        guess = (coords[0][max_ind][0], 2, 100)
        guessp = (coords[0][max_ind][0], 10000, min(toa))
        guess2 = (coords[0][max_ind][0], coords[1][max_ind][0], 2, 100)
        guess2p = (coords[0][max_ind][0], coords[1][max_ind][0], 10000, min(toa))

        # f1d, cov = curve_fit(g1d, coords[0][xslice], tot[xslice], p0=guess)
        # f1p, cov = curve_fit(p1d, coords[0][xslice], toa[xslice], p0=guessp)
        # plt.scatter(coords[0][xslice], f['toa'][where][xslice], c='orange')
        # plt.plot(xrange, p1d(xrange, *f1p), c='orange')
        # plt.plot(xrange, p1d(xrange, *guessp), c='orange')
        # plt.ylim([min(toa)-1000, max(toa)+1000])
        # plt.twinx()
        # plt.scatter(coords[0][xslice], f['tot'][where][xslice])
        # plt.plot(xrange, g1d(xrange, *f1d))
        # plt.plot(xrange, g1d(xrange, *guess))
        fit, cov = curve_fit(g2d, coords, tot, p0=guess2)
        fitp, cov = curve_fit(p2d, coords, toa, p0=guess2p)

        # plt.figure(-(i+1))
        # plt.imshow(p2d((x, y), *fit), cmap='gray')
        # plt.scatter(coords[0], coords[1], c=f['toa'][where])

        xm = np.average(coords[0], weights=tot)
        ym = np.average(coords[1], weights=tot)

        print(i)
        print("Coordinates")
        print("\txf={xf:.2f}, yf={yf:.2f}".format(xf=fit[0], yf=fit[1]))
        print("\txp={xf:.2f}, yp={yf:.2f}".format(xf=fitp[0], yf=fitp[1]))
        print("\txm={xf:.2f}, ym={yf:.2f}".format(xf=xm, yf=ym))
        print("Distance")
        distance[i] = np.sqrt((fitp[0]-xm)**2+(fitp[1]-ym)**2)
        print("\t{}".format(np.sqrt((fit[0]-xm)**2+(fit[1]-ym)**2)))
        print("Uncertainty")
        print("\tx:{dx:.2f}, y:{dy:.2f}".format(dx=np.sqrt(cov[0, 0]), dy=np.sqrt(cov[1, 1])))
        uncertainty[i] = (np.sqrt(cov[0, 0]+cov[1, 1]))
plt.figure(0)
plt.scatter(count, distance)
plt.scatter(count, uncertainty)
