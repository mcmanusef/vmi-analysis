# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:32:31 2022

@author: mcman
"""

from tkinter import filedialog
import tkinter as tk
import numpy as np
import h5py
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.widgets import SpanSelector
from numba import njit
from numba.typed import List
import scipy.interpolate as inter
import scipy.io as spio


def coincedence(t_tof, tof_corr, out_corr, min_tof, max_tof):
    tof_index = np.argwhere((t_tof < max_tof)*(t_tof > min_tof))
    pulses = tof_corr[tof_index][:, 0]

    ind_out_corr = dict((k, i) for i, k in enumerate(out_corr))

    inter = set(ind_out_corr).intersection(set(pulses))
    return [ind_out_corr[x] for x in inter]


@njit
def e_match(etof_corr, pulse_corr, x, y, t_etof):
    xint = List()
    yint = List()
    tint = List()
    p_corr = List()

    for [j, i] in enumerate(etof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(idxr-idxl):
            xint.append(x[idxl:idxr][k])
            yint.append(y[idxl:idxr][k])
            tint.append(t_etof[j])
            p_corr.append(i)
    return xint, yint, tint, p_corr


# @njit
def momentum(x, y, t):
    cxy = 0.000744246051212279

    px = (np.sqrt(cxy)*(x))*np.sqrt(2*0.03675)
    py = (np.sqrt(cxy)*(y))*np.sqrt(2*0.03675)
    # Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
    #      (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0))
    czp = [0, 0, 0.329121139, 0.080619655, 0.00703769]
    czp = [0, 0, 0.424689539, 0.110276512, 0.009581887]
    #[0, 0, 0.069680648, 0.00891148, 0.000873184]
    # czp = [0, 0, 0.102097468608904, 0.00387095928999341, -0.000167356003552665]
    ezp = sum(czp[i]*t**i for i in range(5))
    czm = [0, 0, -0.091548991, 0.006641229, -0.000179857]
    czm = [0, 0, -0.122238822, 0.011358042, -0.000330023]
    #[0, 0, -0.008355474, -0.008362956, 0.000497281]
    # czm = [0, 0, -0.0994289352339583, 0.00672581461331445, -0.000161486025535897]
    ezm = sum(czm[i]*t**i for i in range(5))
    Ez = (ezp*(t < 0) + ezm*(t > 0))
    pz = np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)
    return px, py, Ez  # (px, py, pz)


def rotate(h, edges, theta=-1, phi=0):
    Xc, Yc, Zc = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
    xv = Xc[0, :, 0]
    yv = Yc[:, 0, 0]
    zv = Zc[0, 0, :]
    hp = inter.interpn((xv, yv, zv), h, (Xc*np.cos(theta)+Yc*np.sin(theta),
                                         Yc*np.cos(theta)-Xc*np.sin(theta), Zc),
                       fill_value=0, bounds_error=False)

    h_out = inter.interpn((xv, yv, zv), hp, (Yc*np.cos(phi)-Zc*np.sin(phi), Xc, Zc *
                                             np.cos(phi)+Yc*np.sin(phi)), fill_value=0, bounds_error=False)
    return h_out, edges


# %%


#  #r"Data\xe003_c_cluster.h5""Data\\air_cluster.h5"  #
in_name = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered\xe001_p_cluster.h5"  # r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220404\xe102_cluster.h5"
t = 500
x0, y0, t0 = 125, 125, 28.1
tof_range = [0, 40]
etof_range = [-10, 60]
etof_range[1] = etof_range[1]+(0.26-(etof_range[1]-etof_range[0]) % 0.26)
etof_range = np.array(etof_range)
coin_range = [0, 40]

# maxlen=10000

print(-1)
with h5py.File(in_name, mode='r') as f:
    x = f['Cluster']['x'][()]
    y = f['Cluster']['y'][()]
    pulse_corr = f['Cluster']['pulse_corr'][()]
    print(0)
    t_tof = f['t_tof'][()]
    print(1)
    t_tof = (t_tof-t)/1000
    tof_corr = f['tof_corr'][()]
    print(2)
    t_etof = f['t_etof'][()]
    print(3)
    t_etof = t_etof-t
    etof_corr = f['etof_corr'][()]
    print(4)

# %%
xc, yc, zc, pc = e_match(etof_corr, pulse_corr, x, y, t_etof)
xs = np.asarray(xc)
ys = np.asarray(yc)
zs = np.asarray(zc)


def dist(x, y, x0, y0):
    return np.sqrt((x-x0)**2+(y-y0)**2)


torem = np.concatenate((
    np.where(dist(xs, ys, 195, 234) < 1.5)[0],
    np.where(dist(xs, ys, 203, 185) < 1.5)[0],
    np.where(dist(xs, ys, 253, 110) < 1.5)[0],
    np.where(dist(xs, ys, 23, 255) < 1.5)[0],
    np.where(dist(xs, ys, 204, 187) < 1.5)[0],
    np.where(dist(xs, ys, 98, 163) < 1.5)[0]),
    np.where((zs > etof_range[0])*(zs < etof_range[1]))[0])

noise = np.random.rand(len(zs))*0.26

# %%
# filename = "xe102_hist.mat"
v = np.linspace(-1, 1, 256)

xx = np.delete(xs, torem)
yy = np.delete(ys, torem)
tt = np.delete(zs, torem)
pp = np.delete(np.asarray(pc), torem)

coin = coincedence(t_tof, tof_corr, pp, 0, 128)

raw_hist, raw_edges = np.histogramdd((xx[coin], yy[coin], tt[coin]), bins=256, range=([0, 256], [0, 256], [-10, 60]))

# px, py, pz = momentum(np.delete(xs-x0, torem),
#                       np.delete(ys-y0, torem),
#                       np.delete(zs+noise-t0, torem))
# hist, edges = rotate(*np.histogramdd((px, py, pz),
# bins=256, range=([-1, 1], [-1, 1], [-1, 1])))
# spio.savemat(filename, dict(raw_hist=raw_hist, raw_edges=raw_edges, x=xx[coin], y=yy[coin], t=tt[coin]))  # , p_hist=hist, p_edges=edges))


# def broaden(array, dist):
#     if dist == -1:
#         return np.array([])
#     out = set()
#     for i in range(-dist, dist+1):
#         for j in array:
#             out.add(j+i)
#     return np.array(list(out))


# tokeep = []  # broaden(rem, -1)
# # %%
# corr = np.delete(np.asarray(pc), torem)


# def plot3dh(px, py, pz, index, ax):
#     hist, edges = rotate(*np.histogramdd((px[index], py[index], pz[index]),
#                                          bins=256, range=([-1, 1], [-1, 1], [-1, 1])))
#     print(np.max(hist), hist.shape)
#     a = ax.imshow(np.sum(hist, 1), origin='lower', extent=[-1, 1, -1, 1])
#     v = np.linspace(-1, 1, 256)

#     root = tk.Tk()
#     root.withdraw()

#     # file_path = filedialog.asksaveasfilename()
#     # if file_path:
#     #     spio.savemat(file_path, dict(full_hist=hist, xv=v, yv=v, zv=v))


# f, ((ax1, __), (ax2, __)) = plt.subplots(2, 2)
# ax1.set_title("Ion ToF Spectrum")
# ax2.set_title("Electron ToF Spectrum")
# ax3 = plt.subplot(122)
# ax3.set_title("Polarization Plane (Projection)")
# coin = partial(coincedence, t_tof, tof_corr, corr)
# ecoin = partial(coincedence, t_tof, tof_corr, etof_corr)
# p3d = partial(plot3dh, px, py, pz, ax=ax3)
# etof_bins = int(np.diff(etof_range)/0.26)

# ax1.hist(t_tof, range=tof_range, bins=256)
# ax2.hist(t_etof, range=etof_range, bins=etof_bins)
# plt.figure(2)
# plt.hist(pz**2, bins=1000, range=[0, 1])
# # plot3dh(px, py, pz, slice(len(px)), ax3)
# p3d(slice(len(px)))


# def onselect(xmin, xmax):
#     ax1.clear()
#     ax1.hist(t_tof, range=tof_range, bins=256)
#     ax1.axvline(xmin, c='red')
#     ax1.axvline(xmax, c='red')
#     ax2.clear()
#     temp = coin(xmin, xmax)
#     _, rem_index, _ = np.intersect1d(temp, tokeep, return_indices=True)
#     idx = np.delete(temp, rem_index)
#     ax2.hist(pz[idx], range=etof_range, bins=etof_bins)
#     ax3.clear
#     # print(idx)
#     p3d(idx)
#     plt.figure(-1000)

# # onselect(19, 22)


# span = SpanSelector(
#     ax1,
#     onselect,
#     "horizontal",
#     useblit=True
# )
