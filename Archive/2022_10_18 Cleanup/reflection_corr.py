# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:51:22 2022

@author: mcman
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import scipy.interpolate as inter
from numba import njit
from numba import errors
from numba.typed import List
import functools
from matplotlib.widgets import Slider, Button


import logging
logging.disable(logging.WARNING)

warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)

# %% Initializing
print('Initializing:', datetime.now().strftime("%H:%M:%S"))


@njit
def __e_coincidence(etof_corr, pulse_corr, x, y, t_etof):
    xint = List()
    yint = List()
    tint = List()
    for [j, i] in enumerate(etof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(idxl + 1 == idxr):
            xint.append(x[idxl:idxr][k])
            yint.append(y[idxl:idxr][k])
            tint.append(t_etof[j])
    return xint, yint, tint


@njit
def __i_coincidence(tof_corr, pulse_corr, x, y, t, t_tof):
    xs = List()
    ys = List()
    ts = List()
    tofs = List()
    for [j, i] in enumerate(tof_corr):
        idxr = np.searchsorted(pulse_corr, i, side='right')
        idxl = np.searchsorted(pulse_corr, i, side='left')

        for k in range(idxl+1 == idxr):
            xs.append(x[idxl:idxr][k])
            ys.append(y[idxl:idxr][k])
            ts.append(t[idxl:idxr][k])
            tofs.append(t_tof[j])
    return xs, ys, ts, tofs


mpl.rc('image', cmap='jet')
plt.close('all')

# %% Parameters
name = 'mid'
in_name = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered\xe011_e_cluster.h5"  #

t0 = .5  # in us
t0f = 28.75
tof_range = [0, 40]  # in us
etof_range = [20, 60]  # in ns

do_tof_gate = False

t0 = t0*1000
tof_range = np.array(tof_range)*1000
etof_range[1] = etof_range[1]+(0.26-(etof_range[1]-etof_range[0]) % 0.26)
etof_range = np.array(etof_range)
# %% Loading Data

print('Loading Data:', datetime.now().strftime("%H:%M:%S"))

with h5py.File(in_name, mode='r') as f:
    x = f['Cluster']['x'][()]
    y = f['Cluster']['y'][()]
    pulse_corr = f['Cluster']['pulse_corr'][()]

    t_tof = f['t_tof'][()]
    t_tof = t_tof-t0
    tof_corr = f['tof_corr'][()]

    t_etof = f['t_etof'][()]
    t_etof = t_etof-t0
    etof_corr = f['etof_corr'][()]


# %% e-ToF Coincidence
print('Starting e-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))


etof_index = np.where(np.logical_and(
    t_etof > etof_range[0], t_etof < etof_range[1]))[0]

xint, yint, tint = __e_coincidence(etof_corr[etof_index], pulse_corr, x, y, t_etof[etof_index])

xint, yint, tint = np.array(xint), np.array(yint), np.array(tint)

# %% i-ToF Coincidence
if do_tof_gate:
    print('Starting i-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))
    tof_index = np.where(np.logical_and(
        t_tof > tof_range[0], t_tof < tof_range[1]))[0]

    xs, ys, ts, tofs = __i_coincidence(
        tof_corr[tof_index], pulse_corr, xint, yint, tint, t_tof[tof_index])

    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    tofs = np.array(tofs)
else:
    xs, ys, ts, tofs = xint, yint, tint, t_tof
addnoise = True
# %% Conversion to Momentum
x0 = 119
y0 = 133
torem = np.where(np.logical_or(
    np.logical_and(np.round(xs) == 195, np.round(ys) == 234),
    np.logical_and(np.round(xs) == 98, np.round(ys) == 163)))
xs = np.delete(xs, torem)
ys = np.delete(ys, torem)
ts = np.delete(ts, torem)
if addnoise:
    noise = np.random.rand(len(ts))*0.26
    ts = ts+noise
    addnoise = False

# %% Rotation
print('Rotating:', datetime.now().strftime("%H:%M:%S"))
mpl.rc('image', cmap='jet')

width = 1.5
plot_range = [-width, width]
nbins = 256

h2, edges = np.histogramdd((xs, ys, ts), bins=(nbins, nbins, nbins),
                           range=[[0, 256], [0, 256], etof_range])

# %% Plotting
print('Plotting:', datetime.now().strftime("%H:%M:%S"))

extent = [0, 256, etof_range[0], etof_range[1]]

plt.figure()
plt.imshow(np.sum(h2, 1).transpose(),  # norm=mpl.colors.LogNorm(),
           extent=extent, aspect='auto', origin='lower')

rgi = inter.RegularGridInterpolator(
    (edges[0][:-1], edges[1][:-1], edges[2][:-1]), h2, bounds_error=False, fill_value=0)


if True:
    xx, yy, tt = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
    sm = functools.partial(np.sum, axis=0)
else:
    xx, tt = np.meshgrid(edges[0][:-1], edges[2][:-1])
    yy = 128
    def sm(x): return x.transpose()
plt.figure()
plt.imshow(sm(rgi((xx, yy, tt))).transpose(),
           extent=extent, aspect='auto', origin='lower')


def cor(rgi, xx, yy, tt, dt, scale):
    return rgi((xx, yy, tt))-scale*rgi((xx, yy, tt-dt))


correction = functools.partial(cor, rgi, xx, yy, tt)
fig, ax = plt.subplots()

if 0:
    im = plt.imshow(sm(correction(10.6, .14)).transpose(),  # norm=mpl.colors.LogNorm(),
                    extent=extent, aspect='auto', origin='lower')
else:
    im = plt.imshow(sm(correction(12.6, .04)).transpose(),  # norm=mpl.colors.LogNorm(),
                    extent=extent, aspect='auto', origin='lower')
# plt.subplots_adjust(left=0.25, bottom=0.25)

# ax1 = plt.axes([0.25, 0.1, 0.65, 0.03])
# scale_slider = Slider(
#     ax=ax1,
#     label='Scale',
#     valmin=0,
#     valmax=1,
#     valinit=.1,
# )
# ax2 = plt.axes([0.1, 0.25, 0.0225, 0.63])
# off_slider = Slider(
#     ax=ax2,
#     label='pos',
#     valmin=10,
#     valmax=15,
#     valinit=10.66,
#     orientation="vertical"
# )


# def update(val):
#     im.set(data=sm(correction(off_slider.val, scale_slider.val)).transpose())
#     fig.canvas.draw_idle()


# scale_slider.on_changed(update)
# off_slider.on_changed(update)
