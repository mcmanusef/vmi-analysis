# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:49:48 2022

@author: mcman
"""

from scipy.fft import fftn, fftshift, ifftshift
import warnings
import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as SM
from datetime import datetime
import scipy.interpolate as inter
import scipy.optimize as optim
from skimage import measure
from numba import njit
from numba import errors
from numba.typed import List
from scipy import io as spio


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

        for k in range(idxl < idxr):
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

        for k in range(idxl < idxr):
            xs.append(x[idxl:idxr][k])
            ys.append(y[idxl:idxr][k])
            ts.append(t[idxl:idxr][k])
            tofs.append(t_tof[j])
    return xs, ys, ts, tofs


def P_xy(x):
    return 5.8029e-4*x**2
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


def P_z(t):
    # Ez = -2.433e9*t**5 + 1.482e8*t**4 - 2.937e6*t**3 + 8722*t**2 - 242*t + 0.04998
    Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)
    return (Ez)
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


mpl.rc('image', cmap='jet')
plt.close('all')


# %%
in_name = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220404\xe101_cluster.h5"
out_name = "ppol.mat"
t0 = 252.2  # in us
t0f = 31
tof_range = [0, 40]  # in us
etof_range = [20, 50]  # in ns

do_tof_gate = True

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
    print(etof_corr)


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
etof_range = [20, 60]
tof_range = [0, 40000]
x0 = 121
y0 = 134
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
px = P_xy(xs-x0)
py = P_xy(ys-y0)
pz = P_z(ts-t0f)


# %% Rotation
print('Rotating:', datetime.now().strftime("%H:%M:%S"))
mpl.rc('image', cmap='jet')

width = 1.
plot_range = [-width, width]
nbins = 256

h2, edges = np.histogramdd((xs, ys, ts), bins=(nbins, nbins, nbins),
                           range=[[0, 256], [0, 256], etof_range])

h, edges = np.histogramdd((px, py, pz), bins=(nbins, nbins, nbins),
                          range=[plot_range, plot_range, plot_range])
Xc, Yc, Zc = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1])
h[0, 0, :] = np.zeros_like(h[0, 0, :])
h[-1, -1, :] = np.zeros_like(h[0, 0, :])
numbins = 256

x = Xc[0, :, 0]
y = Yc[:, 0, 0]
z = Zc[0, 0, :]

theta = -1
phi = 176.0/180*np.pi

hp = inter.interpn((x, y, z), h, (Xc*np.cos(theta)+Yc*np.sin(theta),
                                  Yc*np.cos(theta)-Xc*np.sin(theta), Zc),
                   fill_value=0, bounds_error=False)

hp = inter.interpn((x, y, z), hp, (Yc*np.cos(phi)-Zc*np.sin(phi), Xc, Zc *
                                   np.cos(phi)+Yc*np.sin(phi)), fill_value=0, bounds_error=False)

plt.figure('ToF Spectrum')
plt.hist(tofs, bins=100, range=tof_range)

plt.figure('e-ToF Spectrum')
plt.hist(ts, bins=200, range=etof_range)


c = (0, 0, 0)

cx = nbins//2+int(c[0]/np.diff(x)[0])
cy = nbins//2+int(1/np.diff(y)[0]*c[1])
cz = nbins//2+int(1/np.diff(z)[0]*c[2])

w = 7
print(x[cx+w])
outdict = dict()

outdict['xyhist'] = np.sum(hp[:, :, cz-w:cz+w], axis=2)
outdict['xzhist'] = np.sum(hp[cy-w:cy+w, :, :], axis=0)
outdict['x'] = x
outdict['y'] = y
outdict['z'] = z

yzhist = np.sum(hp[:, cx-w:cx+w, :], axis=1)

spio.savemat(out_name, outdict)

plt.figure()
plt.imshow(outdict['xyhist'], extent=(-width, width, -width, width), origin='lower')
plt.xlabel("x")
plt.ylabel("y")


plt.figure()
plt.imshow(outdict['xzhist'], extent=(-width, width, -width, width), origin='lower')
plt.xlabel("x")
plt.ylabel("z")

plt.figure()
plt.imshow(np.sum(h2, 2))
