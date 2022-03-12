# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:29:11 2022

@author: mcman
"""


import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as SM
from datetime import datetime
import scipy.interpolate as inter

mpl.rc('image', cmap='jet')
plt.close('all')

name = 'kr000000'
in_name = name+'_cluster.h5'

t0 = 252.2  # in us
tof_range = [0, 20]  # in us
etof_range = [20, 40]  # in ns

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


# %% e-TOF Coincidence
print('Starting e-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))

etof_index = np.where(np.logical_and(
    t_etof > etof_range[0], t_etof < etof_range[1]))[0]
xint = [[]]*len(etof_index)
yint = [[]]*len(etof_index)
tint = [[]]*len(etof_index)

for [j, i] in enumerate(etof_corr[etof_index]):
    if j % 10000 == 0:
        print("    {num:.2f}%".format(num=100*j/len(etof_index)))
    idxr = np.searchsorted(pulse_corr, i, side='right')
    idxl = np.searchsorted(pulse_corr, i, side='left')

    if idxl < idxr:
        xint[j] = x[idxl:idxr]
        yint[j] = y[idxl:idxr]
        tint[j] = np.array([t_etof[etof_index][j]]*(idxr-idxl))


xint = np.array([a for ll in xint for a in ll])
yint = np.array([a for ll in yint for a in ll])
tint = np.array([a for ll in tint for a in ll])

# %% i-TOF Coincidence

print('Starting i-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))
tof_index = np.where(np.logical_and(
    t_tof > tof_range[0], t_tof < tof_range[1]))[0]

xs = [[]]*len(tof_index)
ys = [[]]*len(tof_index)
ts = [[]]*len(tof_index)
tofs = [[]]*len(tof_index)

for [j, i] in enumerate(tof_corr[tof_index]):
    if j % 10000 == 0:
        print("    {num:.2f}%".format(num=100*j/len(tof_index)))
    idxr = np.searchsorted(pulse_corr, i, side='right')
    idxl = np.searchsorted(pulse_corr, i, side='left')

    if idxl < idxr:
        xs[j] = xint[idxl:idxr]
        ys[j] = yint[idxl:idxr]
        ts[j] = tint[idxl:idxr]
        tofs[j] = np.array([t_tof[tof_index][j]]*(idxr-idxl))

xs = np.array([a for ll in xs for a in ll])
ys = np.array([a for ll in ys for a in ll])
ts = np.array([a for ll in ts for a in ll])
tofs = np.array([a for ll in tofs for a in ll])

# %% Plotting
print('Plotting:', datetime.now().strftime("%H:%M:%S"))

etof_bins = int(min(int(np.diff(etof_range)/0.26), 300))
print(etof_bins)
plt.figure(1)
plt.hist(tofs, bins=300, range=tof_range)

plt.figure(2)
plt.hist(ts, bins=etof_bins, range=etof_range)


nbins = 128
plt.figure(3)
ax = plt.axes(projection='3d')
h, edges = np.histogramdd((xs, ys, ts), bins=[nbins, nbins, etof_bins], range=[
                          [0, 256], [0, 256], etof_range])


xx = np.linspace(0, 256, num=nbins)
yy = np.linspace(0, 256, num=nbins)
zz = np.linspace(etof_range[0], etof_range[1], num=etof_bins)
xc, yc, zc = np.meshgrid(xx, yy, zz)
shape = np.shape(xc)
xc, yc, zc, h = (xc.flatten(), yc.flatten(), zc.flatten(), h.flatten())

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("ToF (ns)")
index = np.where(h > 0)

#mpl.rc('image', cmap='plasma')
cm = SM().to_rgba(h[index])
cm[:, 3] = np.sqrt(h[index]/max(h[index]))
mpl.rc('image', cmap='jet')
ax.scatter3D(yc[index], xc[index], zc[index], color=cm, s=h[index]**2)

xc, yc, zc, h = (np.reshape(xc, shape), np.reshape(yc, shape),
                 np.reshape(zc, shape), np.reshape(h, shape))


def interpolation(x, y, z):
    return inter.interpn((xc[0, :, 0]/256, yc[:, 0, 0]/256, (zc[0, 0, :] - etof_range[0])/np.diff(etof_range)[0]), h, (x, y, z), method="linear")


[xgrid, ygrid, zgrid] = np.mgrid[0:1:nbins*1j, 0:1:nbins*1j, 0:1:etof_bins*1j]
# mlab.contour3d(xgrid, ygrid, zgrid, interpolation(
#     xgrid, ygrid, zgrid), contours=10, transparent=True, extent=[0, 1, 0, 1, 0, 1])

mlab.points3d(xgrid, ygrid, zgrid, interpolation(
    xgrid, ygrid, zgrid)+1, extent=[0, 1, 0, 1, 0, 1])
mlab.axes()

ax.set_xlim(left=0, right=256)
ax.set_ylim(bottom=0, top=256)
ax.set_zlim(bottom=etof_range[0], top=etof_range[1])

xc, yc, zc = np.meshgrid(xx, yy, zz)

xyhist = np.histogramdd((ys, xs), bins=nbins, range=[[0, 256], [0, 256]])[0]
xzhist = np.histogramdd((xs, ts), bins=[nbins, etof_bins], range=[
                        [0, 256], etof_range])[0]
yzhist = np.histogramdd((ys, ts), bins=[nbins, etof_bins], range=[
                        [0, 256], etof_range])[0]


# xy = mlab.imshow(xyhist, extent=[0, 1, 0, 1, 0, 0], name="xy")
# xz = mlab.imshow(xzhist, extent=[0, 1, 0, 1, 0, 0], name="xz")
# yz = mlab.imshow(yzhist, extent=[0, 1, 0, 1, 0, 0], name="yz")

# xy.actor.orientation = [0, 0, 0]
# xz.actor.orientation = [90, 90, 90]
# yz.actor.orientation = [90, 90, 0]

# xz.actor.position = [0.5, 0, 0.5]
# yz.actor.position = [0, 0.5, 0.5]


ax.plot_surface(xc[:, :, 0], yc[:, :, 0], zc[:, :, 0],
                rcount=nbins, ccount=nbins, facecolors=SM().to_rgba(xyhist))
ax.plot_surface(xc[:, 0, :], yc[:, 0, :], zc[:, 0, :],
                rcount=nbins, ccount=etof_bins, facecolors=SM().to_rgba(yzhist))
ax.plot_surface(xc[0, :, :], yc[0, :, :], zc[0, :, :],
                rcount=nbins, ccount=etof_bins, facecolors=SM().to_rgba(xzhist))

plt.figure(4)
plt.suptitle('Time Resolved VMI')

plt.subplot(223)
# plt.imshow(np.histogramdd((ys, xs), bins=256, range=[
#            [0, 256], [0, 256]])[0], interpolation='gaussian')
plt.hist2d(ys, xs, bins=128, range=[[0, 256], [0, 256]])
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
plt.hist2d(ts, xs, bins=[etof_bins, 256], range=[
           etof_range, [0, 256]])
plt.xlabel("t (ns)")
plt.ylabel("x")

plt.subplot(221)
plt.hist2d(ys, ts, bins=[256, etof_bins], range=[
           [0, 256], etof_range])
plt.xlabel("y")
plt.ylabel("t (ns)")
plt.ylabel("t (ns)")
plt.ylabel("t (ns)")
plt.xlabel("y")
plt.ylabel("t (ns)")
plt.ylabel("t (ns)")
plt.ylabel("t (ns)")
plt.ylabel("t (ns)")
