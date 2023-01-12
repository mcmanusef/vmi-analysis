# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:29:11 2022

@author: mcman
"""


import warnings
import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.cm import ScalarMappable as SM
from datetime import datetime
# import scipy.interpolate as inter
from skimage import measure
from numba import jit
from numba import njit
from numba import prange
from numba import errors
from numba.typed import List

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import logging
logging.disable(logging.WARNING)

warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)

# %% Initializing
print('Initializing:', datetime.now().strftime("%H:%M:%S"))


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def double_arrow(ax, com, direction, length):
    """
    Plots a 3d double arrow with a given center, direction and length

    Parameters
    ----------
    ax : Matplotlib Axes
        the axes to add the double arrow.
    com : Float[3]
        Center of the double arrow.
    direction : Float[3]
        Normalized Direction.
    length : num
        The full length of the arrow.

    Returns
    -------
    None.

    """

    a1 = Arrow3D([com[1]-length/3*direction[1], com[1]+length/2*direction[1]],
                 [com[0]-length/3*direction[0], com[0]+length/2*direction[0]],
                 [com[2]-length/3*direction[2], com[2]+length/2*direction[2]],
                 mutation_scale=20, lw=2, arrowstyle="-|>", color="r", zorder=1000)
    ax.add_artist(a1)

    a2 = Arrow3D([com[1]+length/3*direction[1], com[1]-length/2*direction[1]],
                 [com[0]+length/3*direction[0], com[0]-length/2*direction[0]],
                 [com[2]+length/3*direction[2], com[2]-length/2*direction[2]],
                 mutation_scale=20, lw=2, arrowstyle="-|>", color="r", zorder=1000)
    ax.add_artist(a2)


@njit("i8[:],i8[:],f8[:],f8[:],f8[:]")
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


@njit("i8[:],i8[:],f8[:],f8[:],f8[:],f8[:]")
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


mpl.rc('image', cmap='jet')
plt.close('all')

# %% Parameters
name = 'mid'
in_name = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220404\xe101_cluster.h5"  #

t0 = 252.2  # in us
tof_range = [0, 40]  # in us
etof_range = [20, 52]  # in ns

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

xint, yint, tint = __e_coincidence(etof_corr[etof_index], pulse_corr, x, y, t_etof[etof_index])

xint, yint, tint = np.array(xint), np.array(yint), np.array(tint)

# %% i-TOF Coincidence

print('Starting i-ToF Coincidence:', datetime.now().strftime("%H:%M:%S"))
tof_index = np.where(np.logical_and(
    t_tof > tof_range[0], t_tof < tof_range[1]))[0]


xs, ys, ts, tofs = __i_coincidence(
    tof_corr[tof_index], pulse_corr, xint, yint, tint, t_tof[tof_index])

xs = np.array(xs)
ys = np.array(ys)
ts = np.array(ts)
tofs = np.array(tofs)

# %% Plotting
print('Plotting:', datetime.now().strftime("%H:%M:%S"))

etof_bins = int(min(int(np.diff(etof_range)/0.26), 300))
plt.figure(1)
plt.hist(tofs, bins=300, range=tof_range)
# plt.hist(t_tof, bins=300, range=tof_range)

plt.figure(2)
plt.hist(ts, bins=etof_bins, range=etof_range)
# plt.hist(t_etof, bins=etof_bins, range=etof_range)


etof_bins = etof_bins//2
nbins = etof_bins

plt.figure(3)
ax = plt.axes(projection='3d')
h, edges = np.histogramdd((xs, ys, ts), bins=[nbins, nbins, etof_bins], range=[
                          [0, 256], [0, 256], etof_range])


#h = np.log(h+0.5)

xx = np.linspace(0, 256, num=nbins)
yy = np.linspace(0, 256, num=nbins)
zz = np.linspace(etof_range[0], etof_range[1], num=etof_bins)
xc, yc, zc = np.meshgrid(xx, yy, zz)
shape = np.shape(xc)
xc, yc, zc, h = (xc.flatten(), yc.flatten(), zc.flatten(), h.flatten())

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("ToF (ns)")
index = np.where(h >= 0)
mpl.rc('image', cmap='jet')
cm = SM().to_rgba(h[index])
cm[:, 3] = np.sqrt(h[index]/max(h[index]))**1

# ax.scatter3D(yc[index], xc[index], zc[index], color=cm, s=h[index]*100/max(h[index]))

xc, yc, zc, h = (np.reshape(xc, shape), np.reshape(yc, shape),
                 np.reshape(zc, shape), np.reshape(h, shape))

numbins = 100
minbin = 1

numbins = min(numbins, int(h.max())-minbin)

cm = SM().to_rgba(np.array(range(numbins))**1)
cm[:, 3] = (np.array(range(numbins))/numbins)**1.5
mlab.close(all=True)
mlab.figure(figure="surfaces", bgcolor=(1, 1, 1))


for i in range(numbins):
    iso_val = i*(int(h.max())-minbin)/numbins+minbin
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        h, iso_val, spacing=(256/nbins, 256/nbins, np.diff(etof_range)[0]/etof_bins))

    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2]+etof_range[0],
                    color=cm[i], shade=True, zorder=numbins+100-i)

    mlab.triangular_mesh(verts[:, 0]/256, verts[:, 1]/256, verts[:, 2]/np.diff(etof_range)[0],
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
    # mlab.contour3d(h, contours=[iso_val], transparent=True, color=[cm[i][]])
# mlab.axes()
mpl.rc('image', cmap='gray')


mlab.figure(figure="3d", bgcolor=(1, 1, 1))
p = .3

d = np.where(h > 0.08*np.max(h), h/np.max(h), 0)
mlab.pipeline.volume(mlab.pipeline.scalar_field(d**p), vmin=0, vmax=(0.7)**p)

ax.set_xlim(left=0, right=256)
ax.set_ylim(bottom=0, top=256)
ax.set_zlim(bottom=etof_range[0], top=etof_range[1])

# xc, yc, zc = np.meshgrid(xx, yy, zz)

xyhist = np.histogramdd((ys, xs), bins=nbins, range=[[0, 256], [0, 256]])[0]
xyhist[0, 0] = 0
xzhist = np.histogramdd((xs, ts), bins=[nbins, etof_bins], range=[
                        [0, 256], etof_range])[0]
yzhist = np.histogramdd((ys, ts), bins=[nbins, etof_bins], range=[
                        [0, 256], etof_range])[0]

ax.plot_surface(xc[:, :, 0], yc[:, :, 0], zc[:, :, 0],
                rcount=nbins, ccount=nbins, facecolors=SM().to_rgba(xyhist), zorder=-1)
ax.plot_surface(xc[:, 0, :], yc[:, 0, :], zc[:, 0, :],
                rcount=nbins, ccount=etof_bins, facecolors=SM().to_rgba(yzhist), zorder=-2)
ax.plot_surface(xc[0, :, :], yc[0, :, :], zc[0, :, :],
                rcount=nbins, ccount=etof_bins, facecolors=SM().to_rgba(xzhist), zorder=-3)


ax.view_init(elev=15., azim=45.)

plt.figure("Time Resolved VMI (Log)")
plt.suptitle('Time Resolved VMI')

plt.subplot(223)
nhist = 2048
temp = np.histogramdd((ys, xs), bins=nhist, range=[
    [0, 256], [0, 256]])[0]+0.5
temp[0, 0] = 0.5
temp = np.transpose(temp)

plt.imshow(temp, interpolation='gaussian', aspect='auto',
           origin='bottom', extent=(0, 256, 0, 256))
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(224)
temp = np.histogramdd((ts, xs), bins=(etof_bins, nhist), range=[
    etof_range, (0, 256)])[0]+0.5
temp = np.transpose(temp)
plt.imshow(temp, interpolation='gaussian', aspect='auto',
           origin='bottom', extent=(etof_range[0], etof_range[1], 0, 256))
plt.xlabel("t (ns)")
plt.ylabel("x")

plt.subplot(221)
temp = np.histogramdd((ys, ts), bins=[nhist, etof_bins], range=[
    (0, 256), etof_range])[0]+0.5
temp = np.transpose(temp)
plt.imshow(temp, interpolation='gaussian', aspect='auto',
           origin='bottom', extent=(0, 256, etof_range[0], etof_range[1]))
plt.xlabel("y")
plt.ylabel("t (ns)")


xc, yc, zc, h = (xc.flatten(), yc.flatten(), zc.flatten(), h.flatten())

com = [sum(h*xc)/sum(h), sum(h*yc)/sum(h), sum(h*zc)/sum(h)]

xn, yn, zn = ((xc-com[0])/256, (yc-com[1])/256,
              (zc-com[2])/np.diff(etof_range)[0])
# %% Axis Calculations

Ixx = sum(h*yn**2+h*zn**2)
Iyy = sum(h*xn**2+h*zn**2)
Izz = sum(h*yn**2+h*xn**2)

Ixy = sum(h*xn*yn)
Ixz = sum(h*xn*zn)
Iyz = sum(h*zn*yn)

I = [[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]]

evals, evecs = np.linalg.eig(I)

index = np.where(evals == min(evals))[0][0]

# a = np.linspace(-128, 128, num=2)
# ax.plot(com[1]+a*evecs[index][1], com[0]+a*evecs[index]
#         [0], com[2]+a*evecs[index][2], 'r', zorder=1000)


# print(evecs)
# double_arrow(ax, com, evecs[index] * [256, 256, np.diff(etof_range)[0]], 1)

# ax.view_init(elev=45., azim=80.)
# double_arrow(ax, com, evecs[index] * [256, 256, np.diff(etof_range)[0]], 1)

# ax.view_init(elev=45., azim=80.)
# ax.view_init(elev=45., azim=80.)
# ax.view_init(elev=45., azim=80.)
# ax.view_init(elev=45., azim=80.)
# -*- coding: utf-8 -*-
# ax.view_init(elev=45., azim=80.)
# ax.view_init(elev=45., azim=80.)
# -*- coding: utf-8 -*-
print('Finished:', datetime.now().strftime("%H:%M:%S"))
