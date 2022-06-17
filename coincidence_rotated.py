# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:17:41 2022

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


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
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
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


def P_z(t):
    #Ez = -2.433e9*t**5 + 1.482e8*t**4 - 2.937e6*t**3 + 8722*t**2 - 242*t + 0.04998
    Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def getzwidth(z, x, xzhist):
    """
    Get the width of a distribution at various points along the z axis

    Parameters
    ----------
    z : float[:]
        the coordinates at which to get the width
    x : float[:]
        the coordinates perpendicular to z.
    xzhist : float[:,:]
        The data to find the width of.

    Returns
    -------
    zwidth : float[:]
        the width along the z axis.

    """
    zwidth = np.zeros_like(z)
    for i in range(len(z)):
        if sum(xzhist[:, i]) > 0:
            zwidth[i] = weighted_std(x, xzhist[:, i])
    return zwidth


def radial_profile(data, spacing):
    """
    Perform an angular integration to find the radial profile of the data

    Parameters
    ----------
    data : float[:,:]
        The array of data, centered at the array center
    spacing : float
        The coordinate spacing.

    Returns
    -------
    ring_brightness : float[:]
    radius : float[:]
    """

    y, x = np.indices((data.shape))*spacing  # first determine radii of all pixels
    x, y = x-np.max(x)/2, y-np.max(y)/2

    r = np.sqrt(x**2+y**2)
    r_max = np.max(r)

    a = int((r_max/spacing)//1)

    ring_brightness, radius = np.histogram(r, weights=data, bins=a)
    return ring_brightness, radius


def radial_profile_3d(data, spacing):
    """
    Perform an angular integration to find the radial profile of the data

    Parameters
    ----------
    data : float[:,:, :]
        The array of data, centered at the array center
    spacing : float
        The coordinate spacing.

    Returns
    -------
    ring_brightness : float[:]
    radius : float[:]
    """

    y, x, z = np.indices((data.shape))*spacing  # first determine radii of all pixels
    x, y, z = x-np.max(x)/2, y-np.max(y)/2, z-np.max(z)/2

    r = np.sqrt(x**2+y**2+z**2)
    r_max = np.max(r)

    a = int((r_max/spacing)//1)

    ring_brightness, radius = np.histogram(r, weights=data, bins=a)
    return ring_brightness, radius


def angular_profile(data, spacing, rrange):
    """
    Perform an radial integration to find the angular profile of the data

    Parameters
    ----------
    data : float[:,:]
        The array of data, centered at the array center
    spacing : float
        The coordinate spacing.
    rrange: (float, float)
        The bounds of integration

    Returns
    -------
    ring_brightness : float[:]
    radius : float[:]
    """
    y, x = np.indices((data.shape))*spacing  # first determine radii of all pixels\
    x, y = x-np.max(x)/2, y-np.max(y)/2

    theta = np.angle(x+y*1j)
    r = np.sqrt(x**2+y**2)

    band = np.where(np.logical_and(r >= rrange[0], r <= rrange[1]))

    angular_brightness, angle = np.histogram(
        theta[band], weights=data[band], bins=theta.shape[0]//3)
    return angular_brightness, angle


def radial2d(data, spacing):
    """
    Finds a 2d array of data integrated around the cylindrical axis

    Parameters
    ----------
    data : Float[:,:,:]
        Array of data
    spacing : f
        spacing of grid.

    Returns
    -------
    rb : Float[:,:]
        Brightness of each ring at x.
    r : f[:]
        radius of each ring
    """

    l = data.shape[2]
    testrb, r = radial_profile(data[1, :, :], spacing)

    rb = np.zeros((l, len(testrb)))

    for i in range(l):
        rb[i], r = radial_profile(data[:, i, :], spacing)
    return rb, r


def cos2(theta, delta, amp):
    return amp*np.cos(theta+delta)**2


mpl.rc('image', cmap='jet')
plt.close('all')

# %% Parameters
name = 'mid'
in_name = "xe006_e_cluster.h5"  # "J:\\ctgroup\\DATA\\UCONN\\VMI\\VMI\\20220404\\xe104_cluster.h5"  #

t0 = 0.5  # in us
t0f = 30
tof_range = [0, 40]  # in us
etof_range = [20, 40]  # in ns

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

torem = np.where(np.logical_and(xs == 195, ys == 234))
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

width = 1.5
plot_range = [-width, width]
nbins = 256

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

hp = inter.interpn((x, y, z), h, (Xc*np.cos(theta)+Yc*np.sin(theta),
                                  Yc*np.cos(theta)-Xc*np.sin(theta), Zc),
                   fill_value=0, bounds_error=False,)

plt.figure("Rotation")
plt.imshow(np.sum(hp, axis=2))

# %% Plotting
print('Plotting:', datetime.now().strftime("%H:%M:%S"))


plt.figure('ToF Spectrum')
plt.hist(tofs, bins=100, range=tof_range)

plt.figure('e-ToF Spectrum')
plt.hist(ts, bins=200, range=etof_range)

plt.figure('p_z spectrum')
plt.hist(pz, bins=300, range=plot_range)

# plt.figure('Rotated 3d Projection')
# ax = plt.axes(projection='3d')
mlab.close("Rotated 3d Projection")
mlab.figure(figure="Rotated 3d Projection", bgcolor=(0, 0, 0))

minbin = 1
numbins = min(numbins, int(h.max())-minbin)
cm = SM().to_rgba(np.array(range(numbins))**1)
cm[:, 3] = (np.array(range(numbins))/numbins)**1.5

for i in range(numbins):
    iso_val = i*(int(hp.max())-minbin)/numbins+minbin
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        hp, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))

    # ax.plot_trisurf(verts[:, 0]-width, verts[:, 1]-width, faces, verts[:, 2]-width,
    #                 color=cm[i], shade=True, zorder=numbins+100-i)

    mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
    if i == 0:
        mlab.axes()
mpl.rc('image', cmap='gray')

# ax.set_xlim(left=-width, right=width)
# ax.set_ylim(bottom=-width, top=width)
# ax.set_zlim(bottom=-width, top=width)


plt.figure('VMI Cuts')
for j, i in enumerate(np.linspace(-0.3, 0.3, num=9)):
    c = (0, 0, 0)

    cx = nbins//2+int(c[0]/np.diff(x)[0])
    cy = nbins//2+int(1/np.diff(y)[0]*c[1])
    cz = nbins//2+int(1/np.diff(z)[0]*c[2])

    w = nbins//2

    xyhist = np.sum(hp[:, :, cz-w:cz+w], axis=2)
    xzhist = np.sum(hp[cy-w:cy+w, :, :], axis=0)
    yzhist = np.sum(hp[:, cx-w:cx+w, :], axis=1)

    mpl.rc('image', cmap='gray')
    # ax.plot_surface(Xc[:, :, 0], Yc[:, :, 0], Zc[:, :, 0],
    #                 rcount=nbins, ccount=nbins, facecolors=SM().to_rgba(xyhist), zorder=-1)
    # ax.plot_surface(Xc[:, 0, :], Yc[:, 0, :], Zc[:, 0, :],
    #                 rcount=nbins, ccount=nbins, facecolors=SM().to_rgba(yzhist), zorder=-2)
    # ax.plot_surface(Xc[0, :, :], Yc[0, :, :], Zc[0, :, :],
    #                 rcount=nbins, ccount=nbins, facecolors=SM().to_rgba(xzhist), zorder=-3)
    # ax.view_init(elev=15., azim=45.)

    mpl.rc('image', cmap='viridis')
    # plt.subplot(191+j).set_title("$p_x={px:.2f}$".format(px=i))

    plt.subplot(223)
    plt.imshow(xyhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(224)
    plt.imshow(yzhist, extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("z")
    plt.ylabel("y")

    plt.subplot(221)
    plt.imshow(xzhist.transpose(), extent=(-width, width, -width, width), origin='lower')
    plt.xlabel("x")
    plt.ylabel("z")


plt.figure("Fourier Transform")
plt.imshow(np.log(np.abs(ifftshift(fftn(fftshift(xyhist))))))


plt.figure('width')
plt.plot(z, getzwidth(z, x, xzhist), 'b')
plt.twinx()
plt.plot(z, np.sum(xzhist, axis=0), 'r')


plt.figure("Full Radial Profile")
rp, r = radial_profile_3d(hp, np.diff(x)[0])
plt.plot(r[1:], rp)

radii = [[.2, .3], [.35, .45], [.5, .6], [.62, .68]]

plt.figure('Angular Distribution')
ax2 = plt.axes(projection='polar')
f2 = plt.figure("Angular profile XZ")
ax3 = plt.axes()
for i, r in enumerate(radii):
    # plt.subplot(221+i,projection='polar')
    adist, an = angular_profile(yzhist, np.diff(x)[0], r)
    fit = optim.curve_fit(cos2, an[1:], adist)
    ax2.plot(an[1:], adist, label='ATI Ring {}: angle={}'.format(i+1, fit[0][0]))
    #ax2.plot(an, cos2(an, fit[0][0], fit[0][1]))
    ab, angle = angular_profile(xzhist, np.diff(x)[0], r)
    ax3.plot(angle[1:], ab)
ax2.legend()


rb, r = radial2d(hp, np.diff(x)[0])
plt.figure("2d radial profile")
plt.imshow(rb.transpose(), origin='lower', extent=[-width, width, min(r), max(r)])
