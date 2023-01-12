# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:49:55 2022

@author: mcman
"""

import scipy.io
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as spi
from functools import partial
import os
import mayavi.mlab as mlab
import ffmpeg
from scipy import signal
from matplotlib.cm import ScalarMappable as SM
from skimage import measure
import itertools
from matplotlib.widgets import SpanSelector

mpl.rc('image', cmap='jet')
font = {'size': 18}

# plt.rc('font', **font)
# plt.close('all')


@mlab.animate(delay=50, ui=False)
def rotate(f):
    while 1:
        f.scene.camera.azimuth(1)
        f.scene.render()
        yield


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def rotate_save(f, outname, s=256, n=180):
    # f.scene.scene_editor._tool_bar.setVisible(False)
    mlab.view(azimuth=0, elevation=70, roll=30)
    os.makedirs("temp", exist_ok=True)
    for i in range(n):
        print(i)
        f.scene.camera.azimuth(360/n)
        f.scene.render()
        plt.imsave(f"temp/{i:03}.png", mlab.screenshot(figure=f, mode='rgb', antialiased=True))
        # mlab.savefig(f"temp/{i:03}.png", figure=f, size=(s, s))
    if outname in os.listdir():
        os.remove(outname)
    os.remove('temp/000.png')
    try:
        (
            ffmpeg
            .input(r'temp/%03d.png', framerate=25)
            .output(outname)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    for i in os.listdir('temp'):
        os.remove(f'temp/{i}')
    os.rmdir('temp')


def axes(lower, upper, center=(0, 0, 0)):
    xx = yy = zz = np.arange(lower, upper, 0.1)
    xc = np.zeros_like(xx)+center[0]
    yc = np.zeros_like(xx)+center[1]
    zc = np.zeros_like(xx)+center[2]
    mlab.plot3d(xc, yy+yc, zc, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(xc, yc, zz+zc, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(xx+zc, yc, zc, line_width=0.01, tube_radius=0.01)


pi = np.pi
plt.close('all')
mlab.close(all=True)

mpl.rc('image', cmap='jet')
# font = {'size': 16}

# plt.rc('font', **font)
name = "Data\\xe005_e"
d = name+".mat"
filename = d
# for d in sorted(os.listdir("J:\\ctgroup\\DATA\\UCONN\\VMI\\VMI\\20220613\\Analyzed pdfs and matlab workspaces\\")):
# if d[-4:] == ".mat" and f"{d[:-4]}.mp4" not in os.listdir('.'):


# filename = f"J:\\ctgroup\\DATA\\UCONN\\VMI\\VMI\\20220613\\results\\{d}"
# name = d[:-4]
print(name)
data = scipy.io.loadmat(filename, squeeze_me=True)


def gauss3d(sigma):
    x = np.arange(-3, 4, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))


def gauss1d(sigma):
    x = np.arange(-5, 6, 1)   # coordinate arrays -- make sure they contain 0!
    return np.exp(-(x**2)/(2*sigma**2))


def blur3d(array, sigma, width):
    x = np.arange(-width, width+1, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width+1, 1)
    z = np.arange(-width, width+1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return signal.convolve(array, np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)))[width:-width, width:-width, width:-width]


out = np.transpose(blur3d(data["hist"], 2, 6), [1, 0, 2])
x3, y3, z3 = np.meshgrid(data["xv"], data['yv'], data['zv'])
plt.figure()

out_slice = np.where(np.abs(y3) < 0.03, out, 0)
plt.imshow(np.sum(out_slice, axis=0), origin='lower', extent=[np.min(data['xv']), np.max(data['xv']), np.min(data['xv']), np.max(data['xv'])])

hist2d = np.sum(out_slice, axis=0)

xx, yy, zz = np.meshgrid(data['xv'], data['yv'], data['zv'])

r1 = np.sqrt(xx*xx+yy*yy+zz*zz)
interp = spi.RegularGridInterpolator(
    (data['xv'], data['yv'], data['zv']), data['hist'])

rad_hist, edges = np.histogram(r1, weights=data["hist"], bins=1000)


f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)


def average_pairwise(iterable):
    for i in pairwise(iterable):
        yield i[0]/2+i[1]/2


ax1.plot(edges[:-1], rad_hist)
rs = list(average_pairwise(edges))
# ax1.plot(rs, scipy.signal.convolve(rad_hist, gauss1d(1))[3:-3])
ax4.imshow(hist2d, origin='lower', extent=[-1, 1, -1, 1])


def onselect(xmin, xmax):
    print(xmin, xmax)
    xx, yy = np.meshgrid(data['xv'], data['yv'])
    r = np.sqrt(xx*xx+yy*yy)
    theta = np.arctan2(xx, yy)

    mask = np.where((r > xmin)*(r < xmax), 1, 0)

    ax3.imshow(mask*hist2d, origin='lower')
    theta_hist, edges = np.histogram(theta, bins=1000, weights=hist2d*mask)

    smoothed = scipy.signal.convolve(theta_hist, gauss1d(3))[5:-5]

    ax2.plot(list(average_pairwise(edges)), smoothed)

    theta_max = edges[np.where(smoothed == max(smoothed))]
    r_mid = np.average([xmin, xmax])
    loc = np.where(mask*hist2d == np.max(mask*hist2d))
    print(loc)
    x, y = r_mid*np.cos(theta_max), r_mid*np.sin(theta_max)

    x, y = xx[loc], yy[loc]
    print(x)
    ax4.scatter([x, -x], [y, -y])
    f.canvas.draw()


def plot_r(xmin, xmax):
    xx, yy = np.meshgrid(data['xv'], data['yv'])
    ext = np.max(data['xv'])
    r = np.sqrt(xx*xx+yy*yy)
    theta = np.arctan2(xx, yy)
    mask = np.where((r > xmin)*(r < xmax), 1, float("nan"))
    f = plt.figure()
    plt.imshow(mask*hist2d, origin='lower', extent=[-ext, ext, -ext, ext], interpolation='spline36')
    plt.xlim([-xmax, xmax])
    plt.ylim([-xmax, xmax])

    plt.xlabel("$p_x$ (A.U.)", fontdict=font)
    plt.ylabel("$p_y$ (A.U.)", fontdict=font)


span = SpanSelector(
    ax1,
    onselect,
    "horizontal",
    useblit=True
)
