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
name = "xe001_p"
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


out = np.transpose(blur3d(data["full_hist"], 3, 6), [1, 0, 2])
mlab.figure(size=(2000, 2000))

x3, y3, z3 = np.meshgrid(data["xv"], data['yv'], data['zv'])

# xc = np.mean(x3*out)
# yc = np.mean(y3*out)
# zc = np.mean(z3*out)

# dist = np.sqrt(
#     (x3-np.average(x3, weights=out**3))**2 +
#     (y3-np.average(y3, weights=out**3))**2 +
#     (z3-np.average(z3, weights=out**3))**2
# )

# idx = np.where(dist == np.min(dist))
# center = (idx[0][0], idx[1][0], idx[2][0])
# print(center)
# print((x3[center], y3[center], z3[center]))
# print(dist.shape)
# print(out.shape)
# n = 32
# out = out[n:-n, n:-n, n:-n]


out[0, 0, 0] = out[-1, -1, -1] = 100
width = np.max(x3)
nbins = len(out[0, 0])

minbin = 1
numbins = 10
numbins = min(numbins, int(out.max())-minbin)
cm = SM().to_rgba(np.array(range(numbins))**0.5)
cm[:, 3] = (np.array(range(numbins))/numbins)**0.5

for i in range(numbins):
    iso_val = i*(int(out.max())-minbin)/numbins+minbin
    verts, faces, _, _ = measure.marching_cubes(
        out, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))
    mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])
axes(-1, 1)
n = 0
# mlab.figure(2)
# mlab.contour3d(out**0.5, contours=20, extent=[-(1-n/256), (1-n/256), -(1-n/256),
#                                               (1-n/256), -(1-n/256), (1-n/256)], transparent=True)
# # color=(0/256, 128/256, 200/256))


# # (x3[center], y3[center], z3[center])

# # c = (0.06, 0.02, .15)
# # axes(-width*.7, width*.7, c)

# # a = mlab.axes(nb_labels=5, xlabel='$p_y$', ylabel='$p_z$', zlabel='$p_x$')
# # a.font_factor = 0.5

# # mlab.view(azimuth=70, elevation=70, focalpoint=(0, 0, 0), distance=3)
# # axes(-1, 1)

# plt.figure()

# out_slice = np.where(np.abs(y3) < 0.03, out, 0)
# plt.imshow(np.sum(out_slice, axis=0), origin='lower', extent=[np.min(data['xv']), np.max(data['xv']), np.min(data['xv']), np.max(data['xv'])])

# hist2d = np.sum(out_slice, axis=0)

# # plt.imsave(f'{name}_a.png', mlab.screenshot(mode='rgba'))
# # plt.imsave(f'{name}.png', mlab.screenshot(mode='rgb'))
# # mlab.title("P Polarized", size=.4)


# # def circle(r, n=100):
# #     a = np.linspace(0, 2*pi, n, endpoint=True)
# #     return [r*np.cos(a), r*np.sin(a)]


# xx, yy, zz = np.meshgrid(data['xv'], data['yv'], data['zv'])

# r1 = np.sqrt(xx*xx+yy*yy+zz*zz)
# interp = spi.RegularGridInterpolator(
#     (data['xv'], data['yv'], data['zv']), data['hist'])

# rad_hist, edges = np.histogram(r1, weights=data["hist"], bins=1000)


# f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)


# def average_pairwise(iterable):
#     for i in pairwise(iterable):
#         yield i[0]/2+i[1]/2


# ax1.plot(edges[:-1], rad_hist)
# rs = list(average_pairwise(edges))
# # ax1.plot(rs, scipy.signal.convolve(rad_hist, gauss1d(1))[3:-3])
# ax4.imshow(hist2d, origin='lower', extent=[-1, 1, -1, 1])


# def onselect(xmin, xmax):
#     print(xmin, xmax)
#     xx, yy = np.meshgrid(data['xv'], data['yv'])
#     r = np.sqrt(xx*xx+yy*yy)
#     theta = np.arctan2(xx, yy)

#     mask = np.where((r > xmin)*(r < xmax), 1, 0)

#     ax3.imshow(mask*hist2d, origin='lower')
#     theta_hist, edges = np.histogram(theta, bins=1000, weights=hist2d*mask)

#     smoothed = scipy.signal.convolve(theta_hist, gauss1d(3))[5:-5]

#     ax2.plot(list(average_pairwise(edges)), smoothed)

#     theta_max = edges[np.where(smoothed == max(smoothed))]
#     r_mid = np.average([xmin, xmax])
#     loc = np.where(mask*hist2d == np.max(mask*hist2d))
#     print(loc)
#     x, y = r_mid*np.cos(theta_max), r_mid*np.sin(theta_max)

#     x, y = xx[loc], yy[loc]
#     print(x)
#     ax4.scatter([x, -x], [y, -y])
#     f.canvas.draw()


# def plot_r(xmin, xmax):
#     xx, yy = np.meshgrid(data['xv'], data['yv'])
#     ext = np.max(data['xv'])
#     r = np.sqrt(xx*xx+yy*yy)
#     theta = np.arctan2(xx, yy)
#     mask = np.where((r > xmin)*(r < xmax), 1, float("nan"))
#     f = plt.figure()
#     plt.imshow(mask*hist2d, origin='lower', extent=[-ext, ext, -ext, ext], interpolation='spline36')
#     plt.xlim([-xmax, xmax])
#     plt.ylim([-xmax, xmax])


# # for mn, mx in pairwise(np.linspace(0, max(rs))):
# #     onselect(mn, mx)
# span = SpanSelector(
#     ax1,
#     onselect,
#     "horizontal",
#     useblit=True
# )


# # mask = np.where((.0 < r1) & (r1 < .45), 1, 0)

# # f = mlab.figure(size=(1024, 1025))
# # mlab.contour3d(data["full_hist"]*mask, contours=7, transparent=True)
# # mlab.view(elevation=70)

# # # a = rotate(mlab.gcf())

# # r = np.linspace(.2, .35, 25)
# # theta = np.linspace(0, np.pi, 100)
# # phi = np.linspace(-np.pi, np.pi, 100)

# # rr, tt, pp = np.meshgrid(r, theta, phi)
# # z = rr*np.sin(tt)*np.cos(pp)
# # x = rr*np.sin(tt)*np.sin(pp)
# # y = rr*np.cos(tt)

# # mlab.figure()
# # mlab.points3d(x, y, z, interp((x, y, z)))

# # norm2, e2 = np.histogramdd([tt.flatten(), pp.flatten()], bins=50)

# # plt.figure()
# # plt.subplot(321)
# # plt.plot(edges[:-1], hist)
# # plt.xlabel("$p_r$")
# # plt.ylabel("Counts")
# # plt.title(f"{name}: Radial Profile")

# # # plt.figure()
# # plt.subplot(322)
# # plt.imshow(data['yz_proj'], extent=[-1, 1, -1, 1])
# # plt.plot(*circle(min(r)), color='red')
# # plt.plot(*circle(max(r)), color='red')
# # plt.xlabel("$p_y$")
# # plt.ylabel("$p_z$")
# # plt.title(f"{name}: YZ Projection")

# # # plt.figure()
# # plt.subplot(212)
# # plt.imshow(np.sum(interp((x, y, z)), axis=1),
# #            extent=[-np.pi, np.pi, 0, np.pi], aspect='equal')

# # plt.title(f"{name}: 1st ATI Sphere")
# # plt.xlabel("$\phi$")
# # plt.xticks(ticks=[-pi, -pi/2, 0, pi/2, pi],
# #            labels=[r"$-\pi$", r"$-\frac{\pi}{2}$", "$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

# # plt.ylabel(r"$\theta$")
# # plt.yticks(ticks=[0, pi/2, pi], labels=["$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

# # plt.savefig(f"{name}_surface.png")


# # # rotate_save(f, f"{name}.mp4", n=180)

# # # rotate_save(f, f"{name}.mp4", n=180)
