# -*- coding: utf-8 -*-
"""
Renders a 3d image from a mat file

Created on Mon Nov 21 15:56:54 2022

@author: mcman
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import mayavi.mlab as mlab
import ffmpeg
from scipy import signal
from matplotlib.cm import ScalarMappable as SM
from skimage import measure


def axes(lower, upper, center=(0, 0, 0)):
    xx = yy = zz = np.arange(lower, upper, 0.1)
    xc = np.zeros_like(xx)+center[0]
    yc = np.zeros_like(xx)+center[1]
    zc = np.zeros_like(xx)+center[2]
    mlab.plot3d(xc, yy+yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+upper+0.05, center[1], center[2], "y", scale=0.05)
    mlab.plot3d(xc, yc, zz+zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0], center[1]+upper, center[2], "z", scale=0.05)
    mlab.plot3d(xx+zc, yc, zc, line_width=0.01, tube_radius=0.005)
    # mlab.text3d(center[0]+0.025, center[1], center[2]+upper, "x", scale=0.05)


def rotate_save(f, outname, s=256, n=180):
    # f.scene.scene_editor._tool_bar.setVisible(False)
    mlab.view(azimuth=0, elevation=70, distance=3)
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


pi = np.pi
plt.close('all')
mlab.close(all=True)

source = r"r'C:\Users\mcman\Code\VMI\Data'"
name = os.path.join(source, "xe021_e.mat")
# input_file = name+".mat"
filename = d = name

print(name)
data = scipy.io.loadmat(filename)


def gauss3d(sigma):
    x = np.arange(-3, 4, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))


out = signal.convolve(data["hist"], gauss3d(1))[3:-3, 3:-3, 3:-3]
mlab.figure(size=(1000, 1000))

x3, y3, z3 = np.meshgrid(data["xv"], data['yv'], data['zv'])


out[0, 0, 0] = out[-1, -1, -1] = 100
width = np.max(x3)
nbins = len(out[0, 0])

minbin = 1
numbins = 20
numbins = min(numbins, int(out.max())-minbin)
cm = SM(cmap='jet').to_rgba(np.array(range(numbins))**0.7)
cm[:, 3] = (np.array(range(numbins))/numbins)**0.5

for i in range(numbins):
    iso_val = i*(int(out.max())-minbin)/numbins+minbin
    verts, faces, _, _ = measure.marching_cubes(
        out, iso_val, spacing=(2*width/nbins, 2*width/nbins, 2*width/nbins))
    mlab.triangular_mesh(verts[:, 0]-width, verts[:, 1]-width, verts[:, 2]-width,
                         faces, color=tuple(cm[i, :3]), opacity=cm[i, 3])

axes(-width*.7, width*.7)

rotate_save(mlab.gcf(), "test.mp4")
