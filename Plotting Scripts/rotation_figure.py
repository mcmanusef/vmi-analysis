import itertools
import os

import matplotlib
import matplotlib.pyplot as plt  # -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as inter
import scipy.signal as signal
from scipy.io import loadmat

plt.close("all")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \boldmath "  # [r"\usepackage{amsmath}"]
matplotlib.rcParams['font.size'] = 18
"""
Created on Mon Sep 26 11:36:11 2022

@author: mcman
"""

source = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new"


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def blur3d(array, sigma, width):
    x = np.arange(-width, width + 1, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width + 1, 1)
    z = np.arange(-width, width + 1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return signal.convolve(array, np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2)))[width:-width, width:-width, width:-width]


def blur2d(array, sigma, width):
    x = np.arange(-width, width + 1, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width + 1, 1)
    xx, yy = np.meshgrid(x, y)
    return signal.convolve(array, np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)))[width:-width, width:-width]


def threshold(array, n, default=0, fraction=False):
    return np.where(array < n, default, array) if not fraction else np.where(array < n * np.max(array), default, array)


def getmeans(edges):
    return np.asarray(list(map(lambda x: (x[0] + x[1]) / 2, pairwise(edges))))


def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    # plt.figure()
    # plt.scatter(np.cos(adjusted), np.sin(adjusted), c='r', s=weights)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    # print(out)
    # plt.scatter(xavg, yavg, c='b')
    return out / (2 * np.pi) * (high - low) + low


fig, (ax2, ax1) = plt.subplots(2, 1)
fig.set_size_inches(6, 8)
for d, e in [('xe002_s.mat', -0.1),
             ('xe014_e.mat', 0.2),
             ('xe011_e.mat', 0.3),
             ('xe012_e.mat', 0.5),
             ('xe013_e.mat', 0.6),
             ('xe003_c.mat', 0.9)]:
    print(d)
    data = loadmat(os.path.join(source, d))
    xv = data['xv'][0] + 0.001
    cut = np.sum(blur3d(data['hist'], 0.01, 5)[:, (np.abs(data['zv']) < 0.1)[0], :], 1)
    adjustx = -0.02
    fn = inter.RegularGridInterpolator((data['xv'][0] - adjustx, data['yv'][0]), cut, bounds_error=False, fill_value=0)

    xx, yy = np.mgrid[-1:1:0.001, -1:1:0.001]
    resample = fn((xx, yy))
    dr = xx[1, 1] - xx[0, 0]
    n = 1000

    rr, theta = np.mgrid[0:0.8:0.8 / n, -np.pi:np.pi:2 * np.pi / n]

    sample = blur2d(fn((rr * np.cos(-theta * np.sign(e)), rr * np.sin(-theta * np.sign(e)))), 0.01, 10)

    hist, edges = np.histogram(rr.flatten(), weights=sample.flatten(), bins=n, density=True)
    r = getmeans(edges)

    peaks = signal.find_peaks(hist * r)[0]
    widths = np.asarray(signal.peak_widths(hist * r, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(hist * r, peaks)[0])

    peaks = peaks[np.where(proms > 0.02)]
    widths = widths[np.where(proms > 0.02)]

    angles = []
    for peak, width in zip(peaks, widths):
        mask = np.where(np.abs(rr - r[peak]) < r[width], 1, 0)

        angular_dist, angle_edges = np.histogram(theta.flatten() % np.pi, weights=(mask * sample).flatten(), bins=100)
        # mean_angle = angle_edges[np.where(angular_dist == max(angular_dist))[0][0]]

        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angular_dist ** 2) % np.pi

        if angles != [] and abs(angles[-1] - mean_angle) > np.pi / 2:
            angles.append(mean_angle + np.pi * (angles[-1] - mean_angle) / abs(angles[-1] - mean_angle))
        else:
            angles.append(mean_angle)

    plt.sca(ax1)
    plt.plot(r[peaks], (np.asarray(angles)) * 180 / np.pi, label=f"${np.abs(e)}$", linestyle='--', marker='o',
             linewidth=2, markersize=6)
    # plt.yticks(ticks=[-np.pi/2, -np.pi/4, 0, np.pi/4], labels=["$-90$", "$-45$", "$0$", "$45$"])
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\theta$", fontweight="bold")

    plt.annotate("$b$", xy=(0.9, 0.88), xycoords="axes fraction", fontweight='bold')
    # plt.ylim(-100, 19)
    # plt.title("Rotation of Peaks Between ATI Rings", fontweight="bold")
    plt.tight_layout()

    angles_in = []
    max_angles = []
    r_in = []
    n_inner = 11
    index = np.where(hist[peaks] == max(hist[peaks]))[0][0]
    width_coeff = 0.6

    for b in np.linspace(-r[widths[index]] * width_coeff, r[widths[index]] * width_coeff, n_inner):
        mask = np.where(np.abs(rr - r[peaks[index]] - b) < 2 * r[widths[index]] / n_inner, 1, 0)
        angular_dist, angle_edges = np.histogram(theta.flatten() % np.pi, weights=(mask * sample).flatten(), bins=100)
        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angular_dist ** 0.5) % np.pi
        r_in.append(r[peaks[index]] + b)

        if angles_in != [] and abs(angles_in[-1] - mean_angle) > np.pi / 2:
            angles_in.append(mean_angle + np.pi * (angles_in[-1] - mean_angle) / abs(angles_in[-1] - mean_angle))
        else:
            angles_in.append(mean_angle)

    plt.sca(ax2)
    plt.plot(r_in, (np.asarray(angles_in) - angles_in[0]) * 180 / np.pi, label=f"${np.abs(e)}$", linewidth=1,
             marker='o')
    # plt.yticks(ticks=[-np.pi/4, 0, np.pi/4, np.pi/2], labels=["$-\pi/4$", "$0$", "$\pi/4$", "$\pi/2$"])
    # plt.yticks(ticks=[-np.pi/4, 0, np.pi/4, np.pi/2], labels=["$-45$", "$0$", "$45$", "$90$"])
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\Delta\theta$", fontweight="bold")

    plt.annotate("$a$", xy=(0.04, 0.88), xycoords="axes fraction", fontweight='bold')
    # plt.title("First Ring Rotation", fontweight="bold")
    plt.tight_layout()

plt.xlim(right=0.35)
lines_labels = [ax1.get_legend_handles_labels()]
lines, labels = [sum(i, []) for i in zip(*lines_labels)]
fig.legend(lines, labels, fontsize=12)

plt.savefig("rotation_figure.png")
