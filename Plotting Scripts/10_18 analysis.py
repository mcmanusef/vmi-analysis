
import matplotlib.pyplot as plt  # -*- coding: utf-8 -*-
import itertools
from scipy.io import loadmat
import scipy.interpolate as inter
import scipy.signal as signal
import scipy.stats as stats
import numpy as np
import os
import matplotlib
plt.close("all")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \boldmath "  # [r"\usepackage{amsmath}"]
matplotlib.rcParams['font.size'] = 18
"""
Generates a plot of the inter and intra ring rotation, including all intermediate stages with plots
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
    x = np.arange(-width, width+1, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width+1, 1)
    z = np.arange(-width, width+1, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    return signal.convolve(array, np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)))[width:-width, width:-width, width:-width]


def blur2d(array, sigma, width):
    x = np.arange(-width, width+1, 1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width+1, 1)
    xx, yy = np.meshgrid(x, y)
    return signal.convolve(array, np.exp(-(xx**2 + yy**2)/(2*sigma**2)))[width:-width, width:-width]


def threshold(array, n, default=0, fraction=False):
    return np.where(array < n, default, array) if not fraction else np.where(array < n*np.max(array), default, array)


def getmeans(edges):
    return np.asarray(list(map(lambda x: (x[0]+x[1])/2, pairwise(edges))))


def angular_average(angles, low=0, high=2*np.pi, weights=None):
    adjusted = (angles-low)/(high-low)*(2*np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out/(2*np.pi)*(high-low)+low


def angular_dev(angles, low=0, high=2*np.pi, weights=None):
    adjusted = (angles-low)/(high-low)*(2*np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    R = np.sqrt(xavg**2+yavg**2)
    return np.sqrt(-2*np.log(R)) * (high-low) / (2*np.pi)


for d, e in [('xe002_s.mat', -0.1), ('xe014_e.mat', 0.2), ('xe011_e.mat', 0.3), ('xe012_e.mat', 0.5), ('xe013_e.mat', 0.6), ('xe003_c.mat', 0.9), ("theory.mat", 0.6)]:  # , ('xe015_e.mat', -0.3), ('xe016_e.mat', -0.6)]:  # os.listdir(source):
    print(d)
    data = loadmat(os.path.join(source, d))
    xv = data['xv'][0]+0.00031
    cut = np.sum(blur3d(data['hist'], 0.01, 5)[:, (np.abs(data['zv']) < 0.1)[0], :], 1)
    adjustx = -0.0002
    fn = inter.RegularGridInterpolator((data['xv'][0]-adjustx, data['yv'][0]), cut, bounds_error=0, fill_value=0)
    # vals = itertools.starmap(fn, itertools.product(x, y))
    xx, yy = np.mgrid[-1:1:0.001, -1:1:0.001]
    resample = fn((xx, yy))

    # %% Full Image
    f = plt.figure(d)

    plt.title(fr"\Large\boldmath$\epsilon={e}$")
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")

    plt.imshow(threshold(resample, 0.003, fraction=True, default=-1)**0.5,
               extent=[min(xv), max(xv), min(xv), max(xv)], cmap='jet')
    plt.tight_layout()
    plt.savefig(d[:-4]+".png")

    # %% Unwrapped
    rr = np.sqrt(xx**2+yy**2)
    theta = np.arctan2(xx, yy)
    dr = xx[1, 1]-xx[0, 0]

    n = 1000

    rr, theta = np.mgrid[0:0.8:0.8/n, -np.pi:np.pi:2*np.pi/n]

    sample = blur2d(fn((rr*np.cos(theta), rr*np.sin(theta))), 0.01, 10)

    plt.figure(d+" Unwrapped")
    plt.title(fr"\Large\boldmath$\epsilon={e}$ Unwrapped", )
    plt.hist2d(theta.flatten(), rr.flatten(), weights=sample.flatten(), bins=n, cmap='jet')
    plt.xlabel(r"$\theta$")
    plt.ylabel("$p_r$")
    hist, edges = np.histogram(rr.flatten(), weights=sample.flatten(), bins=n, density=True)
    r = getmeans(edges)
    plt.tight_layout()

    peaks = signal.find_peaks(hist*r)[0]
    widths = np.asarray(signal.peak_widths(hist*r, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(hist*r, peaks)[0])

    peaks = peaks[np.where(proms > 0.02)]
    widths = widths[np.where(proms > 0.02)]

    plt.figure(d+" Peaks")
    plt.plot(r, r*hist)
    plt.vlines(r[peaks], 0, max(r*hist), colors='r')
    # plt.vlines(r[peaks+widths], 0, max(r*hist), colors='b')
    # plt.vlines(r[peaks-widths], 0, max(r*hist), colors='b')
    angles = []
    for peak, width in zip(peaks, widths):
        mask = np.where(np.abs(rr-r[peak]) < r[width], 1, 0)
        # cartmask = np.where(np.abs(np.sqrt(xx**2+yy**2)-r[peak]) < r[width], 1, -1)

        # plt.figure()
        # plt.hist2d(xx.flatten(), yy.flatten(), weights=((cartmask*resample)**.5).flatten(), bins=2000, cmap='jet')

        angulardist, angle_edges = np.histogram(theta.flatten() % np.pi, weights=(mask*sample).flatten(), bins=100)
        mean_angle = angle_edges[np.where(angulardist == max(angulardist))[0][0]]

        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist**4) % np.pi

        if angles != [] and abs(angles[-1]-mean_angle) > np.pi/2:
            angles.append(mean_angle+np.pi*(angles[-1]-mean_angle)/abs(angles[-1]-mean_angle))
        else:
            angles.append(mean_angle)
        # plt.figure()

    # %% Inter Rotation
    plt.figure("Inter-Ring Rotation")
    plt.plot(r[peaks], np.asarray(angles)-0*angles[0]-np.pi, label=str(e), linestyle='--', marker='o', linewidth=2, markersize=6)
    plt.yticks(ticks=[-np.pi/2, -np.pi/4, 0, np.pi/4], labels=["-$\pi/2$", "$-\pi/4$", "$0$", "$\pi/4$"])
    plt.legend(fontsize=12)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\theta$", fontweight="bold")
    plt.title("Rotation of Peaks between ATI Rings", fontweight="bold")
    plt.tight_layout()

    plt.figure(d+" Unwrapped", figsize=(4, 4))
    plt.plot(np.asarray(angles), r[peaks], color='red')
    plt.plot(np.asarray(angles)-np.pi, r[peaks], color='red')

    angles_in = []
    max_angles = []
    devs = []
    r_in = []
    n_inner = 11
    index = np.where(hist[peaks] == max(hist[peaks]))[0][0]
    plt.figure(d+" Slice")
    for i, b in enumerate(np.linspace(-r[widths[index]]*.6, r[widths[index]]*.6, n_inner)):
        mask = np.where(np.abs(rr-r[peaks[index]]-b) < 2*r[width]/n_inner, 1, 0)
        # plt.figure()
        # plt.imshow(mask)

        angulardist, angle_edges = np.histogram(theta.flatten() % np.pi, weights=(mask*sample).flatten(), bins=100)
        # max_angle = angle_edges[np.where(angulardist == max(angulardist))[0][0]]
        if i == n_inner//2 or True:
            ad, ae = np.histogram(theta.flatten(), weights=(mask*sample).flatten(), bins=100, normed=True)
            # plt.axes(polar=True)
            plt.plot(getmeans(ae), ad)

        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist**1) % np.pi
        angle_dev = angular_dev(getmeans(angle_edges), high=np.pi, weights=angulardist**1)
        devs.append(angle_dev)
        r_in.append(r[peaks[index]]+b)

        if angles_in != [] and abs(angles_in[-1]-mean_angle) > np.pi/2:
            angles_in.append(mean_angle+np.pi*(angles_in[-1]-mean_angle)/abs(angles_in[-1]-mean_angle))
        else:
            angles_in.append(mean_angle)

        # if angles_in == [] or abs(angles_in[-1]-mean_angle) > np.pi/2:
        #     angles_in.append(mean_angle-np.pi)
        # else:
        #     angles_in.append(mean_angle)

        # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.hist2d(theta.flatten(), rr.flatten(), weights=(mask*sample).flatten(), bins=n, cmap='jet')
        # ax2.plot(list(map(np.average, pairwise(angle_edges))), angulardist, color='red')
        # plt.axvline(mean_angle, color='red')
        # plt.axvline(mean_angle+np.pi, color='red')

        # if max_angles == [] or abs(max_angles[-1]-max_angle) > np.pi/2:
        #     max_angles.append(max_angle-np.pi)
        # else:
        #     max_angles.append(max_angle)
    # %% Intra Rotation
    plt.figure("Mean First Ring Rotation")
    plt.plot(r_in, np.asarray(angles_in)-angles_in[0], label=str(e), linewidth=2)
    # plt.errorbar(r_in, np.asarray(angles_in)-angles_in[0], yerr=devs, label=str(e), linewidth=2)
    plt.yticks(ticks=[-np.pi/4, 0, np.pi/4, np.pi/2], labels=["$-\pi/4$", "$0$", "$\pi/4$", "$\pi/2$"])
    plt.legend(fontsize=12)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\Delta\theta$", fontweight="bold")
    plt.title("First Ring Rotation", fontweight="bold")
    plt.tight_layout()
    # plt.figure("Max First Ring Rotation")
    # plt.plot(r_in, np.asarray(max_angles)-max_angles[0], label=str(e))
    # plt.legend()
    # plt.xlabel("$p_r$")
    # plt.ylabel("θ")

    plt.figure(d+" Unwrapped")
    plt.plot(np.asarray(angles_in), r_in, color='orange')
    plt.plot(np.asarray(angles_in)-np.pi, r_in, color='orange')
    # plt.errorbar(np.asarray(angles_in), r_in, xerr=devs, color='orange')
    # plt.errorbar(np.asarray(angles_in)-np.pi, r_in, xerr=devs, color='orange')

    plt.savefig(d[:-4]+"_polar.png")

plt.figure("Mean First Ring Rotation")
plt.savefig("first_ring.png")

plt.figure("Inter-Ring Rotation")
plt.savefig("inter_ring_absolute.png")
