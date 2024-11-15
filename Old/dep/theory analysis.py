# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:44:02 2023

@author: mcman
"""


import itertools
import os

import matplotlib.pyplot as plt  # -*- coding: utf-8 -*-
import numpy as np
import scipy.fft as fft
import scipy.interpolate as inter
import scipy.signal as signal
from scipy.io import loadmat

plt.close("all")
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \boldmath "  # [r"\usepackage{amsmath}"]
# matplotlib.rcParams['font.size'] = 18
"""
Generates a plot of the inter and intra ring rotation, including all intermediate stages with plots
Created on Mon Sep 26 11:36:11 2022

@author: mcman
"""


source = ""


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


# , ('xe015_e.mat', -0.3), ('xe016_e.mat', -0.6)]:  # os.listdir(source):[('xe002_s.mat', -0.1), ('xe014_e.mat', 0.2), ('xe011_e.mat', 0.3), ('xe012_e.mat', 0.5), ('xe013_e.mat', 0.6), ('xe003_c.mat', 0.9)]:
#
# [('theory_01.mat', 0.1), ('theory_03.mat', 0.3), ('theory_04.mat', 0.4), ('theory_05.mat', 0.5), ('theory_06.mat', 0.6)]:
for d, e in [('theory_03_0.mat', 0.3), ('theory_03_1.mat', 0.3), ('theory_03_2.mat', 0.3), ('theory_03_3.mat', 0.3), ('theory_03_4.mat', 0.3), ('theory_03_5.mat', 0.3)]:
    print(d)
    data = loadmat(os.path.join(source, d))
    xv = data['xv'][0]+0.00031
    cut = np.sum(blur3d(data['hist'], 0.01, 5)[:, (np.abs(data['zv']) < 0.1)[0], :], 1)
    adjustx = -0.0002
    fn = inter.RegularGridInterpolator((data['xv'][0]-adjustx, data['yv'][0]), cut, bounds_error=False, fill_value=0)
    # vals = itertools.starmap(fn, itertools.product(x, y))
    xx, yy = np.mgrid[-1:1:0.001, -1:1:0.001]
    resample = fn((xx, yy))

    # %% Full Image
    f = plt.figure(d)

    plt.title(fr"\Large\boldmath$\epsilon={e}$")
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")

    plt.imshow(threshold(resample, 0.00, fraction=True, default=0)**.5,
               extent=[min(xv), max(xv), min(xv), max(xv)], cmap='jet')
    plt.tight_layout()
    plt.savefig(d[:-4]+".png")

    # %% Unwrapped
    rr = np.sqrt(xx**2+yy**2)
    theta = -np.sign(e)*np.arctan2(xx, yy)
    dr = xx[1, 1]-xx[0, 0]

    n = 1000

    rr, theta = np.mgrid[0:0.8:0.8/n, -np.pi:np.pi:2*np.pi/n]

    sample = blur2d(fn((rr*np.cos(theta), rr*np.sin(theta))), 0.01, 10)

    plt.figure(d+" Unwrapped")
    plt.title(fr"\Large\boldmath$\epsilon={e}$ Unwrapped", )
    tt_deg = (np.degrees(theta.flatten())*-np.sign(e)) % 180
    def dup(x): return np.concatenate((x, x))
    plt.hist2d(np.concatenate((tt_deg, tt_deg-180)), dup(rr.flatten()), weights=dup(sample.flatten()), bins=n, cmap='jet')
    plt.xlabel(r"$\theta$")
    plt.ylabel("$p_r$")
    hist, edges = np.histogram(rr.flatten(), weights=sample.flatten(), bins=n, density=True)
    r = getmeans(edges)
    plt.tight_layout()

    peaks = signal.find_peaks(hist*r**2)[0]
    widths = np.asarray(signal.peak_widths(hist*r**2, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(hist*r**2, peaks)[0])

    peaks = peaks[np.where(proms > 0.01)]
    widths = widths[np.where(proms > 0.01)]

    plt.figure(d+" Peaks")
    plt.plot(r, r**2*hist)
    plt.vlines(r[peaks], 0, max(r*hist), colors='r')
    plt.hlines([0.5]*len(peaks), r[peaks]-r[widths], r[peaks]+r[widths], colors='b')
    angles = []
    for i, (peak, width) in enumerate(zip(peaks, widths)):
        mask = np.where(np.abs(rr-r[peak]) < r[width]*0.1, 1, 0)
        tf = theta.flatten() % np.pi
        angulardist, angle_edges = np.histogram(tf, weights=(mask*sample).flatten(), bins=100)
        # if i == 0:
        # plt.figure()
        # plt.hist2d(tf, rr.flatten(), weights=(mask*sample).flatten(), bins=5000)
        # mean_angle = angle_edges[np.argmax(angulardist)]

        fourier = fft.fft(fft.fftshift(angulardist))

        if i == 0:
            plt.figure()
            def normed(x): return (x-np.mean(x))/max(x-np.mean(x))
            plt.plot(getmeans(angle_edges), normed(angulardist))
            plt.plot(getmeans(angle_edges), np.real(np.exp(-2j*getmeans(angle_edges))*fourier[2]/np.abs(fourier[2])))
        mean_angle = -np.angle(fourier[2])/2 % np.pi

        # mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist**2) % np.pi

        if angles != [] and abs(angles[-1]-mean_angle) > np.pi/2:
            angles.append(mean_angle+np.pi*(angles[-1]-mean_angle)/abs(angles[-1]-mean_angle))
        else:
            angles.append(mean_angle)
        # plt.figure()

    # %% Inter Rotation
    plt.figure("Inter-Ring Rotation")
    plt.plot(r[peaks], np.degrees(np.asarray(angles)-0*angles[0]-np.pi * (e > 0))*-np.sign(e), label=str(e),
             linestyle='--', marker='o', linewidth=2, markersize=6)
    plt.ylim(-10, 150)
    plt.legend(fontsize=12)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\theta$", fontweight="bold")
    plt.title("Rotation of Peaks between ATI Rings", fontweight="bold")
    plt.tight_layout()

    plt.figure(d+" Unwrapped", figsize=(4, 4))
    plt.plot(-np.sign(e)*np.degrees(np.asarray(angles)), r[peaks], color='red')
    plt.plot(-np.sign(e)*np.degrees(np.asarray(angles)-np.pi), r[peaks], color='red')

    angles_in = []
    max_angles = []
    devs = []
    r_in = []
    n_inner = 11
    index = np.where(hist[peaks] == max(hist[peaks]))[0][0]
    for i, b in enumerate(np.linspace(-r[widths[index]]*.6, r[widths[index]]*.6, n_inner)):
        mask = np.where(np.abs(rr-r[peaks[index]]-b) < 2*r[width]/n_inner, 1, 0)
        # plt.figure()
        # plt.imshow(mask)

        angulardist, angle_edges = np.histogram(theta.flatten() % np.pi, weights=(mask*sample).flatten(), bins=100)
        if i == n_inner//2 or True:
            ad, ae = np.histogram(theta.flatten(), weights=(mask*sample).flatten(), bins=100, normed=True)

        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist**1) % np.pi
        angle_dev = angular_dev(getmeans(angle_edges), high=np.pi, weights=angulardist**1)
        devs.append(angle_dev)
        r_in.append(r[peaks[index]]+b)

        if angles_in != [] and abs(angles_in[-1]-mean_angle) > np.pi/2:
            angles_in.append(mean_angle+np.pi*(angles_in[-1]-mean_angle)/abs(angles_in[-1]-mean_angle))
        else:
            angles_in.append(mean_angle)

    # %% Intra Rotation
    plt.figure("Mean First Ring Rotation")
    plt.plot(r_in, -np.sign(e)*np.degrees(np.asarray(angles_in)-angles_in[0]), label=str(e), linewidth=2)
    plt.yticks(ticks=[-45, 0, 45, 90])
    plt.legend(fontsize=12)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\Delta\theta$", fontweight="bold")
    plt.title("First Ring Rotation", fontweight="bold")
    plt.tight_layout()

    plt.figure(d+" Unwrapped")
    plt.plot(-np.sign(e)*np.degrees(np.asarray(angles_in)), r_in, color='orange')
    plt.plot(-np.sign(e)*np.degrees(np.asarray(angles_in)-np.pi), r_in, color='orange')

    plt.savefig(d[:-4]+"_polar.png")

plt.figure("Mean First Ring Rotation")
plt.savefig("first_ring.png")

plt.figure("Inter-Ring Rotation")
plt.savefig("inter_ring_absolute.png")
