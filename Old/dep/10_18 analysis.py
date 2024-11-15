import matplotlib.pyplot as plt  # -*- coding: utf-8 -*-
import itertools as it
from scipy.io import loadmat
import scipy.interpolate as inter
import scipy.signal as signal
import scipy.fft as fft
import numpy as np
import os
import matplotlib as mpl
mpl.rc('image', cmap='jet')
mpl.use('Qt5Agg')
"""
Generates a plot of the inter and intra ring rotation, including all intermediate stages with plots
Created on Mon Sep 26 11:36:11 2022

@author: mcman
"""


def threshold(array, n, default=0, fraction=False):
    return np.where(array < n, default, array) if not fraction else np.where(array < n * np.max(array), default, array)


def getmeans(edges):
    return np.asarray(list(map(lambda x: (x[0] + x[1]) / 2, it.pairwise(edges))))


def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out / (2 * np.pi) * (high - low) + low


def angular_dev(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    R = np.sqrt(xavg ** 2 + yavg ** 2)
    return np.sqrt(-2 * np.log(R)) * (high - low) / (2 * np.pi)


def plot_polar_hist(data, rr, tt, fig_name, title, n, e):
    plt.figure(fig_name)
    plt.title(title)
    plt.hist2d(dup_flat(np.degrees(tt) * -np.sign(e) % 180, off=-180),
               dup_flat(rr),
               weights=dup_flat(data[0]),
               bins=n,
               cmap='jet')
    plt.xlabel(r"$\theta$")
    plt.ylabel("$p_r$")
    plt.tight_layout()
    add_line(data[1], data[2], e, 'orange')
    add_line(data[3], data[4], e, 'red')
    plt.savefig(fig_name + ".png")


def add_line(r, theta, e, color):
    plt.plot(-np.sign(e) * np.degrees(np.asarray(theta)), r, color=color)
    plt.plot(-np.sign(e) * np.degrees(np.asarray(theta) - np.pi), r, color=color)


def plot_inter_ring(r_inter, theta_inter, e):
    plt.figure("Inter-Ring Rotation")
    plt.plot(r_inter,
             np.degrees(np.asarray(theta_inter) - 0 * theta_inter[0] - np.pi * (e > 0)) * -np.sign(e),
             label=str(e),
             linestyle='--', marker='o', linewidth=2, markersize=6)
    plt.legend(fontsize=12)
    plt.ylim(-10, 150)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\theta$", fontweight="bold")
    plt.title("Rotation of Peaks between ATI Rings", fontweight="bold")
    plt.tight_layout()


def plot_intra_ring(r_intra, theta_intra, e):
    plt.figure("Mean First Ring Rotation")
    plt.plot(r_intra,
             -np.sign(e) * np.degrees(np.asarray(theta_intra) - theta_intra[0]),
             label=str(e), linewidth=2)
    plt.yticks(ticks=[-45, 0, 45, 90])
    plt.legend(fontsize=12)
    plt.xlabel("$p_r$", fontweight="bold")
    plt.ylabel(r"$\Delta\theta$", fontweight="bold")
    plt.title("First Ring Rotation", fontweight="bold")
    plt.tight_layout()


def get_intra_ring_profile(rr, tt, polar_sample, peaks, widths, r, peak_index):
    angles_in = []
    r_in = []
    n_inner = 11
    for i, b in enumerate(np.linspace(-r[widths[peak_index]] * .6, r[widths[peak_index]] * .6, n_inner)):
        mask = np.where(np.abs(rr - r[peaks[peak_index]] - b) < 2 * r[widths[peak_index]] / n_inner, 1, 0)

        angulardist, angle_edges = np.histogram(tt.flatten() % np.pi, weights=(mask * polar_sample).flatten(),
                                                bins=100)

        mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist ** 1) % np.pi
        r_in.append(r[peaks[peak_index]] + b)

        if angles_in != [] and abs(angles_in[-1] - mean_angle) > np.pi / 2:
            angles_in.append(mean_angle + np.pi * (angles_in[-1] - mean_angle) / abs(angles_in[-1] - mean_angle))
        else:
            angles_in.append(mean_angle)
    return r_in, angles_in


def get_inter_ring_angles(rr, tt, polar_sample, peaks, widths, r, method):
    angles = []
    for i, (peak, width) in enumerate(zip(peaks, widths)):

        mask = get_ring_mask(peak, r, rr, width, w_mult=0.8 if not method == "max" else 0.1)

        period = 2 * np.pi if method == "fourier" else np.pi

        angulardist, angle_edges = np.histogram(tt.flatten() % period, weights=(mask * polar_sample).flatten(),
                                                bins=100)
        match method:
            case "fourier":
                mean_angle = -np.angle(fft.fft(fft.fftshift(angulardist))[2]) / 2 % np.pi
            case "max":
                mean_angle = getmeans(angle_edges)[np.argmax(angulardist)]
            case "mean":
                mean_angle = angular_average(getmeans(angle_edges), high=np.pi, weights=angulardist ** 2) % np.pi
            case _:
                raise NameError

        if angles != [] and abs(angles[-1] - mean_angle) > np.pi / 2:
            angles.append(mean_angle + np.pi * (angles[-1] - mean_angle) / abs(angles[-1] - mean_angle))
        else:
            angles.append(mean_angle)
    return r[peaks], angles


def get_ring_mask(peak, r, rr, width, w_mult=0.8):
    mask = np.where(np.abs(rr - r[peak]) < r[width] * w_mult, 1, 0)
    return mask


def get_peaks(rr, polar_sample, n, fig_name, prom=0.02):
    hist, edges = np.histogram(rr.flatten(), weights=polar_sample.flatten(), bins=n, density=True)
    r = getmeans(edges)
    peaks = signal.find_peaks(hist * r)[0]
    widths = np.asarray(signal.peak_widths(hist * r, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(hist * r, peaks)[0])
    peaks = peaks[np.where(proms > prom)]
    widths = widths[np.where(proms > prom)]
    plt.figure(fig_name + " Peaks")
    plt.plot(r, r * hist)
    plt.vlines(r[peaks], 0, max(r * hist), colors='r')
    return peaks, widths, r, np.where(hist[peaks] == max(hist[peaks]))[0][0]


def dup_flat(x, off=0):
    x1 = x.flatten()
    return np.concatenate((x1, x1 + off))


def plot_cartesian_hist(data, ext, fig_name, title, thresh=0.003):
    plt.figure(fig_name)
    plt.title(title)
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")
    plt.imshow(threshold(data, thresh, fraction=True, default=0) ** 0.5,
               extent=[*ext, *ext], cmap='jet')
    plt.tight_layout()
    plt.savefig(fig_name + ".png")


def main(files=[('xe002_s.mat', -0.1),
                ('xe014_e.mat', 0.2),
                ('xe011_e.mat', 0.3),
                ('xe012_e.mat', 0.5),
                ('xe013_e.mat', 0.6),
                ('xe003_c.mat', 0.9)],
         source=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new"):

    for d, e in files:
        print(d)
        data = loadmat(os.path.join(source, d))
        xv = data['xv'][0] + 0.00001
        cut = np.sum(data['hist'][:, (np.abs(data['zv']) < 0.1)[0], :], 1)
        print(len(xv))

        fn = inter.RegularGridInterpolator((data['xv'][0], data['yv'][0]),
                                           cut,
                                           bounds_error=False,
                                           fill_value=0)
        n = 2048

        xx, yy = np.mgrid[-1:1:2/n, -1:1:2/n]

        rr, tt = np.mgrid[0:0.8:0.8 / n, -np.pi:np.pi:2 * np.pi / n]

        cartesian_sample = cut  # fn((xx, yy))
        polar_sample = fn((rr * np.cos(tt), rr * np.sin(tt)))*rr

        peaks, widths, r, peak_index = get_peaks(rr, polar_sample, n, d)

        r_inter, theta_inter = get_inter_ring_angles(rr, tt, polar_sample, peaks, widths, r, method="max")

        r_intra, theta_intra = get_intra_ring_profile(rr, tt, polar_sample, peaks, widths, r, peak_index)

        plot_cartesian_hist(cartesian_sample, [min(xv), max(xv)], d[:-4], fr"e={e}")

        plot_polar_hist((polar_sample, r_intra, theta_intra, r_inter, theta_inter),
                        rr, tt, d[:-4] + "_polar", fr"e={e}$ Polar", n, e)

        plot_intra_ring(r_intra, theta_intra, e)

        plot_inter_ring(r_inter, theta_inter, e)
    plt.figure("Mean First Ring Rotation")
    plt.savefig("first_ring.png")
    plt.figure("Inter-Ring Rotation")
    plt.savefig("inter_ring_absolute.png")


if __name__ == "__main__":
    # Theory
    plt.close("all")
    main([('theory_03_3.mat', 0.3), ('theory_06_3.mat', 0.6)], r"C:\Users\mcman\Code\VMI")
    # plt.close("all")
    # Experimental Data
    main([('xe011_e.mat', 0.3),('xe013_e.mat', 0.6),])
