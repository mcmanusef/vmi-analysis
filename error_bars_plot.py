# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:20:10 2023

@author: mcman
"""

import itertools
import os
from multiprocessing import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.stats
from scipy.io import loadmat
from scipy.optimize import curve_fit
from cv3_analysis import load_cv3

mpl.rc('image', cmap='RdYlBu_r')


# %%
def pairwise(iterable):
    """Return an iterator that aggregates elements in pairs. Example: pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def organize(x):
    def filter_valid(x): return tuple(filter(lambda j: j[1] is not None, x))

    def transpose_tuple(x): return tuple(zip(*x))

    return tuple(map(lambda l: tuple(map(transpose_tuple, l)),
                     map(transpose_tuple,
                         map(filter_valid, zip(*x)))))


# @njit
def blur2d(array, sigma, width):
    x = np.arange(-width, width + 1, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-width, width + 1, 1)
    xx, yy = np.meshgrid(x, y)
    return signal.convolve(array, np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)))[width:-width, width:-width]


def get_pol_angle(power_file, angle_file):
    def cos2(theta, delta, a, b): return a * np.cos((theta - delta) * np.pi / 90) + b

    angle = loadmat(angle_file)['angle'][0]
    p = next(v for k, v in loadmat(power_file).items() if not k.startswith('__'))[0]
    fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1],
                    bounds=(0, [180, np.inf, np.inf]))[0]
    return fit[0]


# @njit
def edges_to_centers(edges):
    centers = (edges[:-1] + edges[1:]) / 2
    return centers


# @njit
def angular_average(angles, low=0, high=2 * np.pi, weights=None):
    adjusted = (angles - low) / (high - low) * (2 * np.pi)
    xavg = np.average(np.cos(adjusted), weights=weights)
    yavg = np.average(np.sin(adjusted), weights=weights)
    out = np.arctan2(yavg, xavg)
    return out / (2 * np.pi) * (high - low) + low


# @njit
def get_peak_angles(hist, r, t):
    tt, rr = np.meshgrid(t, r)
    rhist, r_edges = np.histogram(rr, weights=hist * rr, bins=len(r), range=[0, 1], density=True)

    peaks = signal.find_peaks(rhist)[0]
    widths = np.asarray(signal.peak_widths(rhist, peaks)[0], dtype=int)
    proms = np.asarray(signal.peak_prominences(rhist, peaks)[0])
    # for i,p in zip(rhist[peaks],proms):
    #     if p/i > 0.3:
    #         print(i,p)
    # print(proms)
    # print(proms/rhist[peaks] > 0.2)
    for peak, width in itertools.compress(zip(peaks, widths),
                                          np.logical_and(proms / rhist[peaks] > 0.3, rhist[peaks] > 0.3)):
        mask = (np.abs(rr - r[peak]) < r[width] / 2)
        if np.sum(hist * mask) > 0:
            cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask))
            yield r[peak], cm, r[width]


# @njit
def get_peak_profile(hist, rs, r, t):
    tt, rr = np.meshgrid(t, r)
    dr = np.diff(rs)[0]
    for ri in rs:
        mask = (np.abs(rr - ri) < dr / 2)
        # print(hist*mask)
        if np.sum(hist * mask) > 0:
            with np.errstate(invalid='raise'):
                try:
                    cm = angular_average(tt % np.pi, high=np.pi, weights=(hist * mask) ** 0.5)
                except Exception as E:
                    with np.printoptions(threshold=np.inf):
                        print(hist * mask)
                    raise E
            yield ri, cm


# @njit
def get_profiles(data, i, n, pol=0., plotting=True):
    print(i)
    px, py, pz = map(lambda x: x[i::n], data)

    pr, ptheta = np.sqrt(px ** 2 + py ** 2), np.arctan2(-px, py)

    rt_hist, r_edges, t_edges = np.histogram2d(pr, ptheta, bins=500, range=[[0, 1], [-np.pi, np.pi]])

    r = edges_to_centers(r_edges)
    rt_hist = np.maximum(blur2d(rt_hist, 2, 10), 0)

    r_peak, theta_peak, width = zip(*get_peak_angles(rt_hist, edges_to_centers(r_edges), edges_to_centers(t_edges)))

    rs = np.linspace(r_peak[0] - width[0] / 2, r_peak[0] + width[0] / 2, num=10)

    if plotting:
        plt.figure(f"{pol}: {i}")
        plt.subplot(221)
        plt.hist2d(px, py, bins=256, range=[[-1, 1], [-1, 1]])
        plt.subplot(212)
        plt.plot(r, np.sum(rt_hist, 1) / max(np.sum(rt_hist, 1)))
        for rp in r_peak:
            plt.axvline(rp, color='r')
        plt.subplot(222)
        plt.imshow(rt_hist ** 0.5, extent=[-np.pi, np.pi, 0, 1], origin='lower', aspect='auto')
        plt.plot(np.array(theta_peak), r_peak)

    try:
        r_p, ang_p = zip(*get_peak_profile(rt_hist, rs, edges_to_centers(r_edges), edges_to_centers(t_edges)))
    except ValueError as E:
        print("No Peaks for intra")
        # print(list(get_peak_profile(rt_hist, rs, edges_to_centers(r_edges), edges_to_centers(t_edges))))
        return (r_peak, theta_peak), (None, None)

    if plotting:
        plt.plot(np.array(ang_p), r_p)

    return (r_peak, theta_peak), (r_p, ang_p)


# %%

if __name__ == '__main__':
    n = 150
    wdir = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613"
    # inputs = [('xe011_e', 0.5)]
    inputs = [('xe014_e', 0.2), ('xe011_e', 0.3), ('xe012_e', 0.5), ('xe013_e', 0.6), ('xe003_c', 0.9)]
    inter_lines = []
    intra_lines = []
    labels = []

    for name, pol in inputs:
        print(name)
        labels.append(pol)

        angle = get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                              os.path.join(wdir, r"Ellipticity measurements\angle.mat"))
        data = load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"), pol=angle)
        with Pool(4) as p:
            def get_profiles_index(i): return get_profiles(data, i, n, pol=pol, plotting=False)


            inter, intra = organize(map(get_profiles_index, range(n)))


        def cm(i): return scipy.stats.circmean(np.asarray(i) % np.pi, high=np.pi)


        def cs(i): return scipy.stats.circstd(np.asarray(i) % np.pi, high=np.pi)


        r_inter, dr_inter = zip(*((cm(i), cs(i)) for i in inter[0]))
        t_inter, dt_inter = zip(*((cm(i), cs(i)) for i in inter[1]))
        inter_lines.append(tuple(map(np.array, (r_inter, dr_inter, t_inter, dt_inter))))

        r_intra, dr_intra = zip(*((cm(i), cs(i)) for i in intra[0]))
        t_intra, dt_intra = zip(*((cm(i), cs(i)) for i in intra[1]))
        intra_lines.append(tuple(map(np.array, (r_intra, dr_intra, t_intra, dt_intra))))
    # %%
    f, ax = plt.subplots(2, 1)
    plt.sca(ax[0])
    plt.title("inter_ring")
    for (r, dr, t, dt), l in zip(inter_lines, labels):
        plt.errorbar(r, t % np.pi, xerr=dr / np.sqrt(n), yerr=dt / np.sqrt(n), label=l)
    plt.legend()
    plt.sca(ax[1])
    plt.title("intra_ring")
    for (r, dr, t, dt), l in zip(intra_lines, labels):
        plt.errorbar(r, (t % np.pi - t[0] % np.pi), xerr=dr / np.sqrt(n), yerr=dt / np.sqrt(n), label=l)
    plt.tight_layout()
# %%
