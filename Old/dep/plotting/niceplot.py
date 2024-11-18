# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:21:40 2023

@author: mcman
"""
import h5py
from matplotlib.colors import ListedColormap
from dep.plotting import error_bars_plot as ebp
from Old import cv3_analysis as cv3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os

mpl.rc('image', cmap='jet')
mpl.use('Qt5Agg')

def trans_jet(opaque_point=0.15):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return ListedColormap(cmap_colors)


#
def wrap_between(data, low=0, high=180):
    return (data - low) % (high - low) + low


def main(inputs,
         calibrated=False,
         wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613",
         to_load=100000,
         width=0.6,
         bins=512):

    for name, pol in inputs:

        angle, ell, px, py, pz = load_data(name, pol, wdir=wdir, to_load=to_load, calibrated=calibrated)

        print(pol, ell)

        fig_name= f"e={pol}_{calibrated}"
        fig, ax = plt.subplots(num=fig_name)
        make_fig(ax, px, py, ellipse=True, bins=bins, pol=pol, ell=ell, text=True, angle=angle, width=width)

        fig.savefig(f"{name}_nice.png")


def make_fig(ax, px, py, ellipse=True, bins=256, pol=0.001, ell=0, text=True, angle=0., width=0.8,blurring=0):
    hist,xe,ye=np.histogram2d(px, py, bins=bins, range=[[-width, width], [-width, width]], density=True)

    hist=gaussian_filter(hist, sigma=blurring)
    ax.pcolormesh(xe,ye,hist.T, cmap=trans_jet())
    # ax.hist2d(px, py, bins=bins, range=[[-width, width], [-width, width]], cmap=trans_jet())
    ax.set_aspect('equal', 'box')
    ax.grid()
    ax.set_axisbelow(True)
    if ellipse:
        add_ellipse(ax, ell, pol)
    if text:
        add_text(ax, angle)


def load_data(name, pol, wdir, to_load, calibrated):
    if calibrated:
        ell = abs(pol)
        angle = 0
        with h5py.File(os.path.join(wdir, name)) as f:
            py, px, pz = (f["y"][()], f["x"][()], f["z"][()])
    else:
        power_file = os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat")
        angle_file = os.path.join(wdir, r"Ellipticity measurements\angle.mat")
        data_file = os.path.join(wdir, fr"clust_v3\{name}.cv3")

        angle = np.radians(wrap_between(ebp.get_pol_angle(power_file, angle_file) + 4, -90, 90))
        ell = ebp.get_ell(power_file, angle_file)

        py, px, pz = cv3.load_cv3(data_file,
                                  pol=float(angle),
                                  width=.05,
                                  to_load=to_load)
    return angle, ell, px, py, pz


def add_text(ax, angle):
    text_bg = patches.Rectangle(
        (0.6, 0.8),
        0.4,
        0.2,
        linewidth=1,
        edgecolor='black',
        facecolor='white',
        transform=ax.transAxes,
    )
    ax.add_patch(text_bg)
    ax.text(.8, .95, f"Rotation: {angle:.2f}°", ha='center', va='center', transform=ax.transAxes)
    ax.text(.8, .85, f"Into Page", ha='center', va='center', transform=ax.transAxes)


def add_ellipse(ax, ell, pol, rect=False):
    if rect:
        ellipse_bg = patches.Rectangle(
            (0.8, 0),
            0.2,
            0.2,
            linewidth=1,
            edgecolor='black',
            facecolor='white',
            transform=ax.transAxes
        )
        ax.add_patch(ellipse_bg)
    ellipse = patches.Ellipse(
        (0.9, 0.1),
        0.15,
        0.1,
        linewidth=2,
        edgecolor='red',
        facecolor='white',
        transform=ax.transAxes
    )
    ax.add_patch(ellipse)
    left_triangle = patches.Polygon(
        [[0.9 - 0.01 * np.sign(pol), 0.15],
         [0.9 + 0.01 * np.sign(pol), 0.16],
         [0.9 + 0.01 * np.sign(pol), 0.14]],
        linewidth=1,
        edgecolor='red',
        facecolor='red',
        transform=ax.transAxes,
    )
    ax.add_patch(left_triangle)
    right_triangle = patches.Polygon(
        [[0.90 + 0.01 * np.sign(pol), 0.05],
         [0.9 - 0.01 * np.sign(pol), 0.06],
         [0.9 - 0.01 * np.sign(pol), 0.04]],
        linewidth=1,
        edgecolor='red',
        facecolor='red',
        transform=ax.transAxes,
    )
    ax.add_patch(right_triangle)
    ax.text(.9, .1, f"{ell:.1f}", ha='center', va='center', transform=ax.transAxes)


if __name__ == "__main__":
    # main([("theory_03.h5", 0.3),("theory_06.h5",0.6)], wdir=r"C:\Users\mcman\Code\VMI\Data", calibrated=True)
    main(
        [("xe018_e", 0.6)],
        to_load=1000000,
        # wdir=r"C:\Users\mcman\Code\VMI\Data",
        bins=256,
        width=0.8
    )
