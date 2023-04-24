# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:21:40 2023

@author: mcman
"""
import h5py
from matplotlib.colors import ListedColormap
from plotting import error_bars_plot as ebp
import cv3_analysis as cv3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def trans_jet(opaque_point=0.1):
    # Create a colormap that looks like jet
    cmap = plt.cm.jet

    # Create a new colormap that is transparent at low values
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:int(cmap.N * opaque_point), -1] = np.linspace(0, 1, int(cmap.N * opaque_point))
    return ListedColormap(cmap_colors)


#
def wrap_between(data, low=0, high=180):
    return (data - low) % (high - low) + low


def main(inputs=None,
         calibrated=False,
         wdir=r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613",
         to_load=100000,
         width=0.6):
    if inputs is None:
        inputs = [('xe011_e', 0.3), ('xe013_e', 0.6)]

    for name, pol in inputs:
        #     if f"{name}_nice.png" in os.listdir():
        #         continue

        if calibrated:
            ell = abs(pol)
            angle = 0
            with h5py.File(os.path.join(wdir, name)) as f:
                py, px, pz = (f["y"][()], f["x"][()], f["z"][()])
        else:
            angle = wrap_between(ebp.get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                                                   os.path.join(wdir, r"Ellipticity measurements\angle.mat")) + 4, -90,
                                 90)

            ell = ebp.get_ell(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
                              os.path.join(wdir, r"Ellipticity measurements\angle.mat"))
            # noinspection PyTypeChecker
            py, px, pz = cv3.load_cv3(os.path.join(wdir, fr"clust_v3\{name}.cv3"), pol=np.radians(angle), width=.05,
                                      to_load=to_load)

        print(pol, ell)
        fig, ax = plt.subplots(num=f"e={pol}_{calibrated}")

        ax.hist2d(px, py, bins=512, range=[[-width, width], [-width, width]], cmap=trans_jet())
        ax.set_aspect('equal', 'box')
        ax.grid()
        ax.set_axisbelow(True)

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

        # Add and ellipse
        ellipse = patches.Ellipse(
            (0.9, 0.1),
            0.15,
            0.1,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            transform=ax.transAxes
        )
        ax.add_patch(ellipse)

        # Add two triangles

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

        ax.text(.9, .1, f"{ell:.3f}", ha='center', va='center', transform=ax.transAxes)

        text_bg = patches.Rectangle(
            (0.6, 0.8),
            0.4,
            0.2,
            linewidth=1,
            edgecolor='black',
            facecolor='white',
            transform=ax.transAxes,)
        ax.add_patch(text_bg)

        ax.text(.8, .95, f"Rotation: {angle:.2f}°", ha='center', va='center', transform=ax.transAxes)
        ax.text(.8, .85, f"Into Page", ha='center', va='center', transform=ax.transAxes)

        # plt.tight_layout()
        fig.savefig(f"{name}_nice.png")


if __name__ == "__main__":
    # main([("theory_03.h5", 0.3),("theory_06.h5",0.6)], wdir=r"C:\Users\mcman\Code\VMI\Data", calibrated=True)
    main([("xe011_e", 0.3)], to_load=1000000, wdir=r"C:\Users\mcman\Code\VMI\Data")
