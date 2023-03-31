# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:05:19 2023

@author: mcman
"""
import os
import numpy as np
from error_bars_plot import get_pol_angle

wdir = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\Ellipticity measurements"
ang_file = os.path.join(wdir, "angle.mat")
out_file = os.path.join(wdir, "major_axes.txt")

# with open(out_file, mode='w') as f:
# print("file, angle (deg), angle (rad)", file=f)
for power_file in filter(lambda x: x.split('_')[-1] == 'power.mat', os.listdir(wdir)):
    ang = get_pol_angle(os.path.join(wdir, power_file), ang_file, plotting=True)
    # print(f"{power_file.removesuffix('_power.mat')}, {ang:.5f}, {np.radians(ang):.5f}", file=f)
    break
