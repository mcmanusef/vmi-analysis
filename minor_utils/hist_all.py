# -*- coding: utf-8 -*-
"""/[]
Created on Tue Sep 13 13:09:07 2022

@author: mcman
"""

import os
import subprocess
import sys
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

source = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new"
angle = loadmat(r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\Ellipticity measurements\angle.mat")['angle'][0]


def cos2(theta, delta, a, b):
    """Find a cos2 for fit"""
    return a*np.cos((theta-delta)*np.pi/90)+b


with open("angles.txt", 'w') as f:
    for d in os.listdir(source):
        if d[-3:] == '.h5':
            print(d[:-5], end=': ')
            if d[:-5]+'.mat' not in os.listdir(source) or True:
                p = loadmat("J:\\ctgroup\\DATA\\UCONN\\VMI\\VMI\\20220613\\Ellipticity measurements\\" +
                            d[:-5]+"_power.mat")[d[:-5]+"_power"][0]
                fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1], bounds=(0, [180, np.inf, np.inf]))[0]
                a = fit[0] % 180
                print(a)
                # plt.subplots(subplot_kw={'projection': 'polar'})
                # plt.scatter(angle*np.pi/180, p)
                # plt.plot(angle*np.pi/180, cos2(angle, *fit))
                # print(input_file)
                # print(angle[p == max(p)][0] % 180)
                # print(a)
                # print(np.sqrt((fit[2]-fit[1])/(fit[2]+fit[1])))
                out = subprocess.run([sys.executable, "C:/Users/mcman/Code/VMI/get_hist.py",
                                      source+"/"+d, "--pol", str(a+4)], capture_output=True)
                # print(str(out.stdout))
                # print(str(out.stderr))
