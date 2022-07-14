# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:43:23 2022

@author: mcman
"""

import os
import subprocess
import shutil
import sys
import re
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

source = "J:/ctgroup/DATA/UCONN/VMI/VMI/20220613"
working = "C:/Users/mcman/Documents/VMI/Data"
match = re.compile('xe\d\d\d_[ecsp]')
rhp = ["xe006_e", "xe015_e", "xe016_e"]


def cos2(theta, delta, a, b):
    return a*np.cos((theta-delta)*np.pi/90)+b


for d in os.listdir(source):
    if match.match(d) and len(d) == 7:
        print(d)
        if d not in os.listdir(working):
            os.makedirs(working+"/"+d)
            for i in range(10):
                name = "xe{num:06d}.tpx3".format(num=i)
                print("\t"+name)
                shutil.copyfile(source+"/"+d+"/"+name, working+"/"+d+"/"+name)

for d in os.listdir(working):
    if match.match(d) and len(d) == 7:
        print(d)
        print("\tcombining")
        if d+".h5" not in os.listdir(working):
            subprocess.run([sys.executable, "combiner.py", working+"/"+d], shell=True)
        print("\tclustering")
        if d+"_cluster.h5" not in os.listdir(working):
            subprocess.run([sys.executable, "get_cluster.py", working+"/"+d+".h5"], shell=True)


angle = loadmat(source+"/angle.mat")['angle'][0]
for d in os.listdir(source):
    if d[-11:] == "_cluster.h5":
        print(d)
        p = loadmat(source+"/"+d[:-11]+"_power.mat")[d[:-11]+"_power"][0]
        fit = curve_fit(cos2, angle, p, bounds=(0, [180, np.inf, np.inf]))[0]
        # plt.figure(d)
        # plt.plot(angle, p)
        # plt.plot(angle, cos2(angle, fit[0], fit[1], fit[2]))
        # print(fit)
        a = fit[1]+fit[2]
        b = -fit[1]+fit[2]
        eli = -np.sqrt(b/a) if d[:-11] not in rhp else np.sqrt(b/a)
        ang = 176-fit[0]
        print(ang, eli)
        if d[6] == 'c':
            a = subprocess.run([sys.executable, "C:/Users/mcman/Documents/VMI/analyze_pdf.py",
                                source+"/"+d,
                                "--pol", "{}".format(ang), "{:.2f}".format(eli),
                                "--out", "{}/results/".format(working)+d[:7] + ".pdf",
                                "--etof", "10", "50",
                                "--data", "{}/results/".format(working)+d[:7] + ".mat"],
                               capture_output=True, shell=True)
            print(a.stdout)
            print(a.stderr)
