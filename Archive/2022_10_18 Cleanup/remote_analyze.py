# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:55:10 2022

@author: mcman
"""

import os
import subprocess
import sys
import re
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit


parser = argparse.ArgumentParser(
    prog='Remote Analyze', description="Combines and Clusters from J: Drive")


parser.add_argument('--r', dest='reg',
                    default='xe\d\d\d_[ecsp]', help="format of files")
parser.add_argument('--dest', dest='dest',
                    default='clustered', help="destination to save files to (relative)")
parser.add_argument('--pol', dest='pol',
                    default='Ellipticity measurements', help="directory to find polarization data (relative)")
parser.add_argument('--pdf', dest='pdf',
                    default='Analyzed pdfs and matlab workspaces', help="destination to save files to (relative)")
parser.add_argument('--rhp', dest='rhp', nargs='+',
                    default=["xe006_e", "xe015_e", "xe016_e"], help="List of right hand polarized datasets")

parser.add_argument('dir')

args = parser.parse_args()

match = re.compile(args.reg)


def cos2(theta, delta, a, b):
    """Find a cos2 for fit"""
    return a*np.cos((theta-delta)*np.pi/90)+b


for d in sorted(os.listdir(args.dir+"/"+args.dest)):
    print(d)

angle = loadmat(args.dir+"/"+args.pol+"/angle.mat")['angle'][0]

for d in sorted(os.listdir(args.dir)):
    if match.match(d) and match.search(d).end() == len(d):
        print(d)
        if d+"_cluster.h5" not in os.listdir(args.dir+"/"+args.dest):
            print(d)
            print("\tcombining")
            a = subprocess.run([sys.executable, "combiner.py", args.dir+"/"+d,
                                "--out", "temp.h5"], capture_output=True)
            # print(a.stdout)
            print(a.stderr)

            print("\tclustering")
            subprocess.run([sys.executable, "get_cluster.py", "temp.h5",
                            "--out", args.dir+"/"+args.dest+"/"+d+"_cluster.h5"])

        if f"{d}.pdf" not in os.listdir(args.dir+"/"+args.pdf):
            print("\tAnalyzing")
            p = loadmat(args.dir+"/"+args.pol+"/"+d+"_power.mat")[d+"_power"][0]
            fit = curve_fit(cos2, angle, p, bounds=(0, [180, np.inf, np.inf]))[0]
            # plt.figure(d)
            # plt.plot(angle, p)
            # plt.plot(angle, cos2(angle, fit[0], fit[1], fit[2]))
            # print(fit)
            a = fit[1]+fit[2]
            b = -fit[1]+fit[2]

            eli = -np.sqrt(b/a) if d not in args.rhp else np.sqrt(b/a)
            ang = 176-fit[0]

            command = (sys.executable+" " + "analyze_pdf.py"+" " +
                       args.dir+"/"+args.dest+"/"+d+"_cluster.h5"+" " +
                       "--pol"+" " + "{}".format(ang)+" " + "{:.2f}".format(eli)+" " +
                       "--out"+" " + "{}/\"{}\"/".format(args.dir, args.pdf)+d + ".pdf"+" " +
                       "--etof"+" " + "10"+" " + "50"+" " +
                       "--data"+" " + "{}/\"{}\"/".format(args.dir, args.pdf)+d + ".mat")
            print(command)
            os.system(command)
            # a = subprocess.run([sys.executable, "analyze_pdf.py",
            #                     args.dir+"/"+args.dest+"/"+d+"_cluster.h5",
            #                     "--pol", "{}".format(ang), "{:.2f}".format(eli),
            #                     "--out", "{}/\"{}\"/".format(args.dir, args.pdf)+d + ".pdf",
            #                     "--etof", "10", "50",
            #                     "--data", "{}/\"{}\"/".format(args.dir, args.pdf)+d + ".mat"],
            #                    capture_output=True, shell=True)
            # print(a.stdout)
            # print(a.stderr)
