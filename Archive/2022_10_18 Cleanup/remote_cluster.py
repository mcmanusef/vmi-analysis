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
                    default='clustered_new', help="destination to save files to (relative)")
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
        if d+"_c.h5" not in os.listdir(args.dir+"/"+args.dest):
            print(d)
            print("\tcombining")
            combine_command = [sys.executable, "combiner.py", args.dir+"/"+d,
                               "--out", "temp.h5"]
            a = subprocess.run(combine_command, capture_output=True)
            print(' '.join(combine_command))
            print(a.stderr)

            print("\tclustering")
            cluster_command = [sys.executable, "cluster1.py", "temp.h5",
                               "--out", args.dir+"/"+args.dest+"/"+d+"_c.h5"]
            b = subprocess.run(cluster_command)
            print(' '.join(cluster_command))
            print(b.stderr)
