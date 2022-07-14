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

parser = argparse.ArgumentParser(
    prog='Remote Analyze', description="Combines and Clusters from J: Drive")


parser.add_argument('--r', dest='reg',
                    default='xe\d\d\d_[ecsp]', help="format of files")
parser.add_argument('--dest', dest='dest',
                    default='', help="destination to save files to (relative)")
parser.add_argument('dir')

args = parser.parse_args()

match = re.compile(args.reg)

for d in os.listdir(args.dir+"/"+args.dest):
    print(d)

for d in os.listdir(args.dir):
    if match.match(d) and match.search(d).end() == len(d):
        print(d)
        if d+"_cluster.h5" not in os.listdir(args.dir+"/"+args.dest):
            print(d)
            print("\tcombining")
            a = subprocess.run([sys.executable, "combiner.py", args.dir+"/"+d,
                                "--out", "temp.h5"], capture_output=True)
            print(a.stdout)
            print(a.stderr)
            print("\tclustering")
            subprocess.run([sys.executable, "get_cluster.py", "temp.h5",
                            "--out", args.dir+"/"+args.dest+"/"+d+"_cluster.h5"])
