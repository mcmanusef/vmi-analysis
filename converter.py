# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:43:56 2021

@author: mcman
"""
import h5py
import numpy as np
import argparse
import subprocess
from datetime import datetime

parser = argparse.ArgumentParser(prog='converter', description="Converts data from .tpx3 to .h5")
parser.add_argument('--x', dest='executable', default='./a.out', help="The compiled C++ code for converting to .txt")
parser.add_argument('--i', dest='intermediate', default='converted.txt', help="The intermediate text file")
parser.add_argument('--nocomp',action='store_true',help="Do not compensate for 26.8 s jumps in TOA")
parser.add_argument('filename')
args = parser.parse_args()

if __name__ == '__main__':
    filename=args.filename
    in_name=filename
    out_name=filename[:-4]+'h5'
    
    print('Starting C++ Conversion:', datetime.now().strftime("%H:%M:%S"))
    proc = subprocess.run([args.executable, filename], capture_output=True)
    
    print('Collecting Data :', datetime.now().strftime("%H:%M:%S"))
    with open(args.intermediate) as f:
        data=f.readlines()
    data=[x.strip().split() for x in data]
    
    
    x=[]
    y=[]
    toa=[]
    tot=[]
    
    tdc_time=[]
    tdc_type=[]
    
    #print(content)
    for d in data:
        if int(d[0])==0:
            tdc_type.append(int(d[1]))
            tdc_time.append(float(d[2])*1e12)
        elif int(d[0])==1:
            toa.append(float(d[1])*1e12)
            tot.append(int(d[2]))
            x.append(int(d[3]))
            y.append(int(d[4]))
    
    if not args.nocomp:
        print('Starting Compensation:', datetime.now().strftime("%H:%M:%S"))
        diff=np.diff(toa)
        correction=[-sum(diff[0:i][np.where(diff[0:i]<0)]) for i in range(len(diff))]
        toa=[toa[i]+correction[min(i,len(correction)-1)] for i in range(len(toa))]
    
    print('Saving:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(out_name,'w') as f:
        f.create_dataset('x',data=x)
        f.create_dataset('y',data=y)
        f.create_dataset('toa',data=toa)
        f.create_dataset('tot',data=tot)
        f.create_dataset('tdc_time',data=tdc_time)
        f.create_dataset('tdc_type',data=tdc_type)