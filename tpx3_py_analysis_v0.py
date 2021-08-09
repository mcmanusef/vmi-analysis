# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
usage: tpx-to-hdf [-h] [-o OUTPUT_FILE] --layout {quad,single} [--cluster]
                  [--args-file ARGS_FILE] [--keep-filename-order]
                  [--toa-diff TOA_DIFF] [--gaps GAPS]
                  [--cluster-square-size CLUSTER_SQUARE_SIZE]
                  [raw-file [raw-file ...]]

Convert Tpx3 raw data files to HDF5.

positional arguments:
  raw-file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE        output file to write the HDF5 data to
  --layout {quad,single}
                        Use this standard board layout
  --cluster             Cluster the hits and add cluster_index to HDF file
  --args-file ARGS_FILE
                        Read command line arguments from the given file
  --keep-filename-order
                        The filenames need to be processed in the right order
                        for the conversion to work correctly. By default the
                        filenames are processed in lexicographic order, since
                        this gives the right result for filenames containing
                        frame numbers. If you want to process the files
                        exactly in the order presented on the command line,
                        use this option.
  --toa-diff TOA_DIFF   Set maximum TOA difference allowed within a cluster
                        (us)
  --gaps GAPS           Set the gap between chips to this number of pixels
  --cluster-square-size CLUSTER_SQUARE_SIZE
                        To perform clustering, the sensor area is divided in
                        squares to spatially divide the hits. The square can
                        at most hold one active cluster. The default value of
                        the cluster square size is 8, which divides the sensor
                        area in squares of 8x8 pixels. This value should be
                        slightly bigger than the largest expected cluster (so
                        for spherical clusters just larger than the largest
                        diameter).
"""
import itertools
import platform
import argparse
import time
import glob
import os
import sys
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

#This append to the path is the directory where the Amsterdam data analysis package is at
sys.path.append('C:\\Users\\Carlos\\pycode\\timepix3_analysis_v02\\timepix3_analysis')

import tpx3_analysis.standard_layouts as SL
import tpx3_analysis.threaded_sort as TS
from  tpx3_analysis import Converter

#--layout single filein.tpx3 -o fileout.h5 --cluster
#%run tpx3analyze --layout single twoTDCs_000005.tpx3 -o fileout.h5 --cluster --toa-diff 3

boardLayout = SL.singleLayout

delta_shots_max = 900*1e6 #convert \mu s to ps
delta_event_to_tdc2=10*1e6 #convert \mu s to ps

filenamein = 'xe_2tdc_vert_8_000000' 

filenameout = filenamein + '.h5'

# folder_base = r'C:\Users\Carlos\OneDrive - University of Connecticut\DATA\tpxStonyBrook\tpx20210610'
# filepathin = [os.path.join(folder_base,filenamein +'.tpx3' )]

# filepathout = os.path.join(folder_base,filenameout)

maxTOADiff = 1

clusterSquareSize=16

gaps=0
# This is the function that actually makes the conversion from tpx3 to h5
#conv = Converter(filepathin, filepathout, boardLayout, gaps,
 #                clusterSquareSize, maxTOADiff,cluster=True)

"""
os.system("python ./cluster-hdf -o cluster_000007_out_cl16.hdf5 twoTDCs_000007_out_cl16.h5")
  
f16_7=h5py.File('cluster_000007_out_cl16.hdf5');
first_toa16_7 = f16_7['first_toa']
count16_7  = f16_7['count']

plt.plot(first_toa16_7, count16_7)
"""

# Reads the data type fh5.keys() to check the fields of the h5 file
fh5=h5py.File(filenameout);
fh5.keys()
frames = fh5['frame_number']
tdc_time=fh5['tdc_time'] # tdc times are stored in ps
tdc_type = fh5['tdc_type']
x = fh5['x']
y = fh5['y']
tot = fh5['tot']
toa = fh5['toa']
cluster_idx = fh5['cluster_index']
tdc1=np.array([])
tdc2=np.array([])
tdc2_idx=[]
tdc1_idx=[]
tdc2_diff=np.array([])
shot_id=[]
event_id=[]
physical_tdc2_idx=[]
physical_tdc1_idx=[]

tt2=0
last_pixel_idx=0

#tdcs = [idx for idx, element in enumerate(tdc_type) if element ==3]

#numpy array are faster than lists and hdf return lists, it returns a tuple
#so the array of indexes is accessed by tt1np[0]
tdc_time_np = np.array(tdc_time) 
#array of indexes with type=3, tdcd2 rising edge
tt2np = np.where(np.array(tdc_type)==3)

#array of indexes with type=3, tdcd2 rising edge
tt1np = np.where(np.array(tdc_type)==1)

#tdc1 times
tdc1_time_np = tdc_time_np[tt1np[0]]

#tdc2 times
tdc2_time_np = tdc_time_np[tt2np[0]]

tt2np_diff= np.diff(tt2np[0])
tt2_np_single_events = np.where(tt2np_diff >= 4) # if there are ore than 1 tdc_types=3 
#in between tdc1 events then there is more than one tdc2 event between tdc1
# on the contrary if there is only 1 tdc_types=3 in between tdc1 events the difference 
#between indices in tdc_type is less than 4
tt2_np_multiple_events = np.where(tt2np_diff < 4)

# difference in indexes in tdc1, if differences > 2, then there is tdc2 data
tt1np_diff= np.diff(tt1np[0])
tt1_w_tdc2 = np.where(tt1np_diff > 2)
tt1_w_1_tdc2 = np.where(tt1np_diff == 4)
# a difference in index between two tdc_types=1 larger than 2 indicates that 
#there was at least one tdc2 event in between

# These are the tdc1 times where there is at least one tdc2 event
tdc1_time_wtdc2 = tdc1_time_np[tt1_w_tdc2[0]]
# size of tdc1_time_wtdc2 could be smaller or larger than 

# We then need to find the indices of tdc1 that correspond to  tdc1 times
idx_diff_sorted = np.searchsorted(tdc1_time_np, tdc2_time_np)

n, bins, patches=plt.hist(tdc2_time_np-tdc1_time_np[idx_diff_sorted-1], bins=50)

#find differences that are physical
idx_diff_physical = np.where((tdc2_time_np-tdc1_time_np[idx_diff_sorted-1])<delta_shots_max)

#for tdc1_temp in tdc1_time_np:
#    tdc_physical_index = np.logical_and(
#        tdc2_time_np < tdc1_temp+ delta_shots_max,
#        tdc2_time_np > tdc1_temp - delta_shots_max)

# for tt in range(tdc_type.len()):
# #    print tt
#     if tdc_type[tt]==1: # tdc 1 rising edge
#         tdc1_idx.append(tt)
#         tdc1_temp = tdc_time[tt]
#         tdc1=np.append(tdc1,tdc1_temp)
#     if tdc_type[tt]==3: # tdc 2 rising edge
#         tt2=tt2+1
#         tdc2_idx.append(tt)
#         tdc2_temp = tdc_time[tt]
#         tdc2=np.append(tdc2,tdc2_temp)
#         if abs(tdc2_temp-tdc1_temp) < delta_event_to_tdc2:
#             physical_tdc2_idx.append(tt)
#             physical_tdc1_idx.append(tt)
#             tdc2_diff=np.append(tdc2_diff, tdc2_temp-tdc1_temp)
#             # To select physical hits I assume that toa is withing certain time of a physical TDC2 event
            
#  #           while toa[last_pixel_idx] < tdc2_temp +delta_shots_max && toa[last_pixel_idx] > tdc2_temp -delta_shots_max:
#  #               last_pixel_idx=last_pixel_idx+1
                
        
n, bins, patches=plt.hist(np.array(tdc2_diff), bins=50)        



    