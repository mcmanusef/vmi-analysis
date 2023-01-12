# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:37:59 2022

@author: mcman
"""

import argparse
import h5py

parser = argparse.ArgumentParser(
    prog='minimize', description="creates a minimal copy of a clustered h5 file")
parser.add_argument('input')
parser.add_argument('output')


torem = ['x', 'y', 't', 'tot', 'toa', 'tdc_time',
         'tdc_type', 'pulse_corr', 'cluster_index', 'pulse_times']
torem_c = ['tot', 'toa']
args = parser.parse_args()
with h5py.File(args.input, mode='r') as f:
    with h5py.File(args.min, mode='w') as fm:
        for ds in f:
            if ds not in torem:
                if ds == 'Cluster':
                    fm.create_group('Cluster')
                    for ds2 in f[ds]:
                        if ds2 not in torem_c:
                            f.copy(ds+"/"+ds2, fm['Cluster'])
                else:
                    f.copy(ds, fm)
