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
from numba import njit
from numba import prange
import platform
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# %% Initializing


@njit
def numbadiff(x):
    """
    Numba version of np.diff

    Parameters
    ----------
    x : float[:]
        Input Array.

    Returns
    -------
    float[:]
        np.diff(x)

    """
    return x[1:] - x[:-1]


@njit('f8[:](f8[:], i8)')
def compensate(toa, period):
    """
    Compensate for jumps of period in the toa data. Similar to numpy.unwrap().

    Parameters
    ----------
    toa : float[:]
        The list of toa data with jumps to be unwrapped
    period : int
        The size of the jumps in toa.

    Returns
    -------
    toa_comp : float[:]
        The unwrapped toa data.

    """
    # Compensates for jumps down of period in toa
    diff = numbadiff(toa)
    n = len(toa)
    toa_comp = np.zeros_like(toa)
    toa_comp[0] = toa[0]
    for i in range(n-1):
        if diff[i] < -period/2:
            diff[i] += period
        toa_comp[i+1] = toa_comp[i]+diff[i]
    return toa_comp


# @njit(parallel=True)
def string_process(data):
    """
    Strip newline characters and split data into separate numbers.

    Parameters
    ----------
    data : string[:]
        The list of strings to process.

    Returns
    -------
    out : string[:,:]
        A list of the individual numbers in each line, as strings

    """
    out = [['']]*len(data)
    for i in range(len(data)):
        a = data[i].strip().split()
        out[i] = a
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='converter', description="Converts data from .tpx3 to .h5")

    parser.add_argument('--x', dest='executable',
                        default='TPX3_read_and_convert.exe' if platform.system() == 'Windows' else './a.out',
                        help="The compiled C++ code for converting to .txt")

    parser.add_argument('--out', dest='output',
                        default='notset', help="The output HDF5 file")

    parser.add_argument('--nocomp', action='store_true',
                        help="Do not compensate for 26.8 s jumps in ToA and 107.3 s jumps in TDC")

    parser.add_argument('--noread', action='store_true',
                        help="Read directly from intermediate file")

    parser.add_argument('filename')

    args = parser.parse_args()
    filename = args.filename
    in_name = filename

    out_name = filename[:-4]+'h5' if args.output == 'notset' else args.output

    # %% Running C++ Conversion Code
    if not args.noread:
        print('Starting C++ Conversion:', datetime.now().strftime("%H:%M:%S"))
        proc = subprocess.run([args.executable, filename], capture_output=True)

    # %% Collecting Data from Intermediate File
    print('Collecting Data :', datetime.now().strftime("%H:%M:%S"))
    with open('converted.txt') as f:
        data = f.readlines()
    data = string_process(data)

    print('Sorting Data :', datetime.now().strftime("%H:%M:%S"))
    x = []
    y = []
    toa = []
    tot = []

    tdc_time = []
    tdc_type = []

    for d in data:
        # Sorts data
        if int(d[0]) == 0:
            tdc_type.append(int(d[1])+1)
            tdc_time.append(float(d[2])*1e12)
        elif int(d[0]) == 1:
            toa.append(float(d[1])*1e12)
            tot.append(int(d[2])//25)
            x.append(int(d[3]))
            y.append(int(d[4]))

    # %% Compensating for Discontinuities
    if not args.nocomp:
        print('Starting Compensation:', datetime.now().strftime("%H:%M:%S"))
        toa_max = 26843545600000
        toa = compensate(np.array(toa), toa_max)

        tdc_max = 107374182400000
        tdc_time = compensate(np.array(tdc_time), tdc_max)

    # %% Saving Data to H5 File
    print('Saving:', datetime.now().strftime("%H:%M:%S"))
    with h5py.File(out_name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('toa', data=toa)
        f.create_dataset('tot', data=tot)
        f.create_dataset('tdc_time', data=tdc_time)
        f.create_dataset('tdc_type', data=tdc_type)
