# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:51:19 2022

@author: mcman
"""
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.interpolate as inter
import scipy.io


def P_xy(x):
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


def P_z(t):
    Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
          (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0))
    # Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)  # old
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


def smear(array, amount=0.26):
    noise = np.random.rand(len(array))*amount
    return array+noise


def rotate_hist(coord_vecs, hist, theta, phi):
    (xv, yv, zv) = coord_vecs
    xx, yy, zz = np.meshgrid(*coord_vecs)

    hist1 = inter.interpn((xv, yv, zv), hist,
                          (xx*np.cos(theta)+yy*np.sin(theta),
                           yy*np.cos(theta)-xx*np.sin(theta),
                           zz),
                          fill_value=0, bounds_error=False)

    hist2 = inter.interpn((xv, yv, zv), hist1,
                          (yy*np.cos(phi)-zz*np.sin(phi),
                          xx,
                          zz * np.cos(phi)+yy*np.sin(phi)),
                          fill_value=0, bounds_error=False)
    return hist2


def index_by_tuple(to_index, index_tuple):
    return tuple(to_index[i] for i in index_tuple)


def dist(x, y, x0, y0):
    return np.sqrt((x-x0)**2+(y-y0)**2)


parser = argparse.ArgumentParser(prog='Get Histogram',
                                 description="Get a 3d Histogram and Axes from a clustered dataset and save the data to a pdf")
parser.add_argument('filename')

args = parser.parse_args(["Data\\xe003_c_cluster.h5"])


x0, y0, t0 = 128, 128, 355
p_range = [-3, 3]
size = 256
polarization = 0  # 25.35
out_name = args.filename[:-5]+".mat"

if __name__ == '__main__':
    with h5py.File(args.filename) as f:
        x, y, t = f['Cluster']['x'][()], f['Cluster']['y'][()], f['Cluster']['t'][()]

    torem = np.concatenate((
        np.where(dist(x, y, 195, 234) < 1.5)[0],
        np.where(dist(x, y, 203, 185) < 1.5)[0],
        np.where(dist(x, y, 253, 110) < 1.5)[0],
        np.where(dist(x, y, 23, 255) < 1.5)[0],
        np.where(dist(x, y, 204, 187) < 1.5)[0],
        np.where(dist(x, y, 98, 163) < 1.5)[0]))

    px, py, pz = P_xy(np.delete(x, torem)-x0), P_xy(np.delete(y, torem)-y0), P_z(smear(np.delete(t, torem))-t0)

    out_dict = {}

    out_dict['xv'] = np.linspace(p_range[0], p_range[1], size, endpoint=True)
    out_dict['yv'] = np.linspace(p_range[0], p_range[1], size, endpoint=True)
    out_dict['zv'] = np.linspace(p_range[0], p_range[1], size, endpoint=True)

    out_dict['hist'] = rotate_hist(index_by_tuple(out_dict, ('xv', 'yv', 'zv')),
                                   np.histogramdd((px, py, pz), bins=size, range=[p_range]*3)[0],
                                   -1, polarization*np.pi/180)

    plt.figure(0)
    plt.imshow(np.sum(out_dict['hist'], 0))
    plt.figure(1)
    plt.imshow(np.sum(out_dict['hist'], 1))
    plt.figure(2)
    plt.imshow(np.sum(out_dict['hist'], 2))

    scipy.io.savemat(out_name, out_dict)
