# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:32:35 2023

@author: mcman
"""
import h5py
import numpy as np
import functools
import itertools
import matplotlib.pyplot as plt
from numba import njit, vectorize
import os
from scipy.optimize import curve_fit
from scipy.io import loadmat

# %% Functions
# %%% Helper Functions


def iter_dataset(file, dataset, slice_len=None):
    """Return an iterator that yields the elements of dataset in file"""
    if slice_len is None:
        for chunk in file[dataset].iter_chunks():
            yield from file[dataset][chunk]
    else:
        for chunk in (slice(i, i+slice_len) for i in range(0, len(file[dataset]), slice_len)):
            yield from file[dataset][chunk]


@vectorize
def P_xy(x):
    return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675)


@njit(cache=True)
def dist(x, y, x0, y0):
    return np.sqrt((x-x0)**2+(y-y0)**2)


@njit(cache=True)
def in_good_pixels(coords):
    x, y, t = coords
    conditions = np.array([dist(x, y, 195, 234) < 1.5,
                           dist(x, y, 203, 185) < 1.5,
                           dist(x, y, 253, 110) < 1.5,
                           dist(x, y,  23, 255) < 1.5,
                           dist(x, y, 204, 187) < 1.5,
                           dist(x, y,  98, 163) < 1.5])
    return not np.any(conditions)


@vectorize
def P_z(t):
    Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
          (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0))
    # Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)  # old
    return np.sqrt(np.abs(Ez))*((Ez > 0)+0-(Ez < 0))*np.sqrt(2*0.03675)


@njit(cache=True)
def smear(x, amount=0.26):
    return x+np.random.rand()*amount


@njit
def momentum_conversion(coords):
    x, y, t = coords
    return P_xy(x), P_xy(y), P_z(t)


@njit
def centering(x, center=(128, 128, 528.5)):
    return (x[0]-center[0], x[1]-center[1], x[2]-center[2])


@njit(cache=True)
def rotate_coords(coords, theta=-1, phi=0):
    x, y, z = coords
    xp, yp, zp = x*np.cos(theta)+y*np.sin(theta), y*np.cos(theta)-x*np.sin(theta), z
    return zp * np.cos(phi)+yp*np.sin(phi), yp*np.cos(phi)-zp*np.sin(phi), xp


def split_iter(iterator, n=2):
    iters = enumerate(itertools.tee(iterator, n))


def cos2(theta, delta, a, b):
    """Find a cos2 for fit"""
    return a*np.cos((theta-delta)*np.pi/90)+b


def is_in_slice(coords, width=0.05, axis=2):
    return abs(coords[axis]) < width/2

# %%% Main Functions


def get_pol_angle(power_file, angle_file):
    angle = loadmat(angle_file)['angle'][0]
    p = next(v for k, v in loadmat(power_file).items() if not k.startswith('__'))[0]
    fit = curve_fit(cos2, angle, p, p0=[angle[p == max(p)][0] % 180, 1, 1],
                    bounds=(0, [180, np.inf, np.inf]))[0]
    return fit[0]


def load_and_split(f, n, phi=0, width=0.05, it=True):
    for i in range(n):
        if it:
            coords = itertools.islice(zip(iter_dataset(f, 'x'), iter_dataset(f, 'y'), map(lambda x: x/1000, iter_dataset(f, 't'))), i, None, n)
        else:
            coords = itertools.islice(zip(f['x'][()], f['y'][()], map(lambda x: x/1000, f['t'][()])), i, None, n)
        # yield from iter_dataset(f, 'x')
        yield filter(functools.partial(is_in_slice, width=width),
                     map(functools.partial(rotate_coords, phi=phi),
                         map(momentum_conversion,
                             map(centering,
                                 filter(in_good_pixels, coords)))))


# %% Tests

# def get(f, a):
#     return f[0:a]


file = r'J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clustered_new\xe009_e_c.h5'
with h5py.File(file) as f:
    print(f['x'])
#     # a = f['x'].iter_chunks()
#     # b = list(a)
#     for x in list(itertools.islice(iter_dataset(f, 'x', slice_len=10000), 1000000)):
#         print(x)
#     # a = load_and_split(f, 10)
#     # # b = itertools.islice(zip(iter_dataset(f, 'x'), iter_dataset(f, 'y'), map(lambda x: x/1000, iter_dataset(f, 't'))), i, )
#     # for i in range(10):
#     #     b = next(a)
#     #     print(i, b)
#     #     for j in range(5):
#     #         print(f"\t{j}: {next(b)}")


# # %% Run
# if __name__ == '__main__':
#     wdir = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613"
#     for name, pol in [('xe009_e', -0.1)]:
#         angle = get_pol_angle(os.path.join(wdir, fr"Ellipticity measurements\{name}_power.mat"),
#                               os.path.join(wdir, r"Ellipticity measurements\angle.mat"))
#         with h5py.File(os.path.join(wdir, fr"clustered_new\{name}_c.h5")) as f:
#             # print(f['x'][0:1000], f['y'][0:1000], f['t'][0:1000])
#             for p_coords in load_and_split(f, 3, phi=angle, it=False):
#                 #     print(next(p_coords))
#                 x, y, z = zip(*list(p_coords))
#                 plt.hist2d(x, y, bins=100, range=[[-1, 1], [-1, 1]])
#                 break
