# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:08:56 2023

@author: mcman
"""

import h5py
import numpy as np
import numpy.typing as npt
import functools as ft
import itertools
import random
import matplotlib.pyplot as plt
from numba import njit, vectorize
import matplotlib as mpl
from typing import *
from scipy.optimize import minimize

mpl.rc('image', cmap='jet')

Coords = tuple[list[float], list[float], list[float | int]]
Coord = tuple[float, float, float]


def correlate_tof(data_iter, tof_data):
    dc = next(data_iter, None)
    tc = next(tof_data, None)
    while not (dc is None or tc is None):
        if dc[0] > tc[0]:
            tc = next(tof_data, None)
        elif dc[0] < tc[0]:
            dc = next(data_iter, None)
        else:
            yield (dc[0], tuple(list(dc[1]) + [tc[1]]))
            tc = next(tof_data, None)


@njit(cache=True)
def dist(x, y, x0, y0):
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


@njit(cache=True)
def in_good_pixels(coords: Coords) -> bool:
    x, y, t = coords
    conditions = np.array([dist(x, y, 195, 234) < 1.5,
                           dist(x, y, 203, 185) < 1.5,
                           dist(x, y, 253, 110) < 1.5,
                           dist(x, y, 23, 255) < 1.5,
                           dist(x, y, 204, 187) < 1.5,
                           dist(x, y, 36, 243) < 1.5,
                           dist(x, y, 204, 194) < 2.5,
                           dist(x, y, 172, 200) < 1.5,
                           dist(x, y, 98, 163) < 1.5])
    return not np.any(conditions)


# @vectorize
def P_xy(x):
    return np.sqrt(0.000503545) * (x) * np.sqrt(2 * 0.03675)


#     return (np.sqrt(5.8029e-4)*(x))*np.sqrt(2*0.03675) # Old Calibration


# @vectorize
def P_z(t):
    t = float(t)
    pos = np.poly1d([4.51E+04, 1.63E+06, -5.49E+04, 0, 0])
    neg = np.poly1d([0, 0, 8.65E+04, 0, 0])
    Ez = pos(t / 1000) * (t > 0) + neg(t / 1000) * (t < 0)

    #     Ez = ((6.7984E-05*t**4+5.42E-04*t**3+1.09E-01*t**2)*(t < 0) +
    #           (-5.64489E-05*t**4+3.37E-03*t**3-6.94E-02*t**2)*(t > 0)) # Old Calibration
    #     # Ez = 0.074850168*t**2*(t < 0)-0.034706593*t**2*(t > 0)+3.4926E-05*t**4*(t > 0)  # old
    return np.sqrt(np.abs(Ez)) * (0 + (Ez > 0) - (Ez < 0)) * np.sqrt(2 * 0.03675)


@njit(cache=True)
def smear(x: float, amount: float = 0.26) -> float:
    return x + np.random.rand() * amount


# @njit
def momentum_conversion(coords: Coord) -> Coord:
    x, y, t = coords
    return P_xy(x), P_xy(y), P_z(t)


@njit
def centering(x: Coord, center: Coord = (128, 128, 528.5)) -> Coord:
    return (x[0] - center[0], x[1] - center[1], x[2] - center[2])


@njit(cache=True)
def rotate_coords(coords: Coords, theta: float = 1, phi: float = 0) -> Coords:
    # phi=phi+1

    x, y, z = coords
    xp, yp, zp = x * np.cos(theta) + y * np.sin(theta), y * np.cos(theta) - x * np.sin(theta), z
    return xp, yp * np.cos(phi) - zp * np.sin(phi), zp * np.cos(phi) + yp * np.sin(phi)


def partition(list_in: list, n: int) -> list[list]:
    indices = list(range(len(list_in)))
    random.shuffle(indices)
    index_partitions = [sorted(indices[i::n]) for i in range(n)]
    return [[list_in[i] for i in index_partition]
            for index_partition in index_partitions]


def data_conversion(coords,
                    pol: float = 0.,
                    width: float = 0.05,
                    center: Coord = (128, 128, 528.5),
                    raw: bool = False,
                    electrons: str = "all") -> Coords:
    if raw:
        xf, yf, tf = map(np.array, zip(*list(
                    filter(in_good_pixels, coords))))
        return xf, yf, tf
    else:
        match electrons:
            case 'all':
                def right_direction(x): return True
            case 'down':
                def right_direction(x): return x[2] > 0
            case 'up':
                def right_direction(x): return x[2] < 0
            case _:
                def right_direction(x): return True

        px, py, pz = map(np.array, zip(*list(
            filter(lambda x: abs(x[0]) < width,
                   map(ft.partial(rotate_coords, phi=pol),
                       map(momentum_conversion,
                           filter(right_direction,
                                  map(ft.partial(centering, center=center),
                                      filter(in_good_pixels, coords)))))))))
        return py, pz, px


def load_and_correlate(file: str,
                       to_load: Optional[int] = None,
                       ):
    data = {}
    with h5py.File(file) as f:
        for k in f.keys():
            data[k] = list(f[k][()]) if not to_load else list(f[k][:to_load])

    __, coords = tuple(zip(*list(correlate_tof(
        zip(data["cluster_corr"], zip(data["x"], data["y"])),
        zip(data["etof_corr"], map(smear, list(np.array(data["t_etof"]) / 1000)))))))
    return coords


def load_cv3(file: str,
             pol: float = 0.,
             width: float = 1,
             to_load: Optional[int] = None,
             center: Coord = (124.5, 128.5, 528.536596),
             raw: bool = False,
             electrons: str = "all",
             ) -> Coords:
    coords = load_and_correlate(file, to_load=to_load)
    return data_conversion(coords, pol=pol, width=width, center=center, raw=raw, electrons=electrons)

if __name__ == '__main__':
    file = r"J:\ctgroup\DATA\UCONN\VMI\VMI\20220613\clust_v3\xe003_c.cv3"
    ang = 178.95870837170236

    r0 = r1 = (-128, 128)
    r2 = (-10, 10)
    r0 = r1 = r2 = (-1, 1)
    raw_data = load_and_correlate(file, to_load=10000)
    data = data_conversion(raw_data, pol=ang)

    plt.figure()
    plt.hist2d(data[0], data[1], range=[r0, r1], bins=100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure()
    plt.hist2d(data[1], data[2], range=[r1, r2], bins=100)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.figure()
    plt.hist2d(data[2], data[0], range=[r2, r0], bins=100)
    plt.xlabel('z')
    plt.ylabel('x')

    r = (r0, r1, r2)


    def is_in_ranges(point: Coord, ranges: tuple[tuple, tuple, tuple]) -> bool:
        return all(r[0] < c < r[1] for c, r in zip(point, ranges))


    def center_error(c, data):
        data = data_conversion(data, center=c)
        filtered_data = tuple(map(np.array, zip(*filter(ft.partial(is_in_ranges, ranges=r), zip(*data)))))
        return sum(np.mean(d) ** 2 for d in filtered_data)


    # plt.figure()
    # def ce(cs): return [center_error((128, 128, c), raw_data) for c in cs]
    # plt.plot(np.arange(520, 540, .01), ce(np.arange(520, 540, .01)))

    print(minimize(center_error, (128, 128, 530), args=(raw_data,)))

    # print(*(np.mean(input_file) for input_file in filtered_data))

    #
