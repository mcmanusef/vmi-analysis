# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:22:05 2022

@author: mcman
"""


import h5py
import numpy as np
import functools
import itertools


def where_is(a, v):
    il = np.searchsorted(a, v, side="left")
    ir = np.searchsorted(a, v, side="right")
    return list(range(il, ir))


def get_from(dict_list, key):
    return list(itertools.chain(*(d[key] for d in dict_list)))


def get_data_dict(i, params=None):
    (c_corr, e_corr, te, ti, i_corr, x, y) = params
    temp_dict = {}
    temp_dict["t_e"] = list(te[where_is(e_corr, i)])
    temp_dict["t_i"] = list(ti[where_is(i_corr, i)])
    temp_dict["clust"] = list(zip(x[where_is(c_corr, i)], y[where_is(c_corr, i)]))
    return temp_dict


f = h5py.File("small_c.h5")
(c_corr, e_corr, te, ti, i_corr, x, y) = tuple(map(lambda x: f[x][()], f.keys()))

gdd = functools.partial(get_data_dict, params=(c_corr, e_corr, te, ti, i_corr, x, y))
data = list(map(gdd, functools.reduce(np.union1d, (c_corr, e_corr, i_corr))))

get = functools.partial(get_from, data)
