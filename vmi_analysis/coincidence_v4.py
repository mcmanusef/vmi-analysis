import datetime
import os

import h5py
import numpy as np
import scipy


def iter_first(*iterables):
    iterators = [iter(i) for i in iterables]
    last_vals = [next(it) for it in iterators]
    while True:
        min_index = min([(val[0], i) for i, val in enumerate(last_vals)])[1]
        try:
            last_vals[min_index] = next(iterators[min_index])
        except StopIteration:
            break
        yield tuple(last_vals)


def correlate(*iterables):
    for data in iter_first(*iterables):
        corrs, vals = zip(*data)
        if all(c == corrs[0] for c in corrs):
            yield vals


def load_file_coin(filename):
    with h5py.File(filename, mode='r') as f:
        print(f"Loading {filename} (Coincidence):"
              f" {len(f['cluster_corr'])} clusters,"
              f" {len(f['etof_corr'])} etofs,"
              f" {len(f['tof_corr'])} itofs")
        clusters = zip(f['cluster_corr'][()], zip(f['x'][()], f['y'][()], f['t'][()]))
        etof = zip(f['etof_corr'][()], f['t_etof'][()])
        itof = zip(f['tof_corr'][()], f['t_tof'][()])
    c_data, etof, itof = zip(*correlate(clusters, etof, itof))
    x, y, t = zip(*c_data)
    return tuple(map(np.asarray, (x, y, t, etof, itof)))


def load_file_nc(filename):
    with h5py.File(filename, mode='r') as f:
        print(f"Loading {filename} (No Coincidence:"
              f" {len(f['cluster_corr'])} clusters,"
              f" {len(f['etof_corr'])} etofs,"
              f" {len(f['tof_corr'])} itofs")
        clusters = zip(f['cluster_corr'][()], zip(f['x'][()], f['y'][()], f['t'][()]))
        etof = zip(f['etof_corr'][()], f['t_etof'][()])
        itof = zip(f['tof_corr'][()], f['t_tof'][()])
    c_data, etof = zip(*correlate(clusters, etof))
    x, y, t = zip(*c_data)
    return tuple(map(np.asarray, (x, y, t, etof)))


def load_file(filename, ext=None, force_recalc=False, coincidence=True):
    if ext is None:
        ext = lambda x: x.replace('.cv4', '.mat')
    if not coincidence:
        return *load_file_nc(filename), np.array([])
    cache_file = ext(filename)
    if force_recalc or not os.path.exists(cache_file):
        print(f"Calculating {cache_file}")
        coincidence_data = load_file_coin(filename)
        x, y, t, etof, itof = coincidence_data
        print(f"Saving to {cache_file}")
        scipy.io.savemat(cache_file, {'x': x, 'y': y, 't': t, 'etof': etof, 'itof': itof})
        return x, y, t, etof, itof
    else:
        print(f"Loading {cache_file}")
        data = scipy.io.loadmat(cache_file, squeeze_me=True, struct_as_record=False)
        x, y, t, etof, itof = data['x'], data['y'], data['t'], data['etof'], data['itof']
        return x, y, t, etof, itof


def rotate_data(x, y, theta):
    return x * np.cos(theta) + y * np.sin(theta), y * np.cos(theta) - x * np.sin(theta)


#%%
""
