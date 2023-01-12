# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:43:11 2022
@author: mcman
"""
# %% Initializing
import warnings
from numba import errors
import h5py
import numpy as np
# import numpy.typing as npt
import argparse
from numba import njit
from datetime import datetime
# import numba as nb
from numba.typed import List
import functools
import itertools
import sklearn.cluster as skcluster
from multiprocessing import Pool
# from typing import Optional
# array = npt.NDArray
warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)


# %% Functions

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


def h5append(dataset, newdata):
    """
    Append data to the end of a h5 dataset

    Parameters
    ----------
    dataset : h5py Dataset
        The dataset to append data to.
    newdata : Array
        The data to be appended.

    Returns
    -------
    None.

    """
    dataset.resize((dataset.shape[0] + len(newdata)), axis=0)
    dataset[-len(newdata):] = newdata


def is_val(iterable, val):
    for i in iterable:
        yield i == val


def compare_diff(time, cutoff=1, greater=True):
    if greater:
        return (time[1]-time[0] > cutoff)
    else:
        return (time[1]-time[0] < cutoff)


def index_iter(iter_tuple, index):
    return map(lambda x: x[index], iter_tuple)


def split_iter(iter_tuple, n):
    enum = enumerate(itertools.tee(iter_tuple, n))
    return tuple(map(lambda x: index_iter(x[1], x[0]), enum))


def toa_correct(toa_uncorr):
    return np.where(np.logical_and(x >= 194, x < 204), toa_uncorr-25000, toa_uncorr)


def correct_pulse_times(pulse_times, cutoff=(1e9+1.5e4), diff=12666):
    return np.where(np.diff(pulse_times) > (1e9+1.5e4), pulse_times[1:]-diff, pulse_times[1:])


def iter_dataset(file, dataset):
    for chunk in file[dataset].iter_chunks():
        for i in file[dataset][chunk]:
            yield i


def iter_file(file):
    iter_fn = functools.partial(iter_dataset, file)
    datasets = ['tdc_time', 'tdc_type', 'x', 'y', 'tot', 'toa']
    return tuple(map(iter_fn, datasets))


def get_times_iter(tdc_time, tdc_type, mode, cutoff):
    if mode == 'etof':
        return itertools.compress(tdc_time, is_val(tdc_type, 3))

    else:
        times = itertools.compress(pairwise(tdc_time), is_val(tdc_type, 1))
        times, pair_times = itertools.tee(times)
        comp = functools.partial(compare_diff, cutoff=cutoff*1000, greater=(mode == 'pulse'))
        return itertools.compress(index_iter(times, 0), map(comp, pair_times))


def get_t_iter(pulse_times, times):
    pte = enumerate(pulse_times)
    i, t0 = next(pte, (-1, -1))
    for time in times:
        while time < t0:
            i, t0 = next(pte, (-1, -1))
        if i == -1:
            break
        else:
            yield i, time-t0


def read_file(filename):
    with h5py.File(filename, mode='r') as fh5:
        return (fh5['tdc_time'][()], fh5['tdc_type'][()], fh5['x'][()],
                fh5['y'][()], fh5['tot'][()], fh5['toa'][()])


# @njit
def get_times(tdc_time, tdc_type, mode, cutoff):
    """
    Get the array of times of a given type of event from the raw TDC data

    Parameters
    ----------
    tdc_time : array[int]
        The array of TDC events.
    tdc_type : array[int]
        The array of TDC types.
    mode : str
        The type of event to gather: 'pulse', 'etof', or 'itof'.
    cutoff : float|int
        Cutoff between the length of laser trigger and itof events (in ns).

    Raises
    ------
    NotImplementedError
        Other input given for mode.

    Returns
    -------
    times: array[int]
        The array of raw times of the given event type(in ps).

    """
    if mode == 'etof':
        return tdc_time[tdc_type == 3]
    else:
        times = tdc_time[np.where(tdc_type == 1)]
        if not tdc_type[-1] == 1:
            lengths = np.diff(tdc_time)[np.where(tdc_type == 1)]
            if mode == 'pulse':
                return times[np.where(lengths > 1e3*cutoff)]
            elif mode == 'itof':
                return times[np.where(lengths < 1e3*cutoff)]
            else:
                raise NotImplementedError
        else:
            lengths = np.diff(tdc_time)[np.where(tdc_type == 1)[:-1]]
            if mode == 'pulse':
                return times[np.where(lengths > 1e3*cutoff)[:-1]]
            elif mode == 'itof':
                return times[np.where(lengths < 1e3*cutoff)[:-1]]
            else:
                raise NotImplementedError


# @njit
def get_t(pulse_times, times):
    """
    Find the time after a laser trigger for each event in times

    Parameters
    ----------
    pulse_times : array[int]
        Times of the laser pulses in ps.
    times : array[int]
        Times of the events in ps.

    Returns
    -------
    t : array[int]
        Time of the event after a laser pulse, in ps.
    corr : array[int]
        The index of the corresponding laser pulses
    """
    corr = np.searchsorted(pulse_times, times)-1
    t = (times-pulse_times[corr])
    return t, corr


# @njit
def find_where(corr, num, lists):
    idxl = np.searchsorted(corr, num, 'left')
    idxr = np.searchsorted(corr, num, 'right')
    return [x[idxl:idxr] for x in lists]


def cluster(data):
    """
    Find clusters of pixel events in data

    Parameters
    ----------
    data : list[array]
        List of arrays, of which the first two elements provide the coordinates

    Returns
    -------
    clst: array[int]
        The corresponding cluster index
    data: list[array]
        The initial data

    """
    if len(data[0]) == 0:
        return np.array([]), data

    dbscan = skcluster.DBSCAN(eps=1, min_samples=4)
    x = data[0]
    y = data[1]
    db = dbscan.fit(np.column_stack((x, y)))
    clst = db.labels_
    return clst, data


# @njit
def average_i(ind, data, weight, i):
    return tuple([np.average(d[ind == i], weights=weight[ind == i]) for d in data[:-1]])


# @njit
def average_over_cluster(cluster_index, data):
    """
    Average the data over each cluster.

    Parameters
    ----------
    cluster_index : array[int]
        Array of cluster indicies
    data : list[array]
        List of arrays to average over each cluster.

    Returns
    -------
    mean_val: list[tuple]
        List of the averaged values for each cluster

    """
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        count = max(cluster_index)+1
        weight = data[-1]
        av_i = functools.partial(average_i, cluster_index, data, weight)
        mean_vals = list(map(av_i, range(count)))
        return mean_vals
    else:
        return []


@njit
def average_over_cluster_jit(cluster_index, data):
    """
    Average the data over each cluster.

    Parameters
    ----------
    cluster_index : array[int]
        Array of cluster indicies
    data : list[array]
        List of arrays to average over each cluster.

    Returns
    -------
    mean_val: list[tuple]
        List of the averaged values for each cluster

    """
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        count = max(cluster_index)+1
        weight = data[-1]
        mean_vals = List()
        for i in range(count):
            elements = (np.average(data[0][cluster_index == i], weights=weight[cluster_index == i]),
                        np.average(data[1][cluster_index == i], weights=weight[cluster_index == i]),
                        np.average(data[2][cluster_index == i], weights=weight[cluster_index == i]))
            mean_vals.append(elements)
        return mean_vals
    else:
        mean_vals = List([(0., 0., 0.)])
        mean_vals.pop()
        return mean_vals


# @njit
def list_enum(enum):
    i, li = enum[0], enum[1]
    return [(i, x) for x in li]


def correlate_tof(data, t_tof=None, tof_corr=None):
    tof_sel = find_where(tof_corr, data[0], [t_tof])[0]
    if len(tof_sel) != 1:
        return None
    else:
        return (data[0], tuple(list(data[1][:-1])+[tof_sel[0]]))


def filter_exists(iterable):
    return filter(lambda x: x is not None, iterable)


def save(name, data_dict, mode='w'):
    with h5py.File(name, mode) as f:
        for key in data_dict:
            f.create_dataset(key, data=data_dict[key], chunks=True)


def save_iter(name, corr_data, tof_data, groupsize=1000):
    with h5py.File(name, 'w') as f:
        xd = f.create_dataset('x', data=[0.], chunks=True, maxshape=(None,))
        yd = f.create_dataset('y', data=[0.], chunks=True, maxshape=(None,))
        td = f.create_dataset('t', data=[0.], chunks=True, maxshape=(None,))
        corrd = f.create_dataset('cluster_corr', data=[0], chunks=True, maxshape=(None,))
        for split in split_every(groupsize, corr_data):
            (corr, coords) = tuple(zip(*split))
            (x, y, t) = tuple(zip(*coords))
            h5append(xd, x)
            h5append(yd, y)
            h5append(td, t)
            h5append(corrd, corr)

        t_tof_d = f.create_dataset('t_tof', data=[0.], chunks=True, maxshape=(None,))
        tof_corr_d = f.create_dataset('tof_corr', data=[0], chunks=True, maxshape=(None,))
        for split in split_every(groupsize, tof_data):
            (tof_corr, t_tof) = tuple(zip(*split))
            h5append(t_tof_d, t_tof)
            h5append(tof_corr_d, tof_corr)


# %% Argument Parsing
parser = argparse.ArgumentParser(prog='cluster',
                                 description="Clusters data with rejection based on pulse time differences")

parser.add_argument('--g', dest='groupsize',
                    type=int, default=int(1e9),
                    help="Clusters this many pulses at a time")

parser.add_argument('--cutoff', dest='cutoff',
                    type=float, default=500,
                    help="Time cutoff between itof and laser pulses in ns")

parser.add_argument('--single', action='store_true',
                    help="Do not use multithreading to preserve RAM")

parser.add_argument('--out', dest='output',
                    default='', help="The output HDF5 file")

parser.add_argument('filename')
args = parser.parse_args(['small.h5'])

# %% Loading Data
if __name__ == '__main__':

    print('Loading Data:', datetime.now().strftime("%H:%M:%S"))
    p = Pool(8)
    output_name = args.output if args.output else args.filename[:-3]+"_c.h5"

    (tdc_time, tdc_type, x, y, tot, toa) = read_file(args.filename)

    pulse_times = correct_pulse_times(get_times(tdc_time, tdc_type, 'pulse', args.cutoff))
    etof_times = get_times(tdc_time, tdc_type, 'etof', args.cutoff)
    itof_times = get_times(tdc_time, tdc_type, 'itof', args.cutoff)

    t_pixel, pixel_corr = get_t(pulse_times, toa)
    t_etof, etof_corr = get_t(pulse_times, etof_times)
    t_itof, itof_corr = get_t(pulse_times, itof_times)

    find_data = functools.partial(find_where, pixel_corr, lists=[x, y, t_pixel.astype(int), tot])

    formatted_data = map(find_data, range(max(pixel_corr)))

    clustered_data = p.imap(cluster, formatted_data, chunksize=1000)

    averaged_cluster_data = itertools.starmap(average_over_cluster, clustered_data)

    enumerated_data = itertools.chain.from_iterable(map(list_enum, enumerate(averaged_cluster_data)))

    correlate_etof = functools.partial(correlate_tof, t_tof=t_etof, tof_corr=etof_corr)

    correlated_data = filter_exists((map(correlate_etof, enumerated_data)))

    # cluster_corr, cluster_data = (list(i) for i in tuple(zip(*correlated_data)))

    # cluster_x, cluster_y, cluster_t = tuple(np.column_stack(cluster_data))

    # cluster_corr, cluster_data = split_iter(correlated_data, 2)

    # cluster_x, cluster_y, cluster_t = split_iter(cluster_data, 3)

    # to_save = {"x": cluster_x, "y": cluster_y, "t": cluster_t, "cluster_corr": cluster_corr, "tof": t_itof, "tof_corr": itof_corr}
    # save(output_name, to_save)
    save_iter(output_name, correlated_data, zip(itof_corr, t_itof), groupsize=10)
    print('Finished:', datetime.now().strftime("%H:%M:%S"))
