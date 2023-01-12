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
import argparse
from numba import njit
from datetime import datetime
from numba.typed import List
import functools
import itertools
import os
import sklearn.cluster as skcluster
from multiprocessing import Pool
from threading import Lock
warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)


# %% Functions

class safeteeobject(object):
    """tee object wrapped to make it thread-safe"""

    def __init__(self, teeobj, lock):
        self.teeobj = teeobj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.teeobj)

    def __copy__(self):
        return safeteeobject(self.teeobj.__copy__(), self.lock)


def safetee(iterable, n=2):
    """tuple of n independent thread-safe iterators"""
    lock = Lock()
    return tuple(safeteeobject(teeobj, lock) for teeobj in itertools.tee(iterable, n))


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
    enum = enumerate(safetee(iter_tuple, n))
    return tuple(map(lambda x: index_iter(x[1], x[0]), enum))


def toa_correct(toa_uncorr):
    return np.where(np.logical_and(x >= 194, x < 204), toa_uncorr-25000, toa_uncorr)


def correct_pulse_times(pulse_times, cutoff=(1e9+1.5e4), diff=12666):
    return np.where(np.diff(pulse_times) > (1e9+1.5e4), pulse_times[1:]-diff, pulse_times[1:])


def correct_pulse_times_iter(pulse_times, cutoff=(1e9+1.5e4), diff=12666):
    for pt, pt1 in pairwise(pulse_times):
        if pt1-pt < cutoff:
            yield pt1+diff
        else:
            yield pt1


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
    i1, t1 = next(pte, (-1, -1))
    for time in times:
        while time > t1:
            i, t0 = i1, t1
            i1, t1 = next(pte, (-1, -1))
            if i1 == -1:
                break
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


def format_data(corr, iters):
    next(corr)
    pn = next(corr)

    for i in itertools.count():
        n = 0
        while pn == i:
            n += 1
            pn = next(corr, None)
        if pn is None:
            break
        yield [np.asarray(list(itertools.islice(x, n))) for x in iters]

    #     for j in range(i-last_i+1):
    #         yield [np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)]
    #     if i == current:
    #         n += 1
    #     else:
    #         yield [np.asarray(list(itertools.islice(x, n))) for x in iters]
    #         n = 1
    #         current = i
    # yield [np.asarray(list(itertools.islice(x, n))) for x in iters]


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


def correlate_tof_iter(data_iter, tof_data):
    dc = next(data_iter, None)
    tc = next(tof_data, None)
    while not (dc is None or tc is None):
        if dc[0] > tc[0]:
            tc = next(tof_data, None)
        elif dc[0] < tc[0]:
            dc = next(data_iter, None)
        else:
            yield (dc[0], tuple(list(dc[1][:-1])+[tc[1]]))
            tc = next(tof_data, None)


def filter_exists(iterable):
    return filter(lambda x: x is not None, iterable)


def save(name, data_dict, mode='w'):
    with h5py.File(name, mode) as f:
        for key in data_dict:
            f.create_dataset(key, data=data_dict[key], chunks=True)


def save_iter(name, clust_data, etof_data, tof_data, groupsize=1000, maxlen=None):
    with h5py.File(name, 'w') as f:
        first = True
        xd = f.create_dataset('x', [0.], chunks=groupsize, maxshape=(None,))
        yd = f.create_dataset('y', [0.], chunks=groupsize, maxshape=(None,))
        # td = f.create_dataset('t', [0.], chunks=groupsize, maxshape=(None,))
        corrd = f.create_dataset('cluster_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_etof_d = f.create_dataset('t_etof', [0.], chunks=groupsize, maxshape=(None,))
        etof_corr_d = f.create_dataset('etof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_tof_d = f.create_dataset('t_tof', [0.], chunks=groupsize, maxshape=(None,))
        tof_corr_d = f.create_dataset('tof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        for split1, split2, split3 in itertools.zip_longest(split_every(groupsize, clust_data), split_every(groupsize, etof_data), split_every(groupsize, tof_data), fillvalue=None):
            if first:
                split1 = split1[1:]
                split2 = split2[1:]
                split3 = split3[1:]
                first = False

            if split1 is not None:
                (corr, coords) = tuple(zip(*split1))
                (x, y, t) = tuple(zip(*coords))
                h5append(xd, x)
                h5append(yd, y)
                h5append(corrd, corr)

            if split2 is not None:
                (etof_corr, t_etof) = tuple(zip(*split2))
                h5append(t_etof_d, t_etof)
                h5append(etof_corr_d, etof_corr)

            if split3 is not None:
                (tof_corr, t_tof) = tuple(zip(*split2))
                h5append(t_tof_d, t_tof)
                h5append(tof_corr_d, tof_corr)

            if maxlen is not None:
                print(xd.shape)
                if xd.shape[0] > maxlen:
                    break


# %% Argument Parsing
parser = argparse.ArgumentParser(prog='cluster',
                                 description="Clusters data with rejection based on pulse time differences")

parser.add_argument('--g', dest='groupsize',
                    type=int, default=10000,
                    help="Clusters this many pulses at a time")

parser.add_argument('--cutoff', dest='cutoff',
                    type=float, default=500,
                    help="Time cutoff between itof and laser pulses in ns")

parser.add_argument('--single', action='store_true',
                    help="Do not use multithreading to preserve RAM")

parser.add_argument('--out', dest='output',
                    default='', help="The output HDF5 file")

parser.add_argument('--maxlen', dest='maxlen', type=float,
                    default=None, help="The output HDF5 file")

parser.add_argument('filename')
args = parser.parse_args(["small.h5"])

# %% Loading Data
if __name__ == '__main__':
    p = Pool(os.cpu_count())
    print('Loading Data:', datetime.now().strftime("%H:%M:%S"))
    output_name = args.output if args.output else args.filename[:-3]+"_c.h5"

    f_in = h5py.File(args.filename)

    (tdc_time, tdc_type, x, y, tot, toa) = iter_file(f_in)

    pulse_times = correct_pulse_times_iter(get_times_iter(tdc_time, tdc_type, 'pulse', args.cutoff))
    etof_times = get_times_iter(iter_dataset(f_in, 'tdc_time'), iter_dataset(f_in, 'tdc_type'), 'etof', args.cutoff)
    itof_times = get_times_iter(iter_dataset(f_in, 'tdc_time'), iter_dataset(f_in, 'tdc_type'), 'itof', args.cutoff)

    pt1, pt2, pt3 = safetee(pulse_times, 3)
    pixel_corr, t_pixel = split_iter(get_t_iter(pt1, toa), 2)
    etof_data = get_t_iter(pt2, etof_times)
    itof_data = get_t_iter(pt3, itof_times)
    formatted_data = format_data(pixel_corr, [x, y, t_pixel, tot])

    clustered_data = p.imap(cluster, formatted_data, chunksize=1000) if not args.single else map(cluster, formatted_data)

    averaged_cluster_data = itertools.starmap(average_over_cluster, clustered_data)

    enumerated_data = itertools.chain.from_iterable(map(list_enum, enumerate(averaged_cluster_data)))

    save_iter(output_name, enumerated_data, etof_data, itof_data, groupsize=args.groupsize, maxlen=args.maxlen)
    print('Finished:', datetime.now().strftime("%H:%M:%S"))
