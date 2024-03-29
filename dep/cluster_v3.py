# -*- coding: utf-8 -*-
"""
Clusters an unclustered h5 file to a v3 clustered file, with separate pixel, itof, and etof info.
"""
import sys

# %% Initializing
import h5py
import numpy as np
import argparse
from datetime import datetime
from numba.typed import List as nList
import functools as ft
import itertools as it
import os
import sklearn.cluster as skcluster
from multiprocessing import Pool
from threading import Lock


# %%% Functions

# %%%% Iterator Tools

def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


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


def safetee(iterable, n):
    """tuple of n independent thread-safe iterators"""
    lock = Lock()
    return tuple(safeteeobject(teeobj, lock) for teeobj in it.tee(iterable, n))


def split_every(n, iterable):
    """Return an iterator that splits the iterable into chunks of size n. Example: split_every(3, 'ABCDEFGH') --> [
    'A','B','C'] ['D','E','F'] ['G','H']"""
    i = iter(iterable)
    piece = list(it.islice(i, n))
    while piece:
        yield piece
        piece = list(it.islice(i, n))


def is_val(iterable, val):
    """Return an iterator that yields True if the element in the iterable is equal to the given value, otherwise False."""
    for i in iterable:
        yield i == val


def index_iter(iter_tuple, index: int):
    """Return an iterator that yields the element at the given index of each tuple in the iterable."""
    return map(lambda x: x[index], iter_tuple)


def split_iter(iter_tuple, n):
    """Return a tuple of n iterators, each of which yields the nth element of each tuple in the input iterable."""
    enum = enumerate(safetee(iter_tuple, n))
    return tuple(map(lambda x: index_iter(x[1], x[0]), enum))


# %%%% Other Tools


def h5append(dataset, newdata):
    """Append newdata to the end of dataset"""
    dataset.resize((dataset.shape[0] + len(newdata)), axis=0)
    dataset[-len(newdata):] = newdata


def compare_diff(time, cutoff=1, greater=True):
    """Return True where the difference between pairs is greater/less than cutoff depending on greater value"""
    if greater:
        return (time[1] - time[0] > cutoff)
    else:
        return (time[1] - time[0] < cutoff)


def iter_dataset(file, dataset):
    """Return an iterator that yields the elements of dataset in file"""
    for chunk in file[dataset].iter_chunks():
        for i in file[dataset][chunk]:
            yield i


def list_enum(enum):
    """
    Return a list of tuples of the form (i, x) for each x in lst where i is the index of the element.
    Parameters:
        enum (tuple): (i, lst) where i is an integer and lst is a list.
    Returns:
        list: of tuples (i, x)
    """
    i, lst = enum[0], enum[1]
    return [(i, x) for x in lst]


# %%%% Run Specific


def iter_file(file):
    """
    Iterate over datasets in a h5 file.

    Parameters:
        file (h5 file): file to iterate over.

    Returns:
        tuple: tuple of iterators for each dataset in the file.
    """
    iter_fn = ft.partial(iter_dataset, file)
    datasets = ['tdc_time', 'tdc_type', 'x', 'y', 'tot', 'toa']
    return tuple(map(iter_fn, datasets))


# Unused
# def toa_correct(toa_uncorr):
#     return np.where(np.logical_and(x >= 194, x < 204), toa_uncorr-25000, toa_uncorr)


def get_times_iter(tdc_time, tdc_type, mode, cutoff):
    """
   Get an iterator for specific times based on the mode and cutoff.

   Parameters:
       tdc_time (iterator): iterator for tdc time.
       tdc_type (iterator): iterator for tdc type.
       mode (str): mode to filter times by, either 'etof', 'pulse', or 'itof'.
       cutoff (float): cutoff value in nanoseconds.

   Returns:
       iterator: iterator for filtered times.
   """
    if mode == 'etof':
        return it.compress(tdc_time, is_val(tdc_type, 3))

    else:
        times = it.compress(pairwise(tdc_time), is_val(tdc_type, 1))
        times, pair_times = it.tee(times)
        comp = ft.partial(compare_diff, cutoff=cutoff * 1000, greater=(mode == 'pulse'))
        return it.compress(index_iter(times, 0), map(comp, pair_times))


def correct_pulse_times_iter(pulse_times, cutoff=(1e9 + 1.5e4), diff=12666):
    """
    Correct the pulse times so any pulses which are less than [cutoff] away from the prior pulse are delayed by [diff].

    Parameters:
        pulse_times (iterator): iterator for pulse times.
        cutoff (float): cutoff value for time difference in ps, default is (1e9+1.5e4).
        diff (float): time difference value in ps, default is 12666.

    Returns:
        iterator: iterator for corrected pulse times.
    """
    for pt, pt1 in pairwise(pulse_times):
        if pt1 - pt < cutoff:
            yield pt1 + diff
        else:
            yield pt1


def get_t_iter(pulse_times, times):
    """
    Get an iterator for the time difference between events and pulse times.

    Parameters:
        pulse_times (iterator): iterator for pulse times.
        times (iterator): iterator for event times.

    Returns:
        iterator: iterator for the tuple of pulse index and time difference between events and pulse times.
    """
    pte = enumerate(pulse_times)
    i, t0 = next(pte, (-1, -1))
    i1, t1 = next(pte, (-1, -1))
    for time in times:
        while time > t1:
            i, t0 = i1, t1
            i1, t1 = next(pte, (-1, -1))
            if i1 == -1:
                break
            if t1 - t0 > 1e12:
                print(f"Skipping pulse {i}")
                i, t0 = i1, t1
                i1, t1 = next(pte, (-1, -1))
        if i == -1:
            break
        elif time > t0:
            yield i, time - t0
        else:
            yield -1, time-t0


def format_data(corr, iters):
    """
    Format the data for output by grouping events by pulse.

    Parameters:
        corr (iterator): iterator for the corrected pulse index.
        iters (list of iterators): list of iterators for data to format.

    Returns:
        iterator: iterator for formatted data.
            Each element in the iterator is a list of numpy arrays, where each numpy array in the list corresponds
            to a single dataset, and contains data for events that belong to a specific pulse. The number of events
            in each numpy array corresponds to the number of events for that specific pulse.
    """
    next(corr)
    pn = next(corr)

    for i in it.count():
        if pn < i and pn>0:
            raise Exception
        n = 0
        while pn == i or pn == -1:
            n += 1
            pn = next(corr, None)
            while pn == -1:
                pn = next(corr, None)
                _=[next(x) for x in iters]
        if pn is None:
            break
        yield [np.asarray(list(it.islice(x, n))) for x in iters]





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

    dbscan = skcluster.DBSCAN(eps=2, min_samples=5)
    x = data[0]
    y = data[1]
    db = dbscan.fit(np.column_stack((x, y)))
    clst = db.labels_
    return clst, data


def average_over_cluster(cluster_index, data):
    """
   Compute the average of x and y over each cluster and the value of t in each cluster at the max of the ToT.

   Parameters:
       cluster_index (numpy array): Array of cluster indices.
       data (tuple of numpy arrays): Tuple of numpy arrays, containing the data to be processed. The tuple consists of (x, y, toa, tot) arrays.

   Returns:
       list: list of tuple, where each tuple contains the weighted average values of x and y and the minimum value of t, calculated for each cluster index.
   """
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        count = max(cluster_index) + 1
        weight = data[-1]
        mean_vals = nList()
        for i in range(count):
            elements = (np.average(data[0][cluster_index == i], weights=weight[cluster_index == i]),
                        np.average(data[1][cluster_index == i], weights=weight[cluster_index == i]),
                        data[2][cluster_index == i][np.argmax(weight[cluster_index == i])])
            mean_vals.append(elements)
        return mean_vals
    else:
        mean_vals = nList([(0., 0., 0.)])
        mean_vals.pop()
        return mean_vals


def save_iter(name, clust_data, etof_data, tof_data, groupsize=1000, maxlen=None):
    """
    Save data to h5 file

    Parameters:
        name (str): name of the h5 file to be created.
        clust_data (iterable): iterable containing data for clusters.
        etof_data (iterable): iterable containing data for etof.
        tof_data (iterable): iterable containing data for tof.
        groupsize (int): number of data points to be written at a time.
        maxlen (int): maximum number of data points to be written to the file.

    Returns:
        None
    """
    with h5py.File(name, 'w') as f:
        first = True
        xd = f.create_dataset('x', [0.], chunks=groupsize, maxshape=(None,))
        yd = f.create_dataset('y', [0.], chunks=groupsize, maxshape=(None,))
        td = f.create_dataset('t', [0.], chunks=groupsize, maxshape=(None,))
        corrd = f.create_dataset('cluster_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_etof_d = f.create_dataset('t_etof', [0.], chunks=groupsize, maxshape=(None,))
        etof_corr_d = f.create_dataset('etof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_tof_d = f.create_dataset('t_tof', [0.], chunks=groupsize, maxshape=(None,))
        tof_corr_d = f.create_dataset('tof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        lasts = [0, 0, 0]

        split1 = next(split_every(groupsize, clust_data), None)
        split2 = next(split_every(groupsize, etof_data), None)
        split3 = next(split_every(groupsize, tof_data), None)

        while not all((split1 is None, split2 is None, split3 is None)):
            print(lasts)
            if first:
                if split1 is not None:
                    split1 = split1[1:]
                if split2 is not None:
                    split2 = split2[1:]
                if split3 is not None:
                    split3 = split3[1:]
                first = False

            if np.argmin(lasts) == 0:
                if split1 is not None:
                    (corr, coords) = tuple(zip(*split1))
                    (x, y, t) = tuple(zip(*coords))
                    h5append(xd, x)
                    h5append(yd, y)
                    h5append(td, t)
                    h5append(corrd, corr)
                    lasts[0] = corr[-1]
                    split1 = next(split_every(groupsize, clust_data), None)
                else:
                    lasts[0] = sys.maxsize
            elif np.argmin(lasts) == 1:
                if split2 is not None:
                    (etof_corr, t_etof) = tuple(zip(*split2))
                    h5append(t_etof_d, t_etof)
                    h5append(etof_corr_d, etof_corr)
                    lasts[1] = etof_corr[-1]
                    split2 = next(split_every(groupsize, etof_data), None)
                else:
                    lasts[1] = sys.maxsize
            elif np.argmin(lasts) == 2:
                if split3 is not None:
                    (tof_corr, t_tof) = tuple(zip(*split3))
                    h5append(t_tof_d, t_tof)
                    h5append(tof_corr_d, tof_corr)
                    lasts[2] = tof_corr[-1]
                    split3 = next(split_every(groupsize, tof_data), None)
                else:
                    lasts[2] = sys.maxsize

            if maxlen is not None:
                print(xd.shape)
                if xd.shape[0] > maxlen:
                    break


if __name__ == '__main__':
    # %%% Argument Parsing
    parser = argparse.ArgumentParser(prog='cluster',
                                     description="Clusters data into .cv3 file. Does not load full dataset into memory")

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
                        default=None,
                        help="The maximum number of data points to record. Use to partially cluster a dataset.")

    parser.add_argument('filename')
    args = parser.parse_args()

    # %% Running
    if not args.single:
        p = Pool(os.cpu_count())
    print('Beginning Cluster Analysis:', datetime.now().strftime("%H:%M:%S"))
    output_name = args.output if args.output else args.filename[:-3] + ".cv3"

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

    clustered_data = p.imap(cluster, formatted_data, chunksize=1000) if not args.single else map(cluster,
                                                                                                 formatted_data)

    averaged_cluster_data = it.starmap(average_over_cluster, clustered_data)

    enumerated_data = it.chain.from_iterable(map(list_enum, enumerate(averaged_cluster_data)))

    save_iter(output_name, enumerated_data, etof_data, itof_data, groupsize=args.groupsize, maxlen=args.maxlen)
    print('Finished:', datetime.now().strftime("%H:%M:%S"))
