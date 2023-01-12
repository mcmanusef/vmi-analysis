# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:43:11 2022
@author: mcman
"""
# %% Initializing
import h5py
import numpy as np
import numpy.typing as npt
import argparse
import os
import functools
import itertools
import sklearn.cluster as cluster

array = npt.NDArray


def toa_correct(toa_uncorr: array[int]) -> array[int]:
    return np.where(np.logical_and(x >= 194, x < 204), toa_uncorr-25000, toa_uncorr)


def read_file(filename: str) -> tuple[array[int], array[int], array[int], array[int], array[int], array[int]]:
    with h5py.File(filename, mode='r') as fh5:
        return (fh5['tdc_time'][()], fh5['tdc_type'][()], fh5['x'][()],
                fh5['y'][()], fh5['tot'][()], fh5['toa'][()])


def get_times(tdc_time: array[int], tdc_type: array[int], mode: str, cutoff: float | int) -> array[int]:
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


def get_t(pulse_times: array[int], times: array[int]) -> tuple[array[float], array[int]]:
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
        Time of the event after a laser pulse, in ns.
    corr : array[int]
        The index of the corresponding laser pulses
    """
    corr = np.searchsorted(pulse_times, times)-1
    t = 1e-3*(times-pulse_times[corr])
    return t, corr


def find_where(corr: array[int], num: int, lists: list[array]) -> list[array]:
    return [l[corr == num] for l in lists]


def cluster(data: list[array]) -> tuple[array[int], list[array]]:
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

    dbscan = cluster.DBSCAN(eps=1, min_samples=4)
    x = data[0]
    y = data[1]
    db = dbscan.fit(np.column_stack((x, y)))
    clst = db.labels_
    return clst, data


def average_over_cluster(cluster_index: array[int], data: list[array]) -> list[tuple]:
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
    count = max(cluster_index)
    weight = data[-1]
    def average_i(i): return (np.average(
        d[cluster_index == i], weights=weight[i]) for d in data[:-1])
    mean_vals = list(map(average_i, range(count)))

    return mean_vals


def list_enum(enum: tuple[int, list]) -> list[tuple]:
    i, l = enum
    return [(i, x) for x in l]


# %% Argument Parsing
parser = argparse.ArgumentParser(prog='get_cluster',
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
                    default='notset', help="The output HDF5 file")

parser.add_argument('filename')
args = parser.parse_args(['small.h5'])

# %% Loading Data
if __name__ == '__main__':

    (tdc_time, tdc_type, x, y, tot, toa) = read_file(args.filename())

    pulse_times = get_times(tdc_time, tdc_type, 'pulse')
    etof_times = get_times(tdc_time, tdc_type, 'etof')
    itof_times = get_times(tdc_time, tdc_type, 'itof')

    t_pixel, pixel_corr = get_t(pulse_times, toa)
    t_etof, etof_corr = get_t(pulse_times, etof_times)
    t_itof, itof_corr = get_t(pulse_times, itof_times)

    find_data = functools.partial(lambda corr, i, lists: find_where(
        corr == i, lists), corr=pixel_corr, lists=[x, y, t_pixel, tot])

    formatted_data = map(find_data, range(max(pixel_corr)))

    clustered_data = map(cluster, formatted_data)

    averaged_cluster_data = itertools.starmap(average_over_cluster, clustered_data)

    enumerated_data = itertools.chain(*map(list_enum, enumerate(averaged_cluster_data)))

    cluster_corr, cluster_data = (list(i) for i in tuple(zip(*enumerated_data)))
