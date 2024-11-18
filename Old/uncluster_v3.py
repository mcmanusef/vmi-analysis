import argparse
import functools as ft
import itertools as it
import os
from datetime import datetime
from multiprocessing import Pool
from typing import *
import h5py


# %%% Functions

# %%%% Iterator Tools

def split_every(n: int, iterable: Iterable) -> Iterable[list]:
    """Return an iterator that splits the iterable into chunks of size n. Example: split_every(3, 'ABCDEFGH') --> [
    'A','B','C'] ['D','E','F'] ['G','H']"""
    i = iter(iterable)
    piece = list(it.islice(i, n))
    while piece:
        yield piece
        piece = list(it.islice(i, n))


def is_val(iterable: Iterable, val) -> Iterable[bool]:
    """Return an iterator that yields True if the element in the iterable is equal to the given value, otherwise False."""
    for i in iterable:
        yield i == val


def index_iter(iter_tuple: Iterable[tuple], index: int) -> Iterable:
    """Return an iterator that yields the element at the given index of each tuple in the iterable."""
    return map(lambda x: x[index], iter_tuple)


def split_iter(iter_tuple: Iterable[tuple], n: int) -> tuple[Iterable, ...]:
    """Return a tuple of n iterators, each of which yields the nth element of each tuple in the input iterable."""
    enum = enumerate(it.tee(iter_tuple, n))
    return tuple(map(lambda x: index_iter(x[1], x[0]), enum))

# %%%% Other Tools


def h5append(dataset, newdata):
    """Append newdata to the end of dataset"""
    dataset.resize((dataset.shape[0] + len(newdata)), axis=0)
    dataset[-len(newdata):] = newdata


def compare_diff(time, cutoff=1, greater=True):
    """Return True where the difference between pairs is greater/less than cutoff depending on greater value"""
    if greater:
        return (time[1]-time[0] > cutoff)
    else:
        return (time[1]-time[0] < cutoff)


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
        times = it.compress(it.pairwise(tdc_time), is_val(tdc_type, 1))
        times, pair_times = it.tee(times)
        comp = ft.partial(compare_diff, cutoff=cutoff*1000, greater=(mode == 'pulse'))
        return it.compress(index_iter(times, 0), map(comp, pair_times))


def correct_pulse_times_iter(pulse_times, cutoff=(1e9+1.5e4), diff=12666):
    """
    Correct the pulse times so any pulses which are less than [cutoff] away from the prior pulse are delayed by [diff].

    Parameters:
        pulse_times (iterator): iterator for pulse times.
        cutoff (float): cutoff value for time difference in ps, default is (1e9+1.5e4).
        diff (float): time difference value in ps, default is 12666.

    Returns:
        iterator: iterator for corrected pulse times.
    """
    for pt, pt1 in it.pairwise(pulse_times):
        if pt1-pt < cutoff:
            yield pt1+diff
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
            if i % 600000 == 0:
                print(f"Pulse: {i}")
            if i1 == -1:
                break
        if i == -1:
            break
        else:
            yield i, time-t0


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
        totd = f.create_dataset('tot', [0], chunks=groupsize, maxshape=(None,))
        corrd = f.create_dataset('cluster_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_etof_d = f.create_dataset('t_etof', [0.], chunks=groupsize, maxshape=(None,))
        etof_corr_d = f.create_dataset('etof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        t_tof_d = f.create_dataset('t_tof', [0.], chunks=groupsize, maxshape=(None,))
        tof_corr_d = f.create_dataset('tof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

        for split1, split2, split3 in it.zip_longest(split_every(groupsize, clust_data),
                                                     split_every(groupsize, etof_data),
                                                     split_every(groupsize, tof_data), fillvalue=None):

            if first:
                if split1 is not None:
                    split1 = split1[1:]
                if split2 is not None:
                    split2 = split2[1:]
                if split3 is not None:
                    split3 = split3[1:]
                first = False

            if split1 is not None:
                (corr, coords) = tuple(zip(*split1))
                (x, y, t, tot) = tuple(zip(*coords))
                h5append(xd, x)
                h5append(yd, y)
                h5append(td, t)
                h5append(totd, tot)
                h5append(corrd, corr)

            if split2 is not None:
                (etof_corr, t_etof) = tuple(zip(*split2))
                h5append(t_etof_d, t_etof)
                h5append(etof_corr_d, etof_corr)

            if split3 is not None:
                (tof_corr, t_tof) = tuple(zip(*split3))
                h5append(t_tof_d, t_tof)
                h5append(tof_corr_d, tof_corr)

            if maxlen is not None:
                print(xd.shape)
                if xd.shape[0] > maxlen:
                    break


if __name__ == '__main__':
    # %%% Argument Parsing
    parser = argparse.ArgumentParser(prog='uncluster',
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
    p = Pool(os.cpu_count())
    print('Starting:', datetime.now().strftime("%H:%M:%S"))
    output_name = args.output if args.output else args.filename[:-3]+".uv3"

    f_in = h5py.File(args.filename)

    (tdc_time, tdc_type, x, y, tot, toa) = iter_file(f_in)

    # pulse_times = correct_pulse_times_iter(get_times_iter(tdc_time, tdc_type, 'pulse', args.cutoff))
    pulse_times= get_times_iter(tdc_time, tdc_type, 'pulse', args.cutoff)
    etof_times = get_times_iter(iter_dataset(f_in, 'tdc_time'), iter_dataset(f_in, 'tdc_type'), 'etof', args.cutoff)
    itof_times = get_times_iter(iter_dataset(f_in, 'tdc_time'), iter_dataset(f_in, 'tdc_type'), 'itof', args.cutoff)
    # print(list(itof_times))

    pt1, pt2, pt3 = it.tee(pulse_times, 3)
    pixel_corr, t_pixel = split_iter(get_t_iter(pt1, toa), 2)
    etof_data = get_t_iter(pt2, etof_times)
    itof_data = get_t_iter(pt3, itof_times)
    pixel_data = ((corr, (xi, yi, ti, tot)) for corr, xi, yi, ti, tot in zip(pixel_corr,x,y,t_pixel, tot))

    save_iter(output_name, pixel_data, etof_data, itof_data, groupsize=args.groupsize, maxlen=args.maxlen)
    print('Finished:', datetime.now().strftime("%H:%M:%S"))

#%%
