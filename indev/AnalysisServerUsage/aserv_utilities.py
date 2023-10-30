import multiprocessing
from contextlib import contextmanager, ExitStack
from typing import Sequence

import numpy as np
from numba import njit

from indev.AnalysisServerUsage.conversion_utils import pw_jit, split_int


class CircularBuffer(Sequence):
    def __init__(self, max_size, dtypes):
        self.max_size = max_size
        self.dtypes = dtypes
        self.arrays = tuple(multiprocessing.Array(d, max_size) for d in unstructure(dtypes))
        self.index = multiprocessing.Value('L', 0)
        self.size = multiprocessing.Value('L', 0)

    @contextmanager
    def get_lock(self):
        with ExitStack() as stack:
            stack.enter_context(self.index.get_lock())
            stack.enter_context(self.size.get_lock())
            for a in self.arrays:
                stack.enter_context(a.get_lock())
            yield stack

    def put(self, values):
        with self.get_lock():
            for array, value in zip(self.arrays, unstructure(values)):
                array[self.index.value] = value
            self.index.value = (self.index.value + 1) % self.max_size
            if self.size.value < self.max_size:
                self.size.value += 1

    def __getitem__(self, item):
        if item >= self.size.value:
            raise IndexError
        with self.get_lock():
            return structure(
                self.dtypes,
                [a[(self.index.value - self.size.value + item) % self.max_size] for a in self.arrays]
            )

    def __len__(self):
        return self.size.value

    def get_all(self):
        return [self[i] for i in range(len(self))]


class BufferedQueue:
    def __init__(self, *args, buffer_size=1, dtypes=('i',), **kwargs):
        self.buffer = CircularBuffer(buffer_size, dtypes)
        self.queue = multiprocessing.Queue(*args, **kwargs)

    def get(self, block=True, timeout=None):
        out = self.queue.get(block=block, timeout=timeout)
        self.buffer.put(out)
        return out

    def put(self, *args, **kwargs): self.queue.put(*args, **kwargs)

    def qsize(self) -> int: return self.queue.qsize()

    def empty(self) -> bool: return self.queue.empty()

    def full(self) -> bool: return self.queue.full()

    def put_nowait(self, item) -> None: self.queue.put_nowait(item)

    def get_nowait(self): return self.get(False)

    def close(self) -> None: self.queue.close()

    def join_thread(self) -> None: self.queue.join_thread()

    def cancel_join_thread(self) -> None: self.queue.cancel_join_thread()


def structure_map(func, t):
    if isinstance(t, tuple):
        return tuple(structure_map(func, sub_t) for sub_t in t)
    else:
        return func(t)


def structure(template, data):
    d_iter = iter(data)
    return structure_map(lambda _: next(d_iter), template)


def unstructure(t):
    if isinstance(t, tuple):
        for sub_t in t:
            yield from (unstructure(sub_t))
    else:
        yield t


def h5append(dataset, newdata):
    """Append newdata to the end of dataset"""
    dataset.resize((dataset.shape[0] + len(newdata)), axis=0)
    dataset[-len(newdata):] = newdata
    dataset.flush()


@njit
def process_tdc_old(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time, c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels
    return 0, (tdc_type, c_time, f_time, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


def cluster_pixels(pixels, dbscan):
    if not pixels:
        return []
    x, y, toa, tot = map(np.asarray, zip(*pixels))
    toa = np.where(np.logical_and(x >= 194, x < 204), toa - 16, toa)
    toa = fix_toa(toa)
    cluster_index = dbscan.fit(np.column_stack((x, y))).labels_
    return cluster_index, x, y, toa, tot


@njit
def average_over_clusters(cluster_index, x, y, toa, tot):
    period = 25 * 2 ** 30
    clusters = []
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        for i in range(max(cluster_index) + 1):
            clusters.append((
                np.average(toa[cluster_index == i] * 25 / (2 ** 4), weights=tot[cluster_index == i]) % period,
                np.average(x[cluster_index == i], weights=tot[cluster_index == i]),
                np.average(y[cluster_index == i], weights=tot[cluster_index == i]),
            ))
    return clusters


@njit
def sort_tdcs(cutoff, tdcs):
    start_time = 0
    pulses, etof, itof = [], [], []
    for tdc_type, c_time, ftime, _ in tdcs:
        tdc_time = 3.125 * c_time + .260 * ftime
        if tdc_type == 15:
            start_time = tdc_time
        elif tdc_type == 14:
            etof.append(tdc_time)
        elif tdc_type == 10:
            pulse_len = (tdc_time - start_time)
            if pulse_len > cutoff:
                pulses.append(start_time)
            else:
                itof.append(start_time)
    return etof, itof, pulses


@njit
def sort_clusters(clusters):
    period = 25 * 2 ** 30
    t0 = clusters[0][0]
    c_adjust = [((t - t0 + period / 2) % period - period / 2, x, y) for t, x, y in clusters]
    return [((t + t0) % period, x, y) for t, x, y in sorted(c_adjust)]


@njit
def fix_toa(toa):
    if max(toa) - min(toa) < 2 ** 32:
        return toa
    period = 2 ** 34
    t0 = toa[0]
    return (toa - t0 + period / 2) % period + t0 - period / 2
