import asyncio
import datetime
import functools
import itertools
import multiprocessing
import random
import time
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
from time import sleep

import h5py
import matplotlib.pyplot as plt
import matplotlib
import numba
import numpy as np
import sklearn.cluster as skcluster
from numba import njit, NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# matplotlib.use("Qt5Agg")
# plt.switch_backend("Qt5Agg")
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

PERIOD = 25 * 2 ** 30


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
def pw_jit(lst):
    out = []
    for i in range(len(lst) - 1):
        out.append((lst[i], lst[i + 1]))
    return out


@njit
def split_int(num: int, ranges: list[tuple[int, int]]):
    out = []
    for r in ranges:
        out.append((num & (1 << r[1]) - 1) >> r[0])
    return out


@njit
def addr_to_coords(pix_addr):
    dcol = (pix_addr & 0xFE00) >> 8
    spix = (pix_addr & 0x01F8) >> 1
    pix = (pix_addr & 0x0007)
    x = (dcol + pix // 4)
    y = (spix + (pix & 0x3))
    return x, y


@njit
def process_packet(packet: int):
    header = packet >> 60
    reduced = packet & 0x0fff_ffff_ffff_ffff
    if header == 7:
        return process_pixel(reduced)

    elif header == 2:
        return process_tdc(reduced)

    else:
        return -1, (0, 0, 0, 0)


@njit
def process_tdc_old(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time, c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels
    return 0, (tdc_type, c_time, f_time, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F

@njit
def process_tdc(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time,  c_time, _, tdc_type = split_int(packet, pw_jit(split_points))
    c_time = c_time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels

    return 0, (tdc_type, c_time, f_time-1, 0)
    # c_time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


@njit
def process_pixel(packet):
    split_points = (0, 16, 20, 30, 44, 61)
    c_time, f_time, tot, m_time, pix_add = split_int(packet, pw_jit(split_points))
    toa = numba.uint64(c_time * 2 ** 18 + m_time * 2 ** 4)
    toa = numba.uint64(toa - f_time)
    x, y = addr_to_coords(pix_add)
    return 1, (x, y, toa, tot)  # x,y in pixels, toa in units of 25ns/2**4, tot in units of 25 ns


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
def process_chunk(chunk):
    tdcs = []
    pixels = []
    for packet in chunk:
        data_type, packet_data = process_packet(packet)
        if data_type == -1:
            continue
        if data_type == 1:
            pixels.append(packet_data)
        elif data_type == 0:
            tdcs.append(packet_data)
    return pixels, tdcs


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


# %%


class AnalysisServer:
    def __init__(
            self,
            max_size=0,
            buffer_size=10000,
            cluster_loops=1,
            processing_loops=1,
            cutoff=1000,
            filename="test.cv4",
            max_clusters=65536,
            pulse_time_adjust=0,
            expected_pulse_diff=1e6+15,
            pulse_diff_correction=12.666,
            min_samples=6,
            diagnostic_mode=False
                 ):

        self.max_size = max_size
        self.buffer_size = buffer_size
        self.chunk_queue = multiprocessing.Queue(max_size)
        self.min_samples=min_samples
        self.diagnostic_mode=diagnostic_mode

        self.split_pulse_queues = [multiprocessing.Queue(max_size) for _ in range(processing_loops)]
        self.split_etof_queues = [multiprocessing.Queue(max_size) for _ in range(processing_loops)]
        self.split_itof_queues = [multiprocessing.Queue(max_size) for _ in range(processing_loops)]
        self.split_pixel_queues = [multiprocessing.Queue(max_size) for _ in range(processing_loops)]

        self.raw_pulse_queue = multiprocessing.Queue(max_size)
        self.raw_itof_queue = multiprocessing.Queue(max_size)
        self.raw_etof_queue = multiprocessing.Queue(max_size)

        self.pixel_queue = multiprocessing.Queue(max_size)
        self.split_cluster_queues = [multiprocessing.Queue(max_size) for _ in range(cluster_loops)]
        self.raw_cluster_queue = multiprocessing.Queue(max_size)

        self.itof_queue = BufferedQueue(max_size, dtypes=('L', 'd'), buffer_size=buffer_size)
        self.etof_queue = BufferedQueue(max_size, dtypes=('L', 'd'), buffer_size=buffer_size)
        self.cluster_queue = BufferedQueue(max_size, dtypes=('L', ('d', 'd', 'd')), buffer_size=buffer_size)
        self.pulse_queue = BufferedQueue(max_size, dtypes=('L', 'd'), buffer_size=buffer_size)

        self.next = multiprocessing.Array('d', (0, 0, 0, 0))
        self.max_seen = multiprocessing.Array('d', (0, 0, 0, 0))
        self.current = multiprocessing.Array('d', (0, 0, 0, 0))
        self.overflow_loops = multiprocessing.Value('i', 0)
        self.cutoff = cutoff
        self.pulse_time_adjust=pulse_time_adjust
        self.expected_pulse_diff=expected_pulse_diff
        self.pulse_diff_correction=pulse_diff_correction
        self.finished = multiprocessing.Value('i', 0)
        self.max_clusters=max_clusters
        self.cluster_plot=False
        self.desync=multiprocessing.Value('i',0)

        self.threads=[]

        self.filename = filename

    def __enter__(self):
        print("Initializing")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in self.threads:
            t.terminate()

    def get_loops(self):
        processing_loops = [
            functools.partial(self.processing_loop, index=i) for i in range(len(self.split_pulse_queues))
        ]

        clustering_loops = [
            functools.partial(self.clustering_loop, index=i) for i in range(len(self.split_cluster_queues))
        ]

        sorting_pairs = (
            (self.split_pulse_queues, self.raw_pulse_queue),
            (self.split_etof_queues, self.raw_etof_queue),
            (self.split_itof_queues, self.raw_itof_queue),
            (self.split_pixel_queues, self.pixel_queue),
        )

        sorting_loops = [
            functools.partial(self.sorting_loop, in_queues=iq, out_queue=oq) for iq, oq in sorting_pairs
        ]
        if not self.diagnostic_mode:
            loops = [
                *processing_loops,
                *sorting_loops,
                self.monitoring_loop,
                *clustering_loops,
                self.cluster_sorting_loop,
                self.correlating_loop,
                self.saving_loop,
            ]
        else:
            loops = [
                *processing_loops,
                *sorting_loops,
                self.monitoring_loop,
                *clustering_loops,
                self.cluster_sorting_loop,
                self.diagnostic_saving_loop,
            ]
        return loops

    def processing_loop(self, index=0):
        while True:
            cnum, chunk = self.chunk_queue.get()
            pixels, tdcs = process_chunk(chunk)
            etof, itof, pulses = sort_tdcs(self.cutoff, tdcs) if tdcs else ([], [], [])
            if pixels:
                self.split_pixel_queues[index].put((cnum, pixels))
            for p in pulses:
                self.split_pulse_queues[index].put((cnum, p+self.pulse_time_adjust))
            for e in etof:
                self.split_etof_queues[index].put((cnum, e))
            for i in itof:
                self.split_itof_queues[index].put((cnum, i))

    def plot_monitoring_loop(self):
        fig,ax= plt.subplots(1,1)
        plot=ax.imshow(np.zeros((256,256)))
        while True:
            if len(self.cluster_queue.buffer)==0:
                continue
            data=self.cluster_queue.buffer.get_all()
            t,x,y=zip(*data)
            hist,*_=np.histogram2d(x,y,bins=256,range=((0,256),(0,256)))
            plot.set_data(hist)

    def sorting_loop(self, in_queues, out_queue):
        first = [q.get() for q in in_queues]
        last_index = [f[0] for f in first]
        last_data = [f[1] for f in first]
        while True:
            # if self.desync.value == 0:
            idx = last_index.index(min(last_index))
            index, data = in_queues[idx].get()
            out_queue.put(last_data[idx])
            last_data[idx] = data
            last_index[idx] = index

    def diagnostic_saving_loop(self):
        with h5py.File(self.filename, 'w', libver='latest') as f:
            queues = [
                self.raw_pulse_queue,
                self.raw_itof_queue,
                self.raw_etof_queue,
                self.raw_cluster_queue,
            ]
            names = [
                "pulses",
                "itof",
                "etof",
                "toa",
            ]
            for q, name in zip(queues, names):
                f.create_dataset(name, data=[0.], dtype=np.double, chunks=1000, maxshape=(None,))
            f.swmr_mode = True

            while True:
                lens = [q.qsize() for q in queues]
                index = lens.index(max(lens))
                if lens[index] > 1000:
                    q, name = queues[index], names[index]
                    data = (q.get() for _ in range(1000))

                    def flatten(d):
                        for di in d:
                            yield di if not isinstance(di, tuple) else di[0]

                    data = list(flatten(data))
                    h5append(f[name], data)
                    self.finished.value = min(self.finished.value, 1)
                elif self.finished.value > 0:
                    print(f"______________________{self.finished.value}_______________________")
                    time.sleep(1)
                    self.finished.value += 1
                    if self.finished.value >= 5:
                        print("FINISHED")
                        break
  
    def monitoring_loop(self, split_queue_monitoring=False, rate_monitoring=True):
        started = False
        total_pulse_time = 0
        start_time = datetime.datetime.now()
        elapsed_time = 0
        while True:

            queues = (self.chunk_queue,
                      self.pixel_queue,
                      self.raw_pulse_queue,
                      self.raw_cluster_queue,
                      self.raw_etof_queue,
                      self.raw_itof_queue,
                      self.pulse_queue,
                      self.cluster_queue,
                      self.etof_queue,
                      self.itof_queue,
                      )

            names = ("Chunks",
                     "Pixels",
                     "Raw Pulses",
                     "Raw Clusters",
                     "Raw Electrons",
                     "Raw Ions",
                     "Pulses",
                     "Clusters",
                     "Electrons",
                     "Ions",
                     )

            if not started:
                if max(self.current) > 0:
                    started = True
                    start_time = datetime.datetime.now()
                # continue

            last = total_pulse_time
            total_pulse_time = self.current[0] / 1e9 + max(self.max_seen) / 1e9 * self.overflow_loops.value
            if total_pulse_time > last:
                elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

            print(datetime.datetime.now())
            print(f"Next    : {'  | '.join([f'{n/1e6  : 9.3f}' for n in self.next])}")
            print(f"Max     : {'  | '.join([f'{n/1e9  : 9.3f}' for n in self.max_seen])}")
            print(f"Current : {'  | '.join([f'{n/1e9 : 9.3f}' for n in self.current])}")
            print(f"Total   : {total_pulse_time: 8.2f} s | "
                  f"{self.overflow_loops.value: 4d} loops | "
                  f"{elapsed_time: 8.2f} s | "
                  f"{elapsed_time * 100 / (total_pulse_time + 0.0001): 9.2f} %"
                  )
            print("")

            for name, queue in zip(names, queues):
                if isinstance(queue, BufferedQueue):
                    print(f"{name:15}: {queue.qsize(): 6d}/{self.max_size}"
                          f"\t|\tBuffer\t:{len(queue.buffer): 3d}/{self.buffer_size}: {queue.buffer[-1]}")
                else:
                    print(f"{name:15}: {queue.qsize(): 6d}/{self.max_size}")
            print("")

            if split_queue_monitoring:
                for name, qs in (
                        ("pulses", self.split_pulse_queues),
                        ("etof", self.split_etof_queues),
                        ("itof", self.split_itof_queues),
                        ("pixels", self.split_pixel_queues),
                        ("clusters", self.split_cluster_queues)):
                    print(name)
                    for i, q in enumerate(qs):
                        print(f"{i: 15d}: {q.qsize(): 6d}/{self.max_size}")
                    print("")

            if rate_monitoring:
                if len(self.pulse_queue.buffer) == self.buffer_size:
                    pulse_rate = 1e9*(self.pulse_queue.buffer[-1][0]-self.pulse_queue.buffer[0][0])/(
                            (self.pulse_queue.buffer[-1][1]-self.pulse_queue.buffer[0][1])%PERIOD
                    )
                    print(f"Rep Rate      : {pulse_rate: 7.4f} Hz")
                if len(self.etof_queue.buffer) == self.buffer_size:
                    etof_rate = self.buffer_size / (self.etof_queue.buffer[-1][0] - self.etof_queue.buffer[0][0])
                    print(f"e-ToF Rate    : {etof_rate: 7.4f} per shot")
                if len(self.itof_queue.buffer) == self.buffer_size:
                    itof_rate = self.buffer_size / (self.itof_queue.buffer[-1][0] - self.itof_queue.buffer[0][0])
                    print(f"i-ToF Rate    : {itof_rate: 7.4f} per shot")
                if len(self.cluster_queue.buffer) == self.buffer_size:
                    clust_rate = self.buffer_size / (self.cluster_queue.buffer[-1][0] - self.cluster_queue.buffer[0][0])
                    print(f"Cluster Rate  : {clust_rate: 7.4f} per shot")

            sleep(1)
            print("")
            print("")
            print("")

    def dumping_loop(self):
        while True:
            queues = (self.cluster_queue, self.itof_queue, self.etof_queue)
            for q in queues:
                if not q.empty():
                    q.get()

    def clustering_loop(self, index=0):
        dbscan = skcluster.DBSCAN(eps=1.5, min_samples=self.min_samples)
        i=-1
        last=(0,0,0),0
        last_max=0
        while True:
            i+=1
            pixels = self.pixel_queue.get()
            times=[pix[2] for pix in pixels]
            last_max=max(times)
            if not pixels:
                continue
            clust_data = cluster_pixels(pixels, dbscan)
            clusters = average_over_clusters(*clust_data)
            if self.cluster_plot and i%100==0:
                xr,yr,*_=zip(*pixels)
                plt.figure()
                plt.hist2d(xr,yr, bins=256,range=[(0,256)]*2)
                if clusters:
                    _,x,y= zip(*clusters)
                    plt.scatter(x,y, s=1)
                plt.savefig(f"temp_{i}.png")
            if not clusters or len(clusters)>self.max_clusters:
                continue
            for c in sort_clusters(clusters):
                self.split_cluster_queues[index].put(c)

    def cluster_sorting_loop(self):
        loop = 25 * 2 ** 30  # 26.8 second loop time
        last_sent_time = 0
        last_pulses = [q.get() for q in self.split_cluster_queues]
        last_times = [p[0] for p in last_pulses]
        while True:
            deltas = [(lt - last_sent_time+1e6) % loop for lt in last_times]
            index = deltas.index(min(deltas))
            last_sent_time = last_times[index]
            self.raw_cluster_queue.put(last_pulses[index])
            last_pulses[index] = self.split_cluster_queues[index].get()
            last_times[index] = last_pulses[index][0]

    def correlating_loop(self, max_back=1e9, corr_cutoff=(1e6), corr_diff=12.666):
        last_pulse = 0
        next_clust = (0, 0, 0)
        next_times = [0, 0, 0, 0]
        in_queues = [self.raw_pulse_queue, self.raw_itof_queue, self.raw_etof_queue, self.raw_cluster_queue]
        out_queues = [self.pulse_queue, self.itof_queue, self.etof_queue, self.cluster_queue]
        pulse_number = 0

        for loop_number in itertools.cycle(range(100)):
            while last_pulse == 0:
                last_pulse, next_times[0] = next_times[0], self.raw_pulse_queue.get()

            to_pick = np.argmin([(t - last_pulse + max_back) % PERIOD-max_back for t in next_times])
            if loop_number == 0:
                for i, t in enumerate(next_times):
                    self.next[i] = (t - last_pulse + max_back) % PERIOD-max_back
                    self.max_seen[i] = max(t, self.max_seen[i])
                    self.current[i] = t

            if to_pick == 0:
                out_queues[0].put((pulse_number, next_times[0]))
                last_diff=(next_times[0]-last_pulse) % PERIOD

                pulse_number += 1
                last_pulse, next_times[0] = next_times[0], in_queues[0].get()
                # print(next_times[0]-last_pulse-corr_cutoff)
                if (next_times[0]-last_pulse) % PERIOD < corr_cutoff:
                    # print("PULSE TIME")
                    next_times[0]+=corr_diff
                if next_times[0]<=0:
                    next_times[0]=(last_pulse+last_diff) % PERIOD

                if 0 < next_times[0] < last_pulse - 1e9:
                    print(f"Looping from {last_pulse / 1e9:.4f} to {next_times[0] / 1e9:.4f}")
                    self.overflow_loops.value += 1
                    last_pulse, next_times[0] = next_times[0], in_queues[0].get()
                    for i in range(4):
                        while next_times[i] > 1e9 or next_times[i] == 0:
                            next_times[i] = in_queues[i].get()
                            if i == 3:
                                next_clust = next_times[3]
                                next_times[3] = next_clust[0]
            else:
                if next_times[to_pick] > last_pulse:
                    if to_pick < 3:
                        out_queues[to_pick].put((pulse_number, next_times[to_pick] - last_pulse))
                    else:
                        out_queues[3].put(
                            (pulse_number, (next_times[to_pick] - last_pulse, next_clust[1], next_clust[2]))
                        )

                next_times[to_pick] = in_queues[to_pick].get()

                if to_pick == 3:
                    next_clust = next_times[3]
                    next_times[3] = next_clust[0]

    def saving_loop(self):
        with h5py.File(self.filename, 'w', libver='latest') as f:

            groupsize = 1000
            queues = [
                self.cluster_queue,
                self.etof_queue,
                self.itof_queue,
                self.pulse_queue
            ]

            xd = f.create_dataset('x', [0.], chunks=groupsize, maxshape=(None,))
            yd = f.create_dataset('y', [0.], chunks=groupsize, maxshape=(None,))
            td = f.create_dataset('t', [0.], chunks=groupsize, maxshape=(None,))
            corrd = f.create_dataset('cluster_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_etof_d = f.create_dataset('t_etof', [0.], dtype=np.double, chunks=groupsize, maxshape=(None,))
            etof_corr_d = f.create_dataset('etof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_tof_d = f.create_dataset('t_tof', [0.], dtype=np.double, chunks=groupsize, maxshape=(None,))
            tof_corr_d = f.create_dataset('tof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_pulse_d = f.create_dataset('t_pulse', [0.], dtype=np.double, chunks=groupsize, maxshape=(None,))
            pulse_corr_d = f.create_dataset('pulse_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))
            f.swmr_mode = True

            while True:
                lens = [q.qsize() for q in queues]
                index = lens.index(max(lens))
                if lens[index] > groupsize:
                    data = (queues[index].get() for _ in range(groupsize))
                    match index:
                        case 0:
                            corr, clust = zip(*data)
                            t, x, y = zip(*clust)
                            h5append(xd, x)
                            h5append(yd, y)
                            h5append(td, t)
                            h5append(corrd, corr)

                        case 1:
                            corr, t = zip(*data)
                            h5append(t_etof_d, t)
                            h5append(etof_corr_d, corr)

                        case 2:
                            corr, t = zip(*data)
                            h5append(t_tof_d, t)
                            h5append(tof_corr_d, corr)

                        case 3:
                            corr, t = zip(*data)
                            h5append(t_pulse_d, t)
                            h5append(pulse_corr_d, corr)

                    self.finished.value = min(self.finished.value, 1)
                elif self.finished.value > 0:
                    print(f"______________________{self.finished.value}_______________________")
                    time.sleep(1)
                    self.finished.value += 1
                    if self.finished.value >= 5:
                        print("FINISHED")
                        break

    def make_connection_handler(self):
        async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            print(writer.get_extra_info('peername'))
            chunk_num = 0
            while not reader.at_eof():
                try:
                    chunk_packet = await asyncio.wait_for(reader.read(8), timeout=1)
                    _, _, _, _, chip_number, mode, *num_bytes = tuple(chunk_packet)
                    chunk_num += 1
                    num_bytes = int.from_bytes((bytes(num_bytes)), 'little')

                    chunk = await reader.readexactly(num_bytes)
                    packets = [chunk[n:n + 8] for n in range(0, num_bytes, 8)]
                    assert not any(packet[0:3] == b'TPX3' for packet in packets)
                    self.chunk_queue.put(
                        (chunk_num, [int.from_bytes(p, byteorder="little") - 2 ** 62 for p in packets]))
                except asyncio.TimeoutError:
                    print("Reading Failed")
                    time.sleep(1)
                    print(reader.at_eof())
                except asyncio.IncompleteReadError:
                    print("INCOMPLETE CHUNK")
                    time.sleep(1)
                    print(reader.at_eof())
            self.finished.value = 1
            print("Disconnected")

        return handle_connection

    async def start(self, port=1234):
        print("Starting")
        loop_processes = [multiprocessing.Process(target=loop, daemon=True) for loop in self.get_loops()]
        [loop_process.start() for loop_process in loop_processes]
        self.threads=loop_processes
        print("Loops Started")

        server = await asyncio.start_server(self.make_connection_handler(), '127.0.0.1', port=port)
        self.server=server
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        print(f'Serving on {addrs}')
        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    with AnalysisServer(
            max_size=100000,
            cluster_loops=6,
            processing_loops=6,
            max_clusters=2,
            pulse_time_adjust=-500,
            diagnostic_mode=False,
    ) as aserv:
        asyncio.run(aserv.start())

# %%
