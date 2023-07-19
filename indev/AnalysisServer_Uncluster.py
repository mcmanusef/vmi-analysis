import asyncio
import collections
import datetime
import functools
import itertools
import time
from time import sleep
import h5py
import numba
import warnings
import queue
import multiprocessing

import qtconsole.util
from numba import njit, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import sklearn.cluster as skcluster
import numpy as np
from cluster_v3 import h5append

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

PERIOD=25*2**30
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
def process_tdc(packet):
    split_points = (5, 9, 44, 56, 60)
    f_time, time, _, tdc_type = split_int(packet, pw_jit(split_points))
    time = time & 0x1ffffffff  # Remove 2 most significant bit to loop at same point as pixels
    return 0, (tdc_type, time, f_time, 0)
    # time in units of 3.125 f_time in units of 260 ps,
    # tdc_type: 10->TDC1R, 15->TDC1F, 14->TDC2R, 11->TDC2F


@njit
def process_pixel(packet):
    split_points = (0, 16, 20, 30, 44, 61)
    c_time, f_time, tot, time, pix_add = split_int(packet, pw_jit(split_points))
    toa = numba.uint64(c_time * 2 ** 18 + time * 2 ** 4)
    toa = numba.uint64(toa - f_time)
    x, y = addr_to_coords(pix_add)
    return 1, (x, y, toa, tot)  # x,y in pixels, toa in units of 25ns/2**4, tot in units of 25 ns

@njit
def fix_pixel(pixel):
    x, y, toa, tot=pixel
    return (toa*25/2**4, x, y, tot)


@njit
def sort_tdcs(cutoff, tdcs):
    start_time = 0
    pulses, etof, itof = [], [], []
    for tdc_type, time, ftime, _ in tdcs:
        tdc_time = 3.125 * time + 0.260 * ftime
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
            pixels.append(fix_pixel(packet_data))
        elif data_type == 0:
            tdcs.append(packet_data)
    return pixels, tdcs


@njit
def sort_clusters(clusters):
    period = 25 * 2 ** 30
    t0 = clusters[0][0]
    c_adjust = [((t - t0 + period / 2) % period - period / 2, x, y,tot) for t, x, y,tot in clusters]
    return [((t + t0) % period, x, y,tot) for t, x, y,tot in sorted(c_adjust)]


@njit
def fix_toa(toa):
    if max(toa)-min(toa)<2**32:
        return toa
    period = 2 ** 34
    t0 = toa[0]
    return (toa - t0 + period / 2) % period + t0 - period / 2


# %%


class UnclusteredAnalysisServer:
    def __init__(self, max_size=0, cluster_loops=1, processing_loops=1, cutoff=1000, filename="test.cv3"):
        self.max_size = max_size
        manager = multiprocessing
        self.chunk_queue = manager.Queue(max_size)

        self.split_pulse_queues = [manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_etof_queues = [manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_itof_queues = [manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_pixel_queues = [manager.Queue(max_size) for _ in range(processing_loops)]

        self.raw_pulse_queue = manager.Queue(max_size)
        self.raw_itof_queue = manager.Queue(max_size)
        self.raw_etof_queue = manager.Queue(max_size)

        self.pixel_queue = manager.Queue(max_size)
        self.split_cluster_queues = [manager.Queue(max_size) for _ in range(cluster_loops)]
        self.raw_cluster_queue = manager.Queue(max_size)

        self.itof_queue = manager.Queue(max_size)
        self.etof_queue = manager.Queue(max_size)
        self.cluster_queue = manager.Queue(max_size)
        self.pulse_queue = manager.Queue(max_size)

        self.next = manager.Array('d', (0, 0, 0, 0))
        self.max_seen = manager.Array('d', (0, 0, 0, 0))
        self.current = manager.Array('d', (0, 0, 0, 0))
        self.overflow_loops = manager.Value('i', 0)
        self.cutoff = cutoff
        self.finished = manager.Value('i', 0)

        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for process in multiprocessing.active_children():
            process.terminate()

    def get_loops(self):
        processing_loops = [
            functools.partial(self.processing_loop, index=i) for i in range(len(self.split_pulse_queues))
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
        loops = [
            *processing_loops,
            *sorting_loops,
            self.monitoring_loop,
            # self.diagnostic_saving_loop,
            self.correlating_loop,
            self.saving_loop,
            # self.dumping_loop,
        ]
        return loops

    def processing_loop(self, index=0):
        while True:
            cnum, chunk = self.chunk_queue.get()
            pixels, tdcs = process_chunk(chunk)
            etof, itof, pulses = sort_tdcs(self.cutoff, tdcs) if tdcs else ([], [], [])
            if pixels:
                for p in sort_clusters(pixels):
                    self.split_pixel_queues[index].put((cnum, p))
            for p in pulses:
                self.split_pulse_queues[index].put((cnum, p))
            for e in etof:
                self.split_etof_queues[index].put((cnum, e))
            for i in itof:
                self.split_itof_queues[index].put((cnum, i))

    def sorting_loop(self, in_queues, out_queue):
        first = [q.get() for q in in_queues]
        last_index = [f[0] for f in first]
        last_data = [f[1] for f in first]
        while True:
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
                f.create_dataset(name, data=[0.] * 1000, chunks=(1000,), maxshape=(None,))
            f.swmr_mode = True

            while True:
                lens = [q.qsize() for q in queues]
                index = lens.index(max(lens))
                if lens[index] > 1000:
                    q, name = queues[index], names[index]
                    data = (q.get() for _ in range(1000))
                    def flatten(d):
                        for di in d: yield di if not isinstance(di, tuple) else di[0]

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

    def monitoring_loop(self):
        started = False
        total_pulse_time = 0
        last_update = datetime.datetime.now()
        start_time = datetime.datetime.now()
        elapsed_time = 0
        while True:

            queues = (self.chunk_queue,
                      self.pixel_queue,
                      self.raw_pulse_queue,
                      self.raw_itof_queue,
                      self.raw_etof_queue,
                      self.pulse_queue,
                      self.cluster_queue,
                      self.etof_queue,
                      self.itof_queue,
                      )

            names = ("Chunks",
                     "Pixels",
                     "Raw Pulses",
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
            print(f"Next    : {'  | '.join([f'{n / 1e6 : 9.3f}' for n in self.next])}")
            print(f"Max     : {'  | '.join([f'{n / 1e9 : 9.3f}' for n in self.max_seen])}")
            print(f"Current : {'  | '.join([f'{n / 1e9 : 9.3f}' for n in self.current])}")
            print(f"Total   : {total_pulse_time: 8.2f} s | "
                  f"{self.overflow_loops.value: 4d} loops | "
                  f"{elapsed_time: 8.2f} s | "
                  f"{elapsed_time * 100 / (total_pulse_time + 0.0001): 9.2f} %"
                  )
            print("")
            for name, queue in zip(names, queues):
                print(f"{name:15}: {queue.qsize()}/{self.max_size}")
            print("")
            for name, qs in (
                    ("pulses", self.split_pulse_queues),
                    ("etof", self.split_etof_queues),
                    ("itof", self.split_itof_queues),
                    ("pixels", self.split_pixel_queues)):
                print(name)
                for i, q in enumerate(qs):
                    print(f"{i: 15d}: {q.qsize()}/{self.max_size}")
                print("")
            sleep(1)
            print("")
            print("")
            print("")

    def dumping_loop(self):
        while True:
            queues = (self.cluster_queue, self.itof_queue, self.etof_queue)
            for q in queues:
                try:
                    q.get(block=False)
                except Exception:
                    pass

    def correlating_loop(self):
        last_pulse = 0
        next_clust = (0, 0, 0)
        next_times = [0, 0, 0, 0]
        in_queues = [self.raw_pulse_queue, self.raw_itof_queue, self.raw_etof_queue, self.pixel_queue]
        out_queues = [self.pulse_queue, self.itof_queue, self.etof_queue, self.cluster_queue]
        pulse_number = 0
        for loop_number in itertools.cycle(range(100)):
            while last_pulse == 0:
                last_pulse, next_times[0] = next_times[0], self.raw_pulse_queue.get()

            to_pick = np.argmin([abs(t - last_pulse + 1e9) for t in next_times])
            if loop_number == 0:
                for i, t in enumerate(next_times):
                    self.next[i] = t - last_pulse
                    self.max_seen[i] = max(t, self.max_seen[i])
                    self.current[i] = t

            if to_pick == 0:
                out_queues[0].put((pulse_number, next_times[0]))

                pulse_number += 1
                last_pulse, next_times[0] = next_times[0], in_queues[0].get()
                while next_times[0] == 0:
                    next_times[0] = in_queues[0].get()

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
                        print(next_times[i])

            else:
                if next_times[to_pick] > last_pulse:
                    if to_pick < 3:
                        out_queues[to_pick].put((pulse_number, next_times[to_pick] - last_pulse))
                    else:
                        out_queues[3].put(
                            (pulse_number, (next_times[to_pick] - last_pulse, next_clust[1], next_clust[2], next_clust[3]))
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
            totd = f.create_dataset('tot', [0.], chunks=groupsize, maxshape=(None,))
            td = f.create_dataset('t', [0.], chunks=groupsize, maxshape=(None,))
            corrd = f.create_dataset('cluster_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_etof_d = f.create_dataset('t_etof', [0.], chunks=groupsize, maxshape=(None,))
            etof_corr_d = f.create_dataset('etof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_tof_d = f.create_dataset('t_tof', [0.], chunks=groupsize, maxshape=(None,))
            tof_corr_d = f.create_dataset('tof_corr', [0], dtype=int, chunks=groupsize, maxshape=(None,))

            t_pulse_d = f.create_dataset('t_pulse', [0.], chunks=groupsize, maxshape=(None,))
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
                            t, x, y, tot = zip(*clust)
                            h5append(xd, x)
                            h5append(yd, y)
                            h5append(td, t)
                            h5append(totd, tot)
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
        loop_processes = [multiprocessing.Process(target=loop, daemon=True) for loop in self.get_loops()]
        [loop_process.start() for loop_process in loop_processes]
        print("Loops Started")

        server = await asyncio.start_server(self.make_connection_handler(), '127.0.0.1', port=port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        print(f'Serving on {addrs}')
        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    with UnclusteredAnalysisServer(max_size=100000, processing_loops=3,filename="test_uc.h5") as aserv:
        asyncio.run(aserv.start())

# %%
