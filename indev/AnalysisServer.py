import asyncio
import collections
import concurrent
import datetime
import functools
import itertools
import time
from multiprocessing.managers import SyncManager
from time import sleep

import numba
import warnings

import queue
import multiprocessing
from numba import njit, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import sklearn.cluster as skcluster

import numpy as np

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class Timer:
    def __init__(self, num=1000, name=""):
        self.num=num
        self.avg=0
        self.count=0
        self.start_time=0
        self.pause_time=0
        self.name=name

    def start(self):
        self.start_time=time.perf_counter_ns()

    def pause(self):
        self.pause_time=time.perf_counter_ns()

    def unpause(self):
        self.start_time=self.start_time+time.perf_counter_ns()-self.pause_time

    def end(self):
        end=time.perf_counter_ns()
        self.avg,self.count=(self.avg*self.count+end-self.start_time)/(self.count+1), self.count+1
        if self.count==self.num:
            print(f"{self.name}: {self.avg/1000:.2f} us average")
            self.count=0

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
        out.append((num & (1 << r[1] - 1) - 1) >> r[0])
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
    time = time & 0xffffffff  # Remove 3 most significant bit to loop at same point as pixels
    tdc_time = time * 12 + f_time
    return 0, (
    tdc_type, tdc_time, 0, 0)  # tdc_time in units of 260 ps, tdc_type: 7->TDC1R, 2->TDC1F, 6->TDC2R, 3->TDC2F


@njit
def process_pixel(packet):  # TODO: Figure out why looping at 13 seconds instead of 26
    split_points = (0, 16, 20, 30, 44, 60)
    c_time, f_time, tot, time, pix_add = split_int(packet, pw_jit(split_points))
    toa = numba.uint64(c_time << 18 | time << 4)
    toa = toa + ~f_time + 1
    if toa > 2 ** 33: print(toa)
    x, y = addr_to_coords(pix_add)
    return 1, (x, y, toa, tot)  # x,y in pixels, toa in units of 25ns/2**4, tot in units of 25 ns


def cluster_pixels(pixels, dbscan):
    if pixels == []:
        return []
    x, y, toa, tot = map(np.asarray, zip(*pixels))
    cluster_index = dbscan.fit(np.column_stack((x, y))).labels_
    return (cluster_index, x, y, toa, tot)


@njit
def average_over_clusters(cluster_index, x, y, toa, tot):
    clusters = []
    if len(cluster_index) > 0 and max(cluster_index) >= 0:
        for i in range(max(cluster_index) + 1):
            clusters.append((
                np.average(toa[cluster_index == i] * 25 / (2 ** 4), weights=tot[cluster_index == i]),
                np.average(x[cluster_index == i], weights=tot[cluster_index == i]),
                np.average(y[cluster_index == i], weights=tot[cluster_index == i]),
            ))
    return clusters


@njit
def sort_tdcs(cutoff, tdcs):
    start_time = 0
    pulses, etof, itof = [], [], []
    for tdc_type, tdc_time, _, _ in tdcs:
        if tdc_type == 7:
            start_time = tdc_time
        elif tdc_type == 6:
            etof.append(tdc_time * 0.260)
        elif tdc_type == 2:
            pulse_len = (tdc_time - start_time) * 0.260
            if pulse_len > cutoff:
                pulses.append(start_time * 0.260)
            else:
                itof.append(start_time * 0.260)
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


# %%

class MyManager(SyncManager):
    pass


MyManager.register("PriorityQueue", queue.PriorityQueue)  # Register a shared PriorityQueue


class AnalysisServer:
    def __init__(self, max_size=0, cluster_loops=1, processing_loops=1):
        self.max_size = max_size
        manager = multiprocessing
        self.chunk_queue = manager.Queue(max_size)

        self.split_pulse_queues=[manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_etof_queues=[manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_itof_queues=[manager.Queue(max_size) for _ in range(processing_loops)]
        self.split_pixel_queues=[manager.Queue(max_size) for _ in range(processing_loops)]

        self.pulse_queue = manager.Queue(max_size)
        self.raw_itof_queue = manager.Queue(max_size)
        self.raw_etof_queue = manager.Queue(max_size)

        self.pixel_queue = manager.Queue(max_size)
        self.unsorted_pixel_queues=[manager.Queue(max_size) for _ in range(cluster_loops)]
        self.raw_cluster_queue = manager.Queue(max_size)

        self.itof_queue = manager.Queue(max_size)
        self.etof_queue = manager.Queue(max_size)
        self.cluster_queue = manager.Queue(max_size)

        self.next = manager.Array('d', (0, 0, 0, 0))
        self.max_seen = manager.Array('d', (0, 0, 0, 0))
        self.current = manager.Array('d', (0, 0, 0, 0))
        self.overflow_loops = manager.Value('i', 0)
        self.cutoff = 1000

    def get_loops(self):
        processing_loops=[
            functools.partial(self.processing_loop, index=i) for i in range(len(self.split_pulse_queues))
        ]

        clustering_loops=[
            functools.partial(self.clustering_loop, index=i) for i in range(len(self.unsorted_pixel_queues))
        ]

        sorting_pairs=(
            (self.split_pulse_queues,self.pulse_queue),
            (self.split_etof_queues,self.raw_etof_queue),
            (self.split_itof_queues,self.raw_itof_queue),
            (self.split_pixel_queues,self.pixel_queue),
        )

        sorting_loops=[
            functools.partial(self.sorting_loop, in_queues=iq, out_queue=oq) for iq,oq in sorting_pairs
        ]
        loops = [
            * processing_loops,
            * sorting_loops,
            self.monitoring_loop,
            * clustering_loops,
            self.cluster_sorting_loop,
            self.correlating_loop,
            self.dumping_loop,
        ]
        return loops

    def processing_loop(self, index=0):
        while True:
            cnum, chunk = self.chunk_queue.get()
            pixels, tdcs = process_chunk(chunk)
            etof, itof, pulses = sort_tdcs(self.cutoff, tdcs) if tdcs else ([], [], [])
            if pixels:
                self.split_pixel_queues[index].put((cnum,pixels))
            for p in pulses:
                self.split_pulse_queues[index].put((cnum,p))
            for e in etof:
                self.split_etof_queues[index].put((cnum,e))
            for i in itof:
                self.split_itof_queues[index].put((cnum,i))

    def sorting_loop(self,in_queues,out_queue):
        first=[q.get() for q in in_queues]
        last_index=[f[0] for f in first]
        last_data=[f[1] for f in first]
        while True:
            idx=last_index.index(min(last_index))
            index, data=in_queues[idx].get()
            out_queue.put(last_data[idx])
            last_data[idx]=data
            last_index[idx]=index


    def monitoring_loop(self):
        started = False
        total_pulse_time = 0
        last_update = datetime.datetime.now()
        start_time = datetime.datetime.now()
        elapsed_time = 0
        while True:

            queues = (self.chunk_queue,
                      self.pixel_queue,
                      self.pulse_queue,
                      self.raw_cluster_queue,
                      self.raw_itof_queue,
                      self.raw_etof_queue,
                      self.cluster_queue,
                      self.etof_queue,
                      self.itof_queue,
                      )

            names = ("Chunks",
                     "Pixels",
                     "Pulses",
                     "Raw Clusters",
                     "Raw Electrons",
                     "Raw Ions",
                     "Clusters",
                     "Electrons",
                     "Ions",
                     )

            if not started:
                if max(self.current) > 0:
                    started = True
                    start_time = datetime.datetime.now()
                continue

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
                  f"{elapsed_time * 100 / total_pulse_time: 9.2f} %"
                  )
            print("")
            for name, queue in zip(names, queues):
                print(f"{name:15}: {queue.qsize()}/{self.max_size}")
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

    def clustering_loop(self, index=0):
        dbscan = skcluster.DBSCAN(eps=2, min_samples=5)
        while True:
            pixels = self.pixel_queue.get()
            if not pixels: continue
            clust_data = cluster_pixels(pixels, dbscan)
            clusters = average_over_clusters(*clust_data)
            for c in sorted(clusters):
                self.unsorted_pixel_queues[index].put(c)

    def cluster_sorting_loop(self):
        loop=13421772800
        last_times=[0 for _ in self.unsorted_pixel_queues]
        last_sent_time=0
        last_pulses=[(0,0,0) for _ in self.unsorted_pixel_queues]

        while True:
            deltas=[(l-last_sent_time)%loop for l in last_times]
            index=deltas.index(min(deltas))
            last_sent_time=last_times[index]
            self.raw_cluster_queue.put(last_pulses[index])
            last_pulses[index]=self.unsorted_pixel_queues[index].get()
            last_times[index]=last_pulses[index][0]

    def correlating_loop(self):
        last_pulse = 0
        next_clust = (0, 0, 0)
        next_times = [0, 0, 0, 0]
        in_queues = [self.pulse_queue, self.raw_itof_queue, self.raw_etof_queue, self.raw_cluster_queue]
        out_queues = [None, self.itof_queue, self.etof_queue, self.cluster_queue]
        pulse_number=0
        for loop_number in itertools.cycle(range(1000)):
            while last_pulse == 0:
                last_pulse, next_times[0] = next_times[0], self.pulse_queue.get()

            to_pick = np.argmin([abs(t - last_pulse + 1e9) for t in next_times])
            if loop_number==0:
                for i, t in enumerate(next_times):
                    self.next[i] = t - last_pulse
                    self.max_seen[i] = max(t, self.max_seen[i])
                    self.current[i] = t

            if to_pick == 0:
                pulse_number+=1
                last_pulse, next_times[0] = next_times[0], in_queues[0].get()
                while next_times[0] == 0:
                    next_times[0] = in_queues[0].get()

                if 0 < next_times[0] < last_pulse - 1e9:
                    with self.overflow_loops.get_lock():
                        self.overflow_loops.value+=1
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
            i += 1

    def make_connection_handler(self):
        async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            print(writer.get_extra_info('peername'))
            while not reader.at_eof():
                raw_mess = await reader.read(8)
                mess = raw_mess.decode()
                if raw_mess == bytes():
                    continue

                print(mess)
                match mess:
                    case "DUMP":
                        for queue in (self.raw_cluster_queue, self.pulse_queue, self.raw_itof_queue, self.etof_queue):
                            print(f"Dumping Queue, length {queue.qsize()}")
                            while not queue.empty():
                                for a in queue.get():
                                    print(a)

                    case "DATA":
                        chunk_num=0
                        while not reader.at_eof():
                            try:
                                chunk_packet = await reader.read(8)
                                _, _, _, _, chip_number, mode, *num_bytes = tuple(chunk_packet)
                                chunk_num+=1
                                num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
                                # print(f"TPX3: chip {chip_number}, mode {mode}, {num_bytes} bytes")
                                to_read = num_bytes // 8
                                unsorted = [None] * to_read
                                for i in range(to_read):
                                    packet = await reader.readexactly(8)
                                    unsorted[i] = int.from_bytes(packet, byteorder="little") - 0x4000_0000_0000_0000

                                self.chunk_queue.put((chunk_num,unsorted))
                            except asyncio.IncompleteReadError:
                                # self.event_queue.put(unsorted)
                                print("INCOMPLETE CHUNK")

                    case _:
                        self.event_queue.put(data)
            print("Disconnected")

        return handle_connection

    async def start(self, port=1234):
        for loop in self.get_loops():
            loop_process = multiprocessing.Process(target=loop)
            loop_process.start()
        print("Loops Started")

        server = await asyncio.start_server(self.make_connection_handler(),'127.0.0.1',port=port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        print(f'Serving on {addrs}')
        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    aserv = AnalysisServer(max_size=10000,cluster_loops=8,processing_loops=4)
    asyncio.run(aserv.start())

# %%
