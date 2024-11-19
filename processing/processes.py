import collections
import os
import typing

import h5py
import matplotlib
import numpy as np
import sklearn.cluster
from matplotlib import pyplot as plt
matplotlib.use('qt5agg')

from processing.data_types import *
from processing.base_processes import AnalysisStep
from processing.tpx_conversion import *

PIXEL_RES = 25 / 16
PERIOD = 25 * 2 ** 30


class QueueTee(AnalysisStep):
    def __init__(self, input_queue, output_queues):
        super().__init__()
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.input_queues = (input_queue,)

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return
        for q in self.output_queues:
            q.put(data)


class TPXListener(AnalysisStep):
    # TODO: Implement TPXListener class to listen to TPX data over TCP/IP
    host: str
    port: int
    input_queues = ()
    output_queues: (ExtendedQueue[Chunk],)


class TPXFileReader(AnalysisStep):
    path: str
    input_queues = ()
    chunk_queue: ExtendedQueue[Chunk]
    file: typing.IO | None

    def __init__(self, path, chunk_queue, **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXFileReader"
        self.path = path
        self.chunk_queue = chunk_queue
        self.output_queues = (chunk_queue,)
        self.file = None
        self.folder: bool= os.path.isdir(path)
        self.files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tpx3')] if self.folder else [path]
        self.curr_file_idx = 0
        self.holding.value = True

    def initialize(self):
        self.file = open(self.files[self.curr_file_idx], 'rb')
        super().initialize()

    def action(self):
        packet = self.file.read(8)
        if len(packet) < 8:
            self.curr_file_idx += 1
            if self.curr_file_idx >= len(self.files):
                self.shutdown()
                return
            self.file.close()
            self.file = open(self.files[self.curr_file_idx], 'rb')
            return
        _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
        num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
        packets = [int.from_bytes(self.file.read(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
        self.chunk_queue.put(packets)

    def shutdown(self, **kwargs):
        self.file.close() if self.file else None
        super().shutdown(**kwargs)


class TPXConverter(AnalysisStep):
    chunk_queue: ExtendedQueue[Chunk]
    pixel_queue: ExtendedQueue[PixelData]
    tdc_queue: ExtendedQueue[TDCData]

    def __init__(self, chunk_queue: ExtendedQueue[Chunk], pixel_queue: ExtendedQueue[PixelData],
                 tdc_queue: ExtendedQueue[TDCData], **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXConverter"
        self.chunk_queue = chunk_queue
        self.pixel_queue = pixel_queue
        self.tdc_queue = tdc_queue
        self.input_queues = (chunk_queue,)
        self.output_queues = (pixel_queue, tdc_queue)

    def action(self):
        try:
            chunk = self.chunk_queue.get(timeout=1)
        except queue.Empty:
            return

        for packet in chunk:
            data_type, packet_data = process_packet(packet)
            if data_type == 1:
                self.pixel_queue.put(packet_data)
            elif data_type == 0:
                tdc_type, c_time, ftime, _ = packet_data
                tdc_time = 3.125 * c_time + .260 * ftime
                tdc_type = 1 if tdc_type == 10 else 2 if tdc_type == 15 else 3 if tdc_type == 14 else 4 if tdc_type == 11 else 0
                self.tdc_queue.put((tdc_time, tdc_type))


class VMIConverter(AnalysisStep):
    cutoff: float
    chunk_queue: ExtendedQueue[Chunk]
    pixel_queue: ExtendedQueue[list[PixelData]]
    laser_queue: ExtendedQueue[TriggerTime]
    etof_queue: ExtendedQueue[ToF]
    itof_queue: ExtendedQueue[ToF]

    def __init__(self, chunk_queue, pixel_queue, laser_queue, etof_queue, itof_queue, cutoff=300, timewalk_file=None, toa_corr=25, **kwargs):
        super().__init__(**kwargs)
        self.chunk_queue = chunk_queue
        self.pixel_queue = pixel_queue
        self.laser_queue = laser_queue
        self.etof_queue = etof_queue
        self.itof_queue = itof_queue
        self.cutoff = cutoff
        self.output_queues = (pixel_queue, laser_queue, etof_queue, itof_queue)
        self.input_queues = (chunk_queue,)
        self.timewalk_file = timewalk_file
        self.timewalk_correction = None
        self.toa_correction = toa_corr
        self.name = "VMIConverter"

    def initialize(self):
        if self.timewalk_file:
            self.timewalk_correction = np.loadtxt(self.timewalk_file)
        super().initialize()

    def action(self):
        try:
            chunk = self.chunk_queue.get(timeout=1)
        except queue.Empty:
            return
        pixels, tdcs = process_chunk(chunk)
        pixels = [(toa * PIXEL_RES, x, y, tot) for toa, x, y, tot in pixels]

        if pixels:
            if self.timewalk_correction is not None:
                pixels = apply_timewalk(pixels, self.timewalk_correction)
            if self.toa_correction:
                pixels = toa_correction(pixels, self.toa_correction)

        self.pixel_queue.put(pixels) if pixels else None

        etof, itof, pulses = sort_tdcs(self.cutoff, tdcs)
        for t in etof:
            self.etof_queue.put(t)
        for t in itof:
            self.itof_queue.put(t)
        for t in pulses:
            self.laser_queue.put(t)


class Weaver(AnalysisStep):
    input_queues: tuple[ExtendedQueue[T]]
    output_queue: ExtendedQueue[T]

    def __init__(self, input_queues, output_queue):
        super().__init__()
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.current: list[int | float | None] = [None for _ in input_queues]

    def action(self):
        if any(cur is None for cur in self.current):
            for i, q in enumerate(self.input_queues):
                if self.current[i] is None:
                    if q.closed.value and q.empty():
                        self.current[i] = np.inf
                    try:
                        self.current[i] = q.get(timeout=0.1)
                    except queue.Empty:
                        time.sleep(0.1)
                        return
        if all(cur == np.inf for cur in self.current):
            self.shutdown()
            return

        min_idx = self.current.index(min(c for c in self.current if c != np.inf))

        self.output_queue.put(self.current[min_idx])
        self.current[min_idx] = None


class QueueVoid(AnalysisStep):
    def __init__(self, input_queues, **kwargs):
        super().__init__(**kwargs)
        self.input_queues = input_queues
        self.name = "Void"

    def action(self):
        for q in self.input_queues:
            try:
                for _ in range(q.qsize()):
                    q.get(timeout=0.1)
            except queue.Empty:
                continue


class DBSCANClusterer(AnalysisStep):
    pixel_queue: ExtendedQueue[list[PixelData]]
    cluster_queue: ExtendedQueue[ClusterData]

    def __init__(self, pixel_queue, cluster_queue, output_pixel_queue=None, dbscan_params=None, **kwargs):
        super().__init__(**kwargs)
        self.pixel_queue = pixel_queue
        self.cluster_queue = cluster_queue
        self.input_queues = (pixel_queue,)
        self.output_pixel_queue = output_pixel_queue
        self.output_queues = (cluster_queue,) if not output_pixel_queue else (cluster_queue, output_pixel_queue)
        self.dbscan_params = dbscan_params if dbscan_params else {"eps": 1.5, "min_samples": 8}
        self.dbscan = None
        self.group_index = 0

        self.name = "DBSCANClusterer"

    def initialize(self):
        self.dbscan = sklearn.cluster.DBSCAN(**self.dbscan_params)
        super().initialize()

    def action(self):
        try:
            pixels = self.pixel_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            return
        if len(pixels) == 0:
            return
        toa, x, y, tot = map(np.asarray, zip(*pixels))
        if np.max(toa) - np.min(toa) > PERIOD / 2:
            return

        cluster_index = self.dbscan.fit(np.column_stack((x, y))).labels_

        clusters = average_over_clusters(cluster_index, toa, x, y, tot)

        for c in clusters:
            self.cluster_queue.put(c)
        for i, p in enumerate(pixels):
            if self.output_pixel_queue:
                self.output_pixel_queue.put((p, self.group_index + cluster_index[i])) if cluster_index[i] >= 0 else self.output_pixel_queue.put(
                        (p, -1))

        self.group_index += np.max(cluster_index) + 1


class TriggerAnalyzer(AnalysisStep):
    input_trigger_queue: ExtendedQueue[TriggerTime]
    queues_to_index: tuple[ExtendedQueue, ...]
    output_trigger_queue: ExtendedQueue[TriggerTime]
    indexed_queues: tuple[ExtendedQueue[tuple[int, typing.Any]], ...]

    def __init__(self, input_trigger_queue, queues_to_index, output_trigger_queue, indexed_queues, **kwargs):
        super().__init__(**kwargs)
        self.input_trigger_queue = input_trigger_queue
        self.queues_to_index = queues_to_index
        self.output_trigger_queue = output_trigger_queue
        self.indexed_queues = indexed_queues
        self.input_queues = (input_trigger_queue, *queues_to_index)
        self.output_queues = (output_trigger_queue, *indexed_queues)
        self.name = "TriggerAnalyzer"

        self.current: list[list[int | float] | None] = [None for _ in queues_to_index]
        self.current_trigger_idx = 0
        self.last_trigger_time = -1
        self.current_trigger_time = None
        self.current_samples = [None for _ in queues_to_index]

    def action(self):
        while self.current_trigger_time is None:
            try:
                self.current_trigger_time = self.input_trigger_queue.get(timeout=0.1)
            except queue.Empty:
                time.sleep(0.1)
                return

        for i, q in enumerate(self.queues_to_index):
            if self.current[i] is None:
                try:
                    c = q.get(timeout=0.1)
                    if self.current_samples[i] is None:
                        self.current_samples[i] = c
                    self.current[i] = list(unstructure(c))

                except queue.Empty:
                    if q.closed.value and q.empty():
                        self.current[i] = [np.inf, ]
                    time.sleep(0.1)
                    return
        n = 0
        for i, q in enumerate(self.queues_to_index):
            while self.current[i][0] < self.current_trigger_time:
                if n > 1000:
                    return
                n += 1

                out = self.current[i]
                out[0] -= self.last_trigger_time
                self.indexed_queues[i].put((self.current_trigger_idx, structure(self.current_samples[i], out)))
                try:
                    c = q.get(block=False)
                    self.current[i] = list(unstructure(c))
                except queue.Empty:
                    self.current[i] = None
                    break

        if any(c is None for c in self.current):
            return

        self.output_trigger_queue.put(self.last_trigger_time)
        self.last_trigger_time = self.current_trigger_time

        try:
            self.current_trigger_time = self.input_trigger_queue.get(timeout=0.1)
        except queue.Empty:
            self.current_trigger_time = None
        # print(self.current_trigger_time, self.last_trigger_time, self.current_trigger_idx, self.current)
        self.current_trigger_idx += 1


class SaveToH5(AnalysisStep):
    file_path: str
    input_queues: tuple[ExtendedQueue, ...]
    in_queues: dict[str, ExtendedQueue]
    output_queues = ()
    h5_file: h5py.File | None

    def __init__(self, file_path, input_queues, chunk_size=1000, flat: bool | tuple[bool] | dict[str, bool] = True, **kwargs):
        super().__init__(**kwargs)
        self.name = "SaveToH5"
        self.file_path = file_path
        self.input_queues = tuple(input_queues.values())
        self.in_queues = input_queues
        self.chunk_size = chunk_size
        self.flat = flat if isinstance(flat, dict) else {k: flat for k in input_queues.keys()} if isinstance(flat, bool) else {k: f for k, f in flat}
        self.h5_file = None
        self.n = 0

    def initialize(self):
        f = h5py.File(self.file_path, 'w')
        for key, q in self.in_queues.items():
            if self.flat[key]:
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    f.create_dataset(name, (self.chunk_size,), dtype=dtype, maxshape=(None,))
            else:
                g = f.create_group(key)
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    g.create_dataset(name, (self.chunk_size,), dtype=dtype, maxshape=(None,))
        self.h5_file = f
        super().initialize()

    def action(self):
        f = self.h5_file

        sizes = [(q.qsize(), k, q) for k, q in self.in_queues.items()]
        sizes.sort()
        max_queue = sizes[-1][2]
        max_name = sizes[-1][1]
        max_size = sizes[-1][0]
        if len(sizes) > 1:
            size_diff = max_size - sizes[-2][0]
        else:
            size_diff = max_size
        size_diff = min(10 * self.chunk_size, size_diff)
        save_size = size_diff if size_diff else max_size
        if max_size == 0:
            self.n += 1
            time.sleep(1)
            return

        self.n = 0
        to_write = []
        for i in range(save_size):
            try:
                data = max_queue.get(timeout=0.1)
                to_write.append(list(unstructure(data)))
            except queue.Empty:
                break

        data_lists = tuple(zip(*to_write))
        for name, data in zip(unstructure(max_queue.names), data_lists):
            if f[name].shape[0] != self.chunk_size:
                if self.flat[max_name]:
                    f[name].resize(f[name].shape[0] + len(data), axis=0)
                    f[name][-len(data):] = data
                else:
                    g = f[max_name]
                    g[name].resize(g[name].shape[0] + len(data), axis=0)
                    g[name][-len(data):] = data
            else:
                if self.flat[max_name]:
                    f[name].resize(len(data), axis=0)
                    f[name][:] = data
                else:
                    g = f[max_name]
                    g[name].resize(len(data), axis=0)
                    g[name][:] = data

    def shutdown(self, **kwargs):
        self.h5_file.close() if self.h5_file else None
        super().shutdown(**kwargs)

class QueueDecimator(AnalysisStep):
    def __init__(self, input_queue, output_queue,n, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.i=0
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.input_queues = (input_queue,)
        self.output_queues = (output_queue,)

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return
        if self.i % self.n == 0:
            self.output_queue.put(data)
        self.i+=1

class QueueReducer(AnalysisStep):
    def __init__(self, input_queue, output_queue, max_size, record=10000,**kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.input_queues = (input_queue,)
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.max_size = max_size
        self.statuses=collections.deque(maxlen=record)
        self.record=record
        self.ratio=multiprocessing.Value('f',0)

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)

        except queue.Empty:
            return
        if self.output_queue.qsize() < self.max_size:
            self.output_queue.put(data)
            self.statuses.append(1)
        else:
            self.statuses.append(0)
        self.ratio.value=self.statuses.count(1)/self.record

    def status(self):
        stat=super().status()
        stat["ratio"]=self.ratio.value
        return stat

class QueueGrouper(AnalysisStep):

    def __init__(self, input_queues,output_queue,output_empty=True, **kwargs):
        super().__init__(**kwargs)
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.current=0
        self.nexts:list[tuple[int, typing.Any] | None]=[None for _ in input_queues]
        self.out=tuple([] for _ in self.input_queues)
        self.output_empty=output_empty

    def action(self):
        if any(n is None for n in self.nexts):
            for i,q in enumerate(self.input_queues):
                if self.nexts[i] is None:
                    if q.closed.value and q.empty():
                        self.nexts[i]=(np.inf,)
                    try:
                        self.nexts[i]=q.get(timeout=0.1)
                    except queue.Empty:
                        time.sleep(0.1)
                        return

        if all(n[0]==np.inf for n in self.nexts):
            self.shutdown()
            return

        for i,n in enumerate(self.nexts):
            while self.nexts[i][0]==self.current:
                self.out[i].append(self.nexts[i][1])
                try:
                    self.nexts[i]=self.input_queues[i].get(timeout=0.1)
                except queue.Empty:
                    self.nexts[i]=None
                    break
        if any(n is None for n in self.nexts):
            return

        self.output_queue.put(self.out) if any(self.out) or self.output_empty else None
        self.out=tuple([] for _ in self.input_queues)
        self.current+=1


class Display(AnalysisStep):
    def __init__(
            self,
            grouped_queue,
            n
    ):
        super().__init__()
        self.input_queues = (grouped_queue,)
        self.grouped_queue = grouped_queue
        self.name = "Display"
        self.current_data={
            "etof": collections.deque(maxlen=n),
            "itof": collections.deque(maxlen=n),
            "cluster": collections.deque(maxlen=n),
            "timestamps": collections.deque(maxlen=n)
        }
        self.figure=None
        self.ax=None
        self.xy_hist=None
        self.xy_hist_data=np.zeros((256,256))
        self.toa_hist=None
        self.toa_hist_data=np.zeros(2000)
        self.etof_hist=None
        self.etof_hist_data=np.zeros(2000)
        self.itof_hist=None
        self.itof_hist_data=np.zeros(400)
        self.update_interval=1
        self.last_update=0

    def initialize(self):
        self.figure, self.ax = plt.subplots(2,2)
        self.xy_hist=self.ax[0,0].imshow(np.random.random(size=(256,256)), extent=[0,256,0,256], origin='lower')
        self.ax[0,0].set_title("XY")
        self.toa_hist=self.ax[0,1].plot(np.linspace(0, 2000, 2000), np.linspace(0,1,num=2000))[0]
        self.ax[0,1].set_title("ToA")
        self.etof_hist=self.ax[1,0].plot(np.linspace(0, 2000, 2000), np.linspace(0,1,num=2000))[0]

        self.ax[1,0].set_title("eToF")
        self.itof_hist=self.ax[1,1].plot(np.linspace(0, 2e4, 400), np.linspace(0,1,num=400))[0]
        self.ax[1,1].set_title("iToF")
        plt.suptitle("Processing Rate:")
        self.figure.tight_layout()
        self.figure.show()
        super().initialize()

    def action(self):
        try:
            data = self.grouped_queue.get(timeout=0.1)
            # print(data)
        except queue.Empty:
            return

        if len(self.current_data["etof"])==self.current_data["etof"].maxlen:
            etof_rem=self.current_data["etof"].popleft()
            itof_rem=self.current_data["itof"].popleft()
            t_rem,x_rem,y_rem= zip(*last) if (last:=self.current_data["cluster"].popleft()) else ([],[],[])
            last_timestamp=self.current_data["timestamps"].popleft()

            for x,y in zip(x_rem,y_rem):

                self.xy_hist_data[int(x),int(y)]-=1
            for t in t_rem:
                if 0<t<2e4:
                    self.toa_hist_data[int(t/10)]-=1
            for t in etof_rem:
                if 0<t<2e4:
                    self.etof_hist_data[int(t/10)]-=1
            for t in itof_rem:
                if 0<t<2e4:
                    self.itof_hist_data[int(t/50)]-=1

            processing_rate=self.current_data["timestamps"].maxlen/(time.time()-last_timestamp)
            plt.suptitle(f"Processing Rate: {processing_rate:.2f} Hz")



        etof, itof, cluster = data
        self.current_data["etof"].append(etof)
        self.current_data["itof"].append(itof)
        self.current_data["cluster"].append(cluster)
        self.current_data["timestamps"].append(time.time())
        if cluster:
            clust=list(zip(*cluster))
            for x,y in zip(clust[1],clust[2]):
                self.xy_hist_data[int(x),int(y)]+=1 if x<256 and y<256 else 0
            for t in clust[0]:
                if 0<t<2000:
                    self.toa_hist_data[int(t)]+=1

        for t in etof:
            if 0<t<2000:
                self.etof_hist_data[int(t)]+=1
        for t in itof:
            if 0<t<2e4:
                self.itof_hist_data[int(t/50)]+=1

        if time.time()-self.last_update>self.update_interval:
            self.xy_hist.set_data(self.xy_hist_data/np.max(self.xy_hist_data))
            self.toa_hist.set_ydata(self.toa_hist_data/np.max(self.toa_hist_data))
            self.etof_hist.set_ydata(self.etof_hist_data/np.max(self.etof_hist_data))
            self.itof_hist.set_ydata(self.itof_hist_data/np.max(self.itof_hist_data))

            cluster_count_rate=np.mean([len(c) for c in self.current_data["cluster"]])
            etof_count_rate=np.mean([len(c) for c in self.current_data["etof"]])
            itof_count_rate=np.mean([len(c) for c in self.current_data["itof"]])

            self.ax[0,0].set_title(f"XY (Cluster Rate: {cluster_count_rate:.2f})")
            self.ax[0,1].set_title(f"ToA (Count Rate: {cluster_count_rate:.2f})")
            self.ax[1,0].set_title(f"eToF (Count Rate: {etof_count_rate:.2f})")
            self.ax[1,1].set_title(f"iToF (Count Rate: {itof_count_rate:.2f})")

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.last_update=time.time()

class DummyStream(TPXFileReader):
    def __init__(self, path, chunk_queue, delay, **kwargs):
        super().__init__(path, chunk_queue, **kwargs)
        self.delay = delay
        self.name = "DummyStream"

    def action(self):
        try:
            packet = self.file.read(8)
            if len(packet) < 8:
                self.curr_file_idx += 1
                if self.curr_file_idx >= len(self.files):
                    self.curr_file_idx = 0
                self.file.close()
                self.file = open(self.files[self.curr_file_idx], 'rb')
                return
            _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
            num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
            packets = [int.from_bytes(self.file.read(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
            self.chunk_queue.put(packets)
            time.sleep(self.delay)
        except Exception as e:
            print(e)
            self.shutdown()
            return


class FolderStream(TPXFileReader):
    def __init__(self, path, chunk_queue, max_age=0, **kwargs):
        super().__init__(path, chunk_queue, **kwargs)
        self.max_age = max_age
        self.name = "FolderStream"

    def action(self):
        super().action()
        if self.curr_file_idx == 0:
            return
        most_recent_file=sorted(os.listdir(self.path), key=lambda x: os.path.getmtime(os.path.join(self.path, x)))[-1]
        if self.max_age and time.time()-os.path.getmtime(os.path.join(self.path, most_recent_file))>self.max_age:
            return
        self.file.close() if self.file else None
        self.file = open(os.path.join(self.path, most_recent_file), 'rb')
        self.curr_file_idx = 0







def create_process_instances(process_class, n_instances, output_queue, process_args, queue_args=None, process_name="", queue_name=""):
    if queue_args is None:
        queue_args = {}
    queues=tuple([ExtendedQueue(**queue_args) for _ in range(n_instances)])
    args=[]
    for q in queues:
        new_args=process_args.copy()
        out_key=[k for k,v in new_args.items() if v is None][0]
        new_args[out_key]=q
        args.append(new_args)
    processes={f"{process_name}_{i}": process_class(**a) for i,a in enumerate(args)}
    weaver=Weaver(queues, output_queue)
    return (
        {f"{queue_name}_{i}": q for i,q in enumerate(queues)},
        processes,
        weaver
    )


# %%
