import typing
import sklearn.cluster

from .base_process import AnalysisStep
from ..data_types import *
from ..tpx_conversion import *

PIXEL_RES = 25 / 16
PERIOD = 25 * 2 ** 30


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
        except queue.Empty or InterruptedError:
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
        except queue.Empty or InterruptedError:
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
        except queue.Empty or InterruptedError:
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
            except queue.Empty or InterruptedError:
                time.sleep(0.1)
                return

        for i, q in enumerate(self.queues_to_index):
            if self.current[i] is None:
                try:
                    c = q.get(timeout=0.1)
                    if self.current_samples[i] is None:
                        self.current_samples[i] = c
                    self.current[i] = list(unstructure(c))

                except queue.Empty or InterruptedError:
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
                except queue.Empty or InterruptedError:
                    self.current[i] = None
                    break

        if any(c is None for c in self.current):
            return

        self.output_trigger_queue.put(self.last_trigger_time)
        self.last_trigger_time = self.current_trigger_time

        try:
            self.current_trigger_time = self.input_trigger_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            self.current_trigger_time = None
        # print(self.current_trigger_time, self.last_trigger_time, self.current_trigger_idx, self.current)
        self.current_trigger_idx += 1


class TDCFilter(AnalysisStep):
    def __init__(self, tdc_queue, tdc1_queue, tdc2_queue, **kwargs):
        super().__init__(**kwargs)
        self.tdc_queue = tdc_queue
        self.tdc1_queue = tdc1_queue
        self.tdc2_queue = tdc2_queue
        self.input_queues = (tdc_queue,)
        self.output_queues = (tdc1_queue, tdc2_queue)
        self.name = "TDCFilter"

    def action(self):
        try:
            tdc = self.tdc_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            return
        if tdc[1] == 1:
            self.tdc1_queue.put(tdc[0])
        elif tdc[1] == 3:
            self.tdc2_queue.put(tdc[0])