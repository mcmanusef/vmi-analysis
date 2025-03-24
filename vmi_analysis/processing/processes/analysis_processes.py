import queue
import time

import numpy as np
import sklearn.cluster

from vmi_analysis.processing.tpx_conversion import (
    average_over_clusters,
    cluster,
    find_distance_matrix,
    precompute_distance_matrix,
    process_packet,
)
from .base_process import AnalysisStep
from ..data_types import (
    ClusterData,
    Queue,
    StructuredDataQueue as SDQueue,
    Chunk,
    PixelData,
    TDCData,
    Trigger,
    TimestampedData,
    IndexedData,
)
from ..tpx_constants import PERIOD


# from ..tpx_conversion import *


class TPXConverter(AnalysisStep):
    """
    Converts TPX3 data into PixelData and TDCData
    Useful for direct conversion of binary TPX3 data into a format that can be used by the rest of the pipeline
    Not specific to any particular experiment

    Parameters:
    - chunk_queue: The queue containing the binary data chunks
    - pixel_queue: The queue to put PixelData into (Chunked [[(time, x, y, tot), ...],...])
    - tdc_queue: The queue to put TDCData into (Unchunked [(time, type),...]). Type is 1 for TDC1R, 2 for TDC1F, 3 for TDC2R, 4 for TDC2F
    - kwargs: Additional keyword arguments to pass to the AnalysisStep constructor
    """

    def __init__(
        self,
        chunk_queue: Queue[Chunk],
            pixel_queue: SDQueue[PixelData],  # Need to fix. should be SDQueue[list[PixelData]]?
        tdc_queue: SDQueue[TDCData],
    ):
        super().__init__()
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
                self.pixel_queue.put(
                        PixelData(time=packet_data[0], x=packet_data[1], y=packet_data[2], tot=packet_data[3])
                )
            elif data_type == 0:
                tdc_type, c_time, ftime, _ = packet_data
                tdc_time = 3.125 * c_time + 0.260 * ftime
                tdc_type = (
                    1
                    if tdc_type == 10
                    else 2
                    if tdc_type == 15
                    else 3
                    if tdc_type == 14
                    else 4
                    if tdc_type == 11
                    else 0
                )
                self.tdc_queue.put(
                        TDCData(time=tdc_time, type=tdc_type)
                )


class DBSCANClusterer(AnalysisStep):
    """
    Clusters data using DBSCAN, and puts the clusters into the cluster_queue.
    Optionally, can also put the clustered pixels into the output_pixel_queue.

    Parameters:
    - pixel_queue: The queue containing the pixel data to cluster (Chunked [[(toa, x, y, tot), ...], ...])
    - cluster_queue: The queue to put the clustered data into. (Unchunked [(toa, x, y),...])
    - output_pixel_queue: The queue to put the clustered pixel data into. If None, the pixel data is not put into a queue.
    - dbscan_params: The parameters to pass to the DBSCAN algorithm. Default is {"eps": 1.5, "min_samples": 8}

    - kwargs: Additional keyword arguments to pass to the AnalysisStep constructor
    """

    pixel_queue: Queue[list[PixelData]]
    cluster_queue: Queue[ClusterData]
    output_pixel_queue: Queue[IndexedData[PixelData]] | None

    def __init__(
        self,
        pixel_queue,
        cluster_queue,
        output_pixel_queue=None,
        dbscan_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pixel_queue = pixel_queue
        self.cluster_queue = cluster_queue
        self.input_queues = (pixel_queue,)
        self.output_pixel_queue = output_pixel_queue
        self.output_queues = (
            (cluster_queue,)
            if not output_pixel_queue
            else (cluster_queue, output_pixel_queue)
        )
        self.dbscan_params = (
            dbscan_params if dbscan_params else {"eps": 1.5, "min_samples": 8}
        )
        self.dbscan = None
        self.group_index = 0

        self.name = "DBSCANClusterer"

    def initialize(self):
        self.dbscan = sklearn.cluster.DBSCAN(**self.dbscan_params)
        super().initialize()

    def fit(self, x, y):
        assert self.dbscan
        return self.dbscan.fit(np.column_stack((x, y))).labels_

    def action(self):
        try:
            pixels = self.pixel_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            time.sleep(0.1)
            return
        if len(pixels) == 0:
            return

        toa = np.array([p.time for p in pixels])
        x = np.array([p.x for p in pixels])
        y = np.array([p.y for p in pixels])
        tot = np.array([p.tot for p in pixels])

        if np.max(toa) - np.min(toa) > PERIOD / 2:
            return

        cluster_index = self.fit(x, y)

        clusters = average_over_clusters(cluster_index, toa, x, y, tot)

        for c in clusters:
            self.cluster_queue.put(ClusterData(time=c[0], x=c[1], y=c[2]))
        for i, p in enumerate(pixels):
            if self.output_pixel_queue:
                (
                    self.output_pixel_queue.put(IndexedData(index=self.group_index + cluster_index[i], data=p))
                    if cluster_index[i] >= 0
                    else self.output_pixel_queue.put(IndexedData(index=-1, data=p))
                )

        self.group_index += np.max(cluster_index) + 1


class CuMLDBSCANClusterer(DBSCANClusterer):
    def initialize(self):
        import cuml  # type: ignore

        super().initialize()
        self.dbscan = cuml.DBSCAN(
            eps=self.dbscan_params["eps"], min_samples=self.dbscan_params["min_samples"]
        )


class DBSCANClustererPrecomputed(DBSCANClusterer):
    pc_dist = None

    def initialize(self):
        super().initialize()
        self.dbscan = sklearn.cluster.DBSCAN(metric="precomputed", **self.dbscan_params)
        self.pc_dist = precompute_distance_matrix()

    def fit(self, x, y):
        return self.dbscan.fit(find_distance_matrix(x, y, self.pc_dist)).labels_


class HDBSCANClusterer(DBSCANClusterer):
    def initialize(self):
        super().initialize()
        if "eps" in self.dbscan_params.keys():
            del self.dbscan_params["eps"]
        self.dbscan = sklearn.cluster.HDBSCAN(**self.dbscan_params)  # type: ignore

    def fit(self, x, y):
        if len(x) < self.dbscan_params["min_samples"]:
            return np.zeros(len(x)) - 1
        return self.dbscan.fit(np.column_stack((x, y))).labels_


class CustomClusterer(AnalysisStep):
    """
    Clusters data using a custom clustering algorithm, and puts the clusters into the cluster_queue.
    Optionally, can also put the clustered pixels into the output_pixel_queue.

    Parameters:
    - pixel_queue: The queue containing the pixel data to cluster (Chunked [[(toa, x, y, tot), ...], ...])
    - cluster_queue: The queue to put the clustered data into. (Unchunked [(toa, x, y),...])
    - output_pixel_queue: The queue to put the clustered pixel data into. If None, the pixel data is not put into a queue.
    - kwargs: Additional keyword arguments to pass to the AnalysisStep constructor
    """

    pixel_queue: Queue[list[PixelData]]
    cluster_queue: Queue[ClusterData]
    output_pixel_queue: Queue[IndexedData[PixelData]]

    def __init__(
        self,
        pixel_queue,
        cluster_queue,
        output_pixel_queue=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pixel_queue = pixel_queue
        self.cluster_queue = cluster_queue
        self.input_queues = (pixel_queue,)
        self.output_pixel_queue = output_pixel_queue
        self.output_queues = (
            (cluster_queue,)
            if not output_pixel_queue
            else (cluster_queue, output_pixel_queue)
        )
        self.group_index = 0
        self.compiled = None

        self.name = "CustomClusterer"

    def action(self):
        try:
            pixels = self.pixel_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            time.sleep(0.1)
            return
        if len(pixels) == 0:
            return

        toa = np.array([p.time for p in pixels])
        x = np.array([p.x for p in pixels])
        y = np.array([p.y for p in pixels])
        tot = np.array([p.tot for p in pixels])

        if np.max(toa) - np.min(toa) > PERIOD / 2:
            return

        cluster_index = cluster(x, y, 10, 10, 15)
        clusters = average_over_clusters(cluster_index, toa, x, y, tot)

        for c in clusters:
            self.cluster_queue.put(ClusterData(time=c[0], x=c[1], y=c[2]))
        for i, p in enumerate(pixels):
            if self.output_pixel_queue:
                (
                    self.output_pixel_queue.put(IndexedData(index=self.group_index + cluster_index[i], data=p))
                    if cluster_index[i] >= 0
                    else self.output_pixel_queue.put(IndexedData(index=-1, data=p))
                )
        self.group_index += np.max(cluster_index) + 1


class TriggerAnalyzer(AnalysisStep):
    """
    Indexes data based on trigger times, and puts the indexed data into the indexed_queues. Subtracts the trigger time from the
    time of each data point. This is used to synchronize the data with the trigger times, converting the data into a format that
    gives the time since the last trigger.

    Parameters:
    - input_trigger_queue: The queue containing the trigger times.
    - queues_to_index: A tuple of queues containing the data to index. The data should be in some nested tuple format, where the
    first element is the time of the data point, and the rest of the elements are the data associated with that time.
    - output_trigger_queue: A passthrough queue for the trigger times.
    - indexed_queues: A tuple of queues to put the indexed data into. The indexed data is in the format (trigger_index, data) where
    trigger_index is the index of the trigger time associated with the data point, and data is as in the input queues, but with the
    time relative to the trigger time.
    """

    input_trigger_queue: Queue[Trigger]
    queues_to_index: tuple[Queue[TimestampedData], ...]
    output_trigger_queue: Queue[Trigger]
    indexed_queues: tuple[Queue[tuple[int, TimestampedData]], ...]

    def __init__(
        self,
        input_trigger_queue,
        queues_to_index,
        output_trigger_queue,
        indexed_queues,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_trigger_queue = input_trigger_queue
        self.queues_to_index = queues_to_index
        self.output_trigger_queue = output_trigger_queue
        self.indexed_queues = indexed_queues
        self.input_queues = (input_trigger_queue, *queues_to_index)
        self.output_queues = (output_trigger_queue, *indexed_queues)
        self.name = "TriggerAnalyzer"

        # The current value being processed for each queue
        self.current: list[TimestampedData | None] = [None for _ in queues_to_index]
        self.current_trigger_idx = 0
        self.last_trigger_time = -1
        self.current_trigger_time = None

    def action(self):
        while self.current_trigger_time is None:
            try:
                self.current_trigger_time = self.input_trigger_queue.get(timeout=0.1).time
            except queue.Empty or InterruptedError:
                time.sleep(0.1)
                return

        for i, q in enumerate(self.queues_to_index):
            if self.current[i] is None:
                try:
                    self.current[i] = q.get(timeout=0.1)

                except queue.Empty or InterruptedError:
                    if q.closed.value and q.empty():
                        self.current[i] = TimestampedData(time=np.inf)
                    time.sleep(0.1)
                    return
        n = 0
        for i, q in enumerate(self.queues_to_index):
            while self.current[i].time < self.current_trigger_time:
                if n > 1000:
                    return
                n += 1

                out = self.current[i]._replace(time=self.current[i].time - self.current_trigger_time)

                self.indexed_queues[i].put(
                        IndexedData(index=self.current_trigger_idx, data=out)
                )
                try:
                    self.current[i] = q.get(block=False)
                except queue.Empty or InterruptedError:
                    self.current[i] = None
                    break

        if any(c is None for c in self.current):
            return

        self.output_trigger_queue.put(TimestampedData(time=self.last_trigger_time))
        self.last_trigger_time = self.current_trigger_time

        try:
            self.current_trigger_time = self.input_trigger_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            self.current_trigger_time = None
        self.current_trigger_idx += 1


class TDCFilter(AnalysisStep):
    def __init__(
            self,
            tdc_queue: SDQueue[TDCData],
            tdc1_queue: SDQueue[TimestampedData],
            tdc2_queue: SDQueue[TimestampedData],
            **kwargs
    ):
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
            self.tdc1_queue.put(TimestampedData(time=tdc[0]))
        elif tdc[1] == 3:
            self.tdc2_queue.put(TimestampedData(time=tdc[0]))
