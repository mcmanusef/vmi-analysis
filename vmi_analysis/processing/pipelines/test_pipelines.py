import multiprocessing
import socket

import numpy as np
from matplotlib import pyplot as plt

from .. import data_types, processes
from ... import serval
from .base_pipeline import AnalysisPipeline
import threading
import requests
import cv2


class MonitorPipeline(AnalysisPipeline):
    def __init__(
        self,
        saving_path,
        cluster_processes=1,
        timeout=0,
        toa_range=None,
        etof_range=None,
        itof_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.queues = {
            "chunk_stream": data_types.ExtendedQueue(),
            "chunk": data_types.ExtendedQueue(),
            "pixel": data_types.ExtendedQueue(),
            "etof": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "itof": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "pulses": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "clusters": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "t_etof": data_types.ExtendedQueue(),
            "t_itof": data_types.ExtendedQueue(),
            "t_pulse": data_types.ExtendedQueue(),
            "t_cluster": data_types.ExtendedQueue(),
            "grouped": data_types.ExtendedQueue(),
            "reduced_grouped": data_types.ExtendedQueue(),
        }

        self.processes = {
            "ChunkStream": processes.FolderStream(
                saving_path, self.queues["chunk_stream"]
            ).make_process(),
            "Chunk": processes.QueueReducer(
                self.queues["chunk_stream"], self.queues["chunk"], max_size=1000
            ).make_process(),
            "Converter": processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Clusterer": processes.DBSCANClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                self.queues["pulses"],
                (self.queues["etof"], self.queues["itof"], self.queues["clusters"]),
                self.queues["t_pulse"],
                (
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
            ).make_process(),
            "Grouper": processes.QueueGrouper(
                (
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
                self.queues["grouped"],
            ).make_process(),
            "Reducer": processes.QueueReducer(
                self.queues["grouped"], self.queues["reduced_grouped"], max_size=1000
            ).make_process(),
            "Display": processes.Display(
                self.queues["reduced_grouped"],
                1000000,
                toa_range=toa_range,
                etof_range=etof_range,
                itof_range=itof_range,
            ).make_process(),
            "Bin": processes.QueueVoid((self.queues["t_pulse"],)).make_process(),
        }
        self.processes["Reducer"].astep.name = "r2"

        if cluster_processes > 1:
            queues, proc, weaver = processes.create_process_instances(
                processes.DBSCANClusterer,
                cluster_processes,
                self.queues["clusters"],
                process_args={
                    "pixel_queue": self.queues["pixel"],
                    "cluster_queue": None,
                },
                queue_args={"force_monotone": True},
                queue_name="clust",
                process_name="clusterer",
            )

            self.queues.update(queues)
            del self.processes["Clusterer"]
            self.processes.update({n: k.make_process() for n, k in proc.items()})
            self.processes["Weaver"] = weaver.make_process()


class RunMonitorPipeline(AnalysisPipeline):
    def __init__(
        self,
        saving_path,
        cluster_processes=1,
        toa_range=None,
        etof_range=None,
        itof_range=None,
        calibration=None,
        center=(128, 128),
        angle=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(maxsize=1000),
            "pixel": data_types.ExtendedQueue(),
            "etof": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "itof": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "pulses": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "clusters": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "t_etof": data_types.ExtendedQueue(),
            "t_itof": data_types.ExtendedQueue(),
            "t_pulse": data_types.ExtendedQueue(),
            "t_cluster": data_types.ExtendedQueue(),
            "grouped": data_types.ExtendedQueue(),
        }

        self.processes = {
            "ChunkStream": processes.TPXFileReader(
                saving_path, self.queues["chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Clusterer": processes.CustomClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                self.queues["pulses"],
                (self.queues["etof"], self.queues["itof"], self.queues["clusters"]),
                self.queues["t_pulse"],
                (
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
            ).make_process(),
            "Grouper": processes.QueueGrouper(
                (
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
                self.queues["grouped"],
            ).make_process(),
            "Display": processes.Display(
                self.queues["grouped"],
                10000000,
                toa_range=toa_range,
                etof_range=etof_range,
                itof_range=itof_range,
                calibration=calibration,
                center=center,
                angle=angle,
            ).make_process(),
            "Bin": processes.QueueVoid((self.queues["t_pulse"],)).make_process(),
        }

        if cluster_processes > 1:
            queues, proc, weaver = processes.create_process_instances(
                processes.DBSCANClusterer,
                cluster_processes,
                self.queues["clusters"],
                process_args={
                    "pixel_queue": self.queues["pixel"],
                    "cluster_queue": None,
                },
                queue_args={"force_monotone": True},
                queue_name="clust",
                process_name="clusterer",
            )

            self.queues.update(queues)
            del self.processes["Clusterer"]
            self.processes.update({n: k.make_process() for n, k in proc.items()})
            self.processes["Weaver"] = weaver.make_process()


class LiveMonitorPipeline(AnalysisPipeline):
    def __init__(
        self,
        *args,
        local_ip=("localhost", 1234),
        serval_ip=serval.DEFAULT_IP,
        toa_range=None,
        etof_range=None,
        itof_range=None,
        preview_ip_frame=("localhost", 1235),
        preview_ip_total=("localhost", 1236),
        **kwargs,
    ):
        super().__init__()
        self.serval_ip = serval_ip
        self.local_ip = local_ip
        self.preview_ip_frame = preview_ip_frame
        self.preview_ip_total = preview_ip_total

        self.queues = {
            "chunk": data_types.ExtendedQueue(maxsize=1000),
            "pixel": data_types.ExtendedQueue(),
            "etof": data_types.ExtendedQueue(force_monotone=True, maxsize=1000),
            "itof": data_types.ExtendedQueue(force_monotone=True, maxsize=1000),
            "pulses": data_types.ExtendedQueue(force_monotone=True, maxsize=1000),
            "clusters": data_types.ExtendedQueue(
                force_monotone=True, maxsize=1000, loud=True
            ),
            "t_etof": data_types.ExtendedQueue(maxsize=1000),
            "t_itof": data_types.ExtendedQueue(maxsize=1000),
            "t_pulse": data_types.ExtendedQueue(maxsize=1000),
            "t_cluster": data_types.ExtendedQueue(maxsize=1000),
            "grouped": data_types.ExtendedQueue(maxsize=1000),
        }

        self.processes = {
            "Listener": processes.TPXListener(
                self.local_ip, self.queues["chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Clusterer": processes.DBSCANClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            # "printvoid": processes.QueueVoid((
            #     self.queues['clusters'], self.queues['etof'], self.queues['itof'], self.queues['pulses']
            # ), loud=True).make_process(),
        }
        self.latest_frame = np.zeros((256, 256), dtype=np.uint8)
        self.acc_frame = np.zeros((256, 256), dtype=np.uint8)
        self.frame_listener = threading.Thread(
            target=self.process_frame_loop,
            args=(serval_ip, self.latest_frame, self.acc_frame),
        )
        self.frame_listener.start()

    def start(self):
        starting_thread = threading.Thread(target=super().start())
        starting_thread.start()

        serval_destination = {
            "Raw": [
                {
                    "Base": f"tcp://{self.local_ip[0]}:{self.local_ip[1]}",
                    "FilePattern": "",
                }
            ],
            "Preview": {
                "Period": 0.1,
                "SamplingMode": "skipOnFrame",
                "ImageChannels": [
                    {
                        "Base": self.serval_ip,
                        "Format": "png",
                        "Mode": "count",
                    }
                ],
            },
        }

        serval.set_acquisition_parameters(serval_destination, frame_time=1)
        resp = requests.get(self.serval_ip + "/server/destination")
        print(resp.text)
        serval.start_acquisition(block=False)
        starting_thread.join()

    def stop(self):
        serval.stop_acquisition()
        super().stop()

    def process_frame_loop(self, ip, last_frame, acc_frame):
        while True:
            try:
                png = requests.get(ip + "/measurement/image").content
                if (
                    png
                    == b"HTTP is not setup as a destination in the current measurement's destination config.\r\n"
                ):
                    print("HTTP not set up")
                    continue
                frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)

                last_frame[:] = frame
                acc_frame += frame

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectTimeout,
                ConnectionRefusedError,
            ):
                continue


class CorrelatorTestDataPipeline(AnalysisPipeline):
    def __init__(self, input_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "Chunk": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "Pixel": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "Etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                chunk_size=2000,
            ),
            "Itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                chunk_size=2000,
            ),
            "Pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
                chunk_size=10000,
            ),
            "Clusters": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f", "f", "f"),
                names=("toa", "x", "y"),
                force_monotone=True,
                chunk_size=2000,
            ),
        }
        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["Chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
                chunk_queue=self.queues["Chunk"],
                pixel_queue=self.queues["Pixel"],
                laser_queue=self.queues["Pulses"],
                etof_queue=self.queues["Etof"],
                itof_queue=self.queues["Itof"],
            ).make_process(),
            "Clusterer": processes.DBSCANClusterer(
                pixel_queue=self.queues["Pixel"], cluster_queue=self.queues["Clusters"]
            ).make_process(),
            "Pulse Cache": processes.QueueCacheWriter(
                "pulse_cache.pk", self.queues["Pulses"]
            ).make_process(),
            "Cluster Cache": processes.QueueCacheWriter(
                "cluster_cache.pk", self.queues["Clusters"]
            ).make_process(),
            "Etof Cache": processes.QueueCacheWriter(
                "etof_cache.pk", self.queues["Etof"]
            ).make_process(),
            "Itof Cache": processes.QueueCacheWriter(
                "itof_cache.pk", self.queues["Itof"]
            ).make_process(),
        }


class CorrelatorTestPipeline(AnalysisPipeline):
    def __init__(self, chunksize=10000, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "Pulses": data_types.ExtendedQueue(chunk_size=chunksize),
            "Clusters": data_types.ExtendedQueue(chunk_size=chunksize),
            "Etof": data_types.ExtendedQueue(chunk_size=chunksize),
            "Itof": data_types.ExtendedQueue(chunk_size=chunksize),
            "t_pulse": data_types.ExtendedQueue(
                buffer_size=0, dtypes=("i",), names=("t_pulse",), chunk_size=10000
            ),
            "t_etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f",)),
                names=("etof_corr", ("t_etof",)),
                chunk_size=2000,
            ),
            "t_itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f",)),
                names=("itof_corr", ("t_itof",)),
                chunk_size=2000,
            ),
            "t_cluster": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f", "f", "f")),
                names=("clust_corr", ("t", "x", "y")),
                chunk_size=2000,
            ),
        }

        self.processes = {
            "pulse_cache": processes.QueueCacheReader(
                "pulse_cache.pk", self.queues["Pulses"]
            ).make_process(),
            "cluster_cache": processes.QueueCacheReader(
                "cluster_cache.pk", self.queues["Clusters"]
            ).make_process(),
            "etof_cache": processes.QueueCacheReader(
                "etof_cache.pk", self.queues["Etof"]
            ).make_process(),
            "itof_cache": processes.QueueCacheReader(
                "itof_cache.pk", self.queues["Itof"]
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                input_trigger_queue=self.queues["Pulses"],
                queues_to_index=(
                    self.queues["Etof"],
                    self.queues["Itof"],
                    self.queues["Clusters"],
                ),
                output_trigger_queue=self.queues["t_pulse"],
                indexed_queues=(
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
            ).make_process(),
        }


class ClusterTestDataPipeline(AnalysisPipeline):
    def __init__(self, input_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "Chunk": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "Pixel": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "Etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                chunk_size=2000,
            ),
            "Itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                chunk_size=2000,
            ),
            "Pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
                chunk_size=10000,
            ),
        }
        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["Chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
                chunk_queue=self.queues["Chunk"],
                pixel_queue=self.queues["Pixel"],
                laser_queue=self.queues["Pulses"],
                etof_queue=self.queues["Etof"],
                itof_queue=self.queues["Itof"],
            ).make_process(),
            "Pixel Cache": processes.QueueCacheWriter(
                "pixel_cache.pk", self.queues["Pixel"]
            ).make_process(),
        }


class MultiprocessTestPipeline(AnalysisPipeline):
    def __init__(self, n=4, **kwargs):
        super().__init__(**kwargs)

        self.queues = {
            "pixel": data_types.ExtendedQueue(chunk_size=2000),
            "cluster": data_types.ExtendedQueue(chunk_size=2000),
        }
        self.processes = {
            "reader": processes.QueueCacheReader(
                "pixel_cache.pk", self.queues["pixel"]
            ),
            "writer": processes.QueueCacheWriter(
                "cluster_cache.pk", self.queues["cluster"]
            ),
        }

        mp_queues, mp_processes = processes.multithread_process(
            # processes.CuMLDBSCANClusterer,
            processes.DBSCANClusterer,
            # processes.DBSCANClustererPrecomputed,
            {"pixel_queue": self.queues["pixel"]},
            {"cluster_queue": self.queues["cluster"]},
            n,
            in_queue_kw_args={"chunk_size": 2000},
            out_queue_kw_args={"force_monotone": True, "chunk_size": 2000},
            astep_kw_args={"dbscan_params": {"eps": 1.5, "min_samples": 8}},
            name="clusterer",
        )

        self.processes.update(mp_processes)
        self.queues.update(mp_queues)
        for k, v in self.processes.items():
            self.processes[k] = v.make_process()


class VMIConverterTestPipeline(AnalysisPipeline):
    def __init__(self, input_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "Chunk": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=10000
            ),
            "Pixel": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=10000
            ),
            "Etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                chunk_size=10000,
            ),
            "Itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                chunk_size=10000,
            ),
            "Pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
                chunk_size=10000,
            ),
        }
        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["Chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
                chunk_queue=self.queues["Chunk"],
                pixel_queue=self.queues["Pixel"],
                laser_queue=self.queues["Pulses"],
                etof_queue=self.queues["Etof"],
                itof_queue=self.queues["Itof"],
            ).make_process(),
        }
