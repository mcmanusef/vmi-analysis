import uconn_processes
from vmi_analysis.processing import data_types, processes
from vmi_analysis.processing.data_types import PixelData, Chunk, IndexedData, ClusterData, ToF, Trigger, Timestamp
from vmi_analysis.processing.pipelines import BasePipeline, PostProcessingPipeline


class RawVMIConverterPipeline(PostProcessingPipeline):
    """
    Pipeline for converting raw VMI data to h5 files.
    Specific to our VMI setup.
    Converts raw data to pixel data, etof, itof, and pulse data.
    Data are uncorrelated with laser pulses.
    Output Format:
    - pixel: toa, x, y, tot
    - etof: etof
    - itof: itof
    - pulses: pulses
    """

    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        self.queues = {
            "chunk": data_types.Queue[Chunk](),
            "pixel": data_types.MonotonicQueue[PixelData](
                    dtypes=PixelData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y", "tot": "tot"},
                unwrap=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "etof"},
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "itof"},
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.MonotonicQueue(
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "pulse"},
                max_back=1e9,
                chunk_size=10000,
            ),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": uconn_processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "pixel": self.queues["pixel"],
                    "etof": self.queues["etof"],
                    "itof": self.queues["itof"],
                    "pulses": self.queues["pulses"],
                },
            ).make_process(),
        }


class VMIConverterPipeline(PostProcessingPipeline):
    """
    Pipeline for converting raw VMI data to UV4 files (H5 file with specific internal format).
    Specific to our VMI setup.
    Converts raw data to pixel data, etof, itof, and pulse data.
    Data are correlated with laser pulses.
    Output Format:
    - pixel: pixel_corr, t, x, y, tot
    - etof: etof_corr, t_etof
    - itof: itof_corr, t_itof
    - pulses: t_pulse
    """

    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)
        self.queues = {
            "chunk": data_types.StructuredDataQueue(),
            "pixel": data_types.MonotonicQueue[PixelData](
                    dtypes=PixelData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y", "tot": "tot"},
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "etof"},
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "itof"},
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "pulse"},
                max_back=1e9,
                chunk_size=10000,
            ),
            "t_etof": data_types.StructuredDataQueue[IndexedData[Timestamp]](
                    dtypes={**IndexedData.c_dtypes, **Timestamp.c_dtypes},
                    names={"index": "etof_corr", "time": "t_etof"},
                chunk_size=2000,
            ),
            "t_itof": data_types.StructuredDataQueue[Timestamp](
                    dtypes={**IndexedData.c_dtypes, **Timestamp.c_dtypes},
                    names={"index": "itof_corr", "time": "t_itof"},
                chunk_size=2000,
            ),
            "t_pulse": data_types.StructuredDataQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "t_pulse"},
                    chunk_size=10000
            ),
            "t_pixel": data_types.StructuredDataQueue[IndexedData[PixelData]](
                    dtypes=IndexedData.c_dtypes + PixelData.c_dtypes,
                    names={"index": "pixel_corr", "time": "t", "x": "x", "y": "y", "tot": "tot"},
                chunk_size=10000,
            ),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": uconn_processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                self.queues["pulses"],
                (self.queues["etof"], self.queues["itof"], self.queues["pixel"]),
                self.queues["t_pulse"],
                (self.queues["t_etof"], self.queues["t_itof"], self.queues["t_pixel"]),
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "pixel": self.queues["t_pixel"],
                    "etof": self.queues["t_etof"],
                    "itof": self.queues["t_itof"],
                    "pulses": self.queues["t_pulse"],
                },
            ).make_process(),
        }


class CV4ConverterPipeline(PostProcessingPipeline):
    """
    Pipeline for converting raw VMI data to CV4 files (H5 file with specific internal format).
    Specific to our VMI setup.
    Converts raw data to clustered pixel data, etof, itof, and pulse data.
    Data are correlated with laser pulses.
    Output Format:
    - clusters: cluster_corr, t, x, y
    - etof: etof_corr, t_etof
    - itof: itof_corr, t_itof
    - pulses: t_pulse
    """

    def __init__(
        self,
        input_path,
        output_path,
        cluster_processes=1,
        converter_processes=1,
        cluster_class=None,
    ):
        super().__init__(input_path, output_path)

        self.queues = {
            "chunk": data_types.StructuredDataQueue(chunk_size=10000),
            "pixel": data_types.StructuredDataQueue(chunk_size=10000),
            "etof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "etof"},
                    force_monotone=True,
                    chunk_size=10000
            ),
            "itof": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "itof"},
                    force_monotone=True,
                    chunk_size=10000
            ),
            "pulses": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "pulses"},
                    force_monotone=True,
                    chunk_size=10000
            ),
            "clusters": data_types.MonotonicQueue[data_types.ClusterData](
                    dtypes=data_types.ClusterData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y"},
                force_monotone=True,
                chunk_size=10000,
            ),
            "t_etof": data_types.StructuredDataQueue[IndexedData[Timestamp]](
                    dtypes=IndexedData.c_dtypes | Timestamp.c_dtypes,
                    names={"index": "etof_corr", "time": "t_etof"},
                chunk_size=10000,
            ),
            "t_itof": data_types.StructuredDataQueue[IndexedData[Timestamp]](
                    dtypes=IndexedData.c_dtypes | Timestamp.c_dtypes,
                    names={"index": "itof_corr", "time": "t_itof"},
                chunk_size=10000,
            ),
            "t_pulse": data_types.StructuredDataQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "t_pulse"},
                    chunk_size=10000
            ),
            "t_cluster": data_types.StructuredDataQueue[IndexedData[data_types.ClusterData]](

                    dtypes=IndexedData.c_dtypes | data_types.ClusterData.c_dtypes,
                    names={"index": "cluster_corr", "time": "t", "x": "x", "y": "y"},
                chunk_size=10000,
            ),
        }

        cluster_class = (
            processes.DBSCANClusterer if cluster_class is None else cluster_class
        )

        if cluster_processes == 1:
            cluster_processes = {
                "Clusterer": cluster_class(
                        self.queues["pixel"], self.queues["clusters"]
                )
            }

        else:
            cluster_queues, cluster_processes = processes.multithread_process(
                cluster_class,
                    {"pixel_queue": self.queues["pixel"]},
                    {"cluster_queue": self.queues["clusters"]},
                cluster_processes,
                in_queue_kw_args={"chunk_size": 2000},
                out_queue_kw_args={"force_monotone": True, "chunk_size": 10000},
                name="Clusterer",
            )
            self.queues.update(cluster_queues)

        if converter_processes == 1:
            converter_processes = {
                "Converter": uconn_processes.VMIConverter(
                        chunk_queue=self.queues["chunk"],
                        pixel_queue=self.queues["pixel"],
                        laser_queue=self.queues["pulses"],
                        etof_queue=self.queues["etof"],
                        itof_queue=self.queues["itof"],
                )
            }
        else:
            converter_queues, converter_processes = processes.multithread_process(
                uconn_processes.VMIConverter,
                    {"chunk_queue": self.queues["chunk"]},
                {
                    "pixel_queue": self.queues["pixel"],
                    "laser_queue": self.queues["pulses"],
                    "etof_queue": self.queues["etof"],
                    "itof_queue": self.queues["itof"],
                },
                converter_processes,
                in_queue_kw_args={"chunk_size": 2000},
                out_queue_kw_args={
                    "pixel_queue": {"chunk_size": 2000},
                    "laser_queue": {"chunk_size": 10000, "force_monotone": True},
                    "etof_queue": {"chunk_size": 2000, "force_monotone": True},
                    "itof_queue": {"chunk_size": 2000, "force_monotone": True},
                },
                name="Converter",
            )
            self.queues.update(converter_queues)

        self.processes = {
            "Reader": processes.TPXFileReader(
                    input_path, self.queues["chunk"]
            ).make_process(),
            **{n: k.make_process() for n, k in converter_processes.items()},
            **{n: k.make_process() for n, k in cluster_processes.items()},
            "Correlator": processes.TriggerAnalyzer(
                    input_trigger_queue=self.queues["pulses"],
                queues_to_index=(
                    self.queues["etof"],
                    self.queues["itof"],
                    self.queues["clusters"],
                ),
                output_trigger_queue=self.queues["t_pulse"],
                indexed_queues=(
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "t_etof": self.queues["t_etof"],
                    "t_itof": self.queues["t_itof"],
                    "t_pulse": self.queues["t_pulse"],
                    "t_cluster": self.queues["t_cluster"],
                },
            ).make_process(),
        }


class ClusterSavePipeline(PostProcessingPipeline):
    """
    Pipeline for clustering and saving raw VMI data to h5 files.
    Specific to our VMI setup.
    Converts raw data to clustered pixel data, etof, itof, and pulse data.
    Data are not correlated with laser pulses.
    Output Format:
    - clusters: toa, x, y
    - etof: etof
    - itof: itof
    - pulses: pulses
    """

    def __init__(self, input_path, output_path, monotone=False):
        super().__init__(input_path, output_path)
        self.queues = {
            "chunk": data_types.StructuredDataQueue(chunk_size=2000),
            "pixel": data_types.StructuredDataQueue[PixelData](
                    dtypes=PixelData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y", "tot": "tot"},
                    chunk_size=2000
            ),
            "etof": data_types.StructuredDataQueue[Timestamp](

                    dtypes=Timestamp.c_dtypes,
                    names={"time": "etof"},
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "itof": data_types.StructuredDataQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "itof"},
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "pulses": data_types.StructuredDataQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "pulses"},
                force_monotone=monotone,
                chunk_size=10000,
            ),
            "clusters": data_types.StructuredDataQueue[data_types.ClusterData](
                    dtypes=data_types.ClusterData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y"},
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "clustered": data_types.StructuredDataQueue[IndexedData[PixelData]](
                    dtypes=IndexedData.c_dtypes + PixelData.c_dtypes,
                    names={"index": "cluster_pix", "time": "toa_pix", "x": "x_pix", "y": "y_pix", "tot": "tot_pix"},
                force_monotone=monotone,
                chunk_size=2000,
            ),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": uconn_processes.VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Clusterer": processes.DBSCANClusterer(
                pixel_queue=self.queues["pixel"],
                cluster_queue=self.queues["clusters"],
                output_pixel_queue=self.queues["clustered"],
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "clusters": self.queues["clusters"],
                    "etof": self.queues["etof"],
                    "itof": self.queues["itof"],
                    "pulses": self.queues["pulses"],
                    "clustered": self.queues["clustered"],
                },
            ).make_process(),
        }


class RunMonitorPipeline(BasePipeline):
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
    ):
        super().__init__()
        self.queues = {
            "chunk": data_types.Queue[Chunk](maxsize=1000),
            "pixel": data_types.StructuredDataQueue[PixelData](),
            "etof": data_types.MonotonicQueue[ToF](maxsize=50000),
            "itof": data_types.MonotonicQueue[ToF](maxsize=50000),
            "pulses": data_types.MonotonicQueue[Trigger](maxsize=50000),
            "clusters": data_types.MonotonicQueue[ClusterData](maxsize=50000),
            "t_etof": data_types.StructuredDataQueue[IndexedData[ToF]](),
            "t_itof": data_types.StructuredDataQueue[IndexedData[ToF]](),
            "t_pulse": data_types.StructuredDataQueue[Trigger](),
            "t_cluster": data_types.StructuredDataQueue[IndexedData[ClusterData]](),
            "grouped": data_types.Queue(),
        }

        self.processes = {
            "ChunkStream": processes.TPXFileReader(
                saving_path, self.queues["chunk"]
            ).make_process(),
            "Converter": uconn_processes.VMIConverter(
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
            "Display": uconn_processes.Display(
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
