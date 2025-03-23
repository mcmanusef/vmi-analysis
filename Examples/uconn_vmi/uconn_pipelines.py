import uconn_processes
from vmi_analysis.processing import data_types, processes
from vmi_analysis.processing.pipelines import AnalysisPipeline


class RawVMIConverterPipeline(AnalysisPipeline):
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

    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f", "i", "i", "i"),
                names=("toa", "x", "y", "tot"),
                unwrap=True,
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
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


class VMIConverterPipeline(AnalysisPipeline):
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

    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f", "i", "i", "i"),
                names=("toa", "x", "y", "tot"),
                unwrap=True,
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
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
            "t_pulse": data_types.ExtendedQueue(
                buffer_size=0, dtypes=("f",), names=("t_pulse",), chunk_size=10000
            ),
            "t_pixel": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f", "i", "i", "i")),
                names=("pixel_corr", ("t", "x", "y", "tot")),
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


class CV4ConverterPipeline(AnalysisPipeline):
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
        save_pixels=False,
        cluster_processes=1,
        converter_processes=1,
        cluster_class=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.queues = {
            "Chunk": data_types.ExtendedQueue(chunk_size=10000),
            "Pixel": data_types.ExtendedQueue(chunk_size=10000),
            "Etof": data_types.ExtendedQueue(
                dtypes=("f",), names=("etof",), force_monotone=True, chunk_size=10000
            ),
            "Itof": data_types.ExtendedQueue(
                dtypes=("f",), names=("itof",), force_monotone=True, chunk_size=10000
            ),
            "Pulses": data_types.ExtendedQueue(
                dtypes=("f",), names=("pulses",), force_monotone=True, chunk_size=10000
            ),
            "Clusters": data_types.ExtendedQueue(
                dtypes=("f", "f", "f"),
                names=("toa", "x", "y"),
                force_monotone=True,
                chunk_size=10000,
            ),
            "t_etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f",)),
                names=("etof_corr", ("t_etof",)),
                chunk_size=10000,
            ),
            "t_itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f",)),
                names=("tof_corr", ("t_tof",)),
                chunk_size=10000,
            ),
            "t_pulse": data_types.ExtendedQueue(
                buffer_size=0, dtypes=("f",), names=("t_pulse",), chunk_size=10000
            ),
            "t_cluster": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("i", ("f", "f", "f")),
                names=("cluster_corr", ("t", "x", "y")),
                chunk_size=10000,
            ),
        }
        cluster_class = (
            processes.DBSCANClusterer if cluster_class is None else cluster_class
        )

        if cluster_processes == 1:
            cluster_processes = {
                "Clusterer": cluster_class(
                    self.queues["Pixel"], self.queues["Clusters"]
                )
            }
        else:
            cluster_queues, cluster_processes = processes.multithread_process(
                cluster_class,
                {"pixel_queue": self.queues["Pixel"]},
                {"cluster_queue": self.queues["Clusters"]},
                cluster_processes,
                in_queue_kw_args={"chunk_size": 2000},
                out_queue_kw_args={"force_monotone": True, "chunk_size": 10000},
                name="Clusterer",
            )
            self.queues.update(cluster_queues)

        if converter_processes == 1:
            converter_processes = {
                "Converter": uconn_processes.VMIConverter(
                    chunk_queue=self.queues["Chunk"],
                    pixel_queue=self.queues["Pixel"],
                    laser_queue=self.queues["Pulses"],
                    etof_queue=self.queues["Etof"],
                    itof_queue=self.queues["Itof"],
                )
            }
        else:
            converter_queues, converter_processes = processes.multithread_process(
                    uconn_processes.VMIConverter,
                {"chunk_queue": self.queues["Chunk"]},
                {
                    "pixel_queue": self.queues["Pixel"],
                    "laser_queue": self.queues["Pulses"],
                    "etof_queue": self.queues["Etof"],
                    "itof_queue": self.queues["Itof"],
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
                input_path, self.queues["Chunk"]
            ).make_process(),
            **{n: k.make_process() for n, k in converter_processes.items()},
            **{n: k.make_process() for n, k in cluster_processes.items()},
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


class ClusterSavePipeline(AnalysisPipeline):
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

    def __init__(self, input_path, output_path, monotone=False, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "pixel": data_types.ExtendedQueue(
                buffer_size=0, dtypes=(), names=(), chunk_size=2000
            ),
            "etof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("etof",),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "itof": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("itof",),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "pulses": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f",),
                names=("pulses",),
                force_monotone=monotone,
                chunk_size=10000,
            ),
            "clusters": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=("f", "f", "f"),
                names=("toa", "x", "y"),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "clustered": data_types.ExtendedQueue(
                buffer_size=0,
                dtypes=(("f", "i", "i", "i"), "i"),
                names=(("toa_pix", "x_pix", "y_pix", "tot_pix"), "cluster_pix"),
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


class RunMonitorPipeline(AnalysisPipeline):
    def __init__(self, saving_path,
                 cluster_processes=1,
                 toa_range=None,
                 etof_range=None,
                 itof_range=None,
                 calibration=None,
                 center=(128, 128),
                 angle=0,
                 **kwargs):
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
            "ChunkStream": processes.TPXFileReader(saving_path, self.queues['chunk']).make_process(),
            "Converter": uconn_processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'],
                                                      self.queues['itof']).make_process(),
            "Clusterer": processes.CustomClusterer(self.queues['pixel'], self.queues['clusters']).make_process(),
            "Correlator": processes.TriggerAnalyzer(self.queues['pulses'], (self.queues['etof'], self.queues['itof'], self.queues['clusters']),
                                                    self.queues['t_pulse'],
                                                    (self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])).make_process(),
            "Grouper": processes.QueueGrouper((self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster']),
                                              self.queues['grouped']).make_process(),
            "Display": uconn_processes.Display(self.queues['grouped'], 10000000, toa_range=toa_range, etof_range=etof_range,
                                               itof_range=itof_range, calibration=calibration, center=center, angle=angle).make_process(),
            "Bin": processes.QueueVoid((self.queues['t_pulse'],)).make_process(),
        }

        if cluster_processes > 1:
            queues, proc, weaver = processes.create_process_instances(
                    processes.DBSCANClusterer, cluster_processes, self.queues["clusters"],
                    process_args={"pixel_queue": self.queues['pixel'], "cluster_queue": None},
                    queue_args={"force_monotone": True},
                    queue_name="clust", process_name="clusterer")

            self.queues.update(queues)
            del self.processes["Clusterer"]
            self.processes.update({n: k.make_process() for n, k in proc.items()})
            self.processes["Weaver"] = weaver.make_process()
