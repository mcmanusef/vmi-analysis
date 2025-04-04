from .base_pipeline import BasePipeline
from .. import data_types, processes


class PostProcessingPipeline(BasePipeline):
    def __init__(self, input_path: str, output_path: str, **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.output_path = output_path


class TPXFileConverter(PostProcessingPipeline):
    """
    Pipeline for converting TPX3 files to h5 files.
    Not specific to our VMI setup.
    Directly reads the TPX3 files and converts them to h5 files, saving the pixel data and the TDC data with minimal processing.
    Output Format:
    - pixel: toa, x, y, tot
    - tdc: tdc_time, tdc_type
    tdc_types:
    1: tdc1 rising, 2: tdc1 falling, 3: tdc2 rising, 4: tdc2 falling
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        single_process=False,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        if not single_process:
            queues = {
                "chunk": data_types.Queue(),
                "pixel": data_types.StructuredDataQueue[data_types.PixelData](
                    dtypes=("f", "i", "i", "i"),
                    names=("toa", "x", "y", "tot"),
                    chunk_size=10000,
                ),
                "tdc": data_types.StructuredDataQueue[data_types.TDCData](
                    dtypes=("f", "i"),
                    names=("tdc_time", "tdc_type"),
                    chunk_size=10000,
                ),
            }

            self.processes = {
                "reader": processes.TPXFileReader(
                    input_path, self.queues["chunk"]
                ).make_process(),
                "converter": processes.TPXConverter(
                    self.queues["chunk"], queues["pixel"], queues["tdc"]
                ).make_process(),
                "save": processes.SaveToH5(
                    output_path,
                    {"pixel": self.queues["pixel"], "tdc": self.queues["tdc"]},
                ).make_process(),
            }
            self.queues = queues

        if single_process:
            queues = {
                "chunk": data_types.Queue(),
                "pixel": data_types.StructuredDataQueue[data_types.PixelData](
                    dtypes=("f", "i", "i", "i"),
                    names=("toa", "x", "y", "tot"),
                ),
                "tdc": data_types.StructuredDataQueue[data_types.TDCData](
                    dtypes=("f", "i"),
                    names=("tdc_time", "tdc_type"),
                ),
            }

            self.processes = {
                "Combined": processes.CombinedStep(
                    steps=(
                        processes.TPXFileReader(
                            input_path, chunk_queue=self.queues["chunk"]
                        ),
                        processes.TPXConverter(
                            chunk_queue=self.queues["chunk"],
                            pixel_queue=queues["pixel"],
                            tdc_queue=queues["tdc"],
                        ),
                        processes.SaveToH5(
                            output_path,
                            {"pixel": self.queues["pixel"], "tdc": self.queues["tdc"]},
                        ),
                    ),
                    intermediate_queues=(self.queues["pixel"], self.queues["tdc"]),
                    output_queues=(self.queues["chunk"],),
                ).make_process(),
            }
            self.queues = queues


class RawVMIConverterPipeline(BasePipeline):
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
            "chunk": data_types.Queue(),
            "pixel": data_types.StructuredDataQueue(
                dtypes=("f", "i", "i", "i"),
                names=("toa", "x", "y", "tot"),
                unwrap=True,
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.StructuredDataQueue(
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
            "Converter": processes.VMIConverter(
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


class VMIConverterPipeline(BasePipeline):
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
            "chunk": data_types.Queue(),
            "pixel": data_types.StructuredDataQueue(
                dtypes=("f", "i", "i", "i"),
                names=("toa", "x", "y", "tot"),
                unwrap=True,
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "etof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("etof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "itof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("itof",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=2000,
            ),
            "pulses": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("pulses",),
                force_monotone=True,
                max_back=1e9,
                chunk_size=10000,
            ),
            "t_etof": data_types.StructuredDataQueue(
                dtypes=("i", ("f",)),
                names=("etof_corr", ("t_etof",)),
                chunk_size=2000,
            ),
            "t_itof": data_types.StructuredDataQueue(
                dtypes=("i", ("f",)),
                names=("itof_corr", ("t_itof",)),
                chunk_size=2000,
            ),
            "t_pulse": data_types.StructuredDataQueue(
                dtypes=("f",), names=("t_pulse",), chunk_size=10000
            ),
            "t_pixel": data_types.StructuredDataQueue(
                dtypes=("i", ("f", "i", "i", "i")),
                names=("pixel_corr", ("t", "x", "y", "tot")),
                chunk_size=10000,
            ),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": processes.VMIConverter(
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


class ClusterSavePipeline(BasePipeline):
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
            "chunk": data_types.Queue(),
            "pixel": data_types.StructuredDataQueue(
                dtypes=(), names=(), chunk_size=2000
            ),
            "etof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("etof",),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "itof": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("itof",),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "pulses": data_types.StructuredDataQueue(
                dtypes=("f",),
                names=("pulses",),
                force_monotone=monotone,
                chunk_size=10000,
            ),
            "clusters": data_types.StructuredDataQueue(
                dtypes=("f", "f", "f"),
                names=("toa", "x", "y"),
                force_monotone=monotone,
                chunk_size=2000,
            ),
            "clustered": data_types.StructuredDataQueue(
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
            "Converter": processes.VMIConverter(
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


class CV4ConverterPipeline(BasePipeline):
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
            "Chunk": data_types.Queue(chunk_size=10000),
            "Pixel": data_types.StructuredDataQueue(chunk_size=10000),
            "Etof": data_types.StructuredDataQueue(
                dtypes=("f",), names=("etof",), force_monotone=True, chunk_size=10000
            ),
            "Itof": data_types.StructuredDataQueue(
                dtypes=("f",), names=("itof",), force_monotone=True, chunk_size=10000
            ),
            "Pulses": data_types.StructuredDataQueue(
                dtypes=("f",), names=("pulses",), force_monotone=True, chunk_size=10000
            ),
            "Clusters": data_types.StructuredDataQueue(
                dtypes=("f", "f", "f"),
                names=("toa", "x", "y"),
                force_monotone=True,
                chunk_size=10000,
            ),
            "t_etof": data_types.StructuredDataQueue(
                dtypes=("i", ("f",)),
                names=("etof_corr", ("t_etof",)),
                chunk_size=10000,
            ),
            "t_itof": data_types.StructuredDataQueue(
                dtypes=("i", ("f",)),
                names=("tof_corr", ("t_tof",)),
                chunk_size=10000,
            ),
            "t_pulse": data_types.StructuredDataQueue(
                dtypes=("f",), names=("t_pulse",), chunk_size=10000
            ),
            "t_cluster": data_types.StructuredDataQueue(
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
                "Converter": processes.VMIConverter(
                    chunk_queue=self.queues["Chunk"],
                    pixel_queue=self.queues["Pixel"],
                    laser_queue=self.queues["Pulses"],
                    etof_queue=self.queues["Etof"],
                    itof_queue=self.queues["Itof"],
                )
            }
        else:
            converter_queues, converter_processes = processes.multithread_process(
                processes.VMIConverter,
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


class StonyBrookClusterPipeline(BasePipeline):
    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        queues = {
            "chunk": data_types.Queue(),
            "pixel": data_types.StructuredDataQueue[data_types.PixelData](),
            "tdc": data_types.StructuredDataQueue[data_types.TDCData](),
            "pulse": data_types.StructuredDataQueue(),
            "tof": data_types.StructuredDataQueue(),
            "clusters": data_types.StructuredDataQueue(),
            "t_tof": data_types.StructuredDataQueue(
                dtypes=("i", ("f",)), names=("tof_corr", ("t_tof",))
            ),
            "t_cluster": data_types.StructuredDataQueue(
                dtypes=("i", ("f", "f", "f")), names=("cluster_corr", ("t", "x", "y"))
            ),
            "t_pulse": data_types.StructuredDataQueue(
                dtypes=("f",), names=("t_pulse",)
            ),
        }
        self.queues = queues

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": processes.TPXConverter(
                self.queues["chunk"], queues["pixel"], queues["tdc"]
            ).make_process(),
            "Filter": processes.TDCFilter(
                self.queues["tdc"], self.queues["pulse"], self.queues["tof"]
            ).make_process(),
            "Clusterer": processes.DBSCANClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                self.queues["pulse"],
                (self.queues["tof"], self.queues["clusters"]),
                self.queues["t_pulse"],
                (self.queues["t_tof"], self.queues["t_cluster"]),
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "t_tof": self.queues["t_tof"],
                    "t_cluster": self.queues["t_cluster"],
                    "t_pulse": self.queues["t_pulse"],
                },
            ).make_process(),
        }
