from .. import data_types, processes
from .base_pipeline import AnalysisPipeline


class TPXFileConverter(AnalysisPipeline):
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
    def __init__(self, input_path: str, output_path: str, buffer_size: int = 0, single_process=False, **kwargs):
        super().__init__(**kwargs)
        if not single_process:
            self.queues = {
                'chunk': data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                'pixel': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i', 'i', 'i'),
                                                  names=('toa', 'x', 'y', 'tot'), chunk_size=10000),
                'tdc': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i'),
                                                names=('tdc_time', 'tdc_type'), chunk_size=10000),
            }

            self.processes = {
                'reader': processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
                'converter': processes.TPXConverter(self.queues['chunk'], self.queues['pixel'],
                                                    self.queues['tdc']).make_process(),
                'save': processes.SaveToH5(output_path,
                                           {"pixel": self.queues['pixel'], "tdc": self.queues['tdc']}).make_process(),
            }

        if single_process:
            self.queues: dict[str, data_types.ExtendedQueue] = {
                'chunk': data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                'pixel': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i', 'i', 'i'),
                                                  names=('toa', 'x', 'y', 'tot')),
                'tdc': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i'),
                                                names=('tdc_time', 'tdc_type')),
            }

            self.processes = {
                "Combined": processes.CombinedStep(steps=(
                    processes.TPXFileReader(input_path, chunk_queue=self.queues['chunk']),
                    processes.TPXConverter(chunk_queue=self.queues['chunk'],
                                           pixel_queue=self.queues['pixel'],
                                           tdc_queue=self.queues['tdc']),
                    processes.SaveToH5(output_path,
                                       {"pixel": self.queues['pixel'], "tdc": self.queues['tdc']}),
                ), intermediate_queues=(self.queues["pixel"], self.queues['tdc']),
                        output_queues=(self.queues["chunk"],),
                ).make_process(),
            }


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
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'i', 'i', 'i'), names=('toa', 'x', 'y', 'tot'), unwrap=True,
                                              force_monotone=True, max_back=1e9, chunk_size=10000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('etof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('itof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('pulses',), force_monotone=True, max_back=1e9, chunk_size=10000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'],
                                                self.queues['itof']).make_process(),
            "Saver": processes.SaveToH5(output_path, {
                "pixel": self.queues['pixel'],
                "etof": self.queues['etof'],
                "itof": self.queues['itof'],
                "pulses": self.queues['pulses'],
            }).make_process(),
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
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'i', 'i', 'i'), names=('toa', 'x', 'y', 'tot'), unwrap=True,
                                              force_monotone=True, max_back=1e9, chunk_size=10000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('etof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('itof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('pulses',), force_monotone=True, max_back=1e9, chunk_size=10000),

            "t_etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('etof_corr', ('t_etof',)), chunk_size=2000),
            "t_itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('itof_corr', ('t_itof',)), chunk_size=2000),
            "t_pulse": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('t_pulse',), chunk_size=10000),
            "t_pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f', 'i', 'i', 'i')), names=('pixel_corr', ('t', 'x', 'y', 'tot')),
                                                chunk_size=10000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'],
                                                self.queues['itof']).make_process(),
            "Correlator": processes.TriggerAnalyzer(self.queues['pulses'],
                                                    (self.queues['etof'], self.queues['itof'], self.queues['pixel']),
                                                    self.queues['t_pulse'],
                                                    (self.queues['t_etof'], self.queues['t_itof'], self.queues['t_pixel'])
                                                    ).make_process(),
            "Saver": processes.SaveToH5(output_path, {
                "pixel": self.queues['t_pixel'],
                "etof": self.queues['t_etof'],
                "itof": self.queues['t_itof'],
                "pulses": self.queues['t_pulse'],
            }).make_process(),
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
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("etof",), force_monotone=monotone, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("itof",), force_monotone=monotone, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("pulses",), force_monotone=monotone, chunk_size=10000),
            "clusters": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'f', 'f'), names=("toa", "x", "y"), force_monotone=monotone,
                                                 chunk_size=2000),
            "clustered": data_types.ExtendedQueue(buffer_size=0, dtypes=(('f', 'i', 'i', 'i'), 'i'),
                                                  names=(('toa_pix', 'x_pix', 'y_pix', 'tot_pix'), 'cluster_pix'),
                                                  force_monotone=monotone, chunk_size=2000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),

            "Converter": processes.VMIConverter(self.queues['chunk'],
                                                self.queues['pixel'],
                                                self.queues['pulses'],
                                                self.queues['etof'],
                                                self.queues['itof']).make_process(),

            "Clusterer": processes.DBSCANClusterer(pixel_queue=self.queues['pixel'],
                                                   cluster_queue=self.queues['clusters'],
                                                   output_pixel_queue=self.queues['clustered']).make_process(),

            "Saver": processes.SaveToH5(output_path, {
                "clusters": self.queues['clusters'],
                "etof": self.queues['etof'],
                "itof": self.queues['itof'],
                "pulses": self.queues['pulses'],
                "clustered": self.queues['clustered'],
            }).make_process(),
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
    def __init__(self, input_path, output_path, save_pixels=False, cluster_processes=1, **kwargs):
        super().__init__(**kwargs)

        if cluster_processes > 1:
            self.queues = {
                "Chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
                "Pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
                "Etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("etof",), force_monotone=True, chunk_size=2000),
                "Itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("itof",), force_monotone=True, chunk_size=2000),
                "Pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("pulses",), force_monotone=True, chunk_size=10000),
                "Clusters": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'f', 'f'), names=("toa", "x", "y"), force_monotone=True,
                                                     chunk_size=2000),

                "t_etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('etof_corr', ("t_etof",)), chunk_size=2000),
                "t_itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('tof_corr', ("t_tof",)), chunk_size=2000),
                "t_pulse": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('t_pulse',), chunk_size=10000),
                "t_cluster": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f', 'f', 'f')), names=('cluster_corr', ("t", "x", "y")),
                                                      chunk_size=2000),
            }

            queues, proc, weaver = processes.create_process_instances(
                    processes.DBSCANClusterer, cluster_processes, self.queues["Clusters"],
                    process_args={"pixel_queue": self.queues['Pixel'], "cluster_queue": None},
                    queue_args={
                        "buffer_size": 0,
                        "dtypes": ('f', 'f', 'f'),
                        "names": ("toa", "x", "y"),
                        "force_monotone": True,
                        "chunk_size": 2000,
                        "maxsize": 10,
                    },
                    queue_name="clust", process_name="clusterer")
            self.queues.update(queues)
            self.processes = {"Reader": processes.TPXFileReader(input_path, self.queues['Chunk']).make_process(),

                              "Converter": processes.VMIConverter(
                                      chunk_queue=self.queues['Chunk'],
                                      pixel_queue=self.queues['Pixel'],
                                      laser_queue=self.queues['Pulses'],
                                      etof_queue=self.queues['Etof'],
                                      itof_queue=self.queues['Itof']
                              ).make_process(),

                              **{n: k.make_process() for n, k in proc.items()},

                              "Weaver": weaver.make_process(),

                              "Correlator": processes.TriggerAnalyzer(
                                      input_trigger_queue=self.queues['Pulses'],
                                      queues_to_index=(self.queues['Etof'], self.queues['Itof'], self.queues['Clusters']),
                                      output_trigger_queue=self.queues['t_pulse'],
                                      indexed_queues=(self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])
                              ).make_process(),

                              "Saver": processes.SaveToH5(output_path, {
                                  "t_etof": self.queues['t_etof'],
                                  "t_itof": self.queues['t_itof'],
                                  "t_pulse": self.queues['t_pulse'],
                                  "t_cluster": self.queues['t_cluster']
                              }).make_process(),
                              }


        else:
            self.queues = {
                "Chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
                "Pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
                "Etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("etof",), force_monotone=True, chunk_size=2000),
                "Itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("itof",), force_monotone=True, chunk_size=2000),
                "Pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("pulses",), force_monotone=True, chunk_size=10000),
                "Clusters": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'f', 'f'), names=("toa", "x", "y"), force_monotone=True,
                                                     chunk_size=2000),

                "t_etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('etof_corr', ("t_etof",)), chunk_size=2000),
                "t_itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('tof_corr', ("t_tof",)), chunk_size=2000),
                "t_pulse": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('t_pulse',), chunk_size=10000),
                "t_cluster": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f', 'f', 'f')), names=('cluster_corr', ("t", "x", "y")),
                                                      chunk_size=2000),
            }

            self.processes = {
                "Reader": processes.TPXFileReader(input_path, self.queues['Chunk']).make_process(),

                "Converter": processes.VMIConverter(
                        chunk_queue=self.queues['Chunk'],
                        pixel_queue=self.queues['Pixel'],
                        laser_queue=self.queues['Pulses'],
                        etof_queue=self.queues['Etof'],
                        itof_queue=self.queues['Itof']
                ).make_process(),

                "Clusterer": processes.DBSCANClusterer(
                        pixel_queue=self.queues['Pixel'],
                        cluster_queue=self.queues['Clusters']
                ).make_process(),

                "Correlator": processes.TriggerAnalyzer(
                        input_trigger_queue=self.queues['Pulses'],
                        queues_to_index=(self.queues['Etof'], self.queues['Itof'], self.queues['Clusters']),
                        output_trigger_queue=self.queues['t_pulse'],
                        indexed_queues=(self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])
                ).make_process(),

                "Saver": processes.SaveToH5(output_path, {
                    "t_etof": self.queues['t_etof'],
                    "t_itof": self.queues['t_itof'],
                    "t_pulse": self.queues['t_pulse'],
                    "t_cluster": self.queues['t_cluster']
                }).make_process(),

            }

            if save_pixels:
                self.queues["pixels"] = data_types.ExtendedQueue(buffer_size=0, dtypes=(('f', 'i', 'i', 'i'), 'i'),
                                                                 names=(('toa', 'x', 'y', 'tot'), 'cluster_index'), force_monotone=True, max_back=1e9,
                                                                 chunk_size=10000)
                self.queues["t_pixel"] = data_types.ExtendedQueue(buffer_size=0, dtypes=('i', (('f', 'i', 'i', 'i'), 'i')),
                                                                  names=('pixel_corr', (('t', 'x', 'y', 'tot'), 'cluster_index')), chunk_size=10000)

                self.processes["Saver"].astep.in_queues['pixels'] = self.queues["t_pixel"]
                self.processes["Saver"].astep.input_queues += (self.queues["t_pixel"],)
                self.processes["Saver"].astep.flat = {
                    "t_etof": True,
                    "t_itof": True,
                    "t_pulse": True,
                    "t_cluster": True,
                    "pixels": False,
                }
                self.processes["Correlator"].astep.output_queues += (self.queues["t_pixel"],)
                self.processes["Correlator"].astep.indexed_queues += (self.queues["t_pixel"],)
                self.processes["Correlator"].astep.input_queues += (self.queues["pixels"],)
                self.processes["Correlator"].astep.queues_to_index += (self.queues["pixels"],)
                self.processes["Correlator"].astep.current.append(None)
                self.processes["Correlator"].astep.current_samples.append(None)
                self.processes["Clusterer"].astep.output_pixel_queue = self.queues["pixels"]
