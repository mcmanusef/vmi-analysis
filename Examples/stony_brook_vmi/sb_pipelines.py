from sb_processes import *
from vmi_analysis.processing import data_types, processes
from vmi_analysis.processing.data_types import IndexedData
from vmi_analysis.processing.pipelines import BasePipeline


class StonyBrookClusterPipeline(BasePipeline):
    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        queues = {
            "chunk": data_types.StructuredDataQueue(chunk_size=10000),
            "pixel": data_types.StructuredDataQueue(chunk_size=10000),
            "tdc": data_types.MonotonicQueue[data_types.TDCData](chunk_size=10000, ),
            "pulses": data_types.MonotonicQueue[Timestamp](
                    dtypes=Timestamp.c_dtypes,
                    names={"time": "pulses"},
                    force_monotone=True,
                    chunk_size=10000,
            ),
            "clusters": data_types.MonotonicQueue[data_types.ClusterData](
                    dtypes=data_types.ClusterData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y"},
                    force_monotone=True,
                    chunk_size=10000,
            ),
            "t_cluster": data_types.StructuredDataQueue[IndexedData[data_types.ClusterData]](

                    dtypes=IndexedData.c_dtypes | data_types.ClusterData.c_dtypes,
                    names={"index": "cluster_corr", "time": "t", "x": "x", "y": "y"},
                    chunk_size=10000,
            ),
        }

        self.queues = queues

        self.processes = {
            "Reader": processes.TPXFileReader(
                input_path, self.queues["chunk"]
            ).make_process(),
            "Converter": SBVMIConverter(
                self.queues["chunk"], queues["pixel"], queues["tdc"]
            ).make_process(),
            "Clusterer": processes.CustomClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            "Correlator": processes.TriggerAnalyzer(
                    self.queues["tdc"],
                    (self.queues["clusters"],),
                    self.queues["pulses"],
                    (self.queues["t_cluster"],),
            ).make_process(),
            "Saver": processes.SaveToH5(
                output_path,
                {
                    "t_cluster": self.queues["t_cluster"],
                    "pulses": self.queues["pulses"],
                },
            ).make_process(),
        }