import threading

import requests

from vmi_analysis import serval
from vmi_analysis.processing import data_types, processes
from vmi_analysis.processing.pipelines import BasePipeline


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


class SynchronousSBPipeline(BasePipeline):
    def __init__(
        self,
        output_path,
        local_ip=("localhost", 1234),
        serval_ip=serval.DEFAULT_IP,
        input_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_ip = local_ip
        self.serval_ip = serval_ip
        queues = {
            "chunk": data_types.StructuredDataQueue(),
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
            "Listener": processes.TPXListener(
                self.local_ip, self.queues["chunk"]
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
