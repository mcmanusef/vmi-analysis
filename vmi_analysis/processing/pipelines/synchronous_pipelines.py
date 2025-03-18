from .. import data_types, processes
from ... import serval
from .base_pipeline import AnalysisPipeline
import threading
import requests

class SynchronousSBPipeline(AnalysisPipeline):
    def __init__(self,
                 output_path,
                 local_ip=("localhost", 1234),
                 serval_ip=serval.DEFAULT_IP,
                 input_path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.local_ip = local_ip
        self.serval_ip = serval_ip
        self.queues = {
            "chunk": data_types.ExtendedQueue(),
            "pixel": data_types.ExtendedQueue(),
            "tdc": data_types.ExtendedQueue(),
            "pulse": data_types.ExtendedQueue(),
            "tof": data_types.ExtendedQueue(),
            "clusters": data_types.ExtendedQueue(),
            "t_tof": data_types.ExtendedQueue(
                    dtypes=("i", ("f",)), names=("tof_corr", ("t_tof",))
            ),
            "t_cluster": data_types.ExtendedQueue(
                    dtypes=("i", ("f", "f", "f")), names=("cluster_corr", ("t", "x", "y"))
            ),
            "t_pulse": data_types.ExtendedQueue(dtypes=("f",), names=("t_pulse",)),
        }

        self.processes = {
            "Listener": processes.TPXListener(self.local_ip, self.queues['chunk']).make_process(),

            "Converter": processes.TPXConverter(
                    self.queues["chunk"], self.queues["pixel"], self.queues["tdc"]
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
        starting_thread=threading.Thread(target=super().start())
        starting_thread.start()

        serval_destination = {
            "Raw": [{
                "Base": f"tcp://{self.local_ip[0]}:{self.local_ip[1]}",
                "FilePattern": "",
            }],

            "Preview": {
                "Period": 0.1,
                "SamplingMode": "skipOnFrame",
                "ImageChannels": [{
                    "Base": self.serval_ip,
                    "Format": "png",
                    "Mode": "count",
                }]
            }
        }

        serval.set_acquisition_parameters(
                serval_destination,
                frame_time=1
        )
        resp=requests.get(self.serval_ip+"/server/destination")
        print(resp.text)
        serval.start_acquisition(block=False)
        starting_thread.join()

    def stop(self):
        serval.stop_acquisition()
        super().stop()