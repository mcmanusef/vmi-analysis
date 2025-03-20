import abc
import threading

import matplotlib.pyplot as plt
import requests

from .base_pipeline import AnalysisPipeline
from .. import data_types, processes
from ... import serval


class SynchronousPipeline(AnalysisPipeline, abc.ABC):
    def __init__(self, output_path: str, acquisition_time: float, serval_ip=serval.DEFAULT_IP, local_ip=("localhost", 1234), **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.display_fig: plt.Figure | None = None
        self.acquisition_time = acquisition_time
        self.local_ip = local_ip
        self.serval_ip = serval_ip


    @abc.abstractmethod
    def update_display(self):
        pass

    def start(self):
        starting_thread = threading.Thread(target=super().start())
        starting_thread.start()

        serval_destination = {
            "Raw": [{
                "Base": f"tcp://{self.local_ip[0]}:{self.local_ip[1]}",
                "FilePattern": "",
            }],
        }

        serval.set_acquisition_parameters(
                serval_destination,
                frame_time=1
        )
        resp = requests.get(self.serval_ip + "/server/destination")
        print(resp.text)
        serval.start_acquisition(block=False)
        starting_thread.join()


class H5AcquisitionPipeline(SynchronousPipeline):
    def __init__(
            self,
            output_path: str,
            acquisition_time: float,
            serval_ip=serval.DEFAULT_IP,
            local_ip=("localhost", 1234),
    ):
        super().__init__(output_path, acquisition_time, serval_ip, local_ip)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(
                    dtypes=("f", "i", "i", "i"),
                    names=("toa", "x", "y", "tot"),
                    chunk_size=10000,
            ),
            "tdc": data_types.ExtendedQueue(
                    dtypes=("f", "i"),
                    names=("tdc_time", "tdc_type"),
                    chunk_size=10000,
            ),
        }

        self.processes = {
            "reader": processes.TPXListener(
                    self.local_ip,
                    self.queues["chunk"]
            ).make_process(),
            "converter": processes.TPXConverter(
                    self.queues["chunk"],
                    self.queues["pixel"],
                    self.queues["tdc"]
            ).make_process(),
            "save": processes.SaveToH5(
                    output_path,
                    {"pixel": self.queues["pixel"], "tdc": self.queues["tdc"]},
            ).make_process(),
        }
        self.display_fig = plt.figure()

    def update_display(self):
        pass
