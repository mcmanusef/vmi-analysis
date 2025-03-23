import abc
import threading
import os
import multiprocessing
import asyncio

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
        self.on_finish= threading.Event()

        self.server_thread: threading.Thread | None = None
        self.raw_data_queue: data_types.ExtendedQueue = data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=())


    @abc.abstractmethod
    def update_display(self):
        pass

    def initialize(self):
        path_parts = os.path.split(self.output_path)
        if len(path_parts) > 2:
            for i in range(2, len(path_parts)):
                if not os.path.exists(os.path.join(*path_parts[:i])):
                    os.mkdir(os.path.join(*path_parts[:i]))

        self.server_thread = threading.Thread(target=lambda: asyncio.run(self.start_server()))
        self.server_thread.start()

        super().initialize()

    def start(self):
        super().start()
        print("Starting acquisition")
        serval_destination = {
            "Raw": [{
                "Base": f"tcp://{self.local_ip[0]}:{self.local_ip[1]}",
                "FilePattern": "",
            }],
        }

        serval.set_acquisition_parameters(
                serval_destination,
                duration=self.acquisition_time,
                frame_time=1,
        )
        resp = requests.get(self.serval_ip + "/server/destination")
        print(resp.text)
        serval.start_acquisition(block=False)

    def stop(self):
        serval.stop_acquisition()
        self.on_finish.set()
        try:
            self.server_thread.join(timeout=5)
        except TimeoutError as e:
            print("Server thread did not join")
            raise e
        super().stop()

    async def handle_connection(self, reader, writer):
        print("Connected")
        try:
            while True:
                packet = await reader.read(8)
                if not packet:
                    break
                self.raw_data_queue.put(packet)
        except ConnectionResetError:
            pass
        finally:
            self.raw_data_queue.close()
            self.on_finish.set()
            try:
                writer.close()
                await writer.wait_closed()
            except ConnectionResetError:
                pass

    async def start_server(self):
        print(f"Listening on {self.local_ip}")
        server = await asyncio.start_server(
            self.handle_connection, self.local_ip[0], self.local_ip[1]
        )
        print(f"Serving on {server.sockets[0].getsockname()}")
        task=asyncio.create_task(server.serve_forever())
        while not self.on_finish.is_set():
            await asyncio.sleep(0.1)

        server.close()
        task.cancel()


class SynchronousPipelineTest(SynchronousPipeline):
    def update_display(self):
        pass


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
            "raw": self.raw_data_queue,
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "tdc": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
        }

        self.processes = {
            "reader": processes.TPXListener(
                    self.queues["raw"],
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
