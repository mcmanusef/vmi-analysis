import os
import time
import typing
from asyncio import QueueEmpty

import numpy as np

from ..data_types import Queue, Chunk
from .base_process import AnalysisStep


class TPXFileReader(AnalysisStep):
    """
    Reads TPX3 files and puts the data into a queue. The path can be a file or a folder. If it is a folder, the reader
    will read all the files in the folder in alphabetical order.

    Args:
        path (str): Path to the file or folder containing the files.
        chunk_queue (ExtendedQueue[Chunk]): Queue to put the data into.
    """

    path: str
    input_queues = ()
    chunk_queue: Queue[Chunk]
    file: typing.IO | None

    def __init__(self, path, chunk_queue, **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXFileReader"
        self.path = path
        self.chunk_queue = chunk_queue
        self.output_queues = (chunk_queue,)
        self.file = None
        self.folder: bool = os.path.isdir(path)
        self.files = (
            [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tpx3")]
            if self.folder
            else [path]
        )
        self.curr_file_idx = 0
        self.holding.value = True

    def initialize(self):
        self.file = open(self.files[self.curr_file_idx], "rb")
        super().initialize()

    def action(self):
        assert self.file
        try:
            packet = self.file.read(8)
        except ValueError:
            time.sleep(1)
            return

        if len(packet) < 8:
            self.curr_file_idx += 1
            if self.curr_file_idx >= len(self.files):
                self.shutdown()
                return
            self.file.close()
            self.file = open(self.files[self.curr_file_idx], "rb")
            return
        _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
        num_bytes = int.from_bytes((bytes(num_bytes)), "little")
        # packets=[int.from_bytes(self.file.read(8), 'little')-2**62 for _ in range(num_bytes//8)]
        # packets = np.asarray(packets)
        packet_bytes = self.file.read(num_bytes)
        packets = np.frombuffer(packet_bytes, dtype=np.int64) - 2**62
        self.chunk_queue.put(packets)

    def shutdown(self, gentle=False):
        self.file.close() if self.file else None
        super().shutdown(gentle=gentle)


class TPXListener(AnalysisStep):
    """
    Listens for incoming TPX3 data over TCP/IP and puts the data into a queue.

    Args:
        local_ip (tuple[str, int]): Tuple containing the local IP address and port to listen on.
        chunk_queue (ExtendedQueue[Chunk]): Queue to put the data into.

    """

    def __init__(self, raw_data_queue, chunk_queue, **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXListener"
        self.chunk_queue = chunk_queue
        self.output_queues = (chunk_queue,)
        self.raw_data_queue = raw_data_queue
        self.input_queues = (raw_data_queue,)

    def action(self):
        try:
            packet = self.raw_data_queue.get()
            _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
            num_bytes = int.from_bytes((bytes(num_bytes)), "little")
            packets = [
                int.from_bytes(self.raw_data_queue.get(), "little") - 2**62
                for _ in range(num_bytes // 8)
            ]
            self.chunk_queue.put(np.asarray(packets, dtype=np.int64))

        except QueueEmpty:
            time.sleep(0.1)
            return

        except Exception as e:
            self.logger.exception(f"Error in {self.name}: {e}")
            print(f"Error in {self.name}: {e}")
            self.shutdown()
            return
