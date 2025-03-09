import os
import time
import typing

import numpy as np

from ..data_types import ExtendedQueue, Chunk
from .base_process import AnalysisStep
import socket
from numba import njit
import numba


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
    chunk_queue: ExtendedQueue[Chunk]
    file: typing.IO | None

    def __init__(self, path, chunk_queue, **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXFileReader"
        self.path = path
        self.chunk_queue = chunk_queue
        self.output_queues = (chunk_queue,)
        self.file = None
        self.folder: bool = os.path.isdir(path)
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tpx3')] if self.folder else [path]
        self.curr_file_idx = 0
        self.holding.value = True

    def initialize(self):
        self.file = open(self.files[self.curr_file_idx], 'rb')
        super().initialize()

    def action(self):
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
            self.file = open(self.files[self.curr_file_idx], 'rb')
            return
        _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
        num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
        # packets=[int.from_bytes(self.file.read(8), 'little')-2**62 for _ in range(num_bytes//8)]
        # packets = np.asarray(packets)
        packet_bytes = self.file.read(num_bytes)
        packets = np.frombuffer(packet_bytes, dtype=np.int64)-2**62
        self.chunk_queue.put(packets)

    def shutdown(self, **kwargs):
        self.file.close() if self.file else None
        super().shutdown(**kwargs)


class DummyStream(TPXFileReader):
    """
    Test Item Please Ignore
    """
    def __init__(self, path, chunk_queue, delay, **kwargs):
        super().__init__(path, chunk_queue, **kwargs)
        self.delay = delay
        self.name = "DummyStream"

    def action(self):
        try:
            packet = self.file.read(8)
            if len(packet) < 8:
                self.curr_file_idx += 1
                if self.curr_file_idx >= len(self.files):
                    self.curr_file_idx = 0
                self.file.close()
                self.file = open(self.files[self.curr_file_idx], 'rb')
                return
            _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
            num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
            packets = [int.from_bytes(self.file.read(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
            self.chunk_queue.put(packets)
            time.sleep(self.delay)
        except Exception as e:
            print(e)
            self.shutdown()
            return


class FolderStream(TPXFileReader):
    """
    Reads the most recent file in a folder and puts the data into a queue. The reader will repeatedly
    read the most recent file in the folder until it is older than max_age.

    Args:
        path (str): Path to the folder containing the files.
        chunk_queue (ExtendedQueue[Chunk]): Queue to put the data into.
        max_age (int): Maximum age of the file in seconds. If the most recent file is older than this, the reader will shut down.
    """
    def __init__(self, path, chunk_queue, max_age=0, **kwargs):
        super().__init__(path, chunk_queue, **kwargs)
        self.max_age = max_age
        self.name = "FolderStream"

    def action(self):
        super().action()
        if self.curr_file_idx == 0:
            return
        most_recent_file = sorted(os.listdir(self.path), key=lambda x: os.path.getmtime(os.path.join(self.path, x)))[-1]
        if self.max_age and time.time() - os.path.getmtime(os.path.join(self.path, most_recent_file)) > self.max_age:
            self.shutdown()
            return
        self.file.close() if self.file else None
        self.file = open(os.path.join(self.path, most_recent_file), 'rb')
        self.curr_file_idx = 0


class TPXListener(AnalysisStep):
    """
    Listens for incoming TPX3 data over TCP/IP and puts the data into a queue.

    Args:
        local_ip (tuple[str, int]): Tuple containing the local IP address and port to listen on.
        chunk_queue (ExtendedQueue[Chunk]): Queue to put the data into.

    """
    def __init__(self, local_ip: tuple[str,int], chunk_queue, **kwargs):
        super().__init__(**kwargs)
        self.name = "TPXListener"
        self.chunk_queue = chunk_queue
        self.output_queues = (chunk_queue,)
        self.holding.value = True
        self.local_ip = local_ip
        self.sock = None
        self.client = None
        self.client_address = None

    def initialize(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.local_ip)
        self.sock.listen()
        # self.logger.info(f"Listening on {self.local_ip}")
        print(f"Listening on {self.local_ip}")
        super().initialize()

    def action(self):
        if not self.client:
            self.client, self.client_address = self.sock.accept()
            # self.logger.info(f"Connected to {self.client_address}")
            print(f"Connected to {self.client_address}")
        try:
            packet = self.client.recv(8)
            _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
            num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
            packets = [int.from_bytes(self.client.recv(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
            self.chunk_queue.put(np.asarray(packets, dtype=np.int64))

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
            print(f"Error in {self.name}: {e}")
            self.shutdown()
            return

    def shutdown(self, **kwargs):
        self.client.close() if self.client else None
        self.sock.close() if self.sock else None
        super().shutdown(**kwargs)

