import os
import time
import typing
from ..data_types import ExtendedQueue, Chunk
from .base_process import AnalysisStep
import socket


class TPXFileReader(AnalysisStep):
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
        packet = self.file.read(8)
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
        packets = [int.from_bytes(self.file.read(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
        self.chunk_queue.put(packets)

    def shutdown(self, **kwargs):
        self.file.close() if self.file else None
        super().shutdown(**kwargs)


class DummyStream(TPXFileReader):
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
        super().initialize()
        self.logger.info(f"Listening on {self.local_ip}")

    def begin(self):
        self.client, self.client_address = self.sock.accept()
        self.logger.info(f"Connected to {self.client_address}")
        super().begin()

    def action(self):
        try:
            packet = self.client.recv(8)
            if not packet:
                self.shutdown()
                return
            _, _, _, _, chip_number, mode, *num_bytes = tuple(packet)
            num_bytes = int.from_bytes((bytes(num_bytes)), 'little')
            packets = [int.from_bytes(self.client.recv(8), 'little') - 2 ** 62 for _ in range(num_bytes // 8)]
            self.chunk_queue.put(packets)

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
            self.shutdown()
            return

    def shutdown(self, **kwargs):
        self.client.close() if self.client else None
        self.sock.close() if self.sock else None
        super().shutdown(**kwargs)

