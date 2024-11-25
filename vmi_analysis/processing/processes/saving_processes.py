import queue
import time
import h5py

from ..data_types import ExtendedQueue, unstructure
from .base_process import AnalysisStep


class SaveToH5(AnalysisStep):
    file_path: str
    input_queues: tuple[ExtendedQueue, ...]
    in_queues: dict[str, ExtendedQueue]
    output_queues = ()
    h5_file: h5py.File | None

    def __init__(self, file_path, input_queues, chunk_size=1000, flat: bool | tuple[bool] | dict[str, bool] = True, **kwargs):
        super().__init__(**kwargs)
        self.name = "SaveToH5"
        self.file_path = file_path
        self.input_queues = tuple(input_queues.values())
        self.in_queues = input_queues
        self.chunk_size = chunk_size
        self.flat = flat if isinstance(flat, dict) else {k: flat for k in input_queues.keys()} if isinstance(flat, bool) else {k: f for k, f in flat}
        self.h5_file = None
        self.n = 0

    def initialize(self):
        f = h5py.File(self.file_path, 'w')
        for key, q in self.in_queues.items():
            if self.flat[key]:
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    f.create_dataset(name, (self.chunk_size,), dtype=dtype, maxshape=(None,))
            else:
                g = f.create_group(key)
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    g.create_dataset(name, (self.chunk_size,), dtype=dtype, maxshape=(None,))
        self.h5_file = f
        super().initialize()

    def action(self):
        f = self.h5_file

        sizes = [(q.qsize(), k, q) for k, q in self.in_queues.items()]
        sizes.sort()
        max_queue = sizes[-1][2]
        max_name = sizes[-1][1]
        max_size = sizes[-1][0]
        if len(sizes) > 1:
            size_diff = max_size - sizes[-2][0]
        else:
            size_diff = max_size
        size_diff = min(10 * self.chunk_size, size_diff)
        save_size = size_diff if size_diff else max_size
        if max_size == 0:
            self.n += 1
            time.sleep(1)
            return

        self.n = 0
        to_write = []
        for i in range(save_size):
            try:
                data = max_queue.get(timeout=0.1)
                to_write.append(list(unstructure(data)))
            except queue.Empty or InterruptedError:
                break

        data_lists = tuple(zip(*to_write))
        for name, data in zip(unstructure(max_queue.names), data_lists):
            if f[name].shape[0] != self.chunk_size:
                if self.flat[max_name]:
                    f[name].resize(f[name].shape[0] + len(data), axis=0)
                    f[name][-len(data):] = data
                else:
                    g = f[max_name]
                    g[name].resize(g[name].shape[0] + len(data), axis=0)
                    g[name][-len(data):] = data
            else:
                if self.flat[max_name]:
                    f[name].resize(len(data), axis=0)
                    f[name][:] = data
                else:
                    g = f[max_name]
                    g[name].resize(len(data), axis=0)
                    g[name][:] = data

    def shutdown(self, **kwargs):
        self.h5_file.close() if self.h5_file else None
        super().shutdown(**kwargs)
