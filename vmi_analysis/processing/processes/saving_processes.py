import queue
import time
from typing import Any, TypeVar
import h5py

from ..data_types import Queue, unstructure
from .base_process import AnalysisStep


class SaveToH5(AnalysisStep):
    """
    Saves data from queues to an H5 file. The data is saved in chunks of size chunk_size.
    The structure of the data is determined by the structure of the queues. The data is saved in the same order as it
    is received from the queues.

    If flat is True, all data is saved in the root of the H5 file, with the names of datasets given by the
    names attribute of the queues.
    If flat is False, the data is saved in groups, with the names of the queues as the group names, and the
    names of the datasets given by the names attribute of the queues.
    If flat is a tuple, it should be the same length as the number of queues, and each element should be a boolean,
    in which case the data from the corresponding queue is saved according to the corresponding boolean.
    In flat is a dictionary, the keys should be the names of the queues, and the values should be booleans, in which case
    the data from the corresponding queue is saved according to the corresponding boolean.

    Parameters:
    - file_path (str): Path to the H5 file.
    - input_queues (tuple[ExtendedQueue]): Queues to save data from.
    - chunk_size (int): Size of the chunks to save.
    - flat (bool | tuple[bool] | dict[str, bool]): Determines the structure of the saved data.
    - loud (bool): If True, print debug information.
    - **kwargs: Additional keyword arguments.
    """

    file_path: str
    input_queues: tuple[Queue, ...]
    in_queues: dict[str, Queue]
    output_queues = ()
    h5_file: h5py.File | None

    def __init__(
        self,
        file_path,
        input_queues,
        chunk_size=1000,
        flat: bool | tuple[bool] | dict[str, bool] = True,
        loud=False,
        swmr=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "SaveToH5"
        self.file_path = file_path
        self.input_queues = tuple(input_queues.values())
        self.in_queues = input_queues
        self.chunk_size = chunk_size
        # self.flat = (
        #     flat
        #     if isinstance(flat, dict)
        #     else {k: flat for k in input_queues.keys()}
        #     if isinstance(flat, bool)
        #     else {k: f for k, f in flat}
        # )
        match flat:
            case bool():
                self.flat = {k: flat for k in input_queues.keys()}
            case dict():
                self.flat = flat
            case tuple():
                self.flat = {k: f for k, f in zip(input_queues.keys(), flat)}
            case _:
                raise ValueError("Invalid flat argument")
        self.h5_file = None
        self.time_since_last_save = 0
        self.verbose = loud
        self.swmr = swmr

    def initialize(self):
        f = h5py.File(self.file_path, "w")
        for key, q in self.in_queues.items():
            if self.flat[key]:
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    f.create_dataset(
                        name, (self.chunk_size,), dtype=dtype, maxshape=(None,)
                    )
            else:
                g = f.create_group(key)
                for name, dtype in zip(unstructure(q.names), unstructure(q.dtypes)):
                    g.create_dataset(
                        name, (self.chunk_size,), dtype=dtype, maxshape=(None,)
                    )
        self.h5_file = f
        if self.swmr:
            self.h5_file.swmr_mode = True
        print(f"Verbose={self.verbose}")
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
            self.time_since_last_save += 1
            time.sleep(1)
            return

        self.time_since_last_save = 0
        to_write: list[list[Any]] = []
        if self.verbose:
            print(f"Gathering {save_size} from {max_name}")

        # save a chunk of data of size i from the largest queue
        for i in range(save_size):
            try:
                data = max_queue.get(timeout=0.1)
                to_write.append(list(unstructure(data)))
            except queue.Empty or InterruptedError:
                if self.verbose:
                    print(f"Queue {max_name} empty")
                break

        data_lists = tuple(zip(*to_write))
        if self.verbose:
            print(f"Data gathered:{data_lists}")
        for name, data in zip(unstructure(max_queue.names), data_lists):
            if self.verbose:
                print(f"Writing {len(data)} to {name}")
                print(data)
            dataset = h5f[name]
            if not isinstance(dataset, h5py.Dataset):
                raise RuntimeError()
            if dataset.shape[0] != self.chunk_size:
                if self.flat[max_name]:
                    dataset.resize(dataset.shape[0] + len(data), axis=0)
                    dataset[-len(data) :] = data
                    if self.verbose:
                        print(f"Resizing {name} to {dataset.shape[0]}")
                        print(dataset[-len(data) :])
                else:
                    g: h5py.Group = h5f[max_name]  # type: ignore
                    g_dataset: h5py.Dataset = g[name]  # type: ignore
                    g_dataset.resize(g_dataset.shape[0] + len(data), axis=0)
                    g_dataset[-len(data) :] = data
                    if self.verbose:
                        print(f"Resizing {name} to {g_dataset.shape[0]}")
                        print(g_dataset[-len(data) :])
            else:
                if self.flat[max_name]:
                    dataset.resize(len(data), axis=0)
                    dataset[:] = data
                    if self.verbose:
                        print(f"Writing {dataset.shape[0]} to {name}")
                        print(dataset[:])
                else:
                    g: h5py.Group = h5f[max_name]  # type: ignore
                    g_dataset: h5py.Dataset = g[name]  # type: ignore
                    g_dataset.resize(len(data), axis=0)
                    g_dataset[:] = data
                    if self.verbose:
                        print(f"Writing {g_dataset.shape[0]} to {name}")
                        print(g_dataset[:])

    def shutdown(self, gentle=False):
        self.h5_file.close() if self.h5_file else None
        super().shutdown(gentle)
