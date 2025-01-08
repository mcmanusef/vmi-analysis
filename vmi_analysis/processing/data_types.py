import multiprocessing
import queue
import time
from collections.abc import Sequence
from contextlib import contextmanager, ExitStack
from typing import TypeVar, Generic, NamedTuple

T = TypeVar('T')
Chunk = list[int]
PixelData = NamedTuple('PixelData', [('toa', float), ('x', int), ('y', int), ('tot', int)])
TDCData = NamedTuple('TDCData', [('tdc_time', float), ('tdc_type', int)])
ClusterData = NamedTuple('ClusterData', [('toa', float), ('x', float), ('y', float)])
TriggerTime = ToF = float
CameraData = TypeVar('CameraData', PixelData, ClusterData)


def structure_map(func, t):
    if isinstance(t, tuple):
        return tuple(structure_map(func, sub_t) for sub_t in t)
    else:
        return func(t)


def structure(template, data):
    d_iter = iter(data)
    return structure_map(lambda _: next(d_iter), template)


def unstructure(t):
    if isinstance(t, tuple):
        for sub_t in t:
            yield from (unstructure(sub_t))
    else:
        yield t


class CircularBuffer(Sequence):
    def __init__(self, max_size, dtypes):
        self.max_size = max_size
        self.dtypes = dtypes
        self.arrays = tuple(multiprocessing.Array(d, max_size) for d in unstructure(dtypes))
        self.index = multiprocessing.Value('L', 0)
        self.size = multiprocessing.Value('L', 0)

    @contextmanager
    def get_lock(self):
        with ExitStack() as stack:
            stack.enter_context(self.index.get_lock())
            stack.enter_context(self.size.get_lock())
            for a in self.arrays:
                stack.enter_context(a.get_lock())
            yield stack

    def put(self, values):
        with self.get_lock():
            for array, value in zip(self.arrays, unstructure(values)):
                array[self.index.value] = value
            self.index.value = (self.index.value + 1) % self.max_size
            if self.size.value < self.max_size:
                self.size.value += 1

    def __getitem__(self, item):
        if item >= self.size.value:
            raise IndexError
        with self.get_lock():
            return structure(
                    self.dtypes,
                    [a[(self.index.value - self.size.value + item) % self.max_size] for a in self.arrays]
            )

    def __len__(self) -> int:
        return self.size.value

    def get_all(self):
        return [self[i] for i in range(len(self))]


class ExtendedQueue(Generic[T]):
    """
    A queue that can be used in a multi-process environment. It has several additional features:
    - It can be chunked, meaning that it will put multiple items into the queue at once.
    - It can be bound to a process, meaning that it will only be used in a single process.
    - It can be forced to be monotonic, meaning that it will force the output to be monotonically increasing.
    - It can have a buffer, meaning that it will store the last n items that were put into it.
    This is currently an overly complex class, and should be simplified in the future.
    """
    def __init__(self, *args,
                 dtypes=(), names=(),
                 buffer_size=0,
                 chunk_size=0,
                 manager=None,
                 multi_process=True,
                 force_monotone=False,
                 period=25 * 2 ** 30,
                 max_back=1e9,
                 unwrap=False,
                 maxsize=0,
                 loud=False,
                 **kwargs):
        self.buffer = CircularBuffer(buffer_size, dtypes) if buffer_size > 0 else None

        if multi_process:
            self.queue = multiprocessing.Queue(maxsize=maxsize, **kwargs) if manager is None else manager.Queue(maxsize, **kwargs)
        else:
            self.queue = queue.Queue(maxsize=maxsize)

        self.names = names
        self.dtypes = dtypes
        self.chunked = chunk_size > 0
        self.chunk_size = chunk_size
        self.input_buffer = []
        self.output_queue = None
        self.multi_process = multi_process
        self.last = None
        self.current_sum = 0
        self.force_monotone = multiprocessing.Value('b', force_monotone)
        self.period = multiprocessing.Value('f', period)
        self.max_back = multiprocessing.Value('f', max_back)
        self.closed = multiprocessing.Value('b', False)
        self.unwrap = unwrap
        self.loud = loud

    def put(self, obj: T, **kwargs):
        if self.unwrap:
            return [self._put(o, **kwargs) for o in obj]
        return self._put(obj, **kwargs)

    def _put(self, obj: T, **kwargs):
        if self.closed.value:
            raise ValueError("Queue is closed")
        if self.chunked:
            self.input_buffer.append(obj)
            if len(self.input_buffer) >= self.chunk_size:
                self.queue.put(self.input_buffer)
                self.input_buffer = []
        else:
            self.queue.put(obj, **kwargs)

    def _get(self, block=True, timeout=None) -> T:
        if self.chunked:
            if self.output_queue is None:
                self.output_queue = queue.Queue()
            if self.output_queue.empty():
                [self.output_queue.put(datum) for datum in self.queue.get(block=block, timeout=timeout)]
            output = self.output_queue.get(block=block, timeout=timeout)
        else:
            output = self.queue.get(block=block, timeout=timeout)

        if self.buffer:
            self.buffer.put(output)

        if self.loud:
            print(output)

        return output

    def get(self, block=True, timeout=None) -> T:
        if self.force_monotone.value:
            return self.get_monotonic(block=block, timeout=timeout, max_back=self.max_back.value, period=self.period.value)
        return self._get(block=block, timeout=timeout)

    def qsize(self) -> int:
        if self.chunked and self.output_queue:
            return len(self.input_buffer) + self.queue.qsize() * self.chunk_size + self.output_queue.qsize()
        elif self.chunked:
            return len(self.input_buffer) + self.queue.qsize() * self.chunk_size
        else:
            return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty() and not self.input_buffer and (not self.output_queue or self.output_queue.empty())

    def bind_to_process(self):
        if not self.empty():
            raise ValueError("Cannot change to single process while queue is not empty")
        if not self.multi_process:
            raise ValueError("Queue is already single process")
        if self.multi_process:
            self.queue = queue.Queue()
            self.multi_process = False
        return self

    def get_monotonic(self, period=25 * 2 ** 30, max_back=0, **kwargs) -> T:
        if self.last is None:
            out = self._get(**kwargs)
            self.last = list(unstructure(out))[0]
            return out

        current = self._get(**kwargs)
        curr_unstruct = list(unstructure(current))
        curr_unstruct[0] += self.current_sum
        while curr_unstruct[0] < self.last - max_back:
            curr_unstruct[0] += period
            self.current_sum += period

        self.last = curr_unstruct[0]
        return structure(current, curr_unstruct)

    def close(self):
        if self.input_buffer:
            self.queue.put(self.input_buffer)
            self.input_buffer = []
        time.sleep(1)
        self.closed.value = True

    def make_empty(self):
        while not self.queue.empty():
            self.queue.get()
        self.input_buffer = []
        if self.output_queue:
            while not self.output_queue.empty():
                self.output_queue.get()
