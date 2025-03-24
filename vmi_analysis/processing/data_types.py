import collections
import multiprocessing
import multiprocessing.managers
import queue
import time
from typing import (
    Any,
    Iterable,
    NamedTuple,
)

import numpy.typing

Chunk = numpy.typing.NDArray[numpy.signedinteger]


class TimestampedData(NamedTuple):
    time: float
    c_dtypes = {"time": 'f8'}

    def flatten(self):
        return [self.time]

    def to_dict(self):
        return {"time": self.time}


class PixelData(TimestampedData):
    time: float
    x: int
    y: int
    tot: int
    c_dtypes = {"time": 'f8', "x": 'i4', "y": 'i4', "tot": 'i4'}

    def flatten(self):
        return [self.time, self.x, self.y, self.tot]

    def to_dict(self):
        return {"time": self.time, "x": self.x, "y": self.y, "tot": self.tot}


class TDCData(TimestampedData):
    time: float
    type: int
    c_dtypes = {"time": 'f8', "type": 'i4'}

    def flatten(self):
        return [self.time, self.type]

    def to_dict(self):
        return {"time": self.time, "type": self.type}


class ClusterData(TimestampedData):
    time: float
    x: float
    y: float
    c_dtypes = {"time": 'f8', "x": 'f8', "y": 'f8'}

    def flatten(self):
        return [self.time, self.x, self.y]

    def to_dict(self):
        return {"time": self.time, "x": self.x, "y": self.y}


class IndexedData[T: TimestampedData](NamedTuple):
    index: int
    data: T
    c_dtypes = {"index": 'i4'}

    def flatten(self):
        return [self.index] + self.data.flatten()

    def to_dict(self):
        return {"index": self.index, **self.data.to_dict()}


type ToF = TimestampedData
type Trigger = TimestampedData

class DequeQueue:
    def __init__(self, maxlen):
        self.deque = collections.deque(maxlen=maxlen)

    def put(self, data):
        self.deque.append(data)

    def get(self):
        return self.deque.popleft()

    def empty(self):
        return not self.deque

    def qsize(self):
        return len(self.deque)


# class CircularBuffer(Sequence[StructuredData[T]]):
#     def __init__(self, max_size: int, dtypes: StructuredData[str]):
#         self.max_size: int = max_size
#         self.dtypes = dtypes
#         self.arrays: tuple[SynchronizedArray[T], ...] = tuple(
#             multiprocessing.Array(d, max_size) for d in unstructure(dtypes)
#         )
#         self.index_: Synchronized[int] = multiprocessing.Value("L", 0)
#         self.size: Synchronized[int] = multiprocessing.Value("L", 0)
#
#     @contextmanager
#     def get_lock(self):
#         with ExitStack() as stack:
#             stack.enter_context(self.index_.get_lock())
#             stack.enter_context(self.size.get_lock())
#             for a in self.arrays:
#                 stack.enter_context(a.get_lock())
#             yield stack
#
#     def put(self, values: StructuredData):
#         with self.get_lock():
#             for array, value in zip(self.arrays, unstructure(values)):
#                 array[self.index_.value] = value
#             self.index_.value = (self.index_.value + 1) % self.max_size
#             if self.size.value < self.max_size:
#                 self.size.value += 1
#
#     def __getitem__(self, idx: int | slice) -> StructuredData[T]:  # type: ignore
#         if isinstance(idx, slice):
#             raise NotImplementedError
#         if idx >= self.size.value:
#             raise IndexError
#         with self.get_lock():
#             _idx = self.index_.value - self.size.value + idx % self.max_size
#             inner = [a[_idx] for a in self.arrays]
#             return structure(
#                 self.dtypes,
#                 inner,  # type: ignore
#             )
#
#     def __len__(self) -> int:
#         return self.size.value
#
#     def get_all(self):
#         return [self[i] for i in range(len(self))]


class Queue[T]:
    """
    A queue that can be used in a multi-process environment. It has several additional features:
    - It can be chunked, meaning that it will put multiple items into the queue at once.
    - It can be bound to a process, meaning that it will only be used in a single process.
    - It can be forced to be monotonic, meaning that it will force the output to be monotonically increasing.
    - It can have a buffer, meaning that it will store the last n items that were put into it.
    This is currently an overly complex class, and should be simplified in the future.
    """

    def __init__(
        self,
        chunk_size: int = 0,
        manager: multiprocessing.managers.SyncManager | None = None,
        multi_process: bool = True,
        maxsize: int = 0,
        verbose: bool = False,
        ctx: Any = None,
    ):
        if multi_process:
            if manager is None:
                self.queue = multiprocessing.Queue(maxsize=maxsize, ctx=ctx)
            else:
                if ctx:
                    raise ValueError("ctx should be None for managed queue")
                self.queue = manager.Queue(maxsize)
        else:
            self.queue = queue.Queue(maxsize=maxsize)

        self.chunked = chunk_size > 0
        self.chunk_size = chunk_size
        self.input_buffer = []
        self.output_queue = None
        self.multi_process = multi_process
        self.closed = multiprocessing.Value("b", False)
        self.verbose = verbose

    def put_all(
        self, objs: Iterable[T], block: bool = True, timeout: int | float | None = None
    ):
        for o in objs:
            self.put(o, block, timeout)

    def put(
        self,
        obj: T,
        block: bool = True,
        timeout: int | float | None = None,
    ):
        if self.chunked:
            self.input_buffer.append(obj)
            if len(self.input_buffer) >= self.chunk_size:
                self.queue.put(self.input_buffer)
                self.input_buffer = []
        else:
            self.queue.put(obj, block, timeout)

    def get(self, block=True, timeout=None) -> T:
        if self.chunked:
            if self.output_queue is None:
                self.output_queue = DequeQueue(self.chunk_size)
            if self.output_queue.empty():
                [
                    self.output_queue.put(datum)
                    for datum in self.queue.get(block=block, timeout=timeout)
                ]
            output = self.output_queue.get()
        else:
            output = self.queue.get(block=block, timeout=timeout)

        if self.verbose:
            print(output)

        return output

    def qsize(self) -> int:
        if self.chunked and self.output_queue:
            return (
                len(self.input_buffer)
                + self.queue.qsize() * self.chunk_size
                + self.output_queue.qsize()
            )
        elif self.chunked:
            return len(self.input_buffer) + self.queue.qsize() * self.chunk_size
        else:
            return self.queue.qsize()

    def empty(self) -> bool:
        return (
            self.queue.empty()
            and not self.input_buffer
            and (not self.output_queue or self.output_queue.empty())
        )

    def bind_to_process(self):
        if not self.empty():
            raise ValueError("Cannot change to single process while queue is not empty")
        if not self.multi_process:
            raise ValueError("Queue is already single process")
        if self.multi_process:
            self.queue = queue.Queue()
            self.multi_process = False
        return self

    def close(self):
        if self.input_buffer:
            self.queue.put(self.input_buffer)
            self.input_buffer = []
            # if self.multi_process:
            # self.queue.close()
        time.sleep(1)
        self.closed.value = True

    def make_empty(self):
        while not self.queue.empty():
            self.queue.get()
        self.input_buffer = []
        if self.output_queue:
            while not self.output_queue.empty():
                self.output_queue.get()


class StructuredDataQueue[T: TimestampedData](Queue[T]):
    def __init__(
        self,
        chunk_size: int = 0,
        manager: multiprocessing.managers.SyncManager | None = None,
        multi_process: bool = True,
        force_monotone: bool = False,
        period: int = 25 * 2**30,
        max_back: float = 1000000000,
        maxsize: int = 0,
        verbose: bool = False,
            dtypes: dict[str, str] = (),
            names: dict[str, str] = (),
            ctx: Any = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            manager=manager,
            multi_process=multi_process,
            maxsize=maxsize,
            verbose=verbose,
                ctx=ctx,
        )

        self.names = names
        self.dtypes = dtypes
        self.force_monotone = force_monotone
        self.last: float | None = None
        self.current_sum = 0
        self.period = period
        self.max_back = max_back

    def get(self, block=True, timeout=None) -> T:
        if self.force_monotone:
            return self.get_monotonic(
                block=block, timeout=timeout, max_back=self.max_back, period=self.period
            )
        return super().get(block=block, timeout=timeout)

    def get_monotonic(
        self,
        period: int = 25 * 2**30,
        max_back: float = 0,
        block: bool = True,
        timeout: float | None = None,
    ) -> T:

        if self.last is None:
            out = super().get(block, timeout)
            self.last = out.time
            return out

        current = super().get(block, timeout)
        current.time += self.current_sum
        while current.time < self.last - max_back:
            current.time += period
            self.current_sum += period

        self.last = current.time
        return current


class MonotonicQueue[T: TimestampedData](StructuredDataQueue[T]):
    def get(self, block=True, timeout=None) -> T:
        return self.get_monotonic(block=block, timeout=timeout)
