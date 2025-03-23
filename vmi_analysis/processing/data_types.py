import multiprocessing
import multiprocessing.managers
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
import queue
import numpy.typing
import time
from collections.abc import Sequence
import collections
from contextlib import contextmanager, ExitStack
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    NoReturn,
    TypeAlias,
    TypeVar,
    Generic,
    NamedTuple,
    cast,
)

Chunk = numpy.typing.NDArray[numpy.signedinteger]
type PixelData = tuple[int, int, int, int]  # time, x, y, tot
type TDCData = tuple[float, int]  # time, channel
ClusterData = NamedTuple("ClusterData", [("toa", float), ("x", float), ("y", float)])
TriggerTime = ToF = float
CameraData = TypeVar("CameraData", PixelData, ClusterData)

type TpxDataType = int | float | str
T = TypeVar("T", bound=TpxDataType, covariant=True, default=TpxDataType)
U = TypeVar("U", bound=TpxDataType, covariant=True, default=TpxDataType)
type StructuredData[T] = T | tuple["StructuredData[T]", ...]


def i(t: TDCData) -> StructuredData:
    return t


def structure_map(func: Callable[[T], U], t: StructuredData[T]) -> StructuredData[U]:
    """applies function to all elements of a nested tuple"""
    if isinstance(t, tuple):
        return tuple(structure_map(func, sub_t) for sub_t in t)
    else:
        return func(t)


def structure(template: StructuredData, data: Iterable[T]) -> StructuredData[T]:
    d_iter = iter(data)
    return structure_map(lambda _: next(d_iter), template)


def unstructure(t: StructuredData[T]) -> Generator[T, None, None]:
    if isinstance(t, tuple):
        for sub_t in t:
            yield from (unstructure(sub_t))
    else:
        yield t


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


class CircularBuffer(Sequence[StructuredData[T]]):
    def __init__(self, max_size: int, dtypes: StructuredData[str]):
        self.max_size: int = max_size
        self.dtypes = dtypes
        self.arrays: tuple[SynchronizedArray[T], ...] = tuple(
            multiprocessing.Array(d, max_size) for d in unstructure(dtypes)
        )
        self.index_: Synchronized[int] = multiprocessing.Value("L", 0)
        self.size: Synchronized[int] = multiprocessing.Value("L", 0)

    @contextmanager
    def get_lock(self):
        with ExitStack() as stack:
            stack.enter_context(self.index_.get_lock())
            stack.enter_context(self.size.get_lock())
            for a in self.arrays:
                stack.enter_context(a.get_lock())
            yield stack

    def put(self, values: StructuredData):
        with self.get_lock():
            for array, value in zip(self.arrays, unstructure(values)):
                array[self.index_.value] = value
            self.index_.value = (self.index_.value + 1) % self.max_size
            if self.size.value < self.max_size:
                self.size.value += 1

    def __getitem__(self, idx: int | slice) -> StructuredData[T]:  # type: ignore
        if isinstance(idx, slice):
            raise NotImplementedError
        if idx >= self.size.value:
            raise IndexError
        with self.get_lock():
            _idx = self.index_.value - self.size.value + idx % self.max_size
            inner = [a[_idx] for a in self.arrays]
            return structure(
                self.dtypes,
                inner,  # type: ignore
            )

    def __len__(self) -> int:
        return self.size.value

    def get_all(self):
        return [self[i] for i in range(len(self))]


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


class StructuredDataQueue[T: StructuredData](Queue[T]):
    def __init__(
        self,
        *args,
        chunk_size: int = 0,
        manager: multiprocessing.managers.SyncManager | None = None,
        multi_process: bool = True,
        force_monotone: bool = False,
        period: int = 25 * 2**30,
        max_back: float = 1000000000,
        unwrap: bool = False,
        maxsize: int = 0,
        verbose: bool = False,
        dtypes: StructuredData[str] = (),
        names: StructuredData[str] = (),
        **kwargs,
    ):
        super().__init__(
            *args,
            chunk_size=chunk_size,
            manager=manager,
            multi_process=multi_process,
            maxsize=maxsize,
            verbose=verbose,
            **kwargs,
        )

        self.names = names
        self.dtypes = dtypes
        # self.buffer = CircularBuffer(buffer_size, dtypes) if buffer_size > 0 else None
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
        # return super().get_monotonic(period, max_back, block, timeout)
        # some data has maximum timestamp, so if we wrap around to 0, we need to add the period
        if self.last is None:
            out = super().get(block, timeout)
            self.last = next((unstructure(out)))  # type: ignore # assume it's a float
            return out

        current = super().get(block, timeout)
        curr_unstruct = list(unstructure(current))  # type: ignore
        curr_unstruct[0] += self.current_sum  # type: ignore
        while curr_unstruct[0] < self.last - max_back:  # type: ignore
            curr_unstruct[0] += period
            self.current_sum += period

        self.last = curr_unstruct[0]
        return structure(current, curr_unstruct)  # type: ignore
