import multiprocessing
import multiprocessing.managers
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
import queue
import time
from collections.abc import Sequence
import collections
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Generator, Iterable, TypeAlias, TypeVar, Generic, NamedTuple, cast

Chunk = list[int]
PixelData = NamedTuple(
    "PixelData", [("toa", float), ("x", int), ("y", int), ("tot", int)]
)
TDCData = NamedTuple("TDCData", [("tdc_time", float), ("tdc_type", int)])
ClusterData = NamedTuple("ClusterData", [("toa", float), ("x", float), ("y", float)])
TriggerTime = ToF = float
CameraData = TypeVar("CameraData", PixelData, ClusterData)

type TpxDataType = int | float | str
T = TypeVar("T", bound=TpxDataType, covariant=True, default=TpxDataType)
type UnstructurableData[T] = T | tuple['UnstructurableData[T]', ...]


def structure_map[T: TpxDataType, U: TpxDataType](func: Callable[[TpxDataType], U], t: UnstructurableData[T]) -> UnstructurableData[U]:
    """applies function to all elements of a nested tuple"""
    if isinstance(t, tuple):
        return tuple(structure_map(func, sub_t) for sub_t in t)
    else:
        return func(t)


def structure[T: TpxDataType](template: UnstructurableData, data: Iterable[T]) -> UnstructurableData[T]:
    d_iter = iter(data)
    return structure_map(lambda _: next(d_iter), template)


def unstructure[T: TpxDataType](t: UnstructurableData[T]) -> Generator[T, None, None]:
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


class CircularBuffer[T: UnstructurableData](Sequence):
    def __init__(self, max_size: int, dtypes: UnstructurableData[str]):
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

    def put(self, values: UnstructurableData):
        with self.get_lock():
            for array, value in zip(self.arrays, unstructure(values)):
                array[self.index_.value] = value
            self.index_.value = (self.index_.value + 1) % self.max_size
            if self.size.value < self.max_size:
                self.size.value += 1

    def __getitem__(self, idx: int | slice) -> T:
        if isinstance(idx, slice):
            raise NotImplementedError
        if idx >= self.size.value:
            raise IndexError
        with self.get_lock():
            _idx = self.index_.value - self.size.value + idx % self.max_size
            inner = [a[_idx] for a in self.arrays]
            return structure(
                self.dtypes,
                inner,) # type: ignore

    def __len__(self) -> int:
        return self.size.value

    def get_all(self):
        return [self[i] for i in range(len(self))]

T = TypeVar("T", bound=UnstructurableData[TpxDataType], default=UnstructurableData[TpxDataType])
class ExtendedQueue[U]:
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
        *args,
        dtypes: UnstructurableData[str]=(),
        names: UnstructurableData[str]=(),
        buffer_size: int=0,
        chunk_size: int=0,
        manager: multiprocessing.managers.SyncManager | None=None,
        multi_process: bool=True,
        force_monotone: bool=False,
        period: int=25 * 2**30,
        max_back: float=1e9,
        unwrap: bool=False,
        maxsize: int=0,
        verbose: bool=False,
        **kwargs,
    ):
        self.buffer = CircularBuffer(buffer_size, dtypes) if buffer_size > 0 else None

        if multi_process:
            self.queue = (
                multiprocessing.Queue(maxsize=maxsize, **kwargs)
                if manager is None
                else manager.Queue(maxsize, **kwargs)
            )
        else:
            self.queue = queue.Queue(maxsize=maxsize)

        self.names = names
        self.dtypes = dtypes
        self.chunked = chunk_size > 0
        self.chunk_size = chunk_size
        self.input_buffer = []
        self.output_queue = None
        self.multi_process = multi_process
        self.last: float | None = None
        self.current_sum = 0
        self.force_monotone = force_monotone
        self.period = period
        self.max_back = max_back
        self.closed = multiprocessing.Value("b", False)
        self.unwrap = unwrap
        self.verbose = verbose

    def put(self, obj: U | list[U], **kwargs):
        if self.unwrap:
            obj = cast(list[U], obj)
            return [self._put(o, **kwargs) for o in obj]
        return self._put(obj, **kwargs) # type: ignore

    def _put(self, obj: U, **kwargs):
        if self.chunked:
            self.input_buffer.append(obj)
            if len(self.input_buffer) >= self.chunk_size:
                self.queue.put(self.input_buffer)
                self.input_buffer = []
        else:
            self.queue.put(obj, **kwargs)

    def _get(self, block=True, timeout=None) -> U:
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

        if self.buffer:
            self.buffer.put(output)

        if self.verbose:
            print(output)

        return output

    def get(self, block=True, timeout=None) -> U:
        if self.force_monotone:
            return self.get_monotonic(
                block=block, timeout=timeout, max_back=self.max_back, period=self.period
            )
        return self._get(block=block, timeout=timeout)

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

    def get_monotonic(self, period: int=25 * 2**30, max_back: float=0.0, **kwargs) -> U:
        #some data has maximum timestamp, so if we wrap around to 0, we need to add the period
        if self.last is None:
            out = self._get(**kwargs)
            self.last = next((unstructure(out)))
            return out

        current = self._get(**kwargs)
        curr_unstruct = list(unstructure(current))
        curr_unstruct[0] += self.current_sum
        while curr_unstruct[0] < self.last - max_back: # type: ignore
            curr_unstruct[0] += period
            self.current_sum += period

        self.last = curr_unstruct[0]
        return structure(current, curr_unstruct) # type: ignore

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
