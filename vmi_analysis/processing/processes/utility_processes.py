import multiprocessing
import queue
import time
import typing

import numpy as np

from .base_process import AnalysisStep
from ..data_types import (
    Queue,
    StructuredDataQueue as SDQueue, TimestampedData, MonotonicQueue,
)


class QueueTee[T](AnalysisStep):
    """
    Copies data from one queue to multiple queues.

    Parameters:
    - input_queue (ExtendedQueue): Queue to copy data from.
    - output_queues (tuple[ExtendedQueue]): Queues to copy data to.
    """
    input_queue: Queue[T]
    output_queues: tuple[Queue[T]]

    def __init__(self, input_queue, output_queues):
        super().__init__()
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.input_queues = (input_queue,)

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            return
        for q in self.output_queues:
            q.put(data)


class Weaver[T: TimestampedData](AnalysisStep):
    """
    Combines data from multiple sorted queues into a single sorted queue.
    Sorts on the first value of the data.

    Parameters:
    - input_queues (tuple[ExtendedQueue]): Queues to combine data from.
    - output_queue (ExtendedQueue): Queue to put combined data into.
    """
    input_queues: tuple[SDQueue[T]]
    output_queue: SDQueue[T]

    def __init__(
        self,
            input_queues,
            output_queue,
    ):
        super().__init__()
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.current: list[T | None] = [None for _ in input_queues]
        self.name = "Weaver"
        self.checked = False

    def action(self):
        if any(cur is None for cur in self.current):
            for i, q in enumerate(self.input_queues):
                if self.current[i] is None:
                    if q.closed.value and q.empty():
                        self.current[i] = TimestampedData(time=np.inf)
                    try:
                        self.current[i] = q.get(timeout=0.1)

                    except queue.Empty:
                        time.sleep(0.1)
                        return

        if all(cur.time == np.inf for cur in self.current):
            self.shutdown()
            return

        min_idx = np.argmin([cur.time for cur in self.current])
        self.output_queue.put(self.current[min_idx])
        self.current[min_idx] = None


class QueueDecimator[T](AnalysisStep):
    """
    Puts every n-th item from the input queue into the output queue.
    """
    input_queue: Queue[T]
    output_queue: Queue[T]

    def __init__(self, input_queue, output_queue, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.i = 0
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.input_queues = (input_queue,)
        self.output_queues = (output_queue,)

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return
        if self.i % self.n == 0:
            self.output_queue.put(data)
        self.i += 1


class QueueReducer[T](AnalysisStep):
    """
    Reduces the size of the input queue by moving data to the output queue until the output queue is full, then discards the rest.

    """
    input_queue: Queue[T]
    output_queue: Queue[T]

    def __init__(self, input_queue, output_queue, max_size, **kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.input_queues = (input_queue,)
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.max_size = max_size
        self.ratio = multiprocessing.Value("f", 0)
        self.name = "QueueReducer"

    def action(self):
        delta = self.max_size - self.output_queue.qsize()
        for i in range(self.max_size // 2):
            try:
                data = self.input_queue.get(timeout=0.1)
                if i < delta:
                    self.output_queue.put(data, timeout=0)
            except queue.Empty or queue.Full:
                return
        if self.input_queue.qsize() > self.max_size * 5:
            self.input_queue.make_empty()

    def status(self):
        stat = super().status()
        stat["ratio"] = self.ratio.value
        return stat


class QueueGrouper(AnalysisStep):
    def __init__(self, input_queues, output_queue, output_empty=True, **kwargs):
        super().__init__(**kwargs)
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.current = 0
        self.nexts: list[tuple[int | float, typing.Any] | None] = [
            None for _ in input_queues
        ]
        self.out = tuple([] for _ in self.input_queues)
        self.output_empty = output_empty

    def action(self):
        if any(n is None for n in self.nexts):
            for i, q in enumerate(self.input_queues):
                if self.nexts[i] is None:
                    if q.closed.value and q.empty():
                        self.nexts[i] = (np.inf, None)
                    try:
                        self.nexts[i] = q.get(timeout=0.1)
                    except queue.Empty or InterruptedError:
                        time.sleep(0.1)
                        return

        if all(n[0] == np.inf for n in self.nexts):  # type: ignore
            self.shutdown()
            return

        for i, n in enumerate(self.nexts):
            while self.nexts[i][0] == self.current:  # type: ignore
                self.out[i].append(self.nexts[i][1])  # type: ignore
                try:
                    self.nexts[i] = self.input_queues[i].get(timeout=0.1)
                except queue.Empty or InterruptedError:
                    self.nexts[i] = None
                    break
        if any(n is None for n in self.nexts):
            return

        self.output_queue.put(self.out) if any(self.out) or self.output_empty else None
        self.out = tuple([] for _ in self.input_queues)
        self.current += 1


def create_process_instances(
    process_class,
    n_instances,
    output_queue,
    process_args,
    queue_args=None,
    process_name="",
    queue_name="",
):
    if queue_args is None:
        queue_args = {}
    queues = tuple([Queue(**queue_args) for _ in range(n_instances)])
    args = []
    for q in queues:
        new_args = process_args.copy()
        out_key = [k for k, v in new_args.items() if v is None][0]
        new_args[out_key] = q
        args.append(new_args)
    processes = {f"{process_name}_{i}": process_class(**a) for i, a in enumerate(args)}
    weaver = Weaver(queues, output_queue)
    return {f"{queue_name}_{i}": q for i, q in enumerate(queues)}, processes, weaver


class QueueVoid(AnalysisStep):
    """
    Empties the input queues.
    """

    def __init__(self, input_queues, loud=False, **kwargs):
        super().__init__(**kwargs)
        self.input_queues = input_queues
        self.name = "Void"
        self.loud = loud

    def action(self):
        for q in self.input_queues:
            try:
                for _ in range(q.qsize()):
                    print(q.get(timeout=0.1)) if self.loud else q.get(timeout=0.1)
            except queue.Empty or InterruptedError:
                continue


class QueueDistributor(AnalysisStep):
    def __init__(self, input_queue, output_queues, **kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.input_queues = (input_queue,)
        self.i = 0
        self.name = "Distributor"

    def action(self):
        try:
            data = self.input_queue.get(timeout=0.1)
        except queue.Empty or InterruptedError:
            return
        self.output_queues[self.i].put(data)
        self.i = (self.i + 1) % len(self.output_queues)


def multithread_process(
    astep_class: typing.Type[AnalysisStep],
    input_queues_dict: dict[str, Queue],
    output_queues_dict: dict[str, Queue],
    n_threads: int,
    astep_kw_args: dict | None = None,
    in_queue_kw_args: dict[str, typing.Any]
    | dict[str, dict[str, typing.Any]]
    | None = None,
    out_queue_kw_args: dict | None = None,
    name="",
):
    if astep_kw_args is None:
        astep_kw_args = {}
    if in_queue_kw_args is None:
        in_queue_kw_args = {}
    if out_queue_kw_args is None:
        out_queue_kw_args = {}
    sample_process = astep_class(
        **input_queues_dict, **output_queues_dict, **astep_kw_args
    )

    # check if input/output queue kw args are individual or shared
    individual_in_queue_kw_args = in_queue_kw_args.keys() == input_queues_dict.keys()
    individual_out_queue_kw_args = out_queue_kw_args.keys() == output_queues_dict.keys()

    # for each process, a dictionary of queue name : queue
    split_in_queues = []
    for _ in range(n_threads):
        process_queues = {}
        for queue_name in input_queues_dict:
            queue_kwargs: dict[str, typing.Any] = (
                in_queue_kw_args
                if not individual_in_queue_kw_args
                else in_queue_kw_args[queue_name]
            )
            process_queues[queue_name] = Queue(**queue_kwargs)

    split_out_queues = []
    for _ in range(n_threads):
        process_queues = {}
        for queue_name in output_queues_dict:
            queue_kwargs: dict[str, typing.Any] = (
                out_queue_kw_args
                if not individual_out_queue_kw_args
                else out_queue_kw_args[queue_name]
            )
            if "force_monotone" in queue_kwargs:
                monotone = queue_kwargs["force_monotone"]
                queue_kwargs.pop("force_monotone")
                if monotone:
                    process_queues[queue_name] = MonotonicQueue(**queue_kwargs)
                else:
                    process_queues[queue_name] = Queue(**queue_kwargs)

    active_processes = {
        f"{name}_{i}": astep_class(
            **split_in_queues[i], **split_out_queues[i], **astep_kw_args
        )
        for i in range(n_threads)
    }

    distributors = {
        f"{name}_distributor_{i}": QueueDistributor(
            sample_process.input_queues[i],
            [proc.input_queues[i] for proc in active_processes.values()],
        )
        for i in range(len(sample_process.input_queues))
    }
    weavers = {
        f"{name}_weaver_{i}": Weaver(
            [proc.output_queues[i] for proc in active_processes.values()],
            sample_process.output_queues[i],
        )
        for i in range(len(sample_process.output_queues))
    }
    del sample_process

    input_queues = {
        f"input_queue_{i}_{j}": distributors[i].output_queues[j]
        for i in distributors
        for j in range(len(distributors[i].output_queues))
    }
    output_queues = {
        f"output_queue_{i}_{j}": weavers[i].input_queues[j]
        for i in weavers
        for j in range(len(weavers[i].input_queues))
    }
    queues = {**input_queues, **output_queues}
    processes = {**distributors, **active_processes, **weavers}
    return queues, processes
