import collections
import multiprocessing
import queue
import time
import typing
import numpy as np

from ..data_types import ExtendedQueue, T
from .base_process import AnalysisStep


class QueueTee(AnalysisStep):
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


class Weaver(AnalysisStep):
    input_queues: tuple[ExtendedQueue[T]]
    output_queue: ExtendedQueue[T]

    def __init__(self, input_queues, output_queue):
        super().__init__()
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.current: list[int | float | None] = [None for _ in input_queues]

    def action(self):
        if any(cur is None for cur in self.current):
            for i, q in enumerate(self.input_queues):
                if self.current[i] is None:
                    if q.closed.value and q.empty():
                        self.current[i] = np.inf
                    try:
                        self.current[i] = q.get(timeout=0.1)
                    except queue.Empty or InterruptedError:
                        time.sleep(0.1)
                        return
        if all(cur == np.inf for cur in self.current):
            self.shutdown()
            return

        min_idx = self.current.index(min(c for c in self.current if c != np.inf))

        self.output_queue.put(self.current[min_idx])
        self.current[min_idx] = None


class QueueDecimator(AnalysisStep):
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
        except queue.Empty or InterruptedError:
            return
        if self.i % self.n == 0:
            self.output_queue.put(data)
        self.i += 1


class QueueReducer(AnalysisStep):
    def __init__(self, input_queue, output_queue, max_size, record=10000, **kwargs):
        super().__init__(**kwargs)
        self.input_queue = input_queue
        self.input_queues = (input_queue,)
        self.output_queue = output_queue
        self.output_queues = (output_queue,)
        self.max_size = max_size
        self.statuses = collections.deque(maxlen=record)
        self.record = record
        self.ratio = multiprocessing.Value('f', 0)
        self.name = "QueueReducer"

    def action(self):
        delta = self.max_size - self.output_queue.qsize()
        for i in range(self.max_size // 2):
            try:
                data = self.input_queue.get(timeout=0.1)
                if i < delta:
                    self.output_queue.put(data, timeout=0)
            except queue.Empty or queue.Full or InterruptedError:
                return
        if self.input_queue.qsize() > self.max_size * 5:
            self.input_queue.make_empty()

        # self.ratio.value=self.statuses.count(1)/self.record

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
        self.nexts: list[tuple[int, typing.Any] | None] = [None for _ in input_queues]
        self.out = tuple([] for _ in self.input_queues)
        self.output_empty = output_empty

    def action(self):
        if any(n is None for n in self.nexts):
            for i, q in enumerate(self.input_queues):
                if self.nexts[i] is None:
                    if q.closed.value and q.empty():
                        self.nexts[i] = (np.inf,)
                    try:
                        self.nexts[i] = q.get(timeout=0.1)
                    except queue.Empty or InterruptedError:
                        time.sleep(0.1)
                        return

        if all(n[0] == np.inf for n in self.nexts):
            self.shutdown()
            return

        for i, n in enumerate(self.nexts):
            while self.nexts[i][0] == self.current:
                self.out[i].append(self.nexts[i][1])
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


def create_process_instances(process_class, n_instances, output_queue, process_args, queue_args=None, process_name="", queue_name=""):
    if queue_args is None:
        queue_args = {}
    queues = tuple([ExtendedQueue(**queue_args) for _ in range(n_instances)])
    args = []
    for q in queues:
        new_args = process_args.copy()
        out_key = [k for k, v in new_args.items() if v is None][0]
        new_args[out_key] = q
        args.append(new_args)
    processes = {f"{process_name}_{i}": process_class(**a) for i, a in enumerate(args)}
    weaver = Weaver(queues, output_queue)
    return (
        {f"{queue_name}_{i}": q for i, q in enumerate(queues)},
        processes,
        weaver
    )


class QueueVoid(AnalysisStep):
    def __init__(self, input_queues, **kwargs):
        super().__init__(**kwargs)
        self.input_queues = input_queues
        self.name = "Void"

    def action(self):
        for q in self.input_queues:
            try:
                for _ in range(q.qsize()):
                    q.get(timeout=0.1)
            except queue.Empty or InterruptedError:
                continue
