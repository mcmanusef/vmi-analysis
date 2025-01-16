import collections
import pickle
import queue
import time
import numpy as np
from matplotlib import pyplot as plt
from .base_process import AnalysisStep
import matplotlib


class Display(AnalysisStep):
    def __init__(
            self,
            grouped_queue,
            n,
            toa_range=None,
            etof_range=None,
            itof_range=None
    ):
        super().__init__()
        self.input_queues = (grouped_queue,)
        self.grouped_queue = grouped_queue
        self.name = "Display"
        self.current_data = {
            "etof": collections.deque(maxlen=n),
            "itof": collections.deque(maxlen=n),
            "cluster": collections.deque(maxlen=n),
            "timestamps": collections.deque(maxlen=n)
        }
        self.figure = None
        self.ax = None
        self.xy_hist = None
        self.xy_hist_data = np.zeros((256, 256))
        self.toa_hist = None
        self.toa_hist_data = np.zeros(2000)
        self.etof_hist = None
        self.etof_hist_data = np.zeros(2000)
        self.itof_hist = None
        self.itof_hist_data = np.zeros(400)
        self.update_interval = 1
        self.last_update = 0
        self.toa_range = toa_range
        self.etof_range = etof_range
        self.itof_range = itof_range

    def initialize(self):
        self.figure, self.ax = plt.subplots(2, 2)
        self.xy_hist = self.ax[0, 0].imshow(np.random.random(size=(256, 256)), extent=[0, 256, 0, 256], origin='lower', aspect='auto', cmap='jet',
                                            interpolation='nearest')
        self.ax[0, 0].set_title("XY")
        self.toa_hist = self.ax[0, 1].plot(np.linspace(0, 2000, 2000), np.linspace(0, 1, num=2000))[0]
        plt.sca(self.ax[0, 1])
        plt.xlim(0, 2000)
        plt.xlabel("ToA (ns)")
        if self.toa_range is not None:
            plt.axvline(self.toa_range[0], color='r', linestyle='--')
            plt.axvline(self.toa_range[1], color='r', linestyle='--')

        self.ax[0, 1].set_title("ToA")
        self.etof_hist = self.ax[1, 0].plot(np.linspace(0, 2000, 2000), np.linspace(0, 1, num=2000))[0]
        plt.sca(self.ax[1, 0])
        plt.xlim(0, 2000)
        plt.xlabel("ToF (ns)")
        if self.etof_range is not None:
            plt.axvline(self.etof_range[0], color='r', linestyle='--')
            plt.axvline(self.etof_range[1], color='r', linestyle='--')

        self.ax[1, 0].set_title("eToF")
        self.itof_hist = self.ax[1, 1].plot(np.linspace(0, 2e4, 400), np.linspace(0, 1, num=400))[0]
        self.ax[1, 1].set_title("iToF")
        plt.sca(self.ax[1, 1])
        plt.xlim(0, 2e4)
        plt.xlabel("ToF (ns)")
        if self.itof_range is not None:
            plt.axvline(self.itof_range[0], color='r', linestyle='--')
            plt.axvline(self.itof_range[1], color='r', linestyle='--')

        plt.suptitle("Processing Rate:")
        self.figure.tight_layout()
        self.figure.show()
        super().initialize()

    def action(self):
        try:
            data = self.grouped_queue.get(timeout=0.1)
            # print(data)
        except queue.Empty or InterruptedError:
            return

        if len(self.current_data["etof"]) == self.current_data["etof"].maxlen:
            etof_rem = self.current_data["etof"].popleft()
            itof_rem = self.current_data["itof"].popleft()
            t_rem, x_rem, y_rem = zip(*last) if (last := self.current_data["cluster"].popleft()) else ([], [], [])
            last_timestamp = self.current_data["timestamps"].popleft()

            for t, x, y in zip(t_rem, x_rem, y_rem):
                if self.toa_range is None or (self.toa_range[0] < t < self.toa_range[1]):
                    self.xy_hist_data[int(x), int(y)] -= 1
            for t in t_rem:
                if 0 < t < 2e4:
                    self.toa_hist_data[int(t / 10)] -= 1
            for t in etof_rem:
                if 0 < t < 2e4:
                    self.etof_hist_data[int(t / 10)] -= 1
            for t in itof_rem:
                if 0 < t < 2e4:
                    self.itof_hist_data[int(t / 50)] -= 1

            processing_rate = self.current_data["timestamps"].maxlen / (time.time() - last_timestamp)
            plt.suptitle(f"Processing Rate: {processing_rate:.2f} Hz")

        etof, itof, cluster = data

        self.current_data["itof"].append(itof)
        if self.itof_range is not None:
            if len(itof) != 1 or not (self.itof_range[0] < itof[0] < self.itof_range[1]):
                etof = []
                cluster = []

        self.current_data["etof"].append(etof)
        if self.etof_range is not None:
            if len(etof) != 1 or not (self.etof_range[0] < etof[0] < self.etof_range[1]):
                cluster = []

        self.current_data["cluster"].append(cluster)
        self.current_data["timestamps"].append(time.time())

        if cluster:
            for t, x, y in cluster:
                if self.toa_range is None or (self.toa_range[0] < t < self.toa_range[1]):
                    self.xy_hist_data[int(x), int(y)] += 1 if x < 256 and y < 256 else 0
                if 0 < t < 2000:
                    self.toa_hist_data[int(t)] += 1

        for t in etof:
            if 0 < t < 2000:
                self.etof_hist_data[int(t)] += 1
        for t in itof:
            if 0 < t < 2e4:
                self.itof_hist_data[int(t / 50)] += 1

        if time.time() - self.last_update > self.update_interval:
            self.xy_hist.set_data(np.log(self.xy_hist_data + 1) / np.log(np.max(self.xy_hist_data + 1)))
            self.toa_hist.set_ydata(np.log(self.toa_hist_data + 1) / np.log(np.max(self.toa_hist_data + 1)))
            self.etof_hist.set_ydata(np.log(self.etof_hist_data + 1) / np.log(np.max(self.etof_hist_data + 1)))
            self.itof_hist.set_ydata(np.log(self.itof_hist_data + 1) / np.log(np.max(self.itof_hist_data + 1)))

            cluster_count_rate = np.mean([len(c) for c in self.current_data["cluster"]])
            etof_count_rate = np.mean([len(c) for c in self.current_data["etof"]])
            itof_count_rate = np.mean([len(c) for c in self.current_data["itof"]])

            self.ax[0, 0].set_title(f"XY (Cluster Rate: {cluster_count_rate:.2f})")
            self.ax[0, 1].set_title(f"ToA (Count Rate: {cluster_count_rate:.2f})")
            self.ax[1, 0].set_title(f"eToF (Count Rate: {etof_count_rate:.2f})")
            self.ax[1, 1].set_title(f"iToF (Count Rate: {itof_count_rate:.2f})")

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.last_update = time.time()

    def shutdown(self, **kwargs):
        plt.close(self.figure)
        super().shutdown(**kwargs)

class QueueCacheWriter(AnalysisStep):
    def __init__(self, fname, q, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.input_queues=(q,)
        self.fname = fname
        self.cache = []
        self.name = "QueueCache"

    def action(self):
        try:
            data = self.q.get(timeout=0.1)
            self.cache.append(data)
        except queue.Empty:
            return

    def shutdown(self, gentle=False):
        pickle.dump(self.cache, open(self.fname, 'wb'))
        super().shutdown(gentle=gentle)

class QueueCacheReader(AnalysisStep):
    def __init__(self, fname, q, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.output_queues = (q,)
        self.fname = fname
        self.name = "QueueCache"
        self.cache = []
        self.holding.value = True

    def initialize(self):
        self.cache = pickle.load(open(self.fname, 'rb'))
        super().initialize()

    def action(self):
        for data in self.cache:
            self.q.put(data)
        self.cache = []
        self.holding.value = False

    def shutdown(self, gentle=False):
        super().shutdown(gentle=gentle)