import pickle
import queue
from .base_process import AnalysisStep


class QueueCacheWriter(AnalysisStep):
    def __init__(self, fname, q, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.input_queues = (q,)
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
        pickle.dump(self.cache, open(self.fname, "wb"))
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
        self.cache = pickle.load(open(self.fname, "rb"))
        super().initialize()

    def action(self):
        for data in self.cache:
            self.q.put(data)
        self.cache = []
        self.holding.value = False

    def shutdown(self, gentle=False):
        super().shutdown(gentle=gentle)