from vmi_analysis.processing.pipelines import base_pipeline, run_pipeline
from vmi_analysis.processing.processes import *
from vmi_analysis.processing import data_types


class PrintingProcess(AnalysisStep):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.input_queues = [queue]

    def action(self):
        if not self.queue.empty():
            out = self.queue.get()
            if out[1][0] > 20000:
                print(out)


class ClusterTest(base_pipeline.BasePipeline):
    def __init__(self, fname):
        super().__init__()
        self.queues = {
            "chunk": data_types.Queue(maxsize=1000),
            "pixel": data_types.Queue(),
            "etof": data_types.Queue(force_monotone=True, maxsize=50000),
            "itof": data_types.Queue(force_monotone=True, maxsize=50000),
            "pulses": data_types.Queue(force_monotone=True, maxsize=50000),
            "clusters": data_types.Queue(force_monotone=True, maxsize=50000),
            "t_etof": data_types.Queue(
                names=("etof_corr", ("t_etof",)), dtypes=("i", ("f",))
            ),
            "t_itof": data_types.Queue(
                names=("itof_corr", ("t_itof",)), dtypes=("i", ("f",))
            ),
            "t_pulse": data_types.Queue(names=("t_pulse",), dtypes=("i",)),
            "t_cluster": data_types.Queue(
                names=("clust_corr", ("t", "x", "y")), dtypes=("i", ("f", "f", "f"))
            ),
        }
        self.processes = {
            "ChunkStream": TPXFileReader(fname, self.queues["chunk"]).make_process(),
            "Converter": VMIConverter(
                self.queues["chunk"],
                self.queues["pixel"],
                self.queues["pulses"],
                self.queues["etof"],
                self.queues["itof"],
            ).make_process(),
            "Clusterer": DBSCANClusterer(
                self.queues["pixel"], self.queues["clusters"]
            ).make_process(),
            "Correlator": TriggerAnalyzer(
                self.queues["pulses"],
                (self.queues["etof"], self.queues["itof"], self.queues["clusters"]),
                self.queues["t_pulse"],
                (
                    self.queues["t_etof"],
                    self.queues["t_itof"],
                    self.queues["t_cluster"],
                ),
            ).make_process(),
            "Saver": SaveToH5(
                fname + ".h5",
                {
                    "etof": self.queues["t_etof"],
                    "itof": self.queues["t_itof"],
                    "pulse": self.queues["t_pulse"],
                    "cluster": self.queues["t_cluster"],
                },
                loud=True,
            ).make_process(),
        }


if __name__ == "__main__":
    fname = r"J:\ctgroup\Edward\DATA\VMI\20241125\c2h4_p_5W"
    pipeline = ClusterTest(fname)
    run_pipeline(pipeline)
# %%
