from .. import data_types, processes
from .base_pipeline import AnalysisPipeline


class MonitorPipeline(AnalysisPipeline):
    def __init__(self, saving_path, cluster_processes=1, timeout=0, toa_range=None, etof_range=None, itof_range=None, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk_stream": data_types.ExtendedQueue(),
            "chunk": data_types.ExtendedQueue(),
            "pixel": data_types.ExtendedQueue(),
            "etof": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "itof": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "pulses": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),
            "clusters": data_types.ExtendedQueue(force_monotone=True, maxsize=5000),

            "t_etof": data_types.ExtendedQueue(),
            "t_itof": data_types.ExtendedQueue(),
            "t_pulse": data_types.ExtendedQueue(),
            "t_cluster": data_types.ExtendedQueue(),

            "grouped": data_types.ExtendedQueue(),
            "reduced_grouped": data_types.ExtendedQueue(),
        }


        self.processes = {
            "ChunkStream": processes.FolderStream(saving_path, self.queues['chunk_stream']).make_process(),
            "Chunk": processes.QueueReducer(self.queues['chunk_stream'], self.queues['chunk'], max_size=1000).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'], self.queues['itof']).make_process(),
            "Clusterer": processes.DBSCANClusterer(self.queues['pixel'], self.queues['clusters']).make_process(),
            "Correlator": processes.TriggerAnalyzer(self.queues['pulses'], (self.queues['etof'], self.queues['itof'], self.queues['clusters']), self.queues['t_pulse'], (self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])).make_process(),
            "Grouper": processes.QueueGrouper((self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster']), self.queues['grouped']).make_process(),
            "Reducer": processes.QueueReducer(self.queues['grouped'], self.queues['reduced_grouped'], max_size=1000).make_process(),
            "Display": processes.Display(self.queues['reduced_grouped'], 1000000, toa_range=toa_range, etof_range=etof_range, itof_range=itof_range).make_process(),
            "Bin": processes.QueueVoid((self.queues['t_pulse'],)).make_process(),
        }
        self.processes["Reducer"].astep.name="r2"


        if cluster_processes > 1:
            queues, proc, weaver = processes.create_process_instances(
                processes.DBSCANClusterer, cluster_processes, self.queues["clusters"],
                process_args={"pixel_queue": self.queues['pixel'],"cluster_queue": None},
                queue_args={"force_monotone": True},
                queue_name="clust", process_name="clusterer")

            self.queues.update(queues)
            del self.processes["Clusterer"]
            self.processes.update({n: k.make_process() for n, k in proc.items()})
            self.processes["Weaver"] = weaver.make_process()


class RunMonitorPipeline(AnalysisPipeline):
    def __init__(self, saving_path, cluster_processes=1, toa_range=None, etof_range=None, itof_range=None, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(maxsize=1000),
            "pixel": data_types.ExtendedQueue(),
            "etof": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "itof": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "pulses": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),
            "clusters": data_types.ExtendedQueue(force_monotone=True, maxsize=50000),

            "t_etof": data_types.ExtendedQueue(),
            "t_itof": data_types.ExtendedQueue(),
            "t_pulse": data_types.ExtendedQueue(),
            "t_cluster": data_types.ExtendedQueue(),

            "grouped": data_types.ExtendedQueue(),
        }


        self.processes = {
            "ChunkStream": processes.TPXFileReader(saving_path, self.queues['chunk']).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'], self.queues['itof']).make_process(),
            "Clusterer": processes.DBSCANClusterer(self.queues['pixel'], self.queues['clusters']).make_process(),
            "Correlator": processes.TriggerAnalyzer(self.queues['pulses'], (self.queues['etof'], self.queues['itof'], self.queues['clusters']), self.queues['t_pulse'], (self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])).make_process(),
            "Grouper": processes.QueueGrouper((self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster']), self.queues['grouped']).make_process(),
            "Display": processes.Display(self.queues['grouped'], 10000000, toa_range=toa_range, etof_range=etof_range, itof_range=itof_range).make_process(),
            "Bin": processes.QueueVoid((self.queues['t_pulse'],)).make_process(),
        }

        if cluster_processes > 1:
            queues, proc, weaver = processes.create_process_instances(
                processes.DBSCANClusterer, cluster_processes, self.queues["clusters"],
                process_args={"pixel_queue": self.queues['pixel'],"cluster_queue": None},
                queue_args={"force_monotone": True},
                queue_name="clust", process_name="clusterer")

            self.queues.update(queues)
            del self.processes["Clusterer"]
            self.processes.update({n: k.make_process() for n, k in proc.items()})
            self.processes["Weaver"] = weaver.make_process()
