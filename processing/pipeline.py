import processes
import data_types
import logging
import time
import multiprocessing

import base_processes

logger = logging.getLogger()


class AnalysisPipeline:
    queues: dict[str, data_types.ExtendedQueue]
    processes: dict[str, base_processes.AnalysisProcess]
    initialized: bool
    profile: bool

    def __init__(self):
        self.active=multiprocessing.Value('b', True)

    def set_profile(self, profile: bool):
        for process in self.processes.values():
            process.astep.profile = profile
        return self

    def initialize(self):
        logger.info("Initializing pipeline")
        for process in self.processes.values():
            process.astep.pipeline_active = self.active
        for name, process in self.processes.items():
            logger.debug(f"Initializing process {name}")
            process.start()
            while not process.initialized.value:
                time.sleep(0.1)
            logger.debug(f"Process {name} initialized")
        self.initialized = True

    def start(self):
        logger.info("Starting pipeline")
        if not self.initialized:
            self.initialize()
        for name, process in self.processes.items():
            logger.debug(f"Starting process {name}")
            process.begin()
            logger.debug(f"Process {name} started")

    def stop(self):
        self.active.value=False
        logger.info("Stopping pipeline")
        for name, process in self.processes.items():
            if not process.stopped.value:
                logger.debug(f"Stopping process {name}")
                process.shutdown()
            else:
                logger.debug(f"Process {name} already stopped")
            process.join()

    def wait_for_completion(self):
        while any([p.is_holding for p in self.processes.values()]):
            time.sleep(0.1)

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False


class TPXFileConverter(AnalysisPipeline):
    def __init__(self, input_path: str, output_path: str, buffer_size: int = 0, single_process=False, **kwargs):
        super().__init__(**kwargs)
        if not single_process:
            self.queues = {
                'chunk': data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                'pixel': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i', 'i', 'i'),
                                                  names=('toa', 'x', 'y', 'tot'), chunk_size=10000),
                'tdc': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i'),
                                                names=('tdc_time', 'tdc_type'), chunk_size=10000),
            }

            self.processes = {
                'reader': processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
                'converter': processes.TPXConverter(self.queues['chunk'], self.queues['pixel'],
                                                    self.queues['tdc']).make_process(),
                'save': processes.SaveToH5(output_path,
                                           {"pixel": self.queues['pixel'], "tdc": self.queues['tdc']}).make_process(),
            }

        if single_process:
            self.queues: dict[str, data_types.ExtendedQueue] = {
                'chunk': data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                'pixel': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i', 'i', 'i'),
                                                  names=('toa', 'x', 'y', 'tot')),
                'tdc': data_types.ExtendedQueue(buffer_size=buffer_size, dtypes=('f', 'i'),
                                                names=('tdc_time', 'tdc_type')),
            }

            self.processes = {
                "Combined": base_processes.CombinedStep(steps=(
                    processes.TPXFileReader(input_path, chunk_queue=self.queues['chunk']),
                    processes.TPXConverter(chunk_queue=self.queues['chunk'],
                                           pixel_queue=self.queues['pixel'],
                                           tdc_queue=self.queues['tdc']),
                    processes.SaveToH5(output_path,
                                       {"pixel": self.queues['pixel'], "tdc": self.queues['tdc']}),
                ), intermediate_queues=(self.queues["pixel"], self.queues['tdc']),
                        output_queues=(self.queues["chunk"],),
                ).make_process(),
            }


class RawVMIConverterPipeline(AnalysisPipeline):
    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'i', 'i', 'i'), names=('toa', 'x', 'y', 'tot'), unwrap=True, force_monotone=True, max_back=1e9, chunk_size=10000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('etof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('itof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('pulses',), force_monotone=True, max_back=1e9, chunk_size=10000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'], self.queues['itof']).make_process(),
            "Saver": processes.SaveToH5(output_path, {
                "pixel": self.queues['pixel'],
                "etof": self.queues['etof'],
                "itof": self.queues['itof'],
                "pulses": self.queues['pulses'],
            }).make_process(),
        }


class VMIConverterPipeline(AnalysisPipeline):
    def __init__(self, input_path, output_path, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'i', 'i', 'i'), names=('toa', 'x', 'y', 'tot'), unwrap=True, force_monotone=True, max_back=1e9, chunk_size=10000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('etof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('itof',), force_monotone=True, max_back=1e9, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('pulses',), force_monotone=True, max_back=1e9, chunk_size=10000),

            "t_etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('etof_corr', ('t_etof',)), chunk_size=2000),
            "t_itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('itof_corr', ('t_itof',)), chunk_size=2000),
            "t_pulse": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('t_pulse',), chunk_size=10000),
            "t_pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f', 'i', 'i', 'i')), names=('pixel_corr', ('t', 'x', 'y', 'tot')), chunk_size=10000),
            }


        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),
            "Converter": processes.VMIConverter(self.queues['chunk'], self.queues['pixel'], self.queues['pulses'], self.queues['etof'], self.queues['itof']).make_process(),
            "Correlator": processes.TriggerAnalyzer(self.queues['pulses'],
                                                    (self.queues['etof'], self.queues['itof'], self.queues['pixel']),
                                                    self.queues['t_pulse'],
                                                    (self.queues['t_etof'], self.queues['t_itof'], self.queues['t_pixel'])
                                                    ).make_process(),
            "Saver": processes.SaveToH5(output_path, {
                "pixel": self.queues['t_pixel'],
                "etof": self.queues['t_etof'],
                "itof": self.queues['t_itof'],
                "pulses": self.queues['t_pulse'],
            }).make_process(),
        }


class ClusterSavePipeline(AnalysisPipeline):
    def __init__(self, input_path, output_path, monotone=False, **kwargs):
        super().__init__(**kwargs)
        self.queues = {
            "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("etof",), force_monotone=monotone, chunk_size=2000),
            "itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("itof",), force_monotone=monotone, chunk_size=2000),
            "pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("pulses",), force_monotone=monotone, chunk_size=10000),
            "clusters": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'f', 'f'), names=("toa", "x", "y"), force_monotone=monotone, chunk_size=2000),
            "clustered": data_types.ExtendedQueue(buffer_size=0, dtypes=(('f','i','i','i'),'i'),
                                                  names=(('toa_pix','x_pix','y_pix','tot_pix'),'cluster_pix'),
                                                  force_monotone=monotone, chunk_size=2000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['chunk']).make_process(),

            "Converter": processes.VMIConverter(self.queues['chunk'],
                                                self.queues['pixel'],
                                                self.queues['pulses'],
                                                self.queues['etof'],
                                                self.queues['itof']).make_process(),

            "Clusterer": processes.DBSCANClusterer(pixel_queue=self.queues['pixel'],
                                                   cluster_queue=self.queues['clusters'],
                                                   output_pixel_queue=self.queues['clustered']).make_process(),

            "Saver": processes.SaveToH5(output_path, {
                "clusters": self.queues['clusters'],
                "etof": self.queues['etof'],
                "itof": self.queues['itof'],
                "pulses": self.queues['pulses'],
                "clustered": self.queues['clustered'],
            }).make_process(),
        }


class CV4ConverterPipeline(AnalysisPipeline):
    def __init__(self, input_path, output_path, save_pixels=False,**kwargs):
        super().__init__(**kwargs)

        self.queues = {
            "Chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "Pixel": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=(), chunk_size=2000),
            "Etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("etof",), force_monotone=True, chunk_size=2000),
            "Itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("itof",), force_monotone=True, chunk_size=2000),
            "Pulses": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=("pulses",), force_monotone=True, chunk_size=10000),
            "Clusters": data_types.ExtendedQueue(buffer_size=0, dtypes=('f', 'f', 'f'), names=("toa", "x", "y"), force_monotone=True, chunk_size=2000),

            "t_etof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('etof_corr', ("t_etof",)), chunk_size=2000),
            "t_itof": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f',)), names=('tof_corr', ("t_tof",)), chunk_size=2000),
            "t_pulse": data_types.ExtendedQueue(buffer_size=0, dtypes=('f',), names=('t_pulse',), chunk_size=10000),
            "t_cluster": data_types.ExtendedQueue(buffer_size=0, dtypes=('i', ('f', 'f', 'f')), names=('cluster_corr', ("t", "x", "y")), chunk_size=2000),
        }

        self.processes = {
            "Reader": processes.TPXFileReader(input_path, self.queues['Chunk']).make_process(),

            "Converter": processes.VMIConverter(
                    chunk_queue=self.queues['Chunk'],
                    pixel_queue=self.queues['Pixel'],
                    laser_queue=self.queues['Pulses'],
                    etof_queue=self.queues['Etof'],
                    itof_queue=self.queues['Itof']
            ).make_process(),

            "Clusterer": processes.DBSCANClusterer(
                    pixel_queue=self.queues['Pixel'],
                    cluster_queue=self.queues['Clusters']
            ).make_process(),

            "Correlator": processes.TriggerAnalyzer(
                    input_trigger_queue=self.queues['Pulses'],
                    queues_to_index=(self.queues['Etof'], self.queues['Itof'], self.queues['Clusters']),
                    output_trigger_queue=self.queues['t_pulse'],
                    indexed_queues=(self.queues['t_etof'], self.queues['t_itof'], self.queues['t_cluster'])
            ).make_process(),

            "Saver": processes.SaveToH5(output_path, {
                "t_etof": self.queues['t_etof'],
                "t_itof": self.queues['t_itof'],
                "t_pulse": self.queues['t_pulse'],
                "t_cluster": self.queues['t_cluster']
            }).make_process(),

        }
        if save_pixels:
            self.queues["pixels"]=data_types.ExtendedQueue(buffer_size=0, dtypes=(('f', 'i', 'i', 'i'),'i'), names=(('toa', 'x', 'y', 'tot'), 'cluster_index'), force_monotone=True, max_back=1e9, chunk_size=10000)
            self.queues["t_pixel"]=data_types.ExtendedQueue(buffer_size=0, dtypes=('i', (('f', 'i', 'i', 'i'), 'i')), names=('pixel_corr', (('t', 'x', 'y', 'tot'), 'cluster_index')), chunk_size=10000)

            self.processes["Saver"].astep.in_queues['pixels'] = self.queues["t_pixel"]
            self.processes["Saver"].astep.input_queues+=(self.queues["t_pixel"],)
            self.processes["Saver"].astep.flat={
                "t_etof": True,
                "t_itof": True,
                "t_pulse": True,
                "t_cluster": True,
                "pixels": False,
            }
            self.processes["Correlator"].astep.output_queues+=(self.queues["t_pixel"],)
            self.processes["Correlator"].astep.indexed_queues+=(self.queues["t_pixel"],)
            self.processes["Correlator"].astep.input_queues+=(self.queues["pixels"],)
            self.processes["Correlator"].astep.queues_to_index+=(self.queues["pixels"],)
            self.processes["Correlator"].astep.current.append(None)
            self.processes["Correlator"].astep.current_samples.append(None)
            self.processes["Clusterer"].astep.output_pixel_queue=self.queues["pixels"]



def run_pipeline(pipeline: AnalysisPipeline, forever=False):
    print("Initializing pipeline")
    with pipeline:
        for name, process in pipeline.processes.items():
            print(f"{name} initialized correctly: {process.initialized.value}")
        print("Starting pipeline")
        pipeline.start()
        for name, process in pipeline.processes.items():
            print(f"{name} running: {process.running.value}")

        while not all(p.astep.stopped.value for p in pipeline.processes.values()) or forever:
            for name, process in pipeline.processes.items():
                print(f"{name} status: {process.status()}")
                for qname, q in pipeline.queues.items():
                    if q in process.astep.output_queues:
                        print(f"\t{qname} ({'Closed' if q.closed.value else 'Open'}) queue size: {q.qsize()}")
            time.sleep(1)
            print("\n")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s:   %(message)s', level=logging.DEBUG)
    fname=r"D:\Data\xe_03_Scan3W\xe_000000.tpx3"
    # pipeline = ClusterSavePipeline(input_path=r"D:\Data\xe002_s\xe000000.tpx3", output_path="test.h5", monotone=True)
    # pipeline = TPXFileConverter(input_path=r"D:\Data\xe002_s\xe000000.tpx3", output_path="test.h5")
    #                             single_process=False).set_profile(True)
    pipeline = CV4ConverterPipeline(input_path=fname, output_path="test.h5", save_pixels=True)
    # pipeline=VMIConverterPipeline(input_path=r"D:\Data\xe002_s\xe000000.tpx3", output_path="test.h5")
    start = time.time()
    run_pipeline(pipeline)
    print(f"Time taken: {time.time() - start}")
# %%
