import multiprocessing
import time
from .. import data_types, processes
import logging


class AnalysisPipeline:
    queues: dict[str, data_types.ExtendedQueue]
    processes: dict[str, processes.AnalysisProcess]
    initialized: bool
    profile: bool

    def __init__(self, **kwargs):
        self.active = multiprocessing.Value('b', True)

    def set_profile(self, profile: bool):
        for process in self.processes.values():
            process.astep.profile = profile
        return self

    def initialize(self):
        logging.info("Initializing pipeline")
        for process in self.processes.values():
            process.astep.pipeline_active = self.active
        for name, process in self.processes.items():
            logging.debug(f"Initializing process {name}")
            process.start()
        for name, process in self.processes.items():
            while not process.initialized.value:
                time.sleep(0.1)
            logging.debug(f"Process {name} initialized")
        self.initialized = True

    def start(self):
        logging.info("Starting pipeline")
        if not self.initialized:
            self.initialize()
        for name, process in self.processes.items():
            logging.debug(f"Starting process {name}")
            process.begin()
            logging.debug(f"Process {name} started")

    def stop(self):
        self.active.value = False
        logging.info("Stopping pipeline")
        logging.debug("Process Status:")
        [logging.debug(f"{n} - {p.status()}") for n, p in self.processes.items()]
        [logging.debug(f"{n} - {p.status()}\n\t{p.exitcode}") for n, p in self.processes.items()]
        for name, process in self.processes.items():

            if not process.stopped.value:
                logging.debug(f"Stopping process {name}")
                process.shutdown()
            else:
                logging.debug(f"Process {name} already stopped")
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


def run_pipeline(target_pipeline: AnalysisPipeline, forever=False):
    print("Initializing pipeline")
    with target_pipeline:
        for name, process in target_pipeline.processes.items():
            print(f"{name} initialized correctly: {process.initialized.value}")
        print("Starting pipeline")
        target_pipeline.start()
        for name, process in target_pipeline.processes.items():
            print(f"{name} running: {process.running.value}")

        while not all(p.astep.stopped.value for p in target_pipeline.processes.values()) or forever:
            for name, process in target_pipeline.processes.items():
                print(f"{name} status: {process.status()}")
                for qname, q in target_pipeline.queues.items():
                    if q in process.astep.output_queues:
                        print(f"\t{qname} ({'Closed' if q.closed.value else 'Open'}) queue size: {q.qsize()} (internal: {q.queue.qsize()})")
            time.sleep(1)
            print("\n")
