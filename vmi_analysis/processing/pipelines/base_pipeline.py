import multiprocessing
import time
from .. import data_types, processes
import logging


class AnalysisPipeline:
    """
    A class to represent an analysis pipeline. This class is responsible for managing the processes and queues that
    make up the pipeline, and for starting and stopping the pipeline.

    To use this class, create a new instance and add processes and queues to it. Then call the initialize() method to
    initialize the pipeline, the start() method to start it, the wait_for_completion() method to wait for all processes to finish,
    the pipeline, and the stop() method to stop it. The pipeline can also be used as a context manager, in which case it
    will automatically be initialized when entering the context and stopped when exiting it.

    Attributes:
    - queues: A dictionary of queues used by the pipeline. The keys are the names of the queues and the values are the queues themselves.
    - processes: A dictionary of processes used by the pipeline. The keys are the names of the processes and the values are the processes themselves.
    - initialized: A boolean indicating whether the pipeline has been initialized.
    - profile: A boolean indicating whether profiling is enabled for the pipeline.

    Methods:
    - set_profile(profile: bool): Enables or disables profiling for the pipeline, propagating the change to all processes.
    - initialize(): Initializes the pipeline by calling the start() method of all processes.
    - start(): Starts the pipeline by calling the begin() method of all processes.
    - stop(): Stops the pipeline by calling the shutdown() method of all processes.
    - is_running(): Returns True if any of the processes in the pipeline are running, False otherwise.
    - wait_for_completion(): Waits for all processes in the pipeline to finish.
    """

    queues: dict[str, data_types.ExtendedQueue]
    processes: dict[str, processes.AnalysisProcess]
    initialized: bool
    profile: bool

    def __init__(self, **kwargs):
        self.active = multiprocessing.Value("b", True)

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
        [
            logging.debug(f"{n} - {p.status()}\n\t{p.exitcode}")
            for n, p in self.processes.items()
        ]
        for name, process in self.processes.items():
            print(f"Process {name} status: {process.status()}")

            if not process.stopped.value:
                logging.debug(f"Stopping process {name}")
                process.shutdown()
            else:
                logging.debug(f"Process {name} already stopped")
            print(f"Process {name} stopped, joining")
            try:
                process.join(timeout=5)
                print(f"Process {name} joined")
            except TimeoutError:
                logging.error(f"Process {name} failed to join")
                process.terminate()
                process.join()
                logging.error(f"Process {name} terminated")

    def is_running(self):
        return any([p.status()["running"] for p in self.processes.values()])

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

        while (
            not all(p.astep.stopped.value for p in target_pipeline.processes.values())
            or forever
        ):
            for name, process in target_pipeline.processes.items():
                print(f"{name} status: {process.status()}")
                for qname, q in target_pipeline.queues.items():
                    if q in process.astep.output_queues:
                        print(
                            f"\t{qname} ({'Closed' if q.closed.value else 'Open'}) queue size: {q.qsize()} (internal: {q.queue.qsize()})"
                        )
            time.sleep(1)
            print("\n")
