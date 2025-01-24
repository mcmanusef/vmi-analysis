import cProfile
import logging
import multiprocessing
import os
import signal
import threading
import time

from ..data_types import ExtendedQueue


class AnalysisStep:
    """
    Base class for all analysis steps. Analysis steps are the basic building blocks of the analysis pipeline.
    They are designed to be run in parallel and communicate with each other through queues. Each step can have
    multiple input and output queues.

    The action method is the main method that is called in the run loop. The run loop will continue to call the action method
    until the step is stopped. The run loop will also stop if all input queues are closed and empty, and the step is not holding.

    This class can be used as a base class for creating custom analysis steps. To create a custom analysis step, subclass
    this class and implement the start() and action() methods. The start() method should initialize the step, with any code that needs
    to be run within the process. The action() method should contain the main logic of the step, should be designed to run repeatedly,
    and should not block for longer than necessary.

    Holding should be used to indicate that the step is waiting for some external condition to be met before stopping. If holding is True,
    the step will not stop even if all input queues are closed and empty. Holding should be set to False when the external condition is met.
    For example, if a step is receiving data from an external source, holding should be set to True until the external source is finished.

    Attributes:
    - input_queues: A tuple of input queues for the step.
    - output_queues: A tuple of output queues for the step.

    - initialized: A multiprocessing boolean flag indicating whether the step has been initialized.
    - running: A multiprocessing boolean flag indicating whether the step is running.
    - stopped: A multiprocessing boolean flag indicating whether the step has been stopped.
    - holding: A multiprocessing boolean flag indicating whether the step is holding.

    - profile: A boolean flag indicating whether profiling is enabled for the step. If profiling is enabled, the step will
    generate a profile file with the name of the step when it is stopped. This can be useful for debugging performance issues.
    This flag should not be set when running in production, as it will slow down the step significantly.

    - any_queue: A boolean flag indicating whether the step should stop when any input queue is empty and closed, or when all
    input queues are empty and closed. If any_queue is True, the step will stop when any input queue is empty and closed.
    If any_queue is False, the step will stop when all input queues are empty and closed.

    - name: A string containing the name of the step. This is used for logging and debugging purposes.
    - logger: A logger object for the step. If not provided, a default logger will be used.

    Methods:
    - status(): Returns a dictionary containing the status of the step, including whether it is initialized, running, holding, and stopped.
    - initialize(): Initializes the step. This method should be called before starting the step. Code in this method will be run in the
    process and thread that the step will be run in. This method should be used to set up any resources that the step will need.

    - action(): The main method that is called in the run loop. This method should contain the main logic of the step. It should be designed not
    to block for longer than necessary, as this will slow down the step and the pipeline. This method should be implemented in subclasses.

    - begin(): Starts the main loop of the step. This method should be called after the step has been initialized.

    - shutdown(gentle=False): Stops the step. This method should be called when the step is finished. Gentle shutdown indicates that the step
    was stopped due to input queues, which may change how the step is stopped. This method should be implemented in subclasses if necessary.

    - run_loop(): The main loop that runs the step. This method should not be called directly, as it is called by the process or thread that
    the step is run in. This method will call the action method repeatedly until the step is stopped. It will also stop the step if all input
    queues are closed and empty, and the step is not holding.

    - make_process(**kwargs): Creates an AnalysisProcess object for the step, which can be used to run the step in a separate process.
    - make_thread(**kwargs): Creates an AnalysisThread object for the step, which can be used to run the step in a separate thread within the same process.

    """
    def __init__(self, input_queues=(), output_queues=(), profile=False, pipeline_active=None, logger=None, **kwargs):
        self.input_queues: tuple[ExtendedQueue, ...] = input_queues
        self.output_queues: tuple[ExtendedQueue, ...] = output_queues
        self.initialized = multiprocessing.Value('b', False)
        self.running = multiprocessing.Value('b', False)
        self.stopped = multiprocessing.Value('b', False)
        self.holding = multiprocessing.Value('b', False)
        self.profile = profile
        self.any_queue = True
        self.name = ""
        self.pipeline_active = pipeline_active
        self.logger=logger if logger else logging.getLogger()
        self.last_check = time.time()
        self.check_interval = 5

    def status(self):
        return {
            "initialized": bool(self.initialized.value),
            "running": bool(self.running.value),
            "holding": self.is_holding,
            "stopped": bool(self.stopped.value)
        }

    def initialize(self):
        self.initialized.value = True

    def action(self):
        raise NotImplemented

    def begin(self):
        self.running.value = True

    def shutdown(self, gentle=False):
        [q.close() for q in self.output_queues]
        self.running.value = False
        self.holding.value = False
        self.stopped.value = True

    def run_loop(self):
        while not self.running.value and not self.stopped.value:
            time.sleep(0.1)
        if self.profile:
            pr = cProfile.Profile()

        while True:
            if self.profile:
                pr.enable()

            self.action()

            if self.profile:
                pr.disable()

            if self.check_queues():
                if self.profile:
                    pr.dump_stats(f"{self.name}.prof")
                self.shutdown(gentle=True)
                return

    @property
    def is_holding(self):
        return bool(self.holding.value)

    def make_process(self, **kwargs):
        return AnalysisProcess(self, **kwargs)

    def make_thread(self, **kwargs):
        return AnalysisThread(self, **kwargs)

    def check_queues(self):
        if time.time() - self.last_check > self.check_interval:
            self.last_check = time.time()
            return (self.any_queue and (all(q.closed.value and q.empty() for q in self.input_queues))
                or (not self.any_queue and any(q.closed.value and q.empty() for q in self.input_queues))
            ) and not self.is_holding
        return False


class AnalysisProcess(multiprocessing.Process):

    """
    A wrapper around the AnalysisStep class that allows the step to be run in a separate process. This class inherits from the
    multiprocessing.Process class, and overrides the run method to call the run_loop method of the step. This class should be used
    to run analysis steps in parallel in separate processes.

    Attributes:
    - astep: The AnalysisStep object that the process is running.
    - name: A string containing the name of the process. This is set to the name of the step.

    Methods:
    - status(): Returns a dictionary containing the status of the step, including whether it is initialized, running, holding, and stopped.
    - initialize(): Initializes the process by calling the initialize method of the step.
    - begin(): Begins the process by calling the begin method of the step.
    - shutdown(): Shuts down the process by calling the shutdown method of the step.
    - is_holding(): Returns True if the step is holding, False otherwise.
    - initialized(): Returns True if the step is initialized, False otherwise.
    - running(): Returns True if the step is running, False otherwise.
    - stopped(): Returns True if the step is stopped, False otherwise.
    - self_shutdown(): Shuts down the process when a shutdown signal is received, by calling the shutdown method of the step and then killing the process.
    - run(): The main method that is called when the process is started. This method calls the initialize method of the step, and then enters
    a loop that calls the run_loop method of the step until the step is stopped.
    """

    def __init__(self, astep, **kwargs):
        self.astep = astep
        multiprocessing.Process.__init__(self, **kwargs)
        self.name = astep.name

    def status(self):
        return self.astep.status()

    def initialize(self):
        print(f"Initializing Process: {self.name}")
        signal.signal(signal.SIGINT, self.self_shutdown)
        self.astep.initialize()

    def begin(self):
        self.astep.begin()

    @property
    def is_holding(self):
        return self.astep.is_holding

    @property
    def initialized(self):
        return self.astep.initialized

    @property
    def running(self):
        return self.astep.running

    @property
    def stopped(self):
        return self.astep.stopped

    def self_shutdown(self, *args, **kwargs):
        print(f"{self.name} received shutdown signal")
        try:
            self.shutdown()
            print(f"{self.name} shut down")
        finally:
            os.kill(os.getpid(), signal.SIGTERM)

    def shutdown(self):
        self.astep.shutdown()

    def run(self):
        self.initialize()
        while not self.astep.initialized.value:
            time.sleep(0.1)
        while not self.astep.stopped.value:
            self.astep.run_loop()
        print(f"{self.name} Finished")


class AnalysisThread(threading.Thread):
    """
    A wrapper around the AnalysisStep class that allows the step to be run in a separate thread. This class inherits from the
    threading.Thread class, and overrides the run method to call the run_loop method of the step. This class should be used
    to run analysis steps in parallel in separate threads.

    Currently not well tested and may not work as expected. Not recommended for use.
    """
    def __init__(self, astep, **kwargs):
        self.astep = astep
        threading.Thread.__init__(self, **kwargs)
        self.name = astep.name

    def status(self):
        return self.astep.status()

    def initialize(self):
        self.astep.initialize()

    def begin(self):
        self.astep.begin()

    @property
    def is_holding(self):
        return self.astep.is_holding

    @property
    def initialized(self):
        return self.astep.initialized

    @property
    def running(self):
        return self.astep.running

    @property
    def stopped(self):
        return self.astep.stopped

    def shutdown(self):
        self.astep.shutdown()

    def run(self):
        self.astep.initialize()
        while not self.astep.initialized.value:
            time.sleep(0.1)

        while not self.astep.stopped.value:
            self.astep.run_loop()

        logging.info(f"{self.name} Finished")


class CombinedStep(AnalysisStep):
    def __init__(self,
                 steps: tuple[AnalysisStep, ...] = (),
                 input_queues: tuple[ExtendedQueue, ...] = (),
                 output_queues: tuple[ExtendedQueue, ...] = (),
                 intermediate_queues: tuple[ExtendedQueue, ...] = (),
                 name="Combined", **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.threads = []
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.intermediate_queues = intermediate_queues
        self.holding.value = any(p.is_holding for p in self.steps)
        self.monitor_thread = None
        self.name = name

    def initialize(self):
        print(f"Initializing Combined Process: {len(self.steps)} steps")
        self.monitor_thread = threading.Thread(target=self.monitor, name="Monitor")
        if self.profile:
            for step in self.steps:
                step.profile = True
        self.threads = [step.make_thread() for step in self.steps]
        [iq.bind_to_process() for iq in self.intermediate_queues]
        for p in self.threads:
            p.start()
            while not p.initialized.value:
                time.sleep(0.1)
            print(f"Initialized {p.name}: {p.status()}")
        self.monitor_thread.start()
        super().initialize()

    def begin(self):
        for p in self.steps:
            p.begin()
        super().begin()

    def shutdown(self, gentle=False):
        if gentle and not all(p.stopped.value for p in self.threads):
            return

        for p in self.steps:
            if p.stopped.value:
                continue
            print(f"Shutting down {p.name}")
            p.shutdown()
            while not p.stopped.value:
                time.sleep(0.1)
            print(f"Shut down {p.name}")
        super().shutdown()

    def monitor(self):
        while not self.stopped.value:
            if all(p.stopped.value for p in self.threads):
                self.shutdown()
                return
            if not any(p.is_holding for p in self.threads):
                self.holding.value = False
            time.sleep(1)
