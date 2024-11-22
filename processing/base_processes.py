import cProfile
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time

from processing.data_types import ExtendedQueue


class AnalysisStep:
    def __init__(self, input_queues=(), output_queues=(), profile=False, pipeline_active=None, **kwargs):
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
        pass

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

        while self.running.value:

            if not self.pipeline_active.value:
                self.shutdown()
                return

            try:

                if self.profile:
                    pr.enable()

                self.action()

                if self.profile:
                    pr.disable()
            except Exception as e:
                print(f"Error in {self.name}: {e}")
                print(f"Shutting down {self.name}")

                if self.profile:
                    pr.dump_stats(f"{self.name}.prof")

                self.shutdown()
                self.pipeline_active.value = False
                raise e

            if ((all(q.closed.value and q.empty() for q in self.input_queues) and self.any_queue)
                or (any(q.closed.value and q.empty() for q in self.input_queues) and not self.any_queue)
            ) and not self.is_holding:

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


class AnalysisProcess(multiprocessing.Process):
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
