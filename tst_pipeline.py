import time

import matplotlib.pyplot as plt

from vmi_analysis.processing import pipelines
import logging

if __name__ == "__main__":
    pipeline = pipelines.LiveMonitorPipeline()
    logger = logging.getLogger()

    with pipeline:
        print("Pipeline Initialized")
        for process in pipeline.processes.values():
            print(process.status())
        pipeline.start()
        print("Pipeline Started")
        time.sleep(30)
    print("Pipeline Exited")
    plt.figure()
    plt.imshow(pipeline.acc_frame)
    plt.show()
