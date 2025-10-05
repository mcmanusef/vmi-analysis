import queue

import numpy as np

from vmi_analysis.processing.data_types import (
    Queue,
    Chunk,
    PixelData,
    Trigger,
    Timestamp,
)
from vmi_analysis.processing.processes import AnalysisStep
from vmi_analysis.processing.tpx_conversion import (
    process_chunk,
    apply_timewalk,
    toa_correction,
)


def sort_tdcs(tdcs):
    pulses = []
    for tdc_type, c_time, ftime, _ in tdcs:
        tdc_time = 3.125 * c_time + 0.260 * ftime
        if tdc_type == 10:
            pulses.append(tdc_time)
    return pulses


class SBVMIConverter(AnalysisStep):
    """
    Converts binary data from the VMI into PixelData, TriggerTime, electron ToF, and ion ToF, and puts them into the appropriate queues.
    Experiment specific, but can be used as a template for other experiments

    Parameters:
    - chunk_queue: The queue containing the binary data chunks
    - pixel_queue: The queue to put PixelData into (Chunked [[(time, x, y, tot), ...], ...])
    - laser_queue: The queue to put TriggerTime into (Unchunked [time, ...])
    - etof_queue: The queue to put electron ToF into (Unchunked [time, ...])
    - itof_queue: The queue to put ion ToF into (Unchunked [time, ...])

    - cutoff: The cutoff for distinguishing between ion tof and laser pulses, where a TDC1 event with length greater than the cutoff
    is considered a laser pulse, and a TDC1 event with length less than the cutoff is considered an ion pulse.

    - timewalk_file: The file containing the timewalk correction data. Not well tested, but should work.
    - toa_corr: The time of arrival correction to apply to the data. Specific to our experiment, as there is an area of
    artificially high toa values that need to be corrected

    - kwargs: Additional keyword arguments to pass to the AnalysisStep constructor
    """

    cutoff: float
    chunk_queue: Queue[Chunk]
    pixel_queue: Queue[list[PixelData]]
    laser_queue: Queue[Trigger]

    def __init__(
            self,
            chunk_queue: Queue[Chunk],
            pixel_queue: Queue[list[PixelData]],
            laser_queue: Queue[Trigger],
            timewalk_file=None,
            toa_corr=25,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_queue = chunk_queue
        self.pixel_queue = pixel_queue
        self.laser_queue = laser_queue
        self.output_queues = (pixel_queue, laser_queue)
        self.input_queues = (chunk_queue,)
        self.timewalk_file = timewalk_file
        self.timewalk_correction = None
        self.toa_correction = toa_corr
        self.name = "VMIConverter"

    def initialize(self):
        if self.timewalk_file:
            self.timewalk_correction = np.loadtxt(self.timewalk_file)
        super().initialize()

    def action(self):
        try:
            chunk = self.chunk_queue.get(timeout=1)
        except queue.Empty or InterruptedError:
            return
        pixels, tdcs = process_chunk(chunk)

        if pixels:
            if self.timewalk_correction is not None:
                pixels = apply_timewalk(pixels, self.timewalk_correction)
            if self.toa_correction:
                pixels = toa_correction(pixels, self.toa_correction)

        self.pixel_queue.put([PixelData(time=pix[0], x=pix[1], y=pix[2], tot=pix[3]) for pix in pixels]) if pixels else None

        pulses = sort_tdcs(tdcs)

        for t in pulses:
            self.laser_queue.put(Timestamp(t))
