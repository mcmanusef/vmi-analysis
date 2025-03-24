import collections
import queue
import time

import numpy as np
from matplotlib import pyplot as plt

from vmi_analysis.processing.data_types import (
    StructuredDataQueue,
    Queue,
    Chunk,
    PixelData,
    TriggerTime,
    ToF,
)
from vmi_analysis.processing.processes import AnalysisStep
from vmi_analysis.processing.tpx_conversion import (
    process_chunk,
    apply_timewalk,
    toa_correction,
    sort_tdcs,
)


class VMIConverter(AnalysisStep):
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

    def __init__(
        self,
        chunk_queue: Queue[Chunk],
        pixel_queue: Queue[list[PixelData]],
        laser_queue: StructuredDataQueue[TriggerTime],
        etof_queue: StructuredDataQueue[ToF],
        itof_queue: StructuredDataQueue[ToF],
        cutoff: float = 300.0,
        timewalk_file=None,
        toa_corr=25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_queue = chunk_queue
        self.pixel_queue = pixel_queue
        self.laser_queue = laser_queue
        self.etof_queue = etof_queue
        self.itof_queue = itof_queue
        self.cutoff = cutoff
        self.output_queues = (pixel_queue, laser_queue, etof_queue, itof_queue)
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
        # pixels = [(toa * PIXEL_RES, x, y, tot) for toa, x, y, tot in pixels]

        if pixels:
            if self.timewalk_correction is not None:
                pixels = apply_timewalk(pixels, self.timewalk_correction)
            if self.toa_correction:
                pixels = toa_correction(pixels, self.toa_correction)

        self.pixel_queue.put(pixels) if pixels else None

        etof, itof, pulses = sort_tdcs(self.cutoff, tdcs) if tdcs else ([], [], [])
        for t in etof:
            self.etof_queue.put(t)
        for t in itof:
            self.itof_queue.put(t)
        for t in pulses:
            self.laser_queue.put(t)


class Display(AnalysisStep):
    def __init__(
        self,
        grouped_queue,
        n,
        toa_range=None,
        etof_range=None,
        itof_range=None,
        angle=0,
        center=(128, 128),
        calibration=None,
    ):
        super().__init__()
        self.input_queues = (grouped_queue,)
        self.grouped_queue = grouped_queue
        self.name = "Display"
        self.current_data = {
            "etof": collections.deque(maxlen=n),
            "itof": collections.deque(maxlen=n),
            "cluster": collections.deque(maxlen=n),
            "timestamps": collections.deque(maxlen=n),
        }
        self.figure = None
        self.ax = None
        self.xy_hist = None
        self.xy_hist_data = np.zeros((256, 256))
        self.toa_hist = None
        self.toa_hist_data = np.zeros(2000)
        self.etof_hist = None
        self.etof_hist_data = np.zeros(2000)
        self.itof_hist = None
        self.itof_hist_data = np.zeros(400)
        self.update_interval = 1
        self.last_update = 0
        self.toa_range = toa_range
        self.etof_range = etof_range
        self.itof_range = itof_range
        self.angle = angle
        self.center = center
        self.calibration = calibration
        self.calibrated_pol_plane = None
        self.uncalibrated_pol_plane = None
        self.calibrated_pol_plane_data = np.zeros((256, 256))
        self.uncalibrated_pol_plane_data = np.zeros((256, 256))

    def initialize(self):
        self.figure, self.ax = plt.subplots(2, 3)
        self.xy_hist = self.ax[0, 0].imshow(
            np.random.random(size=(256, 256)),
            extent=[0, 256, 0, 256],
            origin="lower",
            aspect="equal",
            cmap="jet",
            interpolation="nearest",
        )
        self.ax[0, 0].set_title("XY")
        self.toa_hist = self.ax[0, 1].plot(
            np.linspace(0, 2000, 2000), np.linspace(0, 1, num=2000)
        )[0]
        plt.sca(self.ax[0, 1])
        plt.xlim(0, 2000)
        plt.xlabel("ToA (ns)")

        if self.toa_range is not None:
            plt.axvline(self.toa_range[0], color="r", linestyle="--")
            plt.axvline(self.toa_range[1], color="r", linestyle="--")

        self.ax[0, 1].set_title("ToA")
        self.etof_hist = self.ax[1, 0].plot(
            np.linspace(0, 2000, 2000), np.linspace(0, 1, num=2000)
        )[0]
        plt.sca(self.ax[1, 0])
        plt.xlim(0, 2000)
        plt.xlabel("ToF (ns)")
        if self.etof_range is not None:
            plt.axvline(self.etof_range[0], color="r", linestyle="--")
            plt.axvline(self.etof_range[1], color="r", linestyle="--")

        self.ax[1, 0].set_title("eToF")
        self.itof_hist = self.ax[1, 1].plot(
            np.linspace(0, 2e4, 400), np.linspace(0, 1, num=400)
        )[0]
        self.ax[1, 1].set_title("iToF")
        plt.sca(self.ax[1, 1])
        plt.xlim(0, 2e4)
        plt.xlabel("ToF (ns)")
        if self.itof_range is not None:
            plt.axvline(self.itof_range[0], color="r", linestyle="--")
            plt.axvline(self.itof_range[1], color="r", linestyle="--")

        self.ax[0, 2].set_title("Polarization Plane (Uncalibrated)")
        self.uncalibrated_pol_plane = self.ax[0, 2].imshow(
            np.random.random(size=(256, 256)),
            extent=[0, 2000, 0, 256],
            origin="lower",
            aspect="auto",
            cmap="jet",
            interpolation="nearest",
        )
        plt.sca(self.ax[0, 2])
        # plt.xlabel("x")
        # plt.ylabel("t")
        #
        self.ax[1, 2].set_title("Polarization Plane (Calibrated)")
        self.calibrated_pol_plane = self.ax[1, 2].imshow(
            np.random.random(size=(256, 256)),
            extent=[-0.6, 0.6, -0.6, 0.6],
            origin="lower",
            aspect="equal",
            cmap="jet",
            interpolation="nearest",
        )
        plt.sca(self.ax[1, 2])
        plt.xlabel("px")
        plt.ylabel("pz")

        plt.suptitle("Processing Rate:")
        self.figure.tight_layout()
        self.figure.show()
        super().initialize()

    def action(self):
        try:
            data = self.grouped_queue.get(timeout=0.1)
            # print(data)
        except queue.Empty or InterruptedError:
            return

        if len(self.current_data["etof"]) == self.current_data["etof"].maxlen:
            etof_rem = self.current_data["etof"].popleft()
            itof_rem = self.current_data["itof"].popleft()
            t_rem, x_rem, y_rem = (
                zip(*last)
                if (last := self.current_data["cluster"].popleft())
                else ([], [], [])
            )
            last_timestamp = self.current_data["timestamps"].popleft()

            for t, x, y in zip(t_rem, x_rem, y_rem):
                if self.toa_range is None or (
                    self.toa_range[0] < t < self.toa_range[1]
                ):
                    self.xy_hist_data[int(x), int(y)] -= 1
            for t in t_rem:
                if 0 < t < 2e4:
                    self.toa_hist_data[int(t / 10)] -= 1
            for t in etof_rem:
                if 0 < t < 2e4:
                    self.etof_hist_data[int(t / 10)] -= 1
            for t in itof_rem:
                if 0 < t < 2e4:
                    self.itof_hist_data[int(t / 50)] -= 1

            if len(x_rem) == 1 and len(etof_rem) == 1:
                if 0 < etof_rem[0] < 2000:
                    self.uncalibrated_pol_plane_data[
                        int(x_rem[0]),
                        np.digitize(etof_rem[0], np.linspace(0, 2000, n=256)),  # type: ignore # honestly idk if this is actually right
                    ] -= 1
                    if self.calibration is not None:
                        px, py, pz = self.calibration(
                            x_rem[0], y_rem[0], etof_rem[0], self.center, self.angle
                        )
                        bins = np.linspace(-0.6, 0.6, num=256)
                        self.calibrated_pol_plane_data[
                            np.digitize(px, bins), np.digitize(pz, bins)
                        ] -= 1

            processing_rate = self.current_data["timestamps"].maxlen / (
                time.time() - last_timestamp
            )
            plt.suptitle(f"Processing Rate: {processing_rate:.2f} Hz")

        etof, itof, cluster = data
        etof = etof + np.random.random(size=len(etof)) * 0.26

        self.current_data["itof"].append(itof)
        if self.itof_range is not None:
            if len(itof) != 1 or not (
                self.itof_range[0] < itof[0] < self.itof_range[1]
            ):
                etof = []
                cluster = []

        self.current_data["etof"].append(etof)
        if self.etof_range is not None:
            if len(etof) != 1 or not (
                self.etof_range[0] < etof[0] < self.etof_range[1]
            ):
                cluster = []

        self.current_data["cluster"].append(cluster)
        self.current_data["timestamps"].append(time.time())

        if cluster:
            for t, x, y in cluster:
                if self.toa_range is None or (
                    self.toa_range[0] < t < self.toa_range[1]
                ):
                    self.xy_hist_data[int(x), int(y)] += 1 if x < 256 and y < 256 else 0
                if 0 < t < 2000:
                    self.toa_hist_data[int(t)] += 1

        for t in etof:
            if 0 < t < 2000:
                self.etof_hist_data[int(t)] += 1
        for t in itof:
            if 0 < t < 2e4:
                self.itof_hist_data[int(t / 50)] += 1

        if len(cluster) == 1 and len(etof) == 1:
            t, x, y = cluster[0]
            if 0 < etof[0] < 2000:
                self.uncalibrated_pol_plane_data[
                    int(x), np.digitize(etof[0], np.linspace(0, 2000, num=256))
                ] += 1
                if self.calibration is not None:
                    px, py, pz = self.calibration(
                        x, y, etof[0], self.center, self.angle
                    )
                    px, py, pz = px[0], py[0], pz[0]
                    if np.abs(px) < 0.6 and np.abs(pz) < 0.6:
                        bins = np.linspace(-0.6, 0.6, num=256)
                        self.calibrated_pol_plane_data[
                            np.digitize(px, bins), np.digitize(pz, bins)
                        ] += 1

        if time.time() - self.last_update > self.update_interval:
            assert (
                self.figure
                and self.ax
                and self.xy_hist
                and self.toa_hist
                and self.etof_hist
                and self.itof_hist
                and self.uncalibrated_pol_plane
                and self.calibrated_pol_plane
            )
            self.xy_hist.set_data(
                np.log(self.xy_hist_data + 1) / np.log(np.max(self.xy_hist_data + 1))
            )
            self.toa_hist.set_ydata(
                np.log(self.toa_hist_data + 1) / np.log(np.max(self.toa_hist_data + 1))
            )
            self.etof_hist.set_ydata(
                np.log(self.etof_hist_data + 1)
                / np.log(np.max(self.etof_hist_data + 1))
            )
            self.itof_hist.set_ydata(
                np.log(self.itof_hist_data + 1)
                / np.log(np.max(self.itof_hist_data + 1))
            )
            self.uncalibrated_pol_plane.set_data(
                np.log(self.uncalibrated_pol_plane_data + 1)
                / np.log(np.max(self.uncalibrated_pol_plane_data + 1))
            )
            self.calibrated_pol_plane.set_data(
                np.log(self.calibrated_pol_plane_data + 1)
                / np.log(np.max(self.calibrated_pol_plane_data + 1))
            )

            cluster_count_rate = np.mean([len(c) for c in self.current_data["cluster"]])
            etof_count_rate = np.mean([len(c) for c in self.current_data["etof"]])
            itof_count_rate = np.mean([len(c) for c in self.current_data["itof"]])
            clust_etof_rate = np.mean(
                np.where(
                    [
                        len(etof) == 1 and len(cluster) == 1
                        for etof, cluster in zip(
                            self.current_data["etof"], self.current_data["cluster"]
                        )
                    ],
                    1,
                    0,
                )
            )

            self.ax[0, 0].set_title(f"XY (Cluster Rate: {cluster_count_rate:.2f})")
            self.ax[0, 1].set_title(f"ToA (Count Rate: {cluster_count_rate:.2f})")
            self.ax[1, 0].set_title(f"eToF (Count Rate: {etof_count_rate:.2f})")
            self.ax[1, 1].set_title(f"iToF (Count Rate: {itof_count_rate:.2f})")
            self.ax[0, 2].set_title(
                f"Polarization Plane (Uncalibrated) (Cluster Rate: {clust_etof_rate:.2f})"
            )

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.last_update = time.time()

    def shutdown(self, gentle=False):
        plt.close(self.figure)
        super().shutdown(gentle)
