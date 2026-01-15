# pip install PySide6 numpy matplotlib

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from uconn_pipelines import DiagnosticQueuePipeline


def run_analysis_pipeline_return_queue(path: str):
    """
    `path` may be a file or a folder. Your DiagnosticQueuePipeline must accept it.
    """
    pipe = DiagnosticQueuePipeline(path)
    threading.Thread(target=pipe.start, daemon=True).start()  # pass function, don't call it
    return pipe.get_grouped_queue()


@dataclass
class Hist1D:
    edges: np.ndarray
    counts: np.ndarray

    @classmethod
    def make(cls, lo: float, hi: float, bins: int):
        edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
        return cls(edges, counts)

    def add(self, values: np.ndarray):
        if values.size == 0:
            return
        idx = np.searchsorted(self.edges, values, side="right") - 1
        valid = (idx >= 0) & (idx < self.counts.size)
        idx = idx[valid]
        if idx.size:
            np.add.at(self.counts, idx, 1)

    @property
    def centers(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])


@dataclass
class Hist2D:
    x_edges: np.ndarray
    y_edges: np.ndarray
    counts: np.ndarray  # (ny, nx)

    @classmethod
    def make(cls, x_lo: float, x_hi: float, x_bins: int, y_lo: float, y_hi: float, y_bins: int):
        x_edges = np.linspace(x_lo, x_hi, x_bins + 1, dtype=np.float64)
        y_edges = np.linspace(y_lo, y_hi, y_bins + 1, dtype=np.float64)
        counts = np.zeros((y_bins, x_bins), dtype=np.int64)
        return cls(x_edges, y_edges, counts)

    def add(self, x: np.ndarray, y: np.ndarray):
        if x.size == 0 or y.size == 0:
            return
        xi = np.searchsorted(self.x_edges, x, side="right") - 1
        yi = np.searchsorted(self.y_edges, y, side="right") - 1
        valid = (xi >= 0) & (xi < self.x_edges.size - 1) & (yi >= 0) & (yi < self.y_edges.size - 1)
        xi = xi[valid]
        yi = yi[valid]
        if xi.size:
            np.add.at(self.counts, (yi, xi), 1)

    @property
    def extent(self):
        return [self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1]]


class DrainWorker(QObject):
    stats_updated = Signal()

    def __init__(self):
        super().__init__()
        self._stop = False
        self.pipeline_queue = None

        self.h_etof = None
        self.h_itof = None
        self.h_t = None
        self.h_xy = None
        self.h_xt = None
        self.h_xetof = None

        # running rates
        self.n_pulses = 0
        self.total_etof = 0
        self.total_itof = 0
        self.total_clusters = 0

        self.max_items_per_tick = 2000

    def configure_hists(self, *, etof_range, itof_range, t_range, x_range, y_range,
                        bins_1d, bins_2d):
        self.h_etof = Hist1D.make(etof_range[0], etof_range[1], bins_1d)
        self.h_itof = Hist1D.make(itof_range[0], itof_range[1], bins_1d)
        self.h_t = Hist1D.make(t_range[0], t_range[1], bins_1d)

        xb, yb = bins_2d, bins_2d
        self.h_xy = Hist2D.make(x_range[0], x_range[1], xb, y_range[0], y_range[1], yb)
        self.h_xt = Hist2D.make(x_range[0], x_range[1], xb, t_range[0], t_range[1], yb)
        self.h_xetof = Hist2D.make(x_range[0], x_range[1], xb, etof_range[0], etof_range[1], yb)

        # reset running rates
        self.n_pulses = 0
        self.total_etof = 0
        self.total_itof = 0
        self.total_clusters = 0

    def start(self, pipeline_queue):
        self.pipeline_queue = pipeline_queue
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop = True

    def _run(self):
        last_emit = time.time()
        while not self._stop:
            drained = 0
            while drained < self.max_items_per_tick:
                try:
                    etofs, itofs, clusters = self.pipeline_queue.get(timeout=0.1)
                except queue.Empty:
                    break

                self.n_pulses += 1
                self.total_etof += len(etofs)
                self.total_itof += len(itofs)
                self.total_clusters += len(clusters)

                etof_vals = np.array([float(t[0]) for t in etofs], dtype=np.float64) if etofs else np.empty(0)
                itof_vals = np.array([float(t[0]) for t in itofs], dtype=np.float64) if itofs else np.empty(0)

                if clusters:
                    t_vals = np.array([float(c[0]) for c in clusters], dtype=np.float64)
                    x_vals = np.array([float(c[1]) for c in clusters], dtype=np.float64)
                    y_vals = np.array([float(c[2]) for c in clusters], dtype=np.float64)
                else:
                    t_vals = x_vals = y_vals = np.empty(0, dtype=np.float64)

                self.h_etof.add(etof_vals)
                self.h_itof.add(itof_vals)
                self.h_t.add(t_vals)

                self.h_xy.add(x_vals, y_vals)
                self.h_xt.add(x_vals, t_vals)

                if (len(clusters) == 1) and (len(etofs) == 1):
                    self.h_xetof.add(np.array([x_vals[0]]), np.array([etof_vals[0]]))

                drained += 1

            now = time.time()
            if drained > 0 and (now - last_emit) > 0.05:
                self.stats_updated.emit()
                last_emit = now

            time.sleep(0.005)


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(constrained_layout=True)
        super().__init__(self.fig)


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._content)

        self._on_toggled(True)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self.setFlat(not checked)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grouped Queue Live Histograms")

        self.worker = DrainWorker()
        self.path = ""  # file or folder
        self.pipeline_q = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Controls
        controls = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls)

        # Row: path entry + browse buttons
        row_path = QHBoxLayout()
        self.ed_path = QLineEdit()
        self.ed_path.setPlaceholderText("Enter file or folder path…")
        self.btn_pick_file = QPushButton("Browse file…")
        self.btn_pick_folder = QPushButton("Browse folder…")
        row_path.addWidget(QLabel("Path:"))
        row_path.addWidget(self.ed_path, 1)
        row_path.addWidget(self.btn_pick_file)
        row_path.addWidget(self.btn_pick_folder)

        # Row: run/stop
        row_run = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.lbl_path_status = QLabel("No path selected")
        self.lbl_path_status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        row_run.addWidget(self.btn_run)
        row_run.addWidget(self.btn_stop)
        row_run.addWidget(self.lbl_path_status, 1)

        controls_layout.addLayout(row_path)
        controls_layout.addLayout(row_run)
        layout.addWidget(controls)

        # Parameters (collapsible)
        params = CollapsibleGroupBox("Histogram parameters (click to collapse)")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Defaults: eToF 0-1000, t 0-1000, iToF 0-20000, x/y 0-256
        self.ed_etof_lo = QDoubleSpinBox();
        self.ed_etof_lo.setRange(-1e12, 1e12);
        self.ed_etof_lo.setValue(0)
        self.ed_etof_hi = QDoubleSpinBox();
        self.ed_etof_hi.setRange(-1e12, 1e12);
        self.ed_etof_hi.setValue(1000)

        self.ed_itof_lo = QDoubleSpinBox();
        self.ed_itof_lo.setRange(-1e12, 1e12);
        self.ed_itof_lo.setValue(0)
        self.ed_itof_hi = QDoubleSpinBox();
        self.ed_itof_hi.setRange(-1e12, 1e12);
        self.ed_itof_hi.setValue(20000)

        self.ed_t_lo = QDoubleSpinBox();
        self.ed_t_lo.setRange(-1e12, 1e12);
        self.ed_t_lo.setValue(0)
        self.ed_t_hi = QDoubleSpinBox();
        self.ed_t_hi.setRange(-1e12, 1e12);
        self.ed_t_hi.setValue(1000)

        self.ed_x_lo = QDoubleSpinBox();
        self.ed_x_lo.setRange(-1e12, 1e12);
        self.ed_x_lo.setValue(0)
        self.ed_x_hi = QDoubleSpinBox();
        self.ed_x_hi.setRange(-1e12, 1e12);
        self.ed_x_hi.setValue(256)

        self.ed_y_lo = QDoubleSpinBox();
        self.ed_y_lo.setRange(-1e12, 1e12);
        self.ed_y_lo.setValue(0)
        self.ed_y_hi = QDoubleSpinBox();
        self.ed_y_hi.setRange(-1e12, 1e12);
        self.ed_y_hi.setValue(256)

        self.sp_bins_1d = QSpinBox();
        self.sp_bins_1d.setRange(16, 4096);
        self.sp_bins_1d.setValue(300)
        self.sp_bins_2d = QSpinBox();
        self.sp_bins_2d.setRange(16, 2048);
        self.sp_bins_2d.setValue(220)
        self.sp_draw_ms = QSpinBox();
        self.sp_draw_ms.setRange(20, 5000);
        self.sp_draw_ms.setValue(150)

        form.addRow("eToF range [lo, hi]", self._row2(self.ed_etof_lo, self.ed_etof_hi))
        form.addRow("iToF range [lo, hi]", self._row2(self.ed_itof_lo, self.ed_itof_hi))
        form.addRow("t range [lo, hi]", self._row2(self.ed_t_lo, self.ed_t_hi))
        form.addRow("x range [lo, hi]", self._row2(self.ed_x_lo, self.ed_x_hi))
        form.addRow("y range [lo, hi]", self._row2(self.ed_y_lo, self.ed_y_hi))
        form.addRow("1D bins", self.sp_bins_1d)
        form.addRow("2D bins", self.sp_bins_2d)
        form.addRow("Redraw period (ms)", self.sp_draw_ms)

        params.content_layout().addLayout(form)
        layout.addWidget(params)

        # Plots + rates box
        plots_row = QHBoxLayout()
        layout.addLayout(plots_row, 1)

        self.canvas = MplCanvas()
        plots_row.addWidget(self.canvas, 1)

        self.rates_box = QGroupBox("Average count rate (per shot)")
        rates_layout = QVBoxLayout(self.rates_box)
        self.lbl_rates = QLabel("eToF: --\niToF: --\nclusters: --\nshots: 0")
        self.lbl_rates.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_rates.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        rates_layout.addWidget(self.lbl_rates)
        plots_row.addWidget(self.rates_box, 0)

        self._setup_axes()

        # Wiring
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_pick_folder.clicked.connect(self.pick_folder)
        self.ed_path.textChanged.connect(self._on_path_text_changed)

        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_stop.clicked.connect(self.stop_pipeline)

        self.worker.stats_updated.connect(self.request_redraw)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.timeout.connect(self.redraw)
        self._redraw_pending = False

    def _row2(self, a, b):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    def _setup_axes(self):
        fig = self.canvas.fig
        fig.clear()

        self.ax_etof = fig.add_subplot(2, 3, 1)
        self.ax_itof = fig.add_subplot(2, 3, 2)
        self.ax_t = fig.add_subplot(2, 3, 3)
        self.ax_xy = fig.add_subplot(2, 3, 4)
        self.ax_xt = fig.add_subplot(2, 3, 5)
        self.ax_xe = fig.add_subplot(2, 3, 6)

        self.ax_etof.set_title("eToF")
        self.ax_itof.set_title("iToF")
        self.ax_t.set_title("t (clusters)")

        self.ax_xy.set_title("x/y (clusters)")
        self.ax_xt.set_title("x/t (clusters)")
        self.ax_xe.set_title("x/eToF (1 cluster & 1 eToF)")

        self._artists_initialized = False
        self.canvas.draw_idle()

    def _on_path_text_changed(self, text: str):
        self.path = text.strip()
        self.lbl_path_status.setText(self.path if self.path else "No path selected")

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data file", "", "All files (*)")
        if not path:
            return
        self.ed_path.setText(path)

    def pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if not path:
            return
        self.ed_path.setText(path)

    def run_pipeline(self):
        path = self.ed_path.text().strip()
        if not path:
            self.lbl_path_status.setText("No path selected")
            return

        # Optional existence check (won't block weird remote paths; just a sanity check)
        if not (os.path.isfile(path) or os.path.isdir(path)):
            self.lbl_path_status.setText(f"Path not found: {path}")
            return

        etof_range = (self.ed_etof_lo.value(), self.ed_etof_hi.value())
        itof_range = (self.ed_itof_lo.value(), self.ed_itof_hi.value())
        t_range = (self.ed_t_lo.value(), self.ed_t_hi.value())
        x_range = (self.ed_x_lo.value(), self.ed_x_hi.value())
        y_range = (self.ed_y_lo.value(), self.ed_y_hi.value())
        bins_1d = int(self.sp_bins_1d.value())
        bins_2d = int(self.sp_bins_2d.value())

        self.worker.configure_hists(
                etof_range=etof_range,
                itof_range=itof_range,
                t_range=t_range,
                x_range=x_range,
                y_range=y_range,
                bins_1d=bins_1d,
                bins_2d=bins_2d
        )

        self.pipeline_q = run_analysis_pipeline_return_queue(path)
        self.worker.start(self.pipeline_q)

        self._artists_initialized = False
        self._setup_axes()

        self.redraw_timer.start(int(self.sp_draw_ms.value()))
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pick_file.setEnabled(False)
        self.btn_pick_folder.setEnabled(False)
        self.ed_path.setEnabled(False)

    def stop_pipeline(self):
        self.worker.stop()
        self.redraw_timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pick_file.setEnabled(True)
        self.btn_pick_folder.setEnabled(True)
        self.ed_path.setEnabled(True)

    def request_redraw(self):
        self._redraw_pending = True

    def redraw(self):
        if not self._redraw_pending:
            return
        self._redraw_pending = False

        w = self.worker
        if w.h_etof is None:
            return

        if not self._artists_initialized:
            (self.l_etof,) = self.ax_etof.plot(w.h_etof.centers, w.h_etof.counts)
            (self.l_itof,) = self.ax_itof.plot(w.h_itof.centers, w.h_itof.counts)
            (self.l_t,) = self.ax_t.plot(w.h_t.centers, w.h_t.counts)

            self.im_xy = self.ax_xy.imshow(w.h_xy.counts, origin="lower", extent=w.h_xy.extent, aspect="auto")
            self.im_xt = self.ax_xt.imshow(w.h_xt.counts, origin="lower", extent=w.h_xt.extent, aspect="auto")
            self.im_xe = self.ax_xe.imshow(w.h_xetof.counts, origin="lower", extent=w.h_xetof.extent, aspect="auto")

            self.ax_xy.set_xlabel("x");
            self.ax_xy.set_ylabel("y")
            self.ax_xt.set_xlabel("x");
            self.ax_xt.set_ylabel("t")
            self.ax_xe.set_xlabel("x");
            self.ax_xe.set_ylabel("eToF")

            self._artists_initialized = True

        self.l_etof.set_ydata(w.h_etof.counts)
        self.l_itof.set_ydata(w.h_itof.counts)
        self.l_t.set_ydata(w.h_t.counts)

        self.ax_etof.relim();
        self.ax_etof.autoscale_view(scalex=False, scaley=True)
        self.ax_itof.relim();
        self.ax_itof.autoscale_view(scalex=False, scaley=True)
        self.ax_t.relim();
        self.ax_t.autoscale_view(scalex=False, scaley=True)

        self.im_xy.set_data(w.h_xy.counts)
        self.im_xt.set_data(w.h_xt.counts)
        self.im_xe.set_data(w.h_xetof.counts)

        xy_max = int(w.h_xy.counts.max()) if w.h_xy.counts.size else 0
        xt_max = int(w.h_xt.counts.max()) if w.h_xt.counts.size else 0
        xe_max = int(w.h_xetof.counts.max()) if w.h_xetof.counts.size else 0

        self.im_xy.set_clim(0, max(1, xy_max))
        self.im_xt.set_clim(0, max(1, xt_max))
        self.im_xe.set_clim(0, max(1, xe_max))

        n = max(1, w.n_pulses)
        avg_etof = w.total_etof / n
        avg_itof = w.total_itof / n
        avg_cl = w.total_clusters / n
        self.lbl_rates.setText(
                f"eToF: {avg_etof:.3f}\n"
                f"iToF: {avg_itof:.3f}\n"
                f"clusters: {avg_cl:.3f}\n"
                f"shots: {w.n_pulses}"
        )

        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            self.stop_pipeline()
        finally:
            event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
