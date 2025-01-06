import sys
import numpy as np
import scipy.io
import scipy.ndimage
from skimage import measure
import logging
import pywt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

np.bool = np.bool_
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer

import cmasher as cmr


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.debug("Initializing MainWindow")
        self.setWindowTitle("3D Contour Visualization")

        # Initial parameters
        self.smoothing = 0
        self.num_contours = 7
        self.gamma_contours = 0.2
        self.gamma_colors = 0.4
        self.gamma_opacities = 0.5
        self.min_contour = 0.3
        self.max_contour = 0.75
        self.filename = ""
        self.pmax = 0.7
        self.cmap = cmr.rainforest

        # Debounce Timer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_plot)

        # Data variables
        self.hist = None
        self.plotter = None

        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Sliders and controls layout
        controls_layout = QVBoxLayout()

        # Filename
        fname_layout = QHBoxLayout()
        fname_label = QLabel("Filename:")
        self.fname_edit = QLineEdit()
        self.fname_edit.editingFinished.connect(lambda: self.fname_edit.setText(self.fname_edit.text().strip("\"")))
        fname_layout.addWidget(fname_label)
        fname_layout.addWidget(self.fname_edit)
        controls_layout.addLayout(fname_layout)

        # Smoothing control
        smoothing_layout = QHBoxLayout()
        smoothing_label = QLabel("Smoothing:")
        self.smoothing_spin = QDoubleSpinBox()
        self.smoothing_spin.setRange(0.0, 5.0)
        self.smoothing_spin.setSingleStep(0.1)
        self.smoothing_spin.setValue(self.smoothing)
        self.smoothing_spin.editingFinished.connect(self.update_plot)
        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addWidget(self.smoothing_spin)
        controls_layout.addLayout(smoothing_layout)

        # Number of contours
        num_contours_layout = QHBoxLayout()
        num_contours_label = QLabel("Number of Contours:")
        self.num_contours_spin = QSpinBox()
        self.num_contours_spin.setRange(1, 50)
        self.num_contours_spin.setValue(self.num_contours)
        self.num_contours_spin.valueChanged.connect(self.delayed_update)
        num_contours_layout.addWidget(num_contours_label)
        num_contours_layout.addWidget(self.num_contours_spin)
        controls_layout.addLayout(num_contours_layout)

        # Min and Max contour sliders
        contour_range_layout = QHBoxLayout()
        min_contour_label = QLabel("Min Contour:")
        self.min_contour_spin = QDoubleSpinBox()
        self.min_contour_spin.setRange(0.0, 1.0)
        self.min_contour_spin.setSingleStep(0.05)
        self.min_contour_spin.setValue(self.min_contour)
        self.min_contour_spin.editingFinished.connect(self.update_plot)

        max_contour_label = QLabel("Max Contour:")
        self.max_contour_spin = QDoubleSpinBox()
        self.max_contour_spin.setRange(0.0, 1.0)
        self.max_contour_spin.setSingleStep(0.05)
        self.max_contour_spin.setValue(self.max_contour)
        self.max_contour_spin.editingFinished.connect(self.update_plot)

        contour_range_layout.addWidget(min_contour_label)
        contour_range_layout.addWidget(self.min_contour_spin)
        contour_range_layout.addWidget(max_contour_label)
        contour_range_layout.addWidget(self.max_contour_spin)
        controls_layout.addLayout(contour_range_layout)

        # Gamma for contours
        gamma_contours_layout = QHBoxLayout()
        gamma_contours_label = QLabel("Gamma Contours:")
        self.gamma_contours_spin = QDoubleSpinBox()
        self.gamma_contours_spin.setRange(0.0, 5.0)
        self.gamma_contours_spin.setSingleStep(0.1)
        self.gamma_contours_spin.setValue(self.gamma_contours)
        self.gamma_contours_spin.editingFinished.connect(self.update_plot)
        gamma_contours_layout.addWidget(gamma_contours_label)
        gamma_contours_layout.addWidget(self.gamma_contours_spin)
        controls_layout.addLayout(gamma_contours_layout)

        # Gamma for colors
        gamma_colors_layout = QHBoxLayout()
        gamma_colors_label = QLabel("Gamma Colors:")
        self.gamma_colors_spin = QDoubleSpinBox()
        self.gamma_colors_spin.setRange(0.0, 5.0)
        self.gamma_colors_spin.setSingleStep(0.1)
        self.gamma_colors_spin.setValue(self.gamma_colors)
        self.gamma_colors_spin.editingFinished.connect(self.update_plot)
        gamma_colors_layout.addWidget(gamma_colors_label)
        gamma_colors_layout.addWidget(self.gamma_colors_spin)
        controls_layout.addLayout(gamma_colors_layout)

        # Gamma for opacities
        gamma_opacities_layout = QHBoxLayout()
        gamma_opacities_label = QLabel("Gamma Opacities:")
        self.gamma_opacities_spin = QDoubleSpinBox()
        self.gamma_opacities_spin.setRange(0.0, 5.0)
        self.gamma_opacities_spin.setSingleStep(0.1)
        self.gamma_opacities_spin.setValue(self.gamma_opacities)
        self.gamma_opacities_spin.editingFinished.connect(self.update_plot)
        gamma_opacities_layout.addWidget(gamma_opacities_label)
        gamma_opacities_layout.addWidget(self.gamma_opacities_spin)
        controls_layout.addLayout(gamma_opacities_layout)

        # Load, Save Screenshot, and Save Video buttons
        buttons_layout = QHBoxLayout()
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        save_button = QPushButton("Save Screenshot")
        save_button.clicked.connect(self.save_screenshot)
        video_button = QPushButton("Save Video")
        video_button.clicked.connect(self.save_video)
        buttons_layout.addWidget(load_button)
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(video_button)
        controls_layout.addLayout(buttons_layout)

        main_layout.addLayout(controls_layout)

        # PyVista Plotter
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.main_layout = main_layout
        self.main_widget = main_widget

    def delayed_update(self):
        self.timer.start(500)  # 500 ms debounce

    def load_data(self):
        logging.debug("Loading data")
        fname = self.fname_edit.text().strip("\"")
        if fname == "":
            logging.warning("Filename is empty")
            return
        data = scipy.io.loadmat(fname, squeeze_me=True)
        px, py, pz = data['px'], data['py'], data['pz']
        mask = (px ** 2 + py ** 2 + pz ** 2) < self.pmax ** 2
        px, py, pz = px[mask], py[mask], pz[mask]
        logging.debug("Data loaded and masked")

        hist, _ = np.histogramdd((px, py, pz),
                                 bins=128,
                                 range=((-self.pmax, self.pmax),
                                        (-self.pmax, self.pmax),
                                        (-self.pmax, self.pmax)))

        self.hist = hist / hist.max()
        logging.debug("Histogram generated and normalized")
        self.update_plot()

    def update_plot(self):
        if self.hist is None:
            logging.warning("Histogram data is None")
            return

        logging.debug("Updating plot")
        smoothing = self.smoothing_spin.value()
        gamma_contours = self.gamma_contours_spin.value()
        gamma_colors = self.gamma_colors_spin.value()
        gamma_opacities = self.gamma_opacities_spin.value()
        min_contour = self.min_contour_spin.value()
        max_contour = self.max_contour_spin.value()

        if min_contour >= max_contour:
            logging.error("Min contour must be less than max contour")
            return

        hist_smooth = scipy.ndimage.gaussian_filter(self.hist, smoothing)
        logging.debug("Histogram smoothed")
        hist_mod = np.nan_to_num(hist_smooth ** (gamma_contours), nan=0.0, posinf=0.0, neginf=0.0)
        logging.debug("Histogram modified with gamma adjustments")
        hist_mod = wavelet_denoise_3d(hist_mod, wavelet='sym4', level=2, threshold_type='soft')
        logging.debug("Histogram denoised with wavelet transform")

        self.plotter.clear()
        QApplication.processEvents()
        levels = np.linspace(min_contour, max_contour, self.num_contours_spin.value())
        logging.debug(f"Levels for contours: {levels}")
        for level in levels:
            logging.debug(f"Processing level {level}")
            if level >= np.max(hist_mod):
                logging.debug(f"Level {level} is greater than maximum value in histogram")
                continue
            verts, faces, _, _ = measure.marching_cubes(hist_mod, level,
                                                        spacing=(2 * self.pmax / 1024, 2 * self.pmax / 1024, 2 * self.pmax / 1024))
            logging.debug("Marching cubes generated")
            faces = [(len(f), *f) for f in faces]
            faces = np.concatenate(faces)
            logging.debug("Marching cubes processed")
            surf = pv.PolyData(verts, faces)
            logging.debug("PolyData created")
            gamma_colors_effective = gamma_colors / gamma_contours
            gamma_opacities_effective = gamma_opacities / gamma_contours
            level_effective = (level - min_contour) / (max_contour - min_contour)
            self.plotter.add_mesh(
                    surf,
                    color=self.cmap(level_effective ** gamma_colors_effective),
                    opacity=level_effective ** gamma_opacities_effective
            )
            logging.debug("Mesh added to plotter")
        logging.debug("All contour levels processed")
        self.plotter.reset_camera()
        self.plotter.show()

    def save_screenshot(self):
        fname = self.fname_edit.text()
        fname=fname.split('.')[0]+"_3D"
        if fname:
            logging.debug(f"Saving screenshot to {fname}.pdf")
            self.plotter.save_graphic(fname + ".pdf")

    def save_video(self):
        # Parameters for the video
        duration = 24.0  # 24 seconds
        fps = 30
        total_frames = int(duration * fps)
        angle_increment = 360.0 / total_frames
        fname = self.fname_edit.text()
        output_filename = fname.split('.')[0] + "_3D.mp4"

        logging.debug("Starting video capture")
        # Set the camera elevation to 5 degrees above xy-plane

        self.plotter.camera.enable_parallel_projection()
        self.plotter.camera.tight()
        self.plotter.camera.elevation = 30
        self.plotter.camera.azimuth = 0
        self.plotter.show()
        # self.plotter.reset_camera(bounds=(-self.pmax, self.pmax, -self.pmax, self.pmax, -self.pmax, self.pmax))

        self.plotter.open_movie(output_filename, quality=9)
        logging.debug("Video file opened")
        logging.debug(f"Capturing {total_frames} frames")
        for i in range(total_frames):
            logging.debug(f"Capturing frame {i} at angle {angle_increment * i}")
            self.plotter.camera.azimuth = angle_increment * i
            self.plotter.render()
            self.plotter.write_frame()
        self.main_layout.removeWidget(self.plotter.interactor)
        self.plotter.close()  # Finalize the video file
        logging.debug(f"Video saved to {output_filename}")
        self.plotter=QtInteractor(self)
        self.main_layout.addWidget(self.plotter.interactor)
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.update_plot()

def wavelet_denoise_3d(data, wavelet='db4', level=2, threshold_type='hard'):
    """
    Perform 3D wavelet-based denoising on volumetric data.

    Parameters:
        data (numpy.ndarray): The noisy 3D input data (e.g., histogram grid).
        wavelet (str): Wavelet type (e.g., 'db4', 'haar', 'sym4').
        level (int): Decomposition level for the wavelet transform.
        threshold_type (str): Thresholding type ('soft' or 'hard').

    Returns:
        numpy.ndarray: Denoised 3D data.
    """
    coeffs = pywt.dwtn(data, wavelet=wavelet, axes=(0, 1, 2))
    detail_coeffs = coeffs['d'*3]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(data.size))

    def thresholding(coeff, threshold, threshold_type):
        if threshold_type == 'soft':
            return pywt.threshold(coeff, threshold, mode='soft')
        elif threshold_type == 'hard':
            return pywt.threshold(coeff, threshold, mode='hard')
        return coeff

    for key in coeffs.keys():
        if 'd' in key:
            coeffs[key] = thresholding(coeffs[key], threshold, threshold_type)

    denoised_data = pywt.idwtn(coeffs, wavelet=wavelet, axes=(0, 1, 2))
    return denoised_data

if __name__ == "__main__":
    logging.debug("Starting application")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
