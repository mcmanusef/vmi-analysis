#SBUTPX Quick Analyze by Kovi Bodek 2026

import sys, math, os, h5py
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QGridLayout,
QCheckBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QDialog,
QComboBox, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import QRectF
from PyQt5 import QtCore
import pyqtgraph as pg

class GlobalVariables:
    def __init__(self):
        self.gate_items = ["xy", "r", "tof", "theta", "delay", "tot", "cluster size", "shot #"]
        self.show_state = [False for item in self.gate_items]
        self.gate_state = [False for item in self.gate_items]
        self.log_state = [True for item in self.gate_items]
        self.default_browse_location = "D:\\data2026"
        self.filepath = ""
        self.dataset = None
        self.static = None
        self.dimension = 1
        self.axes = ["x","y"]
        self.xaxis = "x"
        self.x0 = None
        self.y0 = None
        self.centerx = None
        self.centery = None
        self.r = None
        self.theta = None
        self.choosing_xy = False
        self.tof = None
        self.minor_checkbox = """
            QCheckBox{color: #bbb;}
            QCheckBox:disabled{color: #555;}
            QCheckBox::indicator{margin-top:3px; background-color: #eee;border-radius: 4px;}
            QCheckBox::indicator:checked{background-color: #3f3;}
            QCheckBox::indicator:disabled{background-color: #888;}"""
        self.fancylabel = """
            QLabel{font-size: 24px; color: #eee;}
            QLabel:disabled{color: #444;}
            QCheckBox{color: #bbb;}
            QCheckBox:disabled{color: #555;}
            QCheckBox::indicator{margin-top:3px; background-color: #eee; border-radius: 4px;}
            QCheckBox::indicator:checked{background-color: #3f3}
            QCheckBox::indicator:disabled{background-color: #888;}"""
        self.greenstyle = """
            QPushButton{color: #000; border-radius: 5px; border: 1px solid gray; padding: 4px; background-color: #3f3; font-size: 20px;}
            QPushButton:hover{background-color: #3c3;}
            QPushButton:pressed{background-color: #093;}
            QPushButton:disabled{background-color:#eee; color: gray;}"""
        self.whitestyle = """
            QLineEdit{border: 1px solid #000; margin-top: 1px; font-size: 22px;}
            QPushButton{color: #000; border-radius: 5px; border: 1px solid #222; padding: 4px;background-color: #eef; font-size: 20px;}
            QPushButton:hover{color: #000;background-color: #ccf;}
            QPushButton:pressed{color: #000; background-color: #bbf;}
            QPushButton:disabled{background-color:#555; color: #666}"""
        self.primary_bound_style = pg.mkPen(width=3, color=(200,255,0,100))
        self.secondary_bound_style = pg.mkPen(width=3, color=(40,40,255,100))
        self.tertiary_bound_style = pg.mkPen(width=1.5, color=(255,128,0,200))
        self.x = None
        self.y = None
        self.tot = None
        self.delay = None
        self.t0 = 11189
        self.fs = None
        self.pos_axis = None
        self.main_curve = None
        self.normalize_type = 0
        self.tracenames = []
        self.traces = []
        self.tracestates = []
        self.tracecolors = []
        self.colorkey = ["cyan","red","orange","green","magenta","blue","yellow","gray"]
        self.colorvalues = [(0,255,255,200),(255,0,0,200),(255,128,0,200),(0,220,0,200),(255,0,255,200),(50,70,255,200), (200,255,0,200), (128,128,128,200)]
        self.xy_bx1 = None
        self.xy_bx2 = None
        self.xy_by1 = None
        self.xy_by2 = None
        self.xy_index = None
        self.r_b1 = None
        self.r_b2 = None
        self.r_index = None
        self.tof_b1 = None
        self.tof_b2 = None
        self.tof_index = None
        self.theta_b1 = None
        self.theta_b2 = None
        self.theta_index = None
        self.bigfont = QFont()
        self.bigfont.setBold=True
        self.bigfont.setPointSize(30)
        self.normalfont = QFont()
        self.normalfont.setPointSize(16)
        self.smallfont = QFont()
        self.smallfont.setPointSize(12)
        self.smallfont.setBold=True
        self.boldfont = QFont()
        self.boldfont.setPointSize(20)
        self.boldfont.setBold=True
    def xycenter(self):
        if not (self.x0 is None or self.y0 is None):
            self.centerx = self.x - self.x0
            self.centery = self.y - self.y0
            self.r = (self.centerx**2+self.centery**2)**0.5
            self.theta = np.arctan2(self.centery, self.centerx)
class FileLoader(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_browse_location = ""
        self.text = ""
        self.layout = QGridLayout(self)
        self.dataset_label = QLabel("Dataset:")
        self.dataset_label.setStyleSheet(G.fancylabel)
        self.dataset_textfield = QLineEdit()
        self.dataset_textfield.setStyleSheet(G.whitestyle)
        self.dataset_textfield.setMinimumWidth(300)
        self.dataset_textfield.textChanged.connect(self.dataset_textfield_handler)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setStyleSheet(G.whitestyle)
        self.browse_button.clicked.connect(self.browse_button_handler)
        self.load_button = QPushButton("Load")
        self.load_button.setStyleSheet(G.whitestyle)
        self.load_button.clicked.connect(self.load_button_handler)
        self.layout.addWidget(self.dataset_label, 0,0,1,1)
        self.layout.addWidget(self.dataset_textfield, 0,1,1,1)
        self.layout.addWidget(self.browse_button, 0,2,1,1)
        self.layout.addWidget(self.load_button, 0, 3, 1, 1)
    def dataset_textfield_handler(self):
        self.text = self.dataset_textfield.text()
        G.dataset = None
        self.load_button.setText("Load")
        self.load_button.setStyleSheet(G.whitestyle)
        self.load_button.setEnabled(True)
    def load_button_handler(self):
        try: G.dataset = h5py.File(self.text, "r")
        except Exception as e:
            popup(f"Error when opening {self.text}>> {e}")
            return
        self.load_button.setText("Loaded")
        self.load_button.setStyleSheet(G.greenstyle)
        self.load_button.setEnabled(False)
        self.parent().load()
    def browse_button_handler(self):
        if self.last_browse_location == "": default = G.default_browse_location
        else: default = self.last_browse_location
        files = QFileDialog.getOpenFileNames(self, "Browse",default, filter="*.h5")[0]
        if len(files)>0:
            self.last_browse_location = os.path.dirname(files[0])
            self.dataset_textfield.setText(files[0])

class ThetaPlot(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.vb.setLimits(yMin=0)
        self.plot = pg.PlotDataItem(np.arange(100), np.arange(100))
        self.plot.setPen(pg.mkPen(width=2))
        self.plot_item.addItem(self.plot)
        self.plot_item.setLabel('bottom', 'theta [rad]')
        self.plot_item.getAxis("bottom").label.setFont(G.smallfont)
        self.plot_item.getAxis("bottom").setTickFont(G.smallfont)
        self.plot_item.getAxis("left").label.setFont(G.smallfont)
        self.plot_item.getAxis("left").setTickFont(G.smallfont)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.stateChanged.connect(self.log_checkbox_handler)
        self.log_checkbox.setStyleSheet(G.minor_checkbox)
        self.log_checkbox.setFont(G.smallfont)
        self.layout.addWidget(self.plot_widget, 0, 0, 1, 1)
        self.layout.addWidget(self.log_checkbox, 1, 0, 1, 1)
        self.b1 = pg.InfiniteLine(-np.pi,  angle = 90, movable = True, bounds = (-np.pi,np.pi), pen=G.primary_bound_style)
        self.b2 = pg.InfiniteLine(np.pi,  angle = 90, movable = True, bounds = (-np.pi,np.pi), pen=G.primary_bound_style)
        if not G.theta_b1 is None: self.b1.setValue(G.theta_b1)
        if not G.theta_b2 is None: self.b2.setValue(G.theta_b2)
        self.b1.sigPositionChanged.connect(self.gate_handler)
        self.b2.sigPositionChanged.connect(self.gate_handler)
        if G.gate_state[3]:
            self.plot_item.addItem(self.b1)
            self.plot_item.addItem(self.b2)
        self.gate_handler()
    def gate_handler(self):
        self.update_gate_values()
        self.update_index()
        self.parent().parent().mainplot.load()
    def update_gate_values(self):
        if self.b2.value()>self.b1.value():
            G.theta_b1 = self.b1.value()
            G.theta_b2 = self.b2.value()
        else:
            G.theta_b1 = self.b2.value()
            G.theta_b2 = self.b1.value()
    def update_index(self):
        G.theta_index = (G.theta>G.theta_b1) & (G.theta<G.theta_b2)
    def load(self):
        thetahist, bins = np.histogram(G.theta, bins=np.arange(-np.pi,np.pi,0.02))
        self.plot.setData(bins[:-1], thetahist)
        self.update_gate_values()
        self.update_index()
    def log_checkbox_handler(self):
        if self.log_checkbox.isChecked(): self.plot_item.setLogMode(y=True)
        else: self.plot_item.setLogMode(y=False)
        G.log_state[2] = self.log_checkbox.isChecked()
class TofPlot(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.vb.setLimits(xMin=0,yMin=0)
        self.plot = pg.PlotDataItem(np.arange(100), np.arange(100))
        self.plot.setPen(pg.mkPen(width=2))
        self.plot_item.addItem(self.plot)
        self.plot_item.setLabel('bottom', 'tof [μs]')
        self.plot_item.getAxis("bottom").label.setFont(G.smallfont)
        self.plot_item.getAxis("bottom").setTickFont(G.smallfont)
        self.plot_item.getAxis("left").label.setFont(G.smallfont)
        self.plot_item.getAxis("left").setTickFont(G.smallfont)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.stateChanged.connect(self.log_checkbox_handler)
        self.log_checkbox.setStyleSheet(G.minor_checkbox)
        self.log_checkbox.setFont(G.smallfont)
        self.b1 = pg.InfiniteLine(0,  angle = 90, movable = True, bounds = (0,15), pen=G.primary_bound_style)
        self.b2 = pg.InfiniteLine(20,  angle = 90, movable = True, bounds = (0,15), pen=G.primary_bound_style)
        self.bcenter = pg.InfiniteLine(20, angle = 90, movable = True, bounds = (0.5,14.5), pen=G.secondary_bound_style)
        if not (G.tof_b1 is None or G.tof_b2 is None):
            self.b1.setValue(G.tof_b1)
            self.b2.setValue(G.tof_b2)
            self.bcenter.setValue((G.tof_b1 + G.tof_b2)/2)
        self.b1.sigPositionChanged.connect(self.gate_handler)
        self.b2.sigPositionChanged.connect(self.gate_handler)
        self.b1.sigDragged.connect(self.update_centergate)
        self.b2.sigDragged.connect(self.update_centergate)
        self.bcenter.sigDragged.connect(self.centergate_handler)
        if G.gate_state[2]:
            self.plot_item.addItem(self.b1)
            self.plot_item.addItem(self.b2)
            self.plot_item.addItem(self.bcenter)
        self.gate_handler()
        self.layout.addWidget(self.plot_widget, 0, 0, 1, 1)
        self.layout.addWidget(self.log_checkbox, 1, 0, 1, 1)
    def gate_handler(self):
        self.update_gate_values()
        self.update_index()
        self.parent().parent().mainplot.load()
    def update_centergate(self):
        self.bcenter.setValue((G.tof_b1+G.tof_b2)/2)
    def centergate_handler(self):
        delta = (G.tof_b2-G.tof_b1)/2
        self.b1.setValue(self.bcenter.value()-delta)
        self.b2.setValue(self.bcenter.value()+delta)
    def update_gate_values(self):
        if self.b2.value()>self.b1.value():
            G.tof_b1 = self.b1.value()
            G.tof_b2 = self.b2.value()
        else:
            G.tof_b1 = self.b2.value()
            G.tof_b2 = self.b1.value()
    def update_index(self):
        G.tof_index = (G.tof>G.tof_b1) & (G.tof<G.tof_b2)
    def load(self):
        tofhist, bins = np.histogram(G.tof, bins=np.arange(0,15,0.003))
        self.plot.setData(bins[:-1], tofhist)
        self.update_gate_values()
        self.update_index()
    def log_checkbox_handler(self):
        if self.log_checkbox.isChecked(): self.plot_item.setLogMode(y=True)
        else: self.plot_item.setLogMode(y=False)
        G.log_state[2] = self.log_checkbox.isChecked()
class RPlot(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.vb.setLimits(xMin=0,yMin=0)
        self.plot = pg.PlotDataItem(np.arange(100), np.arange(100)) #this is what you update with data
        self.plot.setPen(pg.mkPen(width=2))
        self.plot_item.addItem(self.plot)
        self.plot_item.setLabel('bottom', 'r [pix]')
        self.plot_item.getAxis("bottom").label.setFont(G.smallfont)
        self.plot_item.getAxis("bottom").setTickFont(G.smallfont)
        self.plot_item.getAxis("left").label.setFont(G.smallfont)
        self.plot_item.getAxis("left").setTickFont(G.smallfont)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.stateChanged.connect(self.log_checkbox_handler)
        self.log_checkbox.setStyleSheet(G.minor_checkbox)
        self.log_checkbox.setFont(G.smallfont)
        self.jacobian_dropdown = QComboBox()
        self.jacobian_dropdown.addItems(["Density", "Yield"])
        self.jacobian_dropdown.currentIndexChanged.connect(self.jacobian_changer)
        self.jacobian_dropdown.setFont(G.normalfont)
        self.density = True
        self.b1 = pg.InfiniteLine(0,  angle = 90, movable = True, bounds = (0,120), pen=G.primary_bound_style)
        self.b2 = pg.InfiniteLine(120,  angle = 90, movable = True, bounds = (0,120), pen=G.primary_bound_style)
        self.bcenter = pg.InfiniteLine(60, angle = 90, movable = True, bounds = (2,118), pen=G.secondary_bound_style)
        if not (G.r_b1 is None or G.r_b2 is None):
            self.b1.setValue(G.r_b1)
            self.b2.setValue(G.r_b2)
            self.bcenter.setValue((G.r_b1 + G.r_b2)/2)
        self.b1.sigPositionChanged.connect(self.gate_handler)
        self.b2.sigPositionChanged.connect(self.gate_handler)
        self.b1.sigDragged.connect(self.update_centergate)
        self.b2.sigDragged.connect(self.update_centergate)
        self.bcenter.sigDragged.connect(self.centergate_handler)
        if G.gate_state[1]:
            self.plot_item.addItem(self.b1)
            self.plot_item.addItem(self.b2)
            self.plot_item.addItem(self.bcenter)
        self.gate_handler()
        self.layout.addWidget(self.plot_widget, 0, 0, 1, 2)
        self.layout.addWidget(self.log_checkbox, 1, 0, 1, 1)
        self.layout.addWidget(self.jacobian_dropdown, 1, 1, 1, 1)
    def gate_handler(self):
        self.update_gate_values()
        self.update_index()
        self.parent().parent().mainplot.load()
    def update_centergate(self):
        self.bcenter.setValue((G.r_b1+G.r_b2)/2)
    def centergate_handler(self):
        delta = (G.r_b2-G.r_b1)/2
        self.b1.setValue(self.bcenter.value()-delta)
        self.b2.setValue(self.bcenter.value()+delta)
    def update_gate_values(self):
        if self.b2.value()>self.b1.value():
            G.r_b1 = self.b1.value()
            G.r_b2 = self.b2.value()
        else:
            G.r_b1 = self.b2.value()
            G.r_b2 = self.b1.value()
    def update_index(self):
        G.r_index = (G.r>G.r_b1) & (G.r<G.r_b2)
    def log_checkbox_handler(self):
        if self.log_checkbox.isChecked(): self.plot_item.setLogMode(y=True)
        else: self.plot_item.setLogMode(y=False)
        G.log_state[1] = self.log_checkbox.isChecked()
    def jacobian_changer(self, i):
        if i==0: self.density = True
        if i==1: self.density = False
        self.load()
    def load(self):
        rhist, bins = np.histogram(G.r, bins=np.arange(0,120,0.5))
        if self.density:
            rhist = rhist.astype(float)
            rhist*=1/bins[1:]
        self.plot.setData(bins[:-1], rhist)
        self.update_gate_values()
        self.update_index()
class XYPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(700)
        self.layout = QGridLayout(self)
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.plot = self.graphics_layout_widget.addPlot(title="XY")
        self.plot.setLabel('bottom', 'x [pix]')
        self.plot.setLabel('left', 'y [pix]')
        self.plot.getAxis("bottom").label.setFont(G.smallfont)
        self.plot.getAxis("bottom").setTickFont(G.smallfont)
        self.plot.getAxis("left").label.setFont(G.smallfont)
        self.plot.getAxis("left").setTickFont(G.smallfont)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.colorbar = self.plot.addColorBar(self.image_item, colorMap='viridis')
        self.choose_xy_button = QPushButton()
        self.choose_xy_button.setStyleSheet(G.whitestyle)
        if G.x0 is None or G.y0 is None:
            self.choose_xy_button.setText("Choose x0 y0")
            self.choose_xy_button.setStyleSheet(G.whitestyle)
        else:
            self.choose_xy_button.setText("x0 y0 calibrated")
            self.choose_xy_button.setStyleSheet(G.greenstyle)
        self.choose_xy_button.clicked.connect(self.choose_xy)
        self.bx0 = pg.InfiniteLine(128,  angle = 90, movable = True, bounds = (0.0,256), pen=G.tertiary_bound_style)
        self.by0 = pg.InfiniteLine(128,  angle = 0, movable = True, bounds = (0.0,256), pen=G.tertiary_bound_style)
        self.x_b1 = pg.InfiniteLine(0,  angle = 90, movable = True, bounds = (0.0,256), pen=G.primary_bound_style)
        self.x_b2 = pg.InfiniteLine(256,  angle = 90, movable = True, bounds = (0.0,256), pen=G.primary_bound_style)
        self.y_b1 = pg.InfiniteLine(0,  angle = 0, movable = True, bounds = (0.0,256), pen=G.primary_bound_style)
        self.y_b2 = pg.InfiniteLine(256,  angle = 0, movable = True, bounds = (0.0,256), pen=G.primary_bound_style)
        if not G.xy_bx1 is None: self.x_b1.setValue(G.xy_bx1)
        if not G.xy_bx2 is None: self.x_b2.setValue(G.xy_bx2)
        if not G.xy_by1 is None: self.y_b1.setValue(G.xy_by1)
        if not G.xy_by2 is None: self.y_b2.setValue(G.xy_by2)
        self.x_b1.sigPositionChanged.connect(self.gate_handler)
        self.x_b2.sigPositionChanged.connect(self.gate_handler)
        self.y_b1.sigPositionChanged.connect(self.gate_handler)
        self.y_b2.sigPositionChanged.connect(self.gate_handler)
        if G.gate_state[0]:
            self.plot.addItem(self.x_b1)
            self.plot.addItem(self.x_b2)
            self.plot.addItem(self.y_b1)
            self.plot.addItem(self.y_b2)
        self.gate_handler()
        self.log = True   #log scale checkbox
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setChecked(True)
        self.log_checkbox.stateChanged.connect(self.log_checkbox_handler)
        self.log_checkbox.setStyleSheet(G.minor_checkbox)
        self.log_checkbox.setFont(G.smallfont)
        self.layout.addWidget(self.graphics_layout_widget, 0, 0, 1, 2)
        self.layout.addWidget(self.log_checkbox, 1, 0, 1, 1)
        self.layout.addWidget(self.choose_xy_button, 1, 1, 1, 1)
    def gate_handler(self):
        self.update_gate_values()
        self.update_index()
        self.parent().parent().mainplot.load()
    def update_gate_values(self):
        if self.x_b2.value()>self.x_b1.value():
            G.xy_bx1 = self.x_b1.value()
            G.xy_bx2 = self.x_b2.value()
        else:
            G.xy_bx1 = self.x_b2.value()
            G.xy_bx2 = self.x_b1.value()
        if self.y_b2.value()>self.y_b1.value():
            G.xy_by1 = self.y_b1.value()
            G.xy_by2 = self.y_b2.value()
        else:
            G.xy_by1 = self.y_b2.value()
            G.xy_by2 = self.y_b1.value()
    def update_index(self):
        G.xy_index = (G.x>G.xy_bx1) & (G.x<G.xy_bx2) & (G.y>G.xy_by1) & (G.y<G.xy_by2)
    def choose_xy(self):
        if G.choosing_xy:
            self.choose_xy_button.setText("x0 y0 calibrated")
            self.choose_xy_button.setStyleSheet(G.greenstyle)
            self.plot.removeItem(self.bx0)
            self.plot.removeItem(self.by0)
            G.x0 = self.bx0.value()
            G.y0 = self.by0.value()
            G.xycenter()
            G.choosing_xy = False
            chooser = self.parent().parent().support_chooser
            chooser.show_checkboxes[1].setEnabled(True)
            chooser.show_checkboxes[3].setEnabled(True)
            chooser.gate_checkboxes[1].setEnabled(True)
            chooser.gate_checkboxes[3].setEnabled(True)
            self.parent().parent().mainparams.set_xy_dependent_options(True)
            self.parent().load()
            self.parent().parent().mainplot.load()
            return
        G.choosing_xy = True
        self.choose_xy_button.setText("Done")
        self.choose_xy_button.setStyleSheet(G.whitestyle)
        self.plot.addItem(self.bx0)
        self.plot.addItem(self.by0)
        if G.x0 is not None: self.bx0.setValue(G.x0)
        if G.y0 is not None: self.by0.setValue(G.y0)
    def load(self):
        histogram, b1, b2 = np.histogram2d(G.x, G.y, bins = (np.arange(0,256,1), np.arange(0,256,1)))
        if self.log:
            idx = (histogram>0)
            histogram = histogram.astype(float)
            histogram[idx] = np.log(histogram[idx])+1
        self.image_item.setImage(histogram)
        self.update_gate_values()
        self.update_index()
        self.colorbar.setLevels((histogram.min(),histogram.max()))
    def log_checkbox_handler(self): #set log mode based on checkbox and update the plots
        if self.log: self.log = False
        else: self.log = True
        G.log_state[0] = self.log
        self.hist.autoHistogramRange()
        self.load()
class MainParameterWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.xitems = ["x", "y", "tof", "r", "theta", "delay"]
        self.yitems = ["y", "x", "tof", "r", "theta", "delay"]
        self.layout=QGridLayout(self)
        self.dimension_label = QLabel("Dimension: ")
        self.dimension_dropdown = QComboBox()
        self.dimension_dropdown.addItems(["1D", "2D", "3D"])
        self.dimension_dropdown.currentIndexChanged.connect(self.dimension_changer)
        self.dimension_label.setStyleSheet(G.fancylabel)
        self.dimension_dropdown.setFont(G.normalfont)
        self.xaxis_label = QLabel("X Axis: ")
        self.xaxis_dropdown = QComboBox()
        self.xaxis_dropdown.addItems(self.xitems)
        self.xaxis_dropdown.currentIndexChanged.connect(self.xaxis_changer)
        self.xaxis_label.setStyleSheet(G.fancylabel)
        self.xaxis_dropdown.setFont(G.normalfont)
        self.yaxis_label = QLabel("Y Axis: ")
        self.yaxis_dropdown = QComboBox()
        self.yaxis_dropdown.addItems(self.yitems)
        self.yaxis_dropdown.currentIndexChanged.connect(self.yaxis_changer)
        self.yaxis_dropdown.setEnabled(False)
        self.yaxis_label.setStyleSheet(G.fancylabel)
        self.yaxis_dropdown.setFont(G.normalfont)
        self.normalize_label = QLabel("Normalize: ")
        self.normalize_dropdown = QComboBox()
        self.normalize_dropdown.addItems(["None", "Max", "Sum"])
        self.normalize_dropdown.currentIndexChanged.connect(self.normalize_changer)
        self.normalize_label.setStyleSheet(G.fancylabel)
        self.normalize_dropdown.setFont(G.normalfont)
        self.timewalk_launcher = QPushButton("Open Timewalk Correction")
        self.timewalk_launcher.clicked.connect(self.launch_timewalk)
        self.timewalk_launcher.setStyleSheet(G.whitestyle)
        self.grid_checkbox = QCheckBox("Show Grid")
        self.checkbox_checkbox = QCheckBox("Gate Controls")
        self.grid_checkbox.setStyleSheet(G.minor_checkbox)
        self.grid_checkbox.setFont(G.smallfont)
        self.checkbox_checkbox.setStyleSheet(G.minor_checkbox)
        self.checkbox_checkbox.setFont(G.smallfont)
        self.grid_checkbox.setChecked(True)
        self.checkbox_checkbox.setChecked(True)
        self.checkbox_checkbox.stateChanged.connect(self.checkbox_checkbox_handler)
        self.grid_checkbox.stateChanged.connect(self.grid_checkbox_handler)
        self.set_xy_dependent_options(False)
        self.dimension_dropdown.model().item(2).setEnabled(False)

        self.layout.addWidget(self.dimension_label, 0,0,1,1)
        self.layout.addWidget(self.dimension_dropdown, 0,1,1,1)
        self.layout.addWidget(self.xaxis_label, 1,0,1,1)
        self.layout.addWidget(self.xaxis_dropdown, 1,1,1,1)
        self.layout.addWidget(self.yaxis_label, 2,0,1,1)
        self.layout.addWidget(self.yaxis_dropdown, 2,1,1,1)
        self.layout.addWidget(self.normalize_label, 3, 0, 1, 1)
        self.layout.addWidget(self.normalize_dropdown, 3, 1, 1, 1)
        self.layout.addWidget(self.timewalk_launcher, 4, 0, 1, 2)
        self.layout.addWidget(self.grid_checkbox, 5, 0, 1, 1)
        self.layout.addWidget(self.checkbox_checkbox, 5, 1, 1, 1)
    def set_xy_dependent_options(self, state):
        for i, item in enumerate(self.xitems):
            if item == "r": self.xaxis_dropdown.model().item(i).setEnabled(state)
            if item == "theta": self.xaxis_dropdown.model().item(i).setEnabled(state)
        for i, item in enumerate(self.yitems):
            if item == "r": self.yaxis_dropdown.model().item(i).setEnabled(state)
            if item == "theta": self.yaxis_dropdown.model().item(i).setEnabled(state)
    def set_delay_dependent_options(self, state):
        for i, item in enumerate(self.xitems):
            if item == "delay": self.xaxis_dropdown.model().item(i).setEnabled(state)
        for i, item in enumerate(self.yitems):
            if item == "delay": self.yaxis_dropdown.model().item(i).setEnabled(state)
    def dimension_changer(self, i):
        if G.dimension != i+1:
            G.dimension = i+1
            self.parent().mainplot.change_dimension(i+1)
        if G.dimension == 2:
            self.yaxis_dropdown.setEnabled(True)
        if G.dimension == 1:
            self.yaxis_dropdown.setEnabled(False)
    def xaxis_changer(self, i):
        G.axes[0] = self.xitems[i]
        self.parent().mainplot.load()
    def yaxis_changer(self, i):
        G.axes[1] = self.yitems[i]
        self.parent().mainplot.load()
    def normalize_changer(self, i):
        G.normalize_type = i
        self.parent().mainplot.load()
        self.parent().mainplot.update_traces()
    def launch_timewalk(self):
        window = TimewalkCorrectorDialog()
        window.resize(1400, 800)
        result = window.exec()
        if result:
            self.parent().update_plot_data()
            self.timewalk_launcher.setStyleSheet(G.greenstyle)
            self.timewalk_launcher.setText("Timewalk Corrected")
            print("Applied timewalk correction")
        else: print("Rejected timewalk correction")
    def grid_checkbox_handler(self):
        self.parent().mainplot.setgrid(self.grid_checkbox.isChecked())
    def checkbox_checkbox_handler(self):
        self.parent().set_subplot_controls(self.checkbox_checkbox.isChecked())
    def load(self):
        self.timewalk_launcher.setStyleSheet(G.whitestyle)
        self.timewalk_launcher.setText("Open Timewalk Correction")
        self.set_delay_dependent_options(not G.static)
class TimewalkCorrectorDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timewalk Corrector")
        self.setMinimumWidth(800)
        self.setMinimumHeight(400)
        self.setStyleSheet(G.whitestyle)
        self.revertbutton = QPushButton("Revert")
        self.revertbutton.clicked.connect(self.revert)
        self.layout = QGridLayout(self)
        self.toftot = MouseInteractivePlot(self)
        self.confirmbutton = QPushButton("Confirm")
        self.confirmbutton.clicked.connect(self.confirm)
        self.cancelbutton = QPushButton("Cancel")
        self.cancelbutton.clicked.connect(self.reject)
        self.confirmbutton.setEnabled(False)
        self.revertbutton.setEnabled(False)
        self.fixtof = G.tof.copy()
        self.layout.addWidget(self.toftot, 0, 0, 1, 3)
        self.layout.addWidget(self.confirmbutton, 1, 0, 1, 1)
        self.layout.addWidget(self.revertbutton, 1, 1, 1, 1)
        self.layout.addWidget(self.cancelbutton, 1, 2, 1, 1)
        self.tof_lut = np.zeros(100)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor('#202020'))
        self.setPalette(p)
        self.revert()
    def confirm(self):
        G.tof = self.fixtof
        self.accept()
    def revert(self):
        self.fixtof = G.tof.copy()
        self.update()
        self.toftot.plot.vb.setRange(xRange=(4.5,6.5),yRange=((0,1)))
        self.toftot.plot.vb.setLimits(xMin=0,xMax=15,yMin=0,yMax=2)
    def fix(self, mask):
        timewalk_hist, b1, b2 = np.histogram2d(self.fixtof, G.tot, bins=(np.arange(0,15,0.005),np.arange(0,25,0.025)))
        timewalk_hist[~mask] = 0
        mom = timewalk_hist*b1[:-1,np.newaxis]
        denom = np.sum(timewalk_hist,axis=0)
        ignore = (denom==0)
        denom[ignore] = 1
        averages = np.sum(mom, axis=0)/denom
        averages-=averages[~ignore].min()
        averages[averages<0]=0
        self.fixtof -= averages[(G.tot*40).astype(int)]
        self.confirmbutton.setEnabled(True)
        self.revertbutton.setEnabled(True)
        self.update()
    def update(self):
        histogram, b1, b2 = np.histogram2d(self.fixtof, G.tot, bins=(np.arange(0,15,0.005),np.arange(0,25,0.025)))
        idx = (histogram>0)
        histogram = histogram.astype(float)
        histogram[idx] = np.log(histogram[idx])+1
        self.toftot.setData(histogram)
        self.toftot.colorbar.setLevels((histogram.min(),histogram.max()))
class MouseInteractivePlot(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mouse_pressed = False
        self.image_item = pg.ImageItem()
        self.image = np.zeros((200,200))
        self.image_item.setImage(self.image)
        self.modsize = 7
        self.plot = self.addPlot()
        self.plot.addItem(self.image_item)
        self.colorbar = self.plot.addColorBar(self.image_item, colorMap='turbo')
        self.validsegment = False
        self.segmentcomplete = False
        self.datapoints = []
        self.scenepoints = []
        self.overlay = pg.PlotDataItem()
        self.overlay.setPen(pg.mkPen(width=3,color=((128,0,255))))
        self.plot.addItem(self.overlay)
        self.plot.vb.setRange(xRange=(4.5,6.5),yRange=((0,1)))
        self.plot.vb.setLimits(xMin=0,xMax=15,yMin=0,yMax=25)
        self.plot.setLabel('bottom', 'tof [μs]')
        self.plot.setLabel('left', 'tot [μs]')
        self.plot.getAxis("bottom").label.setFont(G.smallfont)
        self.plot.getAxis("bottom").setTickFont(G.smallfont)
        self.plot.getAxis("left").label.setFont(G.smallfont)
        self.plot.getAxis("left").setTickFont(G.smallfont)
    def setData(self, image):
        self.image = image
        self.image_item.setImage(self.image)
        self.image_item.setRect(QRectF(0,0,15,25))
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pressed=True
    def mouseMoveEvent(self, event):
        if self.mouse_pressed:
            scene_pos = self.mapToScene(event.pos())
            data_pos = self.plot.vb.mapSceneToView(scene_pos)
            x = data_pos.x()
            y = data_pos.y()
            scenex = scene_pos.x()
            sceney = scene_pos.y()
            lenp = len(self.scenepoints)
            if lenp==0 or math.dist([scenex-self.scenepoints[lenp-1][0]], [sceney-self.scenepoints[lenp-1][1]])>5:
                self.datapoints.append((x,y))
                self.scenepoints.append((scenex, sceney))
                d = math.dist([scenex-self.scenepoints[0][0]],[sceney-self.scenepoints[0][1]])
                if self.validsegment:
                    if d<10:
                        self.overlay.setPen(pg.mkPen(width=3,color=((128,128,255))))
                        self.segmentcomplete = True
                elif d>30: self.validsegment = True
                self.updateOverlay()
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_pressed = False
            if self.segmentcomplete:
                shape = self.image.shape
                inside = polygon_to_mask(shape,self.datapoints,shape[0]/15,shape[1]/25)
                self.parent().fix(inside)
            self.segmentcomplete = False
            self.datapoints = []
            self.scenepoints = []
            self.validsegment = False
            self.overlay.setPen(pg.mkPen(width=3,color=((128,0,255))))
            self.updateOverlay()
    def updateOverlay(self):
        self.overlay.setData(np.array(self.datapoints))
class MainPlotHolder(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout=QGridLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True)
        self.plot = pg.PlotDataItem()
        self.plot.setPen(width=3)
        self.plot_item.addItem(self.plot)
        self.log_checkbox = QCheckBox("Log")
        self.log_checkbox.setStyleSheet(G.minor_checkbox)
        self.log_checkbox.setFont(G.smallfont)
        self.log_checkbox.stateChanged.connect(self.log_checkbox_handler)
        self.log = False
        self.plot_item.getAxis("bottom").label.setFont(G.smallfont)
        self.plot_item.getAxis("bottom").setTickFont(G.smallfont)
        self.plot_item.getAxis("left").label.setFont(G.smallfont)
        self.plot_item.getAxis("left").setTickFont(G.smallfont)
        self.layout.addWidget(self.plot_widget, 0, 0, 1, 1)
        self.layout.addWidget(self.log_checkbox, 1, 0, 1, 1)
        self.additional_traces = []
    def update_traces(self):
        for trace in self.additional_traces:
            self.plot_item.removeItem(trace)
        for i, trace in enumerate(G.traces):
            if G.tracestates[i]:
                x, y = trace
                if G.normalize_type>0:
                    x = x.astype(float)
                    y = y.astype(float)
                    if G.normalize_type == 1:
                        y/=y.max()
                    elif G.normalize_type == 2:
                        y/=y.sum()
                newtrace = pg.PlotDataItem(x, y)
                newtrace.setPen(pg.mkPen(width=3,color=G.colorvalues[G.tracecolors[i]]))
                self.additional_traces.append(newtrace)
                self.plot_item.addItem(newtrace)
    def load(self):
        index = np.full_like(G.x, True, bool)
        if G.gate_state[0]:index &= G.xy_index
        if G.gate_state[1]:index &= G.r_index
        if G.gate_state[2]:index &= G.tof_index
        if G.gate_state[3]:index &= G.theta_index
        if G.dimension==1:
            if G.axes[0] == "x":
                hist, bins = np.histogram(G.x[index], bins=np.arange(0,256))
                self.plot_item.setLabel('bottom', 'x [pix]')
            if G.axes[0] == "y":
                hist, bins = np.histogram(G.y[index], bins=np.arange(0,256))
                self.plot_item.setLabel('bottom', 'y [pix]')
            if G.axes[0] == "r":
                hist, bins = np.histogram(G.r[index], bins=np.arange(0,120,0.5))
                self.plot_item.setLabel('bottom', 'r [pix]')
            if G.axes[0] == "tof":
                hist, bins = np.histogram(G.tof[index], bins=np.arange(0,15,0.003))
                self.plot_item.setLabel('bottom', 'tof [μs]')
            if G.axes[0] == "theta":
                hist, bins = np.histogram(G.theta[index], bins=np.arange(-np.pi,np.pi,0.02))
                self.plot_item.setLabel('bottom', 'theta [rad]')
            if G.axes[0] == "delay":
                hist, bins = np.histogram(G.pos[index], bins=np.arange(len(G.dataset['zaber_micrometers'])+1))
                bins[:-1] = G.fs[bins[:-1]]
                self.plot_item.setLabel('bottom', 'delay [fs]')
            G.main_curve = (bins[:-1], hist)
            if G.normalize_type>0:
                hist = hist.astype(float)
                if G.normalize_type == 1:
                    hist/=hist.max()
                elif G.normalize_type == 2:
                    hist/=hist.sum()
            self.plot.setData(bins[:-1], hist)
            self.update_traces()
        if G.dimension == 2:
            center = [0,0]
            width = [0,0]
            params = [None, None]
            bins = [None, None]
            labels = [None, None]
            for i, ax in enumerate(G.axes):
                match ax:
                    case "x":
                        params[i] = G.x[index]
                        bins[i] = np.arange(0,256)
                        center[i] = 0
                        width[i] = 256
                        labels[i] = 'x [pix]'
                    case "y":
                        params[i] = G.y[index]
                        bins[i] = np.arange(0,256)
                        center[i] = 0
                        width[i] = 256
                        labels[i] = 'y [pix]'
                    case "r":
                        params[i] = G.r[index]
                        bins[i] = np.arange(0,120,0.5)
                        center[i] = 0
                        width[i] = 120
                        labels[i] = 'r [pix]'
                    case "tof":
                        params[i] = G.tof[index]
                        bins[i] = np.arange(0,15,0.003)
                        center[i] = 0
                        width[i] = 20
                        labels[i] = 'tof [μs]'
                    case "theta":
                        params[i] = G.theta[index]
                        bins[i] = np.arange(-np.pi,np.pi,0.02)
                        center[i] = -np.pi
                        width[i] = 2*np.pi
                        labels[i] = 'theta [rad]'
                    case "delay":
                        params[i] = G.pos[index]
                        bins[i] = np.arange(len(G.pos_axis)+1)
                        center[i] = G.fs[0]
                        width[i] = G.fs[-1]-G.fs[0]
                        labels[i] = 'delay [fs]'
            histogram, b1, b2 = np.histogram2d(*params, bins=bins)
            if self.log:
                idx = (histogram>0)
                histogram = histogram.astype(float)
                histogram[idx] = np.log(histogram[idx])+1
            self.image_item.setImage(histogram)
            self.image_item.setRect(QRectF(center[0],center[1],width[0], width[1]))
            self.colorbar.setLevels((histogram.min(),histogram.max()))
            self.plot.setLabel('bottom', labels[0])
            self.plot.setLabel('left', labels[1])
        self.setgrid(self.parent().mainparams.grid_checkbox.isChecked())
    def change_dimension(self, dimension):
        self.layout.removeWidget(self.plot_widget)
        self.plot_widget.deleteLater()
        if dimension == 1:
            self.plot_widget = pg.PlotWidget()
            self.plot_item = self.plot_widget.getPlotItem()
            self.plot = pg.PlotDataItem(np.arange(100), np.arange(100))
            self.plot.setPen(width=3)
            self.plot_item.addItem(self.plot)
            self.layout.addWidget(self.plot_widget, 0, 0, 1, 1)
            self.plot_item.getAxis("bottom").label.setFont(G.smallfont)
            self.plot_item.getAxis("bottom").setTickFont(G.smallfont)
            self.plot_item.getAxis("left").label.setFont(G.smallfont)
            self.plot_item.getAxis("left").setTickFont(G.smallfont)
        if dimension == 2:
            self.plot_widget = pg.GraphicsLayoutWidget()
            self.plot = self.plot_widget.addPlot()
            self.image_item = pg.ImageItem()
            self.plot.addItem(self.image_item)
            self.colorbar = self.plot.addColorBar(self.image_item, colorMap='viridis')
            self.layout.addWidget(self.plot_widget, 0, 0, 1, 1)
            self.plot.getAxis("bottom").label.setFont(G.smallfont)
            self.plot.getAxis("bottom").setTickFont(G.smallfont)
            self.plot.getAxis("left").label.setFont(G.smallfont)
            self.plot.getAxis("left").setTickFont(G.smallfont)
        self.load()
        self.log_checkbox_handler()
    def setgrid(self, state):
        if G.dimension==1:
            self.plot_item.showGrid(x=state, y=state)
        if G.dimension==2:
            self.plot.showGrid(x=state, y=state)
    def log_checkbox_handler(self):
        if G.dimension==1:
            if self.log_checkbox.isChecked(): self.plot_item.setLogMode(y=True)
            else: self.plot_item.setLogMode(y=False)
        self.log = self.log_checkbox.isChecked()
        if not G.dataset is None: self.load()
class TraceManager(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setMaximumWidth(300)
        self.layout=QGridLayout(self)
        self.savebutton = QPushButton("Save Trace")
        self.savebutton.setStyleSheet(G.whitestyle)
        self.savebutton.clicked.connect(self.save)
        self.showcat_checkbox = QCheckBox("Catalog")
        self.showcat_checkbox.setStyleSheet(G.minor_checkbox)
        self.showcat_checkbox.setFont(G.smallfont)
        self.showcat_checkbox.stateChanged.connect(self.showcat_checkbox_handler)
        self.tracecat = TraceCatalog(self)
        self.layout.addWidget(self.savebutton, 0, 0, 1, 1)
        self.layout.addWidget(self.showcat_checkbox, 0, 1, 1, 1)
        self.layout.addWidget(self.tracecat, 1, 0, 1, 2)
        self.tracecat.hide()
    def save(self):
        if G.dataset is None:
            popup("Load a dataset first")
            return
        if G.dimension>1:
            popup("Traces only work in 1D right now")
            return
        window = SaveTraceDialog(self)
        result = window.exec()
        if result:
            self.parent().mainplot.update_traces()
        else: print("Canceled trace save")
    def showcat_checkbox_handler(self):
        self.parent().set_trace_controls(self.showcat_checkbox.isChecked())
class TraceCatalog(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout=QGridLayout(self)
        self.setFont(G.normalfont)
        self.titlelabel = QLabel("Trace Catalog")
        self.titlelabel.setStyleSheet(G.fancylabel)
        self.catalog = QListWidget(self)
        self.catalog.itemPressed.connect(self.trace_select_handler)
        self.showhide_button = QPushButton("Show")
        self.showhide_button.clicked.connect(self.showhide)
        self.showhide_button.setStyleSheet(G.whitestyle)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_trace)
        self.delete_button.setStyleSheet(G.whitestyle)
        self.showhide_button.setFont(G.smallfont)
        self.delete_button.setFont(G.smallfont)
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(G.colorkey)
        self.showhide_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.color_dropdown.setEnabled(False)
        self.color_dropdown.currentIndexChanged.connect(self.color_drop_handler)
        self.selected_trace = -1
        self.layout.addWidget(self.titlelabel, 0, 0, 1, 2)
        self.layout.addWidget(self.catalog, 1, 0, 1, 2)
        self.layout.addWidget(self.showhide_button, 2, 0, 1, 1)
        self.layout.addWidget(self.delete_button, 2, 1, 1, 1)
        self.layout.addWidget(self.color_dropdown, 3, 0, 1, 2)
    def update_traces(self, removeFirst = True):
        while self.catalog.count():
            self.catalog.takeItem(0)
        for i, tracename in enumerate(G.tracenames):
            item = QListWidgetItem(tracename)
            color =  G.colorvalues[G.tracecolors[i]]
            if i==self.selected_trace:
                item.setForeground(QColor(50,50,50))
                item.setBackground(QColor(*[int(c*0.9) for c in color]))
            else:
                item.setForeground(QColor("black"))
                item.setBackground(QColor(*color))
            self.catalog.insertItem(i, item)
    def trace_select_handler(self, item):
        i=self.catalog.row(item)
        self.trace_selector(i)
    def trace_selector(self, i):
        self.selected_trace = i
        if i==-1:
            self.color_dropdown.setEnabled(False)
            self.showhide_button.setEnabled(False)
            self.delete_button.setEnabled(False)
        else:
            self.showhide_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.color_dropdown.setEnabled(True)
            self.color_dropdown.setCurrentIndex(G.tracecolors[i])
            self.update_traces(removeFirst=False)
            self.update_showhide_text()
    def delete_trace(self):
        G.tracenames.pop(self.selected_trace)
        G.traces.pop(self.selected_trace)
        G.tracestates.pop(self.selected_trace)
        G.tracecolors.pop(self.selected_trace)
        if self.selected_trace != 0:
            self.trace_selector(self.selected_trace-1)
        elif len(G.tracenames)==1:
            self.trace_selector(0)
        else:
            self.trace_selector(-1)
            self.update_traces(removeFirst=False)
            if G.dimension==1: self.parent().parent().mainplot.update_traces()
    def showhide(self):
        G.tracestates[self.selected_trace] = not G.tracestates[self.selected_trace]
        self.update_showhide_text()
        self.parent().parent().mainplot.update_traces()
    def update_showhide_text(self):
        if G.tracestates[self.selected_trace]:self.showhide_button.setText("Hide")
        else:self.showhide_button.setText("Show")
    def color_drop_handler(self, i):
        if self.selected_trace!=-1:
            G.tracecolors[self.selected_trace] = i
            self.parent().parent().mainplot.update_traces()
            self.update_traces(removeFirst = False)
class SaveTraceDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Save Trace")
        self.setStyleSheet(G.whitestyle)
        self.setMinimumWidth(300)
        self.setMinimumHeight(100)
        self.savename = ""
        self.layout = QGridLayout(self)
        self.confirmbutton = QPushButton("Confirm")
        self.confirmbutton.clicked.connect(self.confirm)
        self.cancelbutton = QPushButton("Cancel")
        self.cancelbutton.clicked.connect(self.reject)
        self.confirmbutton.setStyleSheet(G.whitestyle)
        self.cancelbutton.setStyleSheet(G.whitestyle)
        self.savename_textfield = QLineEdit()
        self.savename_textfield.setStyleSheet(G.whitestyle)
        self.savename_textfield.textChanged.connect(self.textfield_handler)
        self.layout.addWidget(self.savename_textfield, 0, 0, 1, 2)
        self.layout.addWidget(self.confirmbutton, 1, 0, 1, 1)
        self.layout.addWidget(self.cancelbutton, 1, 1, 1, 1)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor('#202020'))
        self.setPalette(p)
    def textfield_handler(self, s):
        self.savename = s
    def confirm(self):
        if self.savename == "": popup("No name chosen!")
        elif self.savename in G.tracenames: popup("Trace name already used!")
        else:
            print(f"Trace saved as '{self.savename}'")
            G.tracenames.append(self.savename)
            G.traces.append(G.main_curve)
            G.tracestates.append(True)
            G.tracecolors.append(0)
            self.parent().tracecat.update_traces()
            self.parent().tracecat.trace_selector(len(G.traces)-1)
            self.accept()
class SupportPlotChooser(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFont(G.normalfont)
        self.layout = QGridLayout(self)
        self.show_checkboxes = []
        self.gate_checkboxes = []
        self.showlabel = QLabel("Show:")
        self.gatelabel = QLabel("Gate:")
        self.setStyleSheet(G.fancylabel)
        self.layout.addWidget(self.showlabel, 0,0,1,1)
        self.layout.addWidget(self.gatelabel, 1,0,1,1)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor('#303030'))
        self.setPalette(p)
        for i, item in enumerate(G.gate_items):
            self.show_checkboxes.append(QCheckBox(text=item))
            self.gate_checkboxes.append(QCheckBox(text=item))
            self.show_checkboxes[i].stateChanged.connect(self.update_checkbox_states)
            self.gate_checkboxes[i].stateChanged.connect(self.update_checkbox_states)
            self.layout.addWidget(self.show_checkboxes[i], 0,i+1,1,1)
            self.layout.addWidget(self.gate_checkboxes[i], 1,i+1,1,1)
            self.show_checkboxes[i].setEnabled(False)
            self.gate_checkboxes[i].setEnabled(False)
    def update_checkbox_states(self):
        for i, box in enumerate(self.show_checkboxes):
            if box.isChecked(): G.show_state[i] = True
            else: G.show_state[i] = False
        for i, box in enumerate(self.gate_checkboxes):
            if box.isChecked(): G.gate_state[i] = True
            else: G.gate_state[i] = False
        if not any(G.show_state):
            self.parent().mainplot.load() #otherwise subplots will do this
        self.parent().support_container.update_plot_setup()
        if G.dataset is not None: self.parent().support_container.load()
    def load(self):
        for i, item in enumerate(G.gate_items):
            if i!=1 and i!=3 and i<4:
                self.show_checkboxes[i].setEnabled(True)
                self.gate_checkboxes[i].setEnabled(True)
class SupportPlotContainer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
    def update_plot_setup(self):
        while self.layout.count():  #remove all the plots
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for i, p in enumerate(G.show_state):
            if p and G.gate_items[i]=="xy":
                self.xyplot = XYPlot(self)
                self.layout.addWidget(self.xyplot)
                self.xyplot.log_checkbox.setChecked(G.log_state[i])
            if p and G.gate_items[i]=="r":
                self.rplot = RPlot(self)
                self.layout.addWidget(self.rplot)
                self.rplot.log_checkbox.setChecked(G.log_state[i])
            if p and G.gate_items[i]=="tof":
                self.tofplot = TofPlot(self)
                self.layout.addWidget(self.tofplot)
                self.tofplot.log_checkbox.setChecked(G.log_state[i])
            if p and G.gate_items[i]=="theta":
                self.thetaplot = ThetaPlot(self)
                self.layout.addWidget(self.thetaplot)
                self.thetaplot.log_checkbox.setChecked(G.log_state[i])
    def load(self):
        for i, p in enumerate(G.show_state):
            if p and G.gate_items[i]=="tof":self.tofplot.load()
            if p and G.gate_items[i]=="r":self.rplot.load()
            if p and G.gate_items[i]=="xy":self.xyplot.load()
            if p and G.gate_items[i]=="theta":self.thetaplot.load()
class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.loader = FileLoader(self)
        self.mainparams = MainParameterWidget(self)
        self.mainparams.setEnabled(False)
        self.mainplot = MainPlotHolder(self)
        self.tracemanager = TraceManager(self)
        self.support_chooser = SupportPlotChooser(self)
        self.support_chooser.setEnabled(False)
        self.support_container = SupportPlotContainer(self)
        self.mainparams.setMaximumWidth(300)
        self.layout.addWidget(self.loader, 0,0,1,2)
        self.layout.addWidget(self.mainparams, 1,0,1,1)
        self.layout.addWidget(self.mainplot, 1, 1, 1, 2)
        self.layout.addWidget(self.tracemanager, 0, 2, 1, 1)
        self.layout.addWidget(self.support_chooser, 2,0,1,3)
        self.layout.addWidget(self.support_container, 3,0,1,3)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor('#202020'))
        self.setPalette(p)
    def load(self):
        self.mainparams.setEnabled(True)
        G.tof = np.array(G.dataset['tof'])
        G.x = np.array(G.dataset['x'])
        G.y = np.array(G.dataset['y'])
        G.tot = np.array(G.dataset['tot'])
        if "zaber_micrometers" in G.dataset.keys():
            G.static=False
            key = G.dataset['zaber_micrometers']
            G.fs = (np.array(G.dataset['zaber_micrometers'])-G.t0)*2*10/2.99
            G.pos_axis = np.array(G.dataset['zaber_position_index'])
            pos = np.array(G.dataset['pos'])
            G.pos = np.mod(pos, len(key))
            print(f"Loaded time resolved dataset {os.path.basename(self.loader.text)}\n Positions: {len(key)} Repeats: {(np.max(pos)+1)//len(key)}")
        else:
            G.static=True
            print(f"Loaded static dataset {os.path.basename(self.loader.text)}")
        placeholder_index = np.full_like(G.x, True, dtype=bool)
        G.xy_index = placeholder_index
        G.r_index = placeholder_index
        G.tof_index = placeholder_index
        G.theta_index = placeholder_index
        self.update_plot_data()
    def set_subplot_controls(self, state):
        if state:
            self.support_chooser.show()
        else:
            self.support_chooser.hide()
            self.adjustSize()
    def set_trace_controls(self, state):
        if state:
            self.tracemanager.tracecat.show()
            self.layout.removeWidget(self.tracemanager)
            self.layout.removeWidget(self.mainplot)
            self.layout.addWidget(self.tracemanager, 0, 2, 2, 1)
            self.layout.addWidget(self.mainplot, 1, 1, 1, 1)
        else:
            self.tracemanager.tracecat.hide()
            self.layout.removeWidget(self.tracemanager)
            self.layout.removeWidget(self.mainplot)
            self.layout.addWidget(self.tracemanager, 0, 2, 1, 1)
            self.layout.addWidget(self.mainplot, 1, 1, 1, 2)
    def update_plot_data(self):
        G.xycenter()
        self.support_chooser.setEnabled(True)
        self.support_container.load()
        self.support_chooser.load()
        self.mainplot.load()
        self.mainparams.load()

def popup(text):
    alert = QMessageBox()
    alert.setWindowTitle("Alert")
    alert.setText(text)
    alert.setIcon(QMessageBox.Warning)
    alert.setStandardButtons(QMessageBox.Ok)
    alert.exec()
def polygon_to_mask(shape, points, scalex, scaley):
    points = np.array(points).T
    px, py = points[0], points[1]
    px = np.floor(px*scalex)
    py = np.floor(py*scaley)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))  # x=cols, y=rows
    inside = np.zeros((shape[1], shape[0]), dtype=bool)
    for i in range(len(px)):
        denom = (py[i-1] - py[i]) if (py[i-1] - py[i]) != 0 else 1e-12
        condition = ((py[i] > y) != (py[i-1] > y)) & \
                    (x < (px[i-1] - px[i]) * (y - py[i]) / denom + px[i])
        inside ^= condition
    return inside.T
def main():
    app = QApplication(sys.argv)
    w = QMainWindow()
    w.setWindowTitle("SBUTPX Quick Analyze")
    main_widget = MainWidget()
    w.setCentralWidget(main_widget)
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    G = GlobalVariables()
    main()