import datetime
import logging
import os
import tkinter as tk
from tkinter import StringVar, DoubleVar, BooleanVar, OptionMenu
from tkinter import ttk

import requests

from .. import serval
# Constants
DEFAULT_FOLDER_NAME = "test"
DEFAULT_DURATION = 60.0
DEFAULT_DURATION_UNIT = "sec"
DESTINATION_BASE_PATH = ""
STATUS_IDLE = "DA_IDLE"
STATUS_RECORDING = "DA_RECORDING"
STATUS_PREFIX = "DA_"
UPDATE_INTERVAL_MS = 100  # 100 milliseconds
INFINITE_DURATION = 999999999
FRAME_TIME = 1

COLOR_IDLE = "black"
COLOR_BUSY = "green"
COLOR_ERROR = "red"


class AcquisitionUI(ttk.Frame):
    def __init__(self, master, serval_test_dir="C:\\serval_test", default_dir=""):
        super().__init__(master)
        try:
            serval_init = serval.get_dash().Measurement is not None
            if not serval_init:
                try:
                    serval.set_acquisition_parameters(serval_test_dir, 1, frame_time=1)
                    serval.start_acquisition()
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

        # Variables
        self.default_dir = default_dir
        self.folder_name = StringVar(value=os.path.join(datetime.datetime.now().strftime(default_dir), DEFAULT_FOLDER_NAME))
        self.duration_value = DoubleVar(value=DEFAULT_DURATION)
        self.infinite = BooleanVar(value=False)
        self.duration_unit = StringVar(value=DEFAULT_DURATION_UNIT)

        # Server connection state
        self.server_connected = False

        self.status = StringVar(value=STATUS_IDLE)
        self.elapsed_str = StringVar(value="")
        self.start_str = StringVar(value="")
        self.end_str = StringVar(value="")
        self.progress_var = DoubleVar(value=0.0)

        self._initialize_logging()
        self._create_widgets()
        self._bind_events()

        # Attempt server connection
        self._attempt_server_connection()

        # Start status updates
        self.update_status()

    def _initialize_logging(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _create_widgets(self):
        param_frame = ttk.LabelFrame(self, text="Acquisition Parameters", padding=10)
        param_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        status_frame = ttk.LabelFrame(self, text="Status", padding=10)
        status_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        param_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(1, weight=1)

        # Folder name
        ttk.Label(param_frame, text="Folder Name:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.folder_entry = ttk.Entry(param_frame, textvariable=self.folder_name)
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        duration_frame = ttk.Frame(param_frame)
        duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(duration_frame, text="Duration:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.duration_entry = ttk.Entry(
            duration_frame, textvariable=self.duration_value, width=8
        )
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.duration_unit_menu = OptionMenu(
            duration_frame, self.duration_unit, "sec", "min", "hr"
        )
        self.duration_unit_menu.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        self.infinite_check = ttk.Checkbutton(
            duration_frame,
            text="Run Infinitely",
            variable=self.infinite,
            command=self.toggle_infinite_mode,
        )
        self.infinite_check.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        busy_frame = ttk.Frame(status_frame)
        busy_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.busy_canvas = tk.Canvas(
            busy_frame, width=16, height=16, highlightthickness=0
        )
        self.busy_canvas.grid(row=0, column=0, padx=5)
        self.draw_busy_indicator()

        ttk.Label(busy_frame, text="Status:").grid(row=0, column=1, sticky="w")
        status_label = ttk.Label(busy_frame, textvariable=self.status)
        status_label.grid(row=0, column=2, sticky="w", padx=5)

        self.progress_bar = ttk.Progressbar(
            status_frame,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_var,
        )
        self.progress_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )

        ttk.Label(status_frame, text="Start:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        self.start_label = ttk.Label(status_frame, textvariable=self.start_str)
        self.start_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(status_frame, text="End:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        self.end_label = ttk.Label(status_frame, textvariable=self.end_str)
        self.end_label.grid(row=3, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="Elapsed:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        self.elapsed_label = ttk.Label(status_frame, textvariable=self.elapsed_str)
        self.elapsed_label.grid(row=4, column=1, sticky="w", padx=5)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(
            button_frame, text="Start", command=self.start_acquisition
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.stop_acquisition
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=5)

    def _bind_events(self):
        pass

    def draw_busy_indicator(self):
        self.busy_canvas.delete("all")
        match self.status.get():
            case status if status == STATUS_IDLE:
                color = COLOR_IDLE
            case status if status.startswith(STATUS_PREFIX):
                color = COLOR_BUSY
            case _:
                color = COLOR_ERROR

        self.busy_canvas.create_oval(2, 2, 14, 14, fill=color, outline="")

    def convert_duration_to_seconds(self):
        unit = self.duration_unit.get()
        duration = self.duration_value.get()
        if unit == "min":
            return duration * 60
        elif unit == "hr":
            return duration * 3600
        return duration

    def toggle_infinite_mode(self):
        if self.infinite.get():
            self.duration_entry.config(state="disabled")
            self.duration_unit_menu.config(state="disabled")
        else:
            self.duration_entry.config(state="normal")
            self.duration_unit_menu.config(state="normal")

    def _attempt_server_connection(self):
        try:
            self._connect_to_server()
            self.server_connected = True
            logging.info("Successfully connected to the server.")
        except Exception as e:
            logging.error(f"Failed to connect to the server: {e}")
            self.server_connected = False
            self._disable_server_dependent_widgets()

    @staticmethod
    def _connect_to_server():
        try:
            serval.get_dash()
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Failed to connect to the server: {e}")
            raise e

    def _disable_server_dependent_widgets(self):
        # Disable widgets that depend on the server connection
        self.folder_entry.config(state="disabled")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.duration_entry.config(state="disabled")
        self.duration_unit_menu.config(state="disabled")
        self.infinite_check.config(state="disabled")
        self.status.set("Serval connection failed. Acquisition disabled.")
        self.draw_busy_indicator()

    def _enable_server_dependent_widgets(self):
        self.stop_button.config(state="normal")
        self.status.set("DA_IDLE")

    def update_status(self):
        if not self.server_connected:
            # If server is not connected, skip updating status
            self.after(UPDATE_INTERVAL_MS, self.update_status)
            return

        dash = serval.get_dash()
        meas = dash.Measurement
        current_status = meas.Status if meas is not None else None

        if current_status is not None:
            self.status.set(current_status)
            self.draw_busy_indicator()

        if current_status == STATUS_IDLE:
            self._handle_idle_status()
        elif current_status == STATUS_RECORDING:
            self._handle_recording_status(meas)

        self.after(UPDATE_INTERVAL_MS, self.update_status)

    def _handle_idle_status(self):
        self.progress_var.set(0)
        self.start_str.set("")
        self.end_str.set("")
        self.elapsed_str.set("")
        self.progress_bar["value"] = 0

        self.start_button.config(state="normal")
        self.folder_entry.config(state="normal")
        self.infinite_check.config(state="normal")

        if not self.infinite.get():
            self.duration_entry.config(state="normal")
            self.duration_unit_menu.config(state="normal")

    def _handle_recording_status(self, meas):
        self.start_button.config(state="disabled")
        self.folder_entry.config(state="disabled")
        self.infinite_check.config(state="disabled")
        self.duration_entry.config(state="disabled")
        self.duration_unit_menu.config(state="disabled")

        start_ms = meas.StartDateTime
        elapsed = meas.ElapsedTime
        timeleft = meas.TimeLeft
        framecount = meas.FrameCount

        if start_ms is not None:
            start_dt = datetime.datetime.fromtimestamp(start_ms / 1000.0)
            self.start_str.set(self._format_start_time(start_dt))

        if elapsed is not None:
            self.elapsed_str.set(self._format_elapsed_time(elapsed))

        if self.infinite.get():
            self.end_str.set("")
        else:
            self.end_str.set(
                self._calculate_end_time(meas, start_dt) if start_ms is not None else ""
            )

        self._update_progress_bar(meas)

    def _format_start_time(self, start_dt):
        today = datetime.datetime.now().date()
        if start_dt.date() != today:
            return start_dt.strftime("%Y-%m-%d %I:%M:%S %p").replace(" 0", " ")
        else:
            return start_dt.strftime("%I:%M:%S %p").lstrip("0")

    def _format_elapsed_time(self, elapsed):
        elapsed_seconds = int(elapsed)
        milliseconds = int((elapsed - elapsed_seconds) * 1000)

        if elapsed < 60:
            return f"{elapsed_seconds}.{milliseconds:03}"
        elif elapsed < 3600:
            minutes, seconds = divmod(elapsed_seconds, 60)
            return f"{int(minutes)}:{int(seconds):02}.{milliseconds:03}"
        elif elapsed < 86400:
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"
        else:
            days = elapsed // 86400
            hours, remainder = divmod(elapsed_seconds % 86400, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(days)}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"

    def _calculate_end_time(self, meas, start_dt):
        elapsed = meas.ElapsedTime
        timeleft = meas.TimeLeft
        framecount = meas.FrameCount

        if elapsed is not None and timeleft is not None and framecount is not None:
            predicted_offset = timeleft - (elapsed - (FRAME_TIME * framecount))
            end_dt = datetime.datetime.now() + datetime.timedelta(
                seconds=predicted_offset
            )
            if start_dt.date() != end_dt.date():
                return end_dt.strftime("%Y-%m-%d %I:%M:%S %p").replace(" 0", " ")
            else:
                return end_dt.strftime("%I:%M:%S %p").lstrip("0")
        return ""

    def _update_progress_bar(self, meas):
        elapsed = meas.ElapsedTime
        if self.infinite.get():
            self.progress_var.set(0)
        elif elapsed is not None:
            total_duration = self.convert_duration_to_seconds()
            if total_duration > 0:
                fraction = min(max(elapsed / total_duration, 0.0), 1.0)
                self.progress_var.set(fraction * 100)
            else:
                self.progress_var.set(0)

    def start_acquisition(self):
        if not self.server_connected:
            logging.error("Cannot start acquisition: Server is not connected.")
            return

        folder = self.folder_name.get()
        duration = self.convert_duration_to_seconds()
        if self.infinite.get():
            duration = INFINITE_DURATION
        try:
            serval.set_acquisition_parameters(folder, duration)
            serval.start_acquisition(block=False)
            logging.info("Acquisition started.")
        except Exception as e:
            logging.error(f"Failed to start acquisition: {e}")

    def stop_acquisition(self):
        if not self.server_connected:
            logging.error("Cannot stop acquisition: Server is not connected.")
            return

        try:
            serval.stop_acquisition()
            logging.info("Acquisition stopped.")
        except Exception as e:
            logging.error(f"Failed to stop acquisition: {e}")

        self.start_button.config(state="normal")
        self.folder_entry.config(state="normal")
        self.infinite_check.config(state="normal")
        self.toggle_infinite_mode()
