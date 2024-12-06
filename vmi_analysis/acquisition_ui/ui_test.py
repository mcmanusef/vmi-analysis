import os
import time
import datetime
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, DoubleVar, BooleanVar, OptionMenu
import labview_integrations as lv

# Constants
DEFAULT_FOLDER_NAME = "test"
DEFAULT_DURATION = 60.0
DEFAULT_DURATION_UNIT = "sec"
DESTINATION_BASE_PATH = "C:\\DATA"
STATUS_IDLE = "DA_IDLE"
STATUS_RECORDING = "DA_RECORDING"
UPDATE_INTERVAL_MS = 100  # 100 milliseconds
INFINITE_DURATION = 999999999

# Colors
COLOR_IDLE = "black"
COLOR_BUSY = "green"


class AcquisitionUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Acquisition")

        # Initialize variables
        self._initialize_variables()

        # Setup UI
        self._create_widgets()
        self._bind_events()

        # Start status updates
        self._update_status()

    def _initialize_variables(self):
        self.folder_name = StringVar(value=DEFAULT_FOLDER_NAME)
        self.duration_value = DoubleVar(value=DEFAULT_DURATION)
        self.infinite = BooleanVar(value=False)
        self.duration_unit = StringVar(value=DEFAULT_DURATION_UNIT)

        self.destination = StringVar()
        self._update_destination()

        self.status = StringVar(value=STATUS_IDLE)
        self.elapsed_str = StringVar(value="")
        self.start_str = StringVar(value="")
        self.end_str = StringVar(value="")

        self.progress_var = DoubleVar(value=0.0)

    def _create_widgets(self):
        # Main frames
        param_frame = self._create_param_frame()
        status_frame = self._create_status_frame()
        button_frame = self._create_button_frame()

        # Configure grid weights for resizing
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)
        param_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(1, weight=1)

    def _create_param_frame(self):
        param_frame = ttk.LabelFrame(self.master, text="Acquisition Parameters", padding=10)
        param_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Folder Name
        ttk.Label(param_frame, text="Folder Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.folder_entry = ttk.Entry(param_frame, textvariable=self.folder_name)
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Destination
        ttk.Label(param_frame, text="Destination:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        dest_label = ttk.Label(param_frame, textvariable=self.destination, foreground="gray")
        dest_label.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Duration and Infinite Mode
        duration_frame = self._create_duration_frame(param_frame)
        duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Infinite Mode Checkbox
        self.infinite_check = ttk.Checkbutton(
                param_frame,
                text="Run Infinitely",
                variable=self.infinite,
                command=self._toggle_infinite_mode
        )
        self.infinite_check.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        return param_frame

    def _create_duration_frame(self, parent):
        duration_frame = ttk.Frame(parent)
        duration_frame.columnconfigure(0, weight=1)
        duration_frame.columnconfigure(1, weight=0)
        duration_frame.columnconfigure(2, weight=0)
        duration_frame.columnconfigure(3, weight=0)

        ttk.Label(duration_frame, text="Duration:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_value, width=8)
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        self.duration_unit_menu = OptionMenu(duration_frame, self.duration_unit, "sec", "min", "hr")
        self.duration_unit_menu.grid(row=0, column=2, sticky="w", padx=5, pady=5)

        return duration_frame

    def _create_status_frame(self):
        status_frame = ttk.LabelFrame(self.master, text="Status", padding=10)
        status_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Busy Indicator and Status Label
        busy_frame = ttk.Frame(status_frame)
        busy_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.busy_canvas = tk.Canvas(busy_frame, width=16, height=16, highlightthickness=0)
        self.busy_canvas.grid(row=0, column=0, padx=5)
        self._draw_busy_indicator()

        ttk.Label(busy_frame, text="Status:").grid(row=0, column=1, sticky="w")
        status_label = ttk.Label(busy_frame, textvariable=self.status)
        status_label.grid(row=0, column=2, sticky="w", padx=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(
                status_frame,
                orient="horizontal",
                mode="determinate",
                variable=self.progress_var
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Time Information
        self._create_time_labels(status_frame)

        return status_frame

    def _create_time_labels(self, parent):
        time_labels = [
            ("Start:", self.start_str),
            ("End:", self.end_str),
            ("Elapsed:", self.elapsed_str)
        ]

        for idx, (label_text, var) in enumerate(time_labels, start=2):
            ttk.Label(parent, text=label_text).grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            ttk.Label(parent, textvariable=var).grid(row=idx, column=1, sticky="w", padx=5, pady=5)

    def _create_button_frame(self):
        button_frame = ttk.Frame(self.master)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(button_frame, text="Start", command=self._start_acquisition)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_acquisition)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=5)

        return button_frame

    def _bind_events(self):
        self.folder_name.trace_add('write', lambda *args: self._update_destination())

    def _update_destination(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        dest = os.path.join(DESTINATION_BASE_PATH, date_str, self.folder_name.get())
        self.destination.set(dest)

    def _draw_busy_indicator(self):
        self.busy_canvas.delete("all")
        color = COLOR_IDLE if self.status.get() == STATUS_IDLE else COLOR_BUSY
        self.busy_canvas.create_oval(2, 2, 14, 14, fill=color, outline="")

    def _convert_duration_to_seconds(self):
        unit = self.duration_unit.get()
        duration = self.duration_value.get()
        conversion_factors = {
            "sec": 1,
            "min": 60,
            "hr": 3600
        }
        return duration * conversion_factors.get(unit, 1)

    def _toggle_infinite_mode(self):
        state = "disabled" if self.infinite.get() else "normal"
        self.duration_entry.config(state=state)
        self.duration_unit_menu.config(state=state)

    def _update_status(self):
        dash = lv.get_dash()
        meas = dash.get("Measurement", {})
        current_status = meas.get("Status")

        if current_status:
            self.status.set(current_status)
            self._draw_busy_indicator()

        if current_status == STATUS_IDLE:
            self._handle_idle_status()
        elif current_status == STATUS_RECORDING:
            self._handle_recording_status(meas)

        self.master.after(UPDATE_INTERVAL_MS, self._update_status)

    def _handle_idle_status(self):
        self.progress_var.set(0)
        self.start_str.set("")
        self.end_str.set("")
        self.elapsed_str.set("")
        self.progress_bar["value"] = 0

        # Enable inputs
        self.start_button.config(state="normal")
        self.folder_entry.config(state="normal")
        self.infinite_check.config(state="normal")

        if not self.infinite.get():
            self.duration_entry.config(state="normal")
            self.duration_unit_menu.config(state="normal")

    def _handle_recording_status(self, meas):
        # Disable inputs during recording
        self.start_button.config(state="disabled")
        self.folder_entry.config(state="disabled")
        self.infinite_check.config(state="disabled")
        self.duration_entry.config(state="disabled")
        self.duration_unit_menu.config(state="disabled")

        # Update time labels
        self._update_time_labels(meas)

        # Update progress bar
        self._update_progress_bar(meas)

    def _update_time_labels(self, meas):
        start_ms = meas.get("StartDateTime")
        elapsed = meas.get("ElapsedTime")
        timeleft = meas.get("TimeLeft")
        framecount = meas.get("FrameCount")

        if start_ms is not None:
            start_dt = datetime.datetime.fromtimestamp(start_ms / 1000.0)
            self.start_str.set(self._format_start_time(start_dt))

        if elapsed is not None:
            self.elapsed_str.set(self._format_elapsed_time(elapsed))

        if self.infinite.get():
            self.end_str.set("")
        else:
            self.end_str.set(self._calculate_end_time(meas, start_dt) if start_ms else "")

    def _format_start_time(self, start_dt):
        return start_dt.strftime("%Y-%m-%d %I:%M:%S %p").replace(" 0",
                                                                 " ") if start_dt.date() != datetime.datetime.now().date() else start_dt.strftime(
                "%I:%M:%S %p").lstrip("0")

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
        elapsed = meas.get("ElapsedTime")
        timeleft = meas.get("TimeLeft")
        framecount = meas.get("FrameCount")

        if elapsed is not None and timeleft is not None and framecount is not None:
            predicted_offset = timeleft - (elapsed - (10 * framecount))
            end_dt = datetime.datetime.now() + datetime.timedelta(seconds=predicted_offset)
            return end_dt.strftime("%Y-%m-%d %I:%M:%S %p").replace(" 0", " ") if start_dt.date() != end_dt.date() else end_dt.strftime(
                    "%I:%M:%S %p").lstrip("0")
        return ""

    def _update_progress_bar(self, meas):
        elapsed = meas.get("ElapsedTime")
        if self.infinite.get():
            self.progress_var.set(0)
        elif elapsed is not None:
            total_duration = self._convert_duration_to_seconds()
            if total_duration > 0:
                fraction = min(max(elapsed / total_duration, 0.0), 1.0)
                self.progress_var.set(fraction * 100)
            else:
                self.progress_var.set(0)

    def _start_acquisition(self):
        folder = self.folder_name.get()
        duration = self._convert_duration_to_seconds()
        dest = self.destination.get()
        if self.infinite.get():
            duration = INFINITE_DURATION
        lv.acquire_data(dest, duration)

    def _stop_acquisition(self):
        lv.stop_acquisition()
        # Re-enable inputs after stopping
        self.start_button.config(state="normal")
        self.folder_entry.config(state="normal")
        self.infinite_check.config(state="normal")
        self._toggle_infinite_mode()


def main():
    root = tk.Tk()
    app = AcquisitionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
