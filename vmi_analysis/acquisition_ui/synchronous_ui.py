import datetime
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import the synchronous pipeline base class.
# (Make sure the import path matches your project structure.)
from ..processing.pipelines.synchronous_pipelines import SynchronousPipeline

# Constants (adjust these as needed)
INFINITE_DURATION = 999999999
UPDATE_INTERVAL_MS = 500  # UI update interval in milliseconds


class SynchronousUI(ttk.Frame):
    """
    A UI for running a synchronous pipeline. This UI includes:
    - A selection box to choose from multiple pipeline classes.
    - Acquisition parameters (folder name, duration, infinite mode)
    - A progress bar with start/end/elapsed time display
    - A matplotlib figure area to display the synchronous pipeline output
    - A treeview for monitoring process and queue statuses
    - Start and Stop buttons to control the pipeline
    """

    def __init__(self, parent, pipelines: dict[str, type[SynchronousPipeline]] | None = None, default_dir="", **kwargs):
        """
        :param parent: Tkinter parent widget.
        :param pipeline_classes: A dictionary of pipeline classes, where keys are pipeline names.
        :param default_dir: Default directory string.
        """
        super().__init__(parent, **kwargs)
        self.pipeline_classes = pipelines
        self.selected_pipeline = tk.StringVar(value=list(pipelines.keys())[0])
        self.default_dir = default_dir

        # Pipeline instance and thread
        self.pipeline = None
        self.pipeline_thread = None
        self.pipeline_stop_event = threading.Event()

        # Acquisition parameters (similar to AcquisitionUI)
        self.folder_name = tk.StringVar(value=os.path.join(datetime.datetime.now().strftime(default_dir), "test.h5"))
        self.duration_value = tk.DoubleVar(value=60.0)
        self.infinite = tk.BooleanVar(value=False)
        self.duration_unit = tk.StringVar(value="sec")

        # Status variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="IDLE")
        self.start_str = tk.StringVar(value="")
        self.end_str = tk.StringVar(value="")
        self.elapsed_str = tk.StringVar(value="")

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Build UI components
        self._create_widgets()
        # Start periodic monitoring (for progress and process/queue status)
        self._monitor_pipeline()

    def _create_widgets(self):
        # -- Pipeline Selection Frame --
        select_frame = ttk.LabelFrame(self, text="Select Pipeline", padding=10)
        select_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(select_frame, text="Pipeline:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.pipeline_combobox = ttk.Combobox(
                select_frame, textvariable=self.selected_pipeline,
                values=list(self.pipeline_classes.keys()),
                state="readonly"
        )
        self.pipeline_combobox.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        select_frame.columnconfigure(1, weight=1)

        # -- Acquisition Parameters Frame --
        param_frame = ttk.LabelFrame(self, text="Acquisition Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # Folder Name
        ttk.Label(param_frame, text="Folder Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.folder_entry = ttk.Entry(param_frame, textvariable=self.folder_name)
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        param_frame.columnconfigure(1, weight=1)

        # Duration and Infinite Mode
        duration_frame = ttk.Frame(param_frame)
        duration_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(duration_frame, text="Duration:").grid(row=0, column=0, sticky="w", padx=5)
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_value, width=8)
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.duration_unit_menu = ttk.OptionMenu(duration_frame, self.duration_unit, "sec", "sec", "min", "hr")
        self.duration_unit_menu.grid(row=0, column=2, sticky="w", padx=5)
        self.infinite_check = ttk.Checkbutton(
                duration_frame, text="Run Infinitely", variable=self.infinite, command=self._toggle_infinite_mode
        )
        self.infinite_check.grid(row=0, column=3, sticky="w", padx=5)

        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        self.start_button = ttk.Button(buttons_frame, text="Start Pipeline", command=self.start_pipeline)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(buttons_frame, text="Stop Pipeline", command=self.stop_pipeline, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # -- Status Frame (Progress Bar and Time Labels) --
        status_frame = ttk.LabelFrame(self, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(
                status_frame, orient="horizontal", mode="determinate", variable=self.progress_var
        )
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Label(status_frame, text="Start:").grid(row=1, column=0, sticky="w", padx=5, pady=2, columnspan=2)
        self.start_label = ttk.Label(status_frame, textvariable=self.start_str)
        self.start_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(status_frame, text="End:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.end_label = ttk.Label(status_frame, textvariable=self.end_str)
        self.end_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(status_frame, text="Elapsed:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.elapsed_label = ttk.Label(status_frame, textvariable=self.elapsed_str)
        self.elapsed_label.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # -- Figure Display Frame --
        figure_frame = ttk.LabelFrame(self, text="Synchronous Pipeline Display", padding=10)
        figure_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        # Create a placeholder figure (the pipeline will update its display_fig attribute)
        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # -- Process and Queue Monitoring Frame --
        monitor_frame = ttk.LabelFrame(self, text="Processes and Queues", padding=10)
        monitor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tree = ttk.Treeview(monitor_frame, columns=("Type", "Status"), show="headings")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Status", text="Status")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # -- Start and Stop Buttons --

    def _toggle_infinite_mode(self):
        if self.infinite.get():
            self.duration_entry.config(state="disabled")
            self.duration_unit_menu.config(state="disabled")
        else:
            self.duration_entry.config(state="normal")
            self.duration_unit_menu.config(state="normal")

    def convert_duration_to_seconds(self):
        unit = self.duration_unit.get()
        duration = self.duration_value.get()
        if unit == "min":
            return duration * 60
        elif unit == "hr":
            return duration * 3600
        return duration

    def start_pipeline(self):
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            messagebox.showwarning("Warning", "Pipeline is already running.")
            return

        folder = self.folder_name.get()
        duration = self.convert_duration_to_seconds()
        if self.infinite.get():
            duration = INFINITE_DURATION

        # Get the selected pipeline class based on the selection box.
        selected_key = self.selected_pipeline.get()
        pipeline_class = self.pipeline_classes.get(selected_key)
        if not pipeline_class:
            messagebox.showerror("Error", "Selected pipeline is not available.")
            return

        try:
            # For a synchronous pipeline, pass the output_file and acquisition_time.
            self.pipeline = pipeline_class(output_path=folder, acquisition_time=duration)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to instantiate pipeline: {e}")
            logging.exception("Pipeline instantiation failed")
            return

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pipeline_stop_event.clear()
        self.start_str.set(datetime.datetime.now().strftime("%H:%M:%S"))
        self.status.set("RUNNING")

        # Run the pipeline in a separate thread to keep the UI responsive
        self.pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.pipeline_thread.start()

    def _run_pipeline(self):
        try:
            with self.pipeline:
                self.pipeline.start()
                while not self.pipeline.on_finish.is_set() and not self.pipeline_stop_event.is_set():
                    # Update display: call the pipeline's update_display method
                    self.pipeline.update_display()
                    # Update the figure canvas if the pipeline has a display figure
                    if self.pipeline.display_fig:
                        self.canvas.figure = self.pipeline.display_fig
                        self.canvas.draw()

                    # Optionally update the progress bar here.
                    # For demonstration, we simply cycle the progress value.
                    current = self.progress_var.get()
                    self.progress_var.set((current + 1) % 101)

                    # Update process and queue statuses
                    self.update_process_queue_status()

                    # Update elapsed time label (using current time difference)
                    elapsed = (datetime.datetime.now() - datetime.datetime.strptime(self.start_str.get(), "%H:%M:%S")).total_seconds()
                    self.elapsed_str.set(f"{elapsed:.1f} sec")

                    time.sleep(0.5)
                # When the loop exits, set the end time
                self.end_str.set(datetime.datetime.now().strftime("%H:%M:%S"))
                self.status.set("IDLE")
        finally:
            self.pipeline = None
            self.pipeline_thread = None
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def stop_pipeline(self):
        if self.pipeline and self.pipeline.is_running():
            logging.info("Stop signal sent to the pipeline.")
            self.pipeline_stop_event.set()
            self.stop_button.config(state="disabled")
        else:
            messagebox.showwarning("Warning", "No running pipeline to stop.")

    def update_process_queue_status(self):
        # Clear the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not self.pipeline:
            return
        # Populate the treeview with each process and associated queues
        for name, process in self.pipeline.processes.items():
            status_str = str(process.status())
            self.tree.insert("", tk.END, values=(name, status_str))
            for qname, q in self.pipeline.queues.items():
                if q in process.astep.output_queues:
                    q_status = "Closed" if q.closed.value else "Open"
                    q_size = q.qsize()
                    self.tree.insert("", tk.END, values=(f"  {qname}", f"{q_status} (Size: {q_size})"))

    def _monitor_pipeline(self):
        # Periodically update the process/queue statuses (reschedules itself regardless of pipeline state)
        if self.pipeline and self.pipeline.is_running():
            self.update_process_queue_status()
        self.after(UPDATE_INTERVAL_MS, self._monitor_pipeline)


# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Synchronous Pipeline UI Example")
    root.geometry("800x800")

    # Example dictionary of synchronous pipeline classes.
    # Replace 'SynchronousPipeline' with your actual pipeline subclasses.
    example_pipeline_classes = {
        "Pipeline A": SynchronousPipeline,
        "Pipeline B": SynchronousPipeline,
    }

    ui = SynchronousUI(root, pipeline_classes=example_pipeline_classes)
    ui.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
