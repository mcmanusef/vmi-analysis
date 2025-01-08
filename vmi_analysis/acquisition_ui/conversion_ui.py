# conversion_ui.py

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import logging
import queue
import sys
import os
from typing import Dict
from .acquisition_ui import AcquisitionUI
from ..processing.pipelines import (
    TPXFileConverter,
    RawVMIConverterPipeline,
    VMIConverterPipeline,
    ClusterSavePipeline,
    CV4ConverterPipeline,
    StonyBrookClusterPipeline
)
import multiprocessing
import requests
import time

class ConversionUI(ttk.Frame):
    def __init__(self, parent, acquisition_ui: AcquisitionUI, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.acquisition_ui = acquisition_ui  # Reference to AcquisitionUI for default paths

        # Initialize variables
        self._initialize_variables()

        # Setup UI
        self._create_widgets()

        # Setup logging
        self._setup_logging()

        # Trace changes to input_path and selected_pipeline
        self.input_path.trace_add('write', self._update_output_path)
        self.selected_pipeline.trace_add('write', self._update_output_path)

        # Placeholder for the pipeline thread
        self.pipeline_thread = None
        self.pipeline = None
        self.pipeline_stop_event = threading.Event()

        # Start periodic log checking and process monitoring
        # self._poll_logs()
        self._monitor_pipeline()
        self._update_default_paths()

    def _initialize_variables(self):
        self.pipeline_options = {
            "Direct HDF5 Converter": TPXFileConverter,
            "Uncorrelated VMI Converter": RawVMIConverterPipeline,
            "Uncorrelated VMI Converter (Clustered)": ClusterSavePipeline,
            "UV4 Converter (Unclustered VMI Data)": VMIConverterPipeline,
            "CV4 Converter (Clustered VMI Data)": CV4ConverterPipeline,
            "Stony Brook Converter": StonyBrookClusterPipeline
        }

        # Extension map for pipelines
        self.pipeline_extension_map = {
            "Direct HDF5 Converter": ".h5",
            "Uncorrelated VMI Converter": ".h5",
            "Uncorrelated VMI Converter (Clustered)": ".h5",
            "UV4 Converter (Unclustered VMI Data)": ".uv4",
            "CV4 Converter (Clustered VMI Data)": ".cv4",
            "Stony Brook Converter": ".h5"
        }

        self.selected_pipeline = tk.StringVar()
        self.selected_pipeline.set("Direct HDF5 Converter")  # Default selection

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()

    def _create_widgets(self):
        # Top Frame for Pipeline Selection and Paths
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Pipeline Selection
        ttk.Label(top_frame, text="Select Pipeline:").grid(row=0, column=0, sticky="w")
        self.pipeline_combobox = ttk.Combobox(
                top_frame,
                textvariable=self.selected_pipeline,
                values=list(self.pipeline_options.keys()),
                state="readonly"
        )
        self.pipeline_combobox.grid(row=0, column=1, sticky="ew", padx=5)
        self.pipeline_combobox.bind("<<ComboboxSelected>>", self._update_default_paths)

        top_frame.columnconfigure(1, weight=1)

        # Input Path
        ttk.Label(top_frame, text="Input Folder:").grid(row=1, column=0, sticky="w", pady=5)
        self.input_entry = ttk.Entry(top_frame, textvariable=self.input_path)
        self.input_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.input_browse = ttk.Button(top_frame, text="Browse", command=self._browse_input)
        self.input_browse.grid(row=1, column=2, sticky="e", padx=5, pady=5)

        # Output Path
        ttk.Label(top_frame, text="Output File:").grid(row=2, column=0, sticky="w", pady=5)
        self.output_entry = ttk.Entry(top_frame, textvariable=self.output_path)
        self.output_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.output_browse = ttk.Button(top_frame, text="Browse", command=self._browse_output)
        self.output_browse.grid(row=2, column=2, sticky="e", padx=5, pady=5)

        # Start and Stop Buttons Frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(side=tk.TOP, pady=10)

        # Start Pipeline Button
        self.start_button = ttk.Button(buttons_frame, text="Start Pipeline", command=self._start_pipeline)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Stop Pipeline Button
        self.stop_button = ttk.Button(buttons_frame, text="Stop Pipeline", command=self._stop_pipeline, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Middle Frame for Process and Queue Monitoring
        middle_frame = ttk.LabelFrame(self, text="Processes and Queues", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(middle_frame, columns=("Type", "Status"), show="headings")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Status", text="Status")
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _browse_input(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_path.set(folder_selected.replace("/", "\\"))

    def _browse_output(self):
        pipeline_name = self.selected_pipeline.get()
        extension = self.pipeline_extension_map.get(pipeline_name, ".h5")
        if extension == ".h5":
            filetypes = [("HDF5 files", "*.h5"), ("All files", "*.*")]
        else:
            filetypes = [("CV4 files", "*.cv4"), ("All files", "*.*")]
        file_selected = filedialog.asksaveasfilename(defaultextension=extension, filetypes=filetypes)
        if file_selected:
            self.output_path.set(file_selected)

    def _update_default_paths(self, event=None):
        # Set default input path to the acquisition destination
        dest_folder = self.acquisition_ui.destination.get()
        self.input_path.set(dest_folder)
        # The output_path will be automatically updated via tracing

    def _setup_logging(self):
        # Setup logging for stdout
        self.logger_stdout = logging.getLogger('stdout')
        self.logger_stdout.setLevel(logging.INFO)

        # Setup logging for stderr
        self.logger_stderr = logging.getLogger('stderr')
        self.logger_stderr.setLevel(logging.ERROR)

    def _append_text(self, text_widget: scrolledtext.ScrolledText, text: str):
        text_widget.configure(state='normal')
        text_widget.insert(tk.END, text)
        text_widget.see(tk.END)
        text_widget.configure(state='disabled')

    def _update_output_path(self, *args):
        input_path = self.input_path.get()
        pipeline_name = self.selected_pipeline.get()

        if not input_path:
            self.output_path.set("")
            return

        parent_folder = os.path.dirname(os.path.normpath(input_path))
        base_name = os.path.basename(os.path.normpath(input_path))
        extension = self.pipeline_extension_map.get(pipeline_name, ".h5")
        output_filename = f"{base_name}{extension}"
        default_output = os.path.join(parent_folder, output_filename)
        self.output_path.set(default_output)

    def _start_pipeline(self):
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            logging.warning("Pipeline is already running.")
            messagebox.showwarning("Warning", "Pipeline is already running.")
            return

        pipeline_class = self.pipeline_options.get(self.selected_pipeline.get())
        input_path = self.input_path.get()
        output_path = self.output_path.get()

        if not input_path or not output_path:
            messagebox.showerror("Error", "Input and Output paths must be specified.")
            logging.error("Input and Output paths must be specified.")
            return

        try:
            self.pipeline = pipeline_class(input_path=input_path, output_path=output_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to instantiate pipeline: {e}")
            logging.error(f"Failed to instantiate pipeline: {e}")
            return

        # Update button states
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        # Reset stop event
        self.pipeline_stop_event.clear()

        # Start the pipeline in a separate thread
        self.pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.pipeline_thread.start()

    def _run_pipeline(self):
        try:
            with self.pipeline:
                for name, process in self.pipeline.processes.items():
                    initialized = getattr(process, 'initialized', multiprocessing.Value('b', False))
                    logging.info(f"{name} initialized correctly: {initialized.value}")

                logging.info("Starting pipeline")
                self.pipeline.start()
                for name, process in self.pipeline.processes.items():
                    running = getattr(process, 'running', multiprocessing.Value('b', False))
                    logging.info(f"{name} running: {running.value}")

                while not all(getattr(p.astep, 'stopped', multiprocessing.Value('b', True)).value for p in self.pipeline.processes.values()) and not self.pipeline_stop_event.is_set():
                    time.sleep(1)

                logging.info("Initiating Shutdown.")
            logging.info("Pipeline has stopped.")

        finally:
            if self.pipeline_stop_event.is_set():
                logging.info("Pipeline stopped by user.")
            else:
                logging.info("Pipeline has completed.")
            # Update button states back to default
            self.pipeline = None

            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def _stop_pipeline(self):
        if self.pipeline and self.pipeline.is_running():
            logging.info("Stopping the analysis pipeline...")
            self.pipeline_stop_event.set()
            self.stop_button.config(state="disabled")
            logging.info("Stop signal sent to the pipeline.")
        else:
            logging.warning("No running pipeline to stop.")
            messagebox.showwarning("Warning", "No running pipeline to stop.")

    def update_process_queue_status(self):

        # Clear the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.pipeline:
            return

        # Populate the treeview with current process and queue statuses
        for name, process in self.pipeline.processes.items():
            self.tree.insert("", tk.END, values=(f"{name}", process.status()))

            for qname, queue_obj in self.pipeline.queues.items():
                if not queue_obj in process.astep.output_queues:
                    continue
                q_status = "Closed" if queue_obj.closed.value else "Open"
                q_size = queue_obj.qsize()

                self.tree.insert("", tk.END, values=(f"\t{qname}", f"{q_status} (Size: {q_size})"))

    def _monitor_pipeline(self):
        if self.pipeline and self.pipeline.is_running():
            self.update_process_queue_status()
        self.after(100, self._monitor_pipeline)

    def on_destroy(self):
        if self.pipeline and self.pipeline.is_running():
            if messagebox.askokcancel("Quit", "Pipeline is still running. Do you want to stop it and quit?"):
                self.pipeline_stop_event.set()
                self.pipeline.stop()
        self.destroy()
