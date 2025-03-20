import tkinter as tk
from tkinter import ttk
from .acquisition_ui import AcquisitionUI
from .conversion_ui import ConversionUI
from ..processing.pipelines.processing_pipelines import PostProcessingPipeline
from ..processing.pipelines.synchronous_pipelines import SynchronousPipeline


class MainApp(tk.Tk):
    def __init__(
            self,
            processing_pipelines: dict[str, tuple[type[PostProcessingPipeline], str]] | None = None,
            synchronous_pipelines: dict[str, type[SynchronousPipeline]] | None = None,
            test_dir="",
            default_dir=""
    ):

        super().__init__()
        self.title("Data Acquisition and Analysis")
        self.geometry("500x700")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._add_acquisition_tab(serval_test_dir=test_dir, default_dir=default_dir)
        self._add_conversion_tab(pipelines=processing_pipelines)

        self.after(1000, self._update_conversion_monitor)

    def _add_acquisition_tab(self, serval_test_dir="", default_dir=""):
        acquisition_tab = ttk.Frame(self.notebook)
        self.notebook.add(acquisition_tab, text="TPX Acquisition")
        self.acquisition_ui = AcquisitionUI(acquisition_tab,serval_test_dir=serval_test_dir, default_dir=default_dir)
        self.acquisition_ui.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_conversion_tab(self,pipelines=None):
        conversion_tab = ttk.Frame(self.notebook)
        self.notebook.add(conversion_tab, text="File Conversion")
        self.conversion_ui = ConversionUI(conversion_tab, self.acquisition_ui, pipelines=pipelines)
        self.conversion_ui.pack(fill="both", expand=True, padx=10, pady=10)

    def _update_conversion_monitor(self):
        self.conversion_ui.update_process_queue_status()
        self.after(1000, self._update_conversion_monitor)

    def on_closing(self):
        self.conversion_ui.on_destroy()
        self.destroy()


def main(pipelines=None, test_dir=""):
    app = MainApp(processing_pipelines=pipelines, test_dir=test_dir)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
