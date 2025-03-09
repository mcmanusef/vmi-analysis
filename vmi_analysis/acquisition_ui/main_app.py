import tkinter as tk
from tkinter import ttk
from .acquisition_ui import AcquisitionUI
from .conversion_ui import ConversionUI

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Acquisition and Analysis")
        self.geometry("500x700")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._add_acquisition_tab()
        self._add_conversion_tab()

        self.after(1000, self._update_conversion_monitor)

    def _add_acquisition_tab(self):
        acquisition_tab = ttk.Frame(self.notebook)
        self.notebook.add(acquisition_tab, text="Acquisition")
        self.acquisition_ui = AcquisitionUI(acquisition_tab)
        self.acquisition_ui.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_conversion_tab(self):
        conversion_tab = ttk.Frame(self.notebook)
        self.notebook.add(conversion_tab, text="File Conversion")
        self.conversion_ui = ConversionUI(conversion_tab, self.acquisition_ui)
        self.conversion_ui.pack(fill="both", expand=True, padx=10, pady=10)

    def _update_conversion_monitor(self):
        self.conversion_ui.update_process_queue_status()
        self.after(1000, self._update_conversion_monitor)

    def on_closing(self):
        self.conversion_ui.on_destroy()
        self.destroy()

def main():
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
