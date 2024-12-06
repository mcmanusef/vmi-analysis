import tkinter as tk
from tkinter import ttk
from .acquisition_frame import AcquisitionUI


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Application with Tabs")
        self.geometry("800x600")

        # Create Notebook (Tabbed Interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        # Add Tabs
        self._add_acquisition_tab()
        self._add_other_tabs()

    def _add_acquisition_tab(self):
        acquisition_tab = ttk.Frame(self.notebook)
        self.notebook.add(acquisition_tab, text="Acquisition")

        # Instantiate AcquisitionUI within the tab
        self.acquisition_ui = AcquisitionUI(acquisition_tab)
        self.acquisition_ui.pack(fill="both", expand=True, padx=10, pady=10)

    def _add_other_tabs(self):
        # Example of adding another tab
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Settings")

        # Populate settings_tab as needed
        ttk.Label(settings_tab, text="Settings go here.").pack(padx=10, pady=10)

        # You can add more tabs similarly
        info_tab = ttk.Frame(self.notebook)
        self.notebook.add(info_tab, text="Info")

        ttk.Label(info_tab, text="Application Information.").pack(padx=10, pady=10)

def main():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()
