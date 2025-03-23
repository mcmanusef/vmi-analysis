from tkinter import Frame

from vmi_analysis.serval import DEFAULT_IP


class ConnectionUI(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.serval_ip = DEFAULT_IP
        self.BPC_file = ""
        self.DACS_file = ""
        self.connected = False
        self._create_widgets()

    def _create_widgets(self):
        pass
