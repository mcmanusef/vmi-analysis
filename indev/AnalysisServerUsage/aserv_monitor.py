import tkinter as tk
from tkinter import ttk
import threading
import time
from indev.AnalysisServerUsage import AnalysisServer
import functools
import datetime
class Application(tk.Frame):
    def __init__(self, analysis_server: AnalysisServer.AnalysisServer, master=None):
        self.analysis_server=analysis_server
        queues = (self.analysis_server.chunk_queue,
                  self.analysis_server.pixel_queue,
                  self.analysis_server.raw_pulse_queue,
                  self.analysis_server.raw_cluster_queue,
                  self.analysis_server.raw_etof_queue,
                  self.analysis_server.raw_itof_queue,
                  self.analysis_server.pulse_queue,
                  self.analysis_server.cluster_queue,
                  self.analysis_server.etof_queue,
                  self.analysis_server.itof_queue,
                  )

        names = ("Chunks",
                 "Pixels",
                 "Raw Pulses",
                 "Raw Clusters",
                 "Raw Electrons",
                 "Raw Ions",
                 "Pulses",
                 "Clusters",
                 "Electrons",
                 "Ions",
                 )
        self.queue_dict=dict(zip(names,queues))


        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()


    def create_widgets(self):
        current_row=0
        # 4x3 Grid of Numerical Labels
        headers = ["Pulse", "e-ToF", "i-Tof", "Cluster"]
        rows = ["Next", "Max", "Current"]

        started = False
        total_pulse_time = 0
        start_time = datetime.datetime.now()
        elapsed_time = 0

        for i, header in enumerate(headers):
            tk.Label(self, text=header).grid(row=current_row, column=i+1)
        current_row+=1

        for j, row in enumerate(rows):
            tk.Label(self, text=row).grid(row=current_row, column=0)
            for i, header in enumerate(headers):
                lbl = tk.Label(self, text="0")
                lbl.grid(row=current_row, column=i+1)
                self.after(1000, functools.partial(self.update_grid_data,row=row,col=i), lbl)
            current_row+=1

        current_row+=1
        ttk.Separator(self,orient='horizontal').grid(row=current_row,columnspan=5, sticky='ew')
        current_row+=1



        for c,label in enumerate(["Data Time", "Loops", "Runtime", "Ratio"]):
            l1 = tk.Label(self, text=label)
            l1.grid(row=current_row,column=c+1)
            l2 = tk.Label(self, text="0")
            l2.grid(row=current_row+1,column=c+1)
            self.after(1000, functools.partial(self.update_timing,col=label), l2)

        current_row+=2

        current_row+=1
        ttk.Separator(self,orient='horizontal').grid(row=current_row,columnspan=5, sticky='ew')
        current_row+=1

        # Progress Bars
        progress_labels = ["Chunks", "Pixels", "Raw Pulses", "Raw Clusters", "Raw Electrons",
                           "Raw Ions", "Pulses", "Clusters", "Electrons", "Ions"]

        for i, label in enumerate(progress_labels):
            tk.Label(self, text=label).grid(row=current_row, column=0, sticky=tk.W)
            progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=200, mode='determinate')
            progress.grid(row=current_row, column=1, columnspan=3, sticky=tk.W+tk.E)
            self.after(1000, functools.partial(self.update_progress, element=label), progress)
            current_row+=1

        current_row+=1
        ttk.Separator(self,orient='horizontal').grid(row=current_row,columnspan=5, sticky='ew')
        current_row+=1

        # Additional Numerical Displays
        extra_labels = ["Repetition rate", "Cluster rate", "i-ToF rate", "e-ToF rate"]

        for i, label in enumerate(extra_labels):
            tk.Label(self, text=label).grid(row=current_row, column=0, sticky=tk.W)
            lbl = tk.Label(self, text="0")
            lbl.grid(row=current_row, column=1, sticky=tk.W)
            self.after(1000,functools.partial(self.update_rate, rate=label), lbl)
            current_row+=1

    def update_rate(self, label, rate=""):
        buffer_size = self.analysis_server.buffer_size
        match rate:
            case "Repetition rate":
                buffer = self.analysis_server.pulse_queue.buffer
                if len(buffer)==buffer_size:
                    pulse_rate = 1e9 * (buffer[-1][0] - buffer[0][0]) / (
                            (buffer[-1][1]-buffer[0][1])+1% (25 * 2 ** 30))
                    label.config(text = f"{pulse_rate: 7.4f}")
            case "e-ToF rate":
                buffer = self.analysis_server.etof_queue.buffer
                if len(buffer)==buffer_size:
                    etof_rate = buffer_size / (buffer[-1][0] - buffer[0][0])
                    label.config(text = f"{etof_rate: 7.4f}")
            case "i-ToF rate":
                buffer = self.analysis_server.itof_queue.buffer
                if len(buffer)==buffer_size:
                    itof_rate = buffer_size / (buffer[-1][0] - buffer[0][0])
                    label.config(text = f"{itof_rate: 7.4f}")
            case "Cluster rate":
                buffer = self.analysis_server.cluster_queue.buffer
                if len(buffer)==buffer_size:
                    clust_rate = buffer_size / (buffer[-1][0] - buffer[0][0])
                    label.config(text = f"{clust_rate: 7.4f}")
        self.after(1000,functools.partial(self.update_rate, rate=rate), label)

    def update_grid_data(self, label, row=None, col=None):
        match row:
            case "Next":
                label.config(text=f"{self.analysis_server.next[col]/1e6  : 9.3f}")
            case "Max":
                label.config(text=f"{self.analysis_server.max_seen[col]/1e9  : 9.3f}")
            case "Current":
                label.config(text=f"{self.analysis_server.current[col]/1e9  : 9.3f}")
        self.after(1000, functools.partial(self.update_grid_data,row=row,col=col), label)
    def update_progress(self, progress, element=None):
        progress['value'] = self.queue_dict[element].qsize()/self.analysis_server.max_size*100
        self.after(1000, functools.partial(self.update_progress, element=element), progress)
    def update_timing(self, label, col=None,total_pulse_time=0):
        if self.analysis_server.start_time.value != 0:
            last = total_pulse_time
            total_pulse_time = self.analysis_server.current[0] / 1e9 \
                               + max(self.analysis_server.max_seen) / 1e9 \
                               * self.analysis_server.overflow_loops.value
            if total_pulse_time > last:
                elapsed_time = (datetime.datetime.now() -
                                datetime.datetime.fromtimestamp(self.analysis_server.start_time.value)
                                ).total_seconds()
            else:
                elapsed_time=0
            match col:
                case "Data Time":
                    label.config(text=f"{total_pulse_time  : 9.3f}")
                case "Loops":
                    label.config(text=self.analysis_server.overflow_loops.value)
                case "Runtime":
                    label.config(text=f"{elapsed_time: 9.3f}")
                case "Ratio":
                    label.config(text=f"{elapsed_time / (total_pulse_time + 1): 9.3f}")
        self.after(1000, functools.partial(self.update_timing,col=col, total_pulse_time=total_pulse_time), label)


def run_gui(aserv):
    root = tk.Tk()
    root.title("TPX3 Analysis Monitor")
    app = Application(aserv,master=root)
    app.mainloop()

if __name__ == '__main__':

    # Running the GUI in a separate thread
    threading.Thread(target=run_gui).start()

    # You can do other stuff in the main thread or just sleep
    while True:
        time.sleep(10)

#%%
