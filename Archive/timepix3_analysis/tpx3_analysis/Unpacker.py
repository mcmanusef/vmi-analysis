from threading import Thread
import numpy as np
import time
import queue

from . import numba_read as NR

class Unpacker(Thread):
    
    def __init__(self, input_q, gaps):
        super().__init__()
        
        self.input_q = input_q
        self.output_q = queue.Queue(10)
        self.numbaUnpacker = NR.NumbaUnpacker(gaps)
         
        self.bytesUnpacked = 0
        self.elapsedTime = 0
        self.stop = False

        self.start()

    def getCurrentFrameIndex(self):
        return self.numbaUnpacker.currentFrameIndex()
    
    def run(self):
        
        startTime = time.time()
        
        while not self.stop:
            try:
                toUnpack, criteria, chips = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue
             
            unpacked = self.numbaUnpacker.unpackSortedData(toUnpack, criteria, chips)
            
            for (key, data) in unpacked.items():
                self.output_q.put( (key, data) )
             
            self.elapsedTime = time.time() - startTime
            self.bytesUnpacked += toUnpack.nbytes
            self.input_q.task_done()


