from threading import Thread
import time
import queue

import numpy as np

from . import threaded_sort as TS
from .constants import *

class Sorter(Thread):
    
    def __init__(self, input_q):
        super().__init__()
        
        self.input_q = input_q
        self.output_q = queue.Queue(40)

        self.bytesRead = 0
        self.elapsedTime = 0
        self.stop = False
        
        self.start()
    
    def run(self):

        startTime = time.time()
        
        while not self.stop:
            try:
                toUnpack, criteria, chips = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if toUnpack.size == CHUNK_N:
                # Do a multi threaded sort. 
                TS.sort(toUnpack, criteria, chips)
            else:
                # We have read the last chunk of the file. We take 
                # the 'slow' path and do a straightforward sort.
                # This is to avoid complicating the threaded sort
                # with block sizes that are not the right size.
                sortIndices = np.argsort(criteria)
                
                toUnpack = toUnpack[sortIndices] 
                criteria = criteria[sortIndices]
                chips = chips[sortIndices]

            self.output_q.put( (toUnpack, criteria, chips) )
             
            self.bytesRead += toUnpack.nbytes
            self.elapsedTime = time.time() - startTime
             
            self.input_q.task_done()


