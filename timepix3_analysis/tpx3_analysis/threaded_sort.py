import threading
import numpy as np
import queue
import time

from .constants import *
from . import numba_read as NR

class SorterThread(threading.Thread):
    
    def __init__(self):
        super().__init__(daemon=True)
        
        self.input_q = queue.Queue()
        self.output_q = queue.Queue()
        self.elapsedTime = 0
        
        self.start()

    def run(self):
        
        startTime = time.time()
        
        while True:
            criteria, chips, data = self.input_q.get()

            assert len(criteria) == BLOCK_N
            
            startSort = time.time()
            indices = np.argsort(criteria, kind='mergesort')
            endSort = time.time()
            
            assert len(indices) == BLOCK_N
             
            NR.reorder(criteria, chips, data, indices) 
             
            self.elapsedTime = time.time() - startTime
            self.input_q.task_done()

threads_ = [SorterThread() for i in range(SORT_WORKERS)]

blockSlices = [slice(i*BLOCK_N, (i+1)*BLOCK_N) for i in range(SORT_WORKERS)]

assert blockSlices[0].stop - blockSlices[0].start == BLOCK_N

def sort(data, criteria, chips):

    assert criteria.size == CHUNK_N
    assert chips.size == CHUNK_N
    assert data.size == CHUNK_N
     
    for thread, s in zip(threads_, blockSlices):
        thread.input_q.put( (criteria[s], chips[s], data[s]) )

    for t in threads_:
        t.input_q.join()

def get_sorting_time():
    return np.mean([t.elapsedTime for t in threads_])
     
if __name__ == '__main__':
    # Benchmark of sorting performance
     
    N = 45

    data = [np.random.randint(0, 1e15, size=CHUNK_N, dtype=np.uint64) for i in range(N)]
    criteria = [np.random.randint(0, 1e15, size=CHUNK_N, dtype=np.uint64) for i in range(N)]
    chips = [np.random.randint(0, 5, size=CHUNK_N, dtype=np.uint8) for i in range(N)]
     
    for i in range(N):
        startTime = time.time()     
         
        sort(data[i], criteria[i], chips[i])
         
        endTime = time.time()     
        
        print('Throughput: ', data[i].nbytes/1e6/(endTime - startTime))
     
    


    
