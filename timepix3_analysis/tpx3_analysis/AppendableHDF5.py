from threading import Thread
import time
import queue

import numpy as np
import h5py

from .constants import *

class AppendableHDF5(Thread):
    
    def __init__(self, input_q, filename, metadata, cluster):
        super().__init__()
         
        self.input_q = input_q
        self.metadata = metadata
        self.filename = filename
        self.stop = False
        
        self.bytesWritten = 0
        self.elapsedTime = 0
        
        self.hitsWritten = 0
        self.tdcsWritten = 0

        self.cluster = cluster
         
        self.start()
    
    def create_dset(self, dname, dtype):
        return self.file.create_dataset(dname, (0,), dtype=dtype, chunks=(BLOCK_N,), maxshape=(None,))
    
    def run(self):
        
        startTime = time.time()

        with h5py.File(self.filename, 'w') as _file:
            
            self.file = _file
            
            for (key, value) in self.metadata.items():
                self.file.attrs[key] = value
             
            self.dsets = {
                'tdc_type': self.create_dset('tdc_type', np.uint8),
                'tdc_time' : self.create_dset('tdc_time', np.int64),
                'x': self.create_dset('x', np.uint16),
                'y': self.create_dset('y', np.uint16),
                'tot': self.create_dset('tot', np.uint16),
                'toa': self.create_dset('toa', np.int64), 
                'frame_number': self.create_dset('frame_number', np.uint32)
            }
             
            if self.cluster:
                self.dsets['cluster_index'] = self.create_dset('cluster_index', np.uint32)
            
            while not self.stop:
                try:
                    dname, chunk = self.input_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                  
                dset = self.dsets[dname]
                assert dset.dtype == chunk.dtype
                oldSize = dset.size
                dset.resize(oldSize + chunk.size, axis=0)
                dset[oldSize:] = chunk
                 
                if dname == 'x':
                    self.hitsWritten += chunk.size
                if dname == 'tdc_type':
                    self.tdcsWritten += chunk.size
                
                self.bytesWritten += chunk.nbytes
                self.elapsedTime = time.time() - startTime
                
                self.input_q.task_done()


