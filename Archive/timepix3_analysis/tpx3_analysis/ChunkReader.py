import queue
from threading import Thread
import time
import os.path

import numpy as np

from . import numba_read as NR
from .constants import *

class ChunkReader(Thread):

    def __init__(self, files):
        super().__init__()
            
        self.files = files
        self.input_size = np.sum([os.path.getsize(f) for f in files])
        self.output_q = queue.Queue(10)
        self.currentFile = None
        self.sortCriteriaGen = NR.SortCriteriaGenerator()
        
        self.bytesRead = 0
        self.elapsedTime = 0
        self.stop = False
        
        self.start()

    partitionSlices = [slice((i+1)*BLOCK_N - UNSORTED,
                             (i+1)*BLOCK_N + UNSORTED) for i in range(SORT_WORKERS)]
    
    def partition(data, criteria, chips):
        # Partition the blocks at the block boundaries.
        # This makes sure the blocks are sorted with respect to each other.
        # (But they are not yet sorted themselves).
        for slice_ in ChunkReader.partitionSlices:
            partitionIndices = np.argpartition(criteria[slice_], UNSORTED)
             
            NR.reorder(criteria[slice_],
                   chips[slice_],
                   data[slice_], partitionIndices)
     
    def run(self):

        startTime = time.time()
        
        forNextChunk = [np.zeros(0, dtype=np.uint64),
                    np.zeros(0, dtype=np.uint64),
                    np.zeros(0, dtype=np.uint8)]
         
        for (i, self.currentFile) in enumerate(self.files):
             
            offset = 0

            if self.stop:
                break
            
            while not self.stop:
                
                                
                # We need to read UNSORTED more than we actually need.
                # This is to make sure that all blocks are at least sorted
                # with respect to each other (see partition function).
                toRead = SORT_WORKERS * BLOCK_N + UNSORTED - len(forNextChunk[0])
                
                chunk = np.fromfile(self.currentFile, dtype='<u8', count=toRead, offset=offset*8)
                
                self.elapsedTime = time.time() - startTime
                self.bytesRead += chunk.nbytes
                offset += chunk.size
                
                assert chunk.size <= toRead
                 
                criteriaChunk, chipsChunk = self.sortCriteriaGen.process(chunk)
                
                assert criteriaChunk.size == chunk.size
                assert criteriaChunk.size == chipsChunk.size
                
                toUnpack = np.append(forNextChunk[0], chunk)
                criteria = np.append(forNextChunk[1], criteriaChunk)
                chips = np.append(forNextChunk[2], chipsChunk)

                if len(chunk) != toRead:
                    # We are at the last chunk of the file.
                    
                    if self.currentFile != self.files[-1]:
                        
                        if len(toUnpack) > 2*UNSORTED:
                            # Partition with respect to the next file

                            partitionCriteria = criteria[-2*UNSORTED:]
                            indx = np.argpartition(partitionCriteria, UNSORTED)
                             
                            NR.reorder(toUnpack[-2*UNSORTED:], criteria[-2*UNSORTED:], chips[-2*UNSORTED:], indx)
                         
                        # This still has to be sorted with respect to the next file.
                        forNextChunk = [
                            toUnpack[-UNSORTED:],
                            criteria[-UNSORTED:],
                            chips[-UNSORTED:]]

                        self.output_q.put( (toUnpack[:-UNSORTED], criteria[:-UNSORTED], chips[:-UNSORTED]) )
                    else:
                        # Last chunk of the last file.
                        self.output_q.put( (toUnpack, criteria, chips) )

                    break
                else:
                    assert len(toUnpack) == SORT_WORKERS*BLOCK_N + UNSORTED
                    
                    ChunkReader.partition(toUnpack, criteria, chips)
                     
                    # This data is too close to the right boundary.
                    # This means that we need to sort this data also with
                    # respect to the next chunk. Therefore reserve it for the
                    # next loop.
                    forNextChunk = [
                        toUnpack[-UNSORTED:],
                        criteria[-UNSORTED:],
                        chips[-UNSORTED:]]

                    chunks = (toUnpack[:-UNSORTED], criteria[:-UNSORTED], chips[:-UNSORTED])
                    
                    assert chunks[0].size == CHUNK_N

                    self.output_q.put(chunks)
        
        self.elapsedTime = time.time() - startTime



