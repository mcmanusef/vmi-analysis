import argparse
import time
from os import path
import cProfile
import queue
from threading import Thread

from . import numba_read as NR

from . import standard_layouts as SL
from .ChunkReader import ChunkReader
from .Sorter import Sorter
from .Unpacker import Unpacker
from .AppendableHDF5 import AppendableHDF5

class Converter(Thread):
    
    def __init__(self, raw_files, hdf_file,  boardLayout, gaps=2, clusterSquareSize=8, maxTOADiff=5, cluster=False):
        super().__init__()
        
        if boardLayout == 'single':
            layout = SL.singleLayout
        elif boardLayout == 'quad':
            layout = SL.quadLayout
        else:
            assert isinstance(boardLayout, list), "Board layout should be 'single', 'quad' or a custom board layout represented by a list"
            layout = boardLayout

        NR.setBoardLayout(layout)
         
        NR.setClusterEnabled(cluster)
        NR.setMaxTOADiff(maxTOADiff * 1000000) # ps -> us
        NR.setClusterSquareSize(clusterSquareSize)

        startCompile = time.time()
        NR.compileFunctions()
        self.compilationTime = time.time() - startCompile
        
        metadata = {
            'BoardLayout': str(layout),
            'ConverterVersion': 'v0.2',
            'Gaps': gaps
        }
         
        self.chunkReader = ChunkReader(raw_files)
        self.sorter = Sorter(self.chunkReader.output_q)
        self.unpacker = Unpacker(self.sorter.output_q, gaps)
        self.hdf_file = AppendableHDF5(self.unpacker.output_q, hdf_file, metadata, cluster)
        
        self.start()
    
    def run(self):
        self.chunkReader.join()
         
        # Todo: also catch a crash of chunkReader
        crashed = not (self.sorter.is_alive() and self.unpacker.is_alive() and self.hdf_file.is_alive())
         
        if not crashed: 
            self.sorter.input_q.join()
            self.unpacker.input_q.join()
            self.hdf_file.input_q.join()
             
            self.stop() 
    
    def getCurrentFrameIndex(self):
        return self.unpacker.getCurrentFrameIndex()

    def getCurrentFile(self):
        return path.basename(self.chunkReader.currentFile)

    def getHitsWritten(self):
        return self.hdf_file.hitsWritten

    def getTDCsWritten(self):
        return self.hdf_file.tdcsWritten
     
    def has_crashed(self):
        # One of the threads died before processing all files
        error = self.chunkReader.is_alive() and (not self.sorter.is_alive() or \
                                                 not self.unpacker.is_alive() or \
                                                 not self.hdf_file.is_alive())
         
        return error
     
    def stop(self):
        self.chunkReader.stop = True
        self.sorter.stop = True
        self.unpacker.stop = True
        self.hdf_file.stop = True
    
    def getProgress(self):
        return (self.chunkReader.bytesRead / self.chunkReader.input_size) * 100
        


