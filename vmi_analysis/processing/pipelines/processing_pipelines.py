from .base_pipeline import BasePipeline
from .. import data_types, processes
from ..data_types import PixelData, TDCData, Chunk


class PostProcessingPipeline(BasePipeline):
    def __init__(self, input_path: str, output_path: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path


class TPXFileConverter(PostProcessingPipeline):
    """
    Pipeline for converting TPX3 files to h5 files.
    Not specific to our VMI setup.
    Directly reads the TPX3 files and converts them to h5 files, saving the pixel data and the TDC data with minimal processing.
    Output Format:
    - pixel: toa, x, y, tot
    - tdc: tdc_time, tdc_type
    tdc_types:
    1: tdc1 rising, 2: tdc1 falling, 3: tdc2 rising, 4: tdc2 falling
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
    ):
        super().__init__(input_path, output_path)
        queues = {
            "chunk": data_types.Queue[Chunk](),
            "pixel": data_types.StructuredDataQueue[PixelData](
                    dtypes=PixelData.c_dtypes,
                    names={"time": "toa", "x": "x", "y": "y", "tot": "tot"},
                    chunk_size=10000,
            ),
            "tdc": data_types.StructuredDataQueue[TDCData](
                    dtypes=TDCData.c_dtypes,
                    names={"time": "tdc_time", "type": "tdc_type"},
                    chunk_size=10000,
            ),
        }

        self.processes = {
            "reader": processes.TPXFileReader(
                    input_path, queues["chunk"]
            ).make_process(),
            "converter": processes.TPXConverter(
                    queues["chunk"], queues["pixel"], queues["tdc"]
            ).make_process(),
            "save": processes.SaveToH5(
                    output_path,
                    {"pixel": queues["pixel"], "tdc": queues["tdc"]},
            ).make_process(),
        }
        self.queues = queues
