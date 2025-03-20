from .base_pipeline import AnalysisPipeline
from .. import data_types, processes


class PostProcessingPipeline(AnalysisPipeline):
    def __init__(self, input_path: str, output_path: str, **kwargs):
        super().__init__(**kwargs)
        self.input_path=input_path
        self.output_path=output_path


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
        buffer_size: int = 0,
        single_process=False,
        **kwargs,
    ):
        super().__init__(input_path,output_path,**kwargs)
        if not single_process:
            self.queues = {
                "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                "pixel": data_types.ExtendedQueue(
                    buffer_size=buffer_size,
                    dtypes=("f", "i", "i", "i"),
                    names=("toa", "x", "y", "tot"),
                    chunk_size=10000,
                ),
                "tdc": data_types.ExtendedQueue(
                    buffer_size=buffer_size,
                    dtypes=("f", "i"),
                    names=("tdc_time", "tdc_type"),
                    chunk_size=10000,
                ),
            }

            self.processes = {
                "reader": processes.TPXFileReader(
                    input_path, self.queues["chunk"]
                ).make_process(),
                "converter": processes.TPXConverter(
                    self.queues["chunk"], self.queues["pixel"], self.queues["tdc"]
                ).make_process(),
                "save": processes.SaveToH5(
                    output_path,
                    {"pixel": self.queues["pixel"], "tdc": self.queues["tdc"]},
                ).make_process(),
            }

        if single_process:
            self.queues: dict[str, data_types.ExtendedQueue] = {
                "chunk": data_types.ExtendedQueue(buffer_size=0, dtypes=(), names=()),
                "pixel": data_types.ExtendedQueue(
                    buffer_size=buffer_size,
                    dtypes=("f", "i", "i", "i"),
                    names=("toa", "x", "y", "tot"),
                ),
                "tdc": data_types.ExtendedQueue(
                    buffer_size=buffer_size,
                    dtypes=("f", "i"),
                    names=("tdc_time", "tdc_type"),
                ),
            }

            self.processes = {
                "Combined": processes.CombinedStep(
                    steps=(
                        processes.TPXFileReader(
                            input_path, chunk_queue=self.queues["chunk"]
                        ),
                        processes.TPXConverter(
                            chunk_queue=self.queues["chunk"],
                            pixel_queue=self.queues["pixel"],
                            tdc_queue=self.queues["tdc"],
                        ),
                        processes.SaveToH5(
                            output_path,
                            {"pixel": self.queues["pixel"], "tdc": self.queues["tdc"]},
                        ),
                    ),
                    intermediate_queues=(self.queues["pixel"], self.queues["tdc"]),
                    output_queues=(self.queues["chunk"],),
                ).make_process(),
            }


