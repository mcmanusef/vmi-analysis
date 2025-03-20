from vmi_analysis.processing.pipelines.base_pipeline import AnalysisPipeline
import abc
import matplotlib.pyplot as plt


class SynchronousPipeline(AnalysisPipeline, abc.ABC):
    def __init__(self, output_file: str, **kwargs):
        super().__init__(**kwargs)
        self.output_file = output_file
        self.display_fig: plt.Figure | None = None

    @abc.abstractmethod
    def update_display(self):
        pass

