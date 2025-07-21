import uconn_pipelines
from vmi_analysis.processing.pipelines import run_pipeline
from vmi_analysis.processing.processes import CustomClusterer

if __name__ == "__main__":
    pipeline = uconn_pipelines.CV4ConverterPipeline(input_path=r"J:\ctgroup\Edward\DATA\VMI\20250522\515 Ellipticity Scan\0",
                                                    output_path="test.cv4", cluster_class=CustomClusterer)
    run_pipeline(pipeline)
