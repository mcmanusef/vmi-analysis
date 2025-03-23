import uconn_pipelines
import vmi_analysis.processing.pipelines as pl
from vmi_analysis.processing.processes import CustomClusterer

if __name__ == "__main__":
    fname = r"C:\DATA\20250123\prop_oxide_3W_5,7e-7\0"
    # pipeline=pl.CV4ConverterPipeline(fname,fname+".cv4", cluster_class=CustomClusterer)
    # pipeline=pl.MultiprocessTestPipeline(n=8)
    # pipeline=pl.VMIConverterTestPipeline(fname)
    pipeline = uconn_pipelines.CV4ConverterPipeline(
        fname, fname + ".cv4", cluster_class=CustomClusterer, cluster_processes=8
    )
    pipeline.set_profile(True)
    pl.run_pipeline(pipeline)
