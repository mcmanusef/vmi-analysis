import vmi_analysis.processing.pipelines as pl
from vmi_analysis.processing.processes import CustomClusterer

if __name__ == '__main__':
    fname = r"C:\DATA\20250122\prop_oxide_3,2W_1e-6\135"
    pipeline=pl.CV4ConverterPipeline(fname,fname+".cv4", cluster_class=CustomClusterer)
    # pipeline=pl.MultiprocessTestPipeline(n=8)
    pipeline.set_profile(True)
    pl.run_pipeline(pipeline)