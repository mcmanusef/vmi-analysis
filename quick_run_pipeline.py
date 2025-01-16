import vmi_analysis.processing.pipelines as pl

if __name__ == '__main__':
    fname = r"/mnt/d/Data/testds"
    # pipeline=pl.CV4ConverterPipeline(fname,fname+".cv4",cluster_processes=1,converter_processes=1, cuda=True)
    pipeline=pl.MultiprocessTestPipeline(n=8)
    pipeline.set_profile(True)
    pl.run_pipeline(pipeline)