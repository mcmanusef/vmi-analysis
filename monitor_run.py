from vmi_analysis.processing.pipelines import RunMonitorPipeline, run_pipeline
import os

if __name__ == '__main__':
    fname = r"J:\ctgroup\Edward\DATA\VMI\20241125\c2h4_p_5W"
    pipeline = RunMonitorPipeline(fname, cluster_processes=8)
    run_pipeline(pipeline)
