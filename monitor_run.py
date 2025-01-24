from vmi_analysis.processing.pipelines import RunMonitorPipeline, run_pipeline
import os
from vmi_analysis.calibrations import calibration_20241120

if __name__ == '__main__':
    fname = r"C:\DATA\20250123\Propylene Oxide 2W\45"
    pipeline = RunMonitorPipeline(fname, cluster_processes=1, toa_range=(200,250), calibration=calibration_20241120, center=(133.2, 131.7, 496), angle=-0.43)
    run_pipeline(pipeline)

#%%
