from vmi_analysis.processing.pipelines import run_pipeline
from uconn_pipelines import RunMonitorPipeline
from calibrations import calibration_20241120

if __name__ == "__main__":
    fname = r"C:\DATA\20250227\xe_2,5W\0"
    pipeline = RunMonitorPipeline(
        fname,
        cluster_processes=4,
        toa_range=(200, 250),
        calibration=calibration_20241120,
        center=(133.2, 131.7, 496),
        angle=-0.43,
    )
    run_pipeline(pipeline)

# %%
