from vmi_analysis.processing.pipelines import CV4ConverterPipeline, run_pipeline
from vmi_analysis.processing.processes import *
import os


def convert_cv4(fname):
    pipeline = CV4ConverterPipeline(fname, fname + ".cv4", cluster_processes=1,converter_processes=1, cluster_class=CustomClusterer)
    run_pipeline(pipeline)


def bulk_convert_cv4(dirname):
    for f in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, f)) and not os.path.exists(os.path.join(dirname, f + ".cv4")):
            fname = os.path.join(dirname, f)
        else:
            continue
        convert_cv4(fname)


if __name__ == '__main__':
    fname = r"C:\DATA\20250123\Propylene Oxide 2W"
    bulk_convert_cv4(fname)
#%%
