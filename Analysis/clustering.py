from sb_pipelines import StonyBrookClusterPipeline
from uconn_pipelines import CV4ConverterPipeline, VMIConverterPipeline
from vmi_analysis.processing.pipelines import run_pipeline
from vmi_analysis.processing.processes import *


def convert_cv4(fname):
    pipeline = CV4ConverterPipeline(
        fname,
        fname + ".cv4",
            cluster_processes=1,
        converter_processes=1,
        cluster_class=CustomClusterer,
    )
    run_pipeline(pipeline)

def convert_uv4(fname):
    pipeline = VMIConverterPipeline(
        fname,
        fname + ".uv4",
    )
    run_pipeline(pipeline)



def convert_stonybrook(fname):
    pipeline = StonyBrookClusterPipeline(
            fname,
            fname[:-4] + "cv4",
    )
    run_pipeline(pipeline)


def bulk_convert_cv4(dirname):
    for f in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, f)) and not os.path.exists(
            os.path.join(dirname, f + ".cv4")
        ):
            fname = os.path.join(dirname, f)
        else:
            continue
        convert_cv4(fname)

if __name__ == "__main__":
    # file = r"J:\ctgroup\Edward\DATA\VMI\20260430\PO_P_Cal"
    # convert_cv4(file)
    # folder= r"J:\ctgroup\Edward\DATA\VMI\20260430"
    folder = r"J:\ctgroup\Edward\DATA\VMI\20260722\(r) PO Circular -"
    convert_cv4(folder)
# %%
