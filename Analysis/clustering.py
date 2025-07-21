from uconn_pipelines import CV4ConverterPipeline
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


def bulk_convert_cv4(dirname):
    for f in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, f)) and not os.path.exists(
            os.path.join(dirname, f + ".cv4")
        ):
            fname = os.path.join(dirname, f)
        else:
            continue
        convert_cv4(fname)


def continuous_bulk_convert_cv4(dirname):
    while True:
        for f in os.listdir(dirname):
            if not os.path.isdir(os.path.join(dirname, f)):
                continue
            for file in os.listdir(os.path.join(dirname, f)):
                if not file.endswith(".tpx3") or os.path.exists(
                    os.path.join(dirname, f, file + ".cv4")
                ):
                    continue
                if os.path.getsize(os.path.join(dirname, f, file)) < 1000000:
                    continue
                fname = os.path.join(dirname, f, file)
                print(f"Converting {fname}")
                convert_cv4(fname)


if __name__ == "__main__":
    file = r"J:\ctgroup\Edward\DATA\VMI\20250701\1,5_W_Pump__3_W_Probe"
    bulk_convert_cv4(file)
    # continuous_bulk_convert_cv4(file)
# %%
