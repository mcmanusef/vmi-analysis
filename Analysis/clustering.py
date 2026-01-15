from sb_pipelines import StonyBrookClusterPipeline
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


def convert_stonybrook(fname):
    pipeline = StonyBrookClusterPipeline(
            fname,
            fname + "cv4",
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
    # dir =r"C:\DATA\StonyBrookCollab\2025_11_05_tpx"
    # # file = r"C:\DATA\StonyBrookCollab\2025_11_05_tpx\prop_oxide_rotated_t0_2025-11-05_18-12"
    # for folder in os.listdir(dir):
    #     for f in os.listdir(os.path.join(dir, folder)):
    #         if f.endswith(".tpx3"):
    #             if not os.path.exists(os.path.join(dir,folder, f[:-4] + "cv4")):
    #                 convert_stonybrook(os.path.join(dir, folder, f))

    folder = r"C:\DATA\StonyBrookCollab\2025_11_05_tpx\prop_oxide_coarse_2025-11-05_16-34"
    convert_stonybrook(folder)


    # convert_stonybrook(file)
    # convert_cv4(file)
    # bulk_convert_cv4(file)x
    # continuous_bulk_convert_cv4(file)
# %%
