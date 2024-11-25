
from vmi_analysis.processing.pipelines import CV4ConverterPipeline, run_pipeline
import os
def convert_cv4(fname):
    pipeline = CV4ConverterPipeline(fname, fname + ".cv4", cluster_processes=8)
    run_pipeline(pipeline)

def bulk_convert_cv4(dirname):
    for f in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, f)) and not os.path.exists(os.path.join(dirname, f+".cv4")):
            fname = os.path.join(dirname, f)
        else:
            continue
        convert_cv4(fname)

if __name__ == '__main__':
    fname=r"J:\ctgroup\Edward\DATA\VMI\20241120\c2h4_p_1,5W"
    convert_cv4(fname)
#%%
