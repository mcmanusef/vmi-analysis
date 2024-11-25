import os
from processing.pipelines import CV4ConverterPipeline, run_pipeline
import logging
import time

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s:   %(message)s', level=logging.INFO)
    dirname = r"J:\ctgroup\Edward\DATA\VMI\20241120"
    for f in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, f)) and not os.path.exists(os.path.join(dirname, f+".cv4")):
            fname = os.path.join(dirname, f)
        else:
            continue
        # fname=r"J:\ctgroup\Edward\DATA\VMI\20241120\c2h4_p_1,2W\ds_000000.tpx3"

        pipeline= CV4ConverterPipeline(fname, fname + ".cv4", cluster_processes=8).set_profile(True)
        start = time.time()
        run_pipeline(pipeline)
        print(f"Time taken: {time.time() - start}")
# %%
