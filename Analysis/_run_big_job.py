"""Detached single-job runner for the one clustering job that exceeds the
background-task time limit: PO (-) e0.5 Long / pos_002_+26.5700 (144 folders).
Same behaviour as batch_cluster_and_calibrate's clustering pass for this job.

Must be __main__-guarded: the pipeline uses spawn-based multiprocessing, and
on Windows every child re-imports this module.
"""
import os
import sys

import batch_cluster_and_calibrate as b

ROOT = r"J:\ctgroup\Edward\DATA\VMI\20260820"
EXPERIMENT = "PO (-) e0.5 Long"
LABEL = "pos_002_+26.5700"


def main():
    jobs = b.discover_jobs(ROOT)
    job = next(j for j in jobs if j.experiment == EXPERIMENT and j.label == LABEL)
    print(f"job: {job!r}", flush=True)
    if job.done:
        print("already done, nothing to do", flush=True)
        return

    staging_root = os.path.join(ROOT, "_cluster_staging")
    b.cluster_job(
        job,
        staging_root,
        cluster_processes=1,
        converter_processes=1,
        cluster_class="custom",
        keep_staging=False,
    )
    print("DONE" if job.done else "FINISHED-BUT-NO-OUTPUT", flush=True)


if __name__ == "__main__":
    main()
