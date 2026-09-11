r"""Batch-cluster and calibrate every run under a VMI data-day folder.

Built for folders like ``J:\ctgroup\Edward\DATA\VMI\20260820``, where each
immediate subdirectory is one experiment, and each experiment uses one of
three raw-data layouts:

  1. Flat raw run
         <experiment>/raw/*.tpx3
     -> one CV4: <experiment>/<experiment name>.cv4
     (if any non-partial "*.cv4" already sits directly in <experiment>/,
     that counts as done regardless of its exact name)

  2. Raw sweep (calibration + repeated position passes)
         <experiment>/raw/calibration/*.tpx3            (optional, may be empty)
         <experiment>/raw/pos_XXX_<value>/*.tpx3         (one folder per pass)
     Folders are grouped by <value> (the position, e.g. "+26.5700") -- a
     sweep typically revisits the same positions many times. All the pass
     folders for a given position, plus the calibration folder's shots (if
     any), are merged and clustered into ONE CV4:
         <experiment>/<lowest-indexed pos folder for that value>.cv4

  3. Bare position folders (no raw/ wrapper)
         <experiment>/<position>/*.tpx3
     -> one CV4 per folder: <experiment>/<position>.cv4

A CV4 only counts as "done" if the exact non-partial "<name>.cv4" file
exists -- "<name>.partial.cv4" is ignored, per instructions.

The merge for layout 2 is done by hard-linking the relevant .tpx3 files
(original filenames preserved, so they still sort into acquisition order)
into a staging folder and running the normal clustering pipeline over that
folder, then removing the staging folder. Hard links cost no extra disk
space and require no special privileges (as long as the staging folder is
on the same volume as the source data, which it is here).

Two passes:
  1. Cluster every job that isn't fully clustered yet (skips completed ones).
  2. Calibrate every job that *is* fully clustered (including ones this run
     just finished), skipping ones already calibrated. By default the ion-ToF
     channel is skipped for experiments whose name starts with "Xe" (no usable
     itof coincidence data) and used for everything else (e.g. "PO"); override
     with --ignore-itof/--use-itof to force one behavior for every dataset.

Usage:
    python batch_cluster_and_calibrate.py --dry-run
    python batch_cluster_and_calibrate.py
    python batch_cluster_and_calibrate.py --cluster-only --cluster-processes 4
    python batch_cluster_and_calibrate.py --calibrate-only --calibration calibration_20260820
"""

import argparse
import datetime
import inspect
import os
import re
import shutil
from collections import defaultdict

import calibrate_data_cli
import clustering_cli

DEFAULT_ROOT = r"J:\ctgroup\Edward\DATA\VMI\20260820"
POS_RE = re.compile(r"^pos_(\d+)_([+-]?\d+(?:\.\d+)?)$")


# ---------------------------------------------------------------------------
# Discovery / classification
# ---------------------------------------------------------------------------


def _tpx3_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".tpx3")]


class ClusterJob:
    """One clustering job: merge `inputs` (raw folders) and write to `output`."""

    def __init__(self, experiment, label, inputs, output):
        self.experiment = experiment
        self.label = label
        self.inputs = inputs
        self.output = output

    @property
    def done(self):
        return os.path.isfile(self.output)

    @property
    def calibrated_output(self):
        return self.output[: -len(".cv4")] + "_calibrated.h5"

    @property
    def calibrated(self):
        return os.path.isfile(self.calibrated_output)

    def __repr__(self):
        return (
            f"ClusterJob({self.experiment!r}, {self.label!r}, "
            f"inputs={len(self.inputs)}, output={self.output!r}, done={self.done})"
        )


def plan_flat_raw(experiment_dir, raw_dir):
    name = os.path.basename(experiment_dir)
    existing = [
        f
        for f in os.listdir(experiment_dir)
        if f.endswith(".cv4") and not f.endswith(".partial.cv4")
    ]
    output = os.path.join(experiment_dir, existing[0] if existing else name + ".cv4")
    return [ClusterJob(name, name, [raw_dir], output)]


def plan_sweep(experiment_dir, raw_dir):
    name = os.path.basename(experiment_dir)
    entries = sorted(os.listdir(raw_dir))

    calibration_dir = None
    position_dirs = []
    for entry in entries:
        full = os.path.join(raw_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry.lower() == "calibration":
            calibration_dir = full
            continue
        m = POS_RE.match(entry)
        if m:
            position_dirs.append((int(m.group(1)), m.group(2), entry, full))
        else:
            print(f"  [warn] {name}: unrecognized raw subfolder '{entry}', skipping")

    # Only worth merging in if it actually has data.
    if calibration_dir is not None and not _tpx3_files(calibration_dir):
        calibration_dir = None

    groups = defaultdict(list)
    for idx, value, entry, full in position_dirs:
        groups[value].append((idx, entry, full))

    jobs = []
    for members in groups.values():
        members.sort(key=lambda m: m[0])
        _, lead_name, _ = members[0]
        output = os.path.join(experiment_dir, lead_name + ".cv4")
        inputs = [full for _, _, full in members]
        if calibration_dir is not None:
            inputs.append(calibration_dir)
        jobs.append(ClusterJob(name, lead_name, inputs, output))
    return jobs


def plan_position_group(experiment_dir):
    name = os.path.basename(experiment_dir)
    jobs = []
    for entry in sorted(os.listdir(experiment_dir)):
        full = os.path.join(experiment_dir, entry)
        if not os.path.isdir(full) or not _tpx3_files(full):
            continue
        output = os.path.join(experiment_dir, entry + ".cv4")
        jobs.append(ClusterJob(name, entry, [full], output))
    return jobs


def classify_and_plan(experiment_dir):
    name = os.path.basename(experiment_dir)
    raw_dir = os.path.join(experiment_dir, "raw")

    if os.path.isdir(raw_dir):
        sub_entries = os.listdir(raw_dir)
        subdirs = [e for e in sub_entries if os.path.isdir(os.path.join(raw_dir, e))]
        files = [e for e in sub_entries if e.endswith(".tpx3")]
        if subdirs:
            return plan_sweep(experiment_dir, raw_dir)
        if files:
            return plan_flat_raw(experiment_dir, raw_dir)
        print(f"  [warn] {name}: raw/ has neither .tpx3 files nor subfolders, skipping")
        return []

    subdirs = [e for e in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, e))]
    if subdirs:
        return plan_position_group(experiment_dir)
    print(f"  [warn] {name}: no raw/ folder and no subfolders, skipping")
    return []


def discover_jobs(root):
    jobs = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full) or entry.startswith("_"):
            continue
        jobs.extend(classify_and_plan(full))
    return jobs


# ---------------------------------------------------------------------------
# Merging multiple raw folders into one staging folder (hard links)
# ---------------------------------------------------------------------------


def _stage_inputs(job, staging_root):
    staging_dir = os.path.join(staging_root, job.experiment, job.label)
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    seen = set()
    for folder in job.inputs:
        prefix = os.path.basename(folder.rstrip("\\/"))
        for f in _tpx3_files(folder):
            src = os.path.join(folder, f)
            # Keep original filenames (they sort into acquisition order); only
            # disambiguate on an actual collision.
            name = f if f not in seen else f"{prefix}__{f}"
            seen.add(name)
            dst = os.path.join(staging_dir, name)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    return staging_dir


# ---------------------------------------------------------------------------
# Clustering / calibration passes
# ---------------------------------------------------------------------------


def cluster_job(job, staging_root, cluster_processes, converter_processes, cluster_class, keep_staging):
    if job.done:
        print(f"[skip]     {job.experiment}/{job.label}: already clustered ({job.output})")
        return

    staged = None
    if len(job.inputs) == 1:
        input_path = job.inputs[0]
    else:
        input_path = staged = _stage_inputs(job, staging_root)

    print(f"[cluster]  {job.experiment}/{job.label}: {len(job.inputs)} input folder(s) -> {job.output}")
    clustering_cli.cluster_to_cv4(
        input_path,
        job.output,
        cluster_processes=cluster_processes,
        converter_processes=converter_processes,
        cluster_class=cluster_class,
    )

    if staged and not keep_staging:
        shutil.rmtree(staged, ignore_errors=True)


def default_ignore_itof(experiment_name):
    """Xe runs have no usable ion-ToF coincidence data; PO (and anything else) do."""
    return experiment_name.strip().lower().startswith("xe")


def calibrate_job(job, calibration, itof_calibration, symmetrize, ignore_itof):
    if not job.done:
        print(f"[skip]     {job.experiment}/{job.label}: not fully clustered yet, skipping calibration")
        return
    if job.calibrated:
        print(f"[skip]     {job.experiment}/{job.label}: already calibrated ({job.calibrated_output})")
        return

    if ignore_itof is None:
        ignore_itof = default_ignore_itof(job.experiment)

    print(f"[calibrate] {job.experiment}/{job.label}: {job.output} -> {job.calibrated_output}")
    try:
        calibrate_data_cli.process_file(
            job.output,
            job.calibrated_output,
            calibration=calibration,
            symmetrize=symmetrize,
            itof_calibration=itof_calibration,
            ignore_itof=ignore_itof,
        )
    except Exception as exc:  # one bad cv4 shouldn't block the rest of the batch
        print(f"[error]    {job.experiment}/{job.label}: calibration failed: {exc!r}")
        if os.path.isfile(job.calibrated_output):
            os.remove(job.calibrated_output)


def write_calibration_info(root, calibration, itof_calibration, symmetrize, ignore_itof_arg):
    """Write/refresh a plain-text summary of the calibrations applied to this
    data folder's *_calibrated.h5 files, so it's documented alongside the data."""
    path = os.path.join(root, "calibration_info.txt")
    lines = [
        "Calibration info for the *_calibrated.h5 files in this folder.",
        f"Generated by batch_cluster_and_calibrate.py on {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Momentum/position calibration: {calibration.__name__}",
        inspect.getsource(calibration).rstrip(),
        "",
        f"Ion time-of-flight calibration: {itof_calibration.__name__}",
        inspect.getsource(itof_calibration).rstrip(),
        "",
        "  t = t0 + scale * sqrt(m/z)   (t0=582, scale=1735, ns)",
        "  inverted (as applied):  m/z = ((t - t0) / scale) ** 2",
        "",
        f"symmetrize: {symmetrize}",
    ]
    if ignore_itof_arg is None:
        lines.append(
            "ignore_itof: auto per dataset -- skipped for experiments named starting with "
            "'Xe' (no usable ion-ToF coincidence), applied for everything else (e.g. 'PO...')."
        )
    else:
        lines.append(f"ignore_itof: forced to {ignore_itof_arg} for every dataset (--ignore-itof/--use-itof).")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[info]     wrote calibration info to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-cluster and calibrate every run under a VMI data-day folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"Data-day folder to process (default: {DEFAULT_ROOT}).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print the discovered jobs and their status; do nothing."
    )
    parser.add_argument(
        "--skip", action="append", default=[], metavar="SUBSTR",
        help="Drop any job whose '<experiment>/<label>' contains this (case-insensitive) substring, "
        "from both passes. Repeatable.",
    )
    parser.add_argument(
        "--max-input-folders", type=int, default=None, metavar="N",
        help="Drop any clustering job that merges more than N raw folders (use to skip the huge sweeps).",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cluster-only", action="store_true", help="Only run the clustering pass.")
    mode.add_argument("--calibrate-only", action="store_true", help="Only run the calibration pass.")

    parser.add_argument("--cluster-processes", type=int, default=1, help="Clustering worker processes per job.")
    parser.add_argument("--converter-processes", type=int, default=1, help="Converter worker processes per job.")
    parser.add_argument(
        "--cluster-class", choices=sorted(clustering_cli.CLUSTER_CLASSES), default="custom",
        help="Clustering algorithm to use (default: custom).",
    )
    parser.add_argument(
        "--keep-staging", action="store_true",
        help="Keep the merged-input staging folders (under <root>/_cluster_staging) after clustering, for debugging.",
    )

    parser.add_argument(
        "--calibration", default="calibration_20260820",
        help="Name of the calibration function in calibrations.py to apply (default: calibration_20260820).",
    )
    parser.add_argument(
        "--itof-calibration", default="itof_calibration_20260820",
        help="Name of the itof calibration function to apply (default: itof_calibration_20260820, "
        "t = 582 + 1735*sqrt(m/z)).",
    )
    parser.add_argument("--symmetrize", action="store_true", help="Symmetrize the momentum distribution.")

    itof = parser.add_mutually_exclusive_group()
    itof.add_argument(
        "--ignore-itof", dest="ignore_itof", action="store_true", default=None,
        help="Force-skip the ion-ToF channel for every dataset.",
    )
    itof.add_argument(
        "--use-itof", dest="ignore_itof", action="store_false",
        help="Force-load/calibrate the ion-ToF channel for every dataset.",
    )
    # Default (neither flag given): auto -- ignore itof for Xe datasets, use it for
    # everything else (see default_ignore_itof). Override per run with the flags above.

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    print(f"Scanning {args.root} ...")
    jobs = discover_jobs(args.root)
    print(f"Found {len(jobs)} position/run group(s) across the experiments in {args.root}.")

    skips = [s.lower() for s in args.skip]
    kept = []
    for job in jobs:
        key = f"{job.experiment}/{job.label}"
        reason = None
        if any(s in key.lower() for s in skips):
            reason = "--skip"
        elif args.max_input_folders is not None and len(job.inputs) > args.max_input_folders:
            reason = f"--max-input-folders ({len(job.inputs)} > {args.max_input_folders})"
        if reason:
            print(f"[exclude]  {key}: {reason}")
        else:
            kept.append(job)
    jobs = kept
    print(f"{len(jobs)} job(s) after exclusions.\n")

    if args.dry_run:
        for job in jobs:
            status = "done" if job.done else "PENDING"
            print(f"[{status:7}] {job.experiment}/{job.label}: {len(job.inputs)} folder(s) -> {job.output}")
        return

    if not args.calibrate_only:
        staging_root = os.path.join(args.root, "_cluster_staging")
        for job in jobs:
            cluster_job(
                job,
                staging_root,
                args.cluster_processes,
                args.converter_processes,
                args.cluster_class,
                args.keep_staging,
            )

    if not args.cluster_only:
        calibration = calibrate_data_cli._resolve(calibrate_data_cli.CALIBRATIONS, args.calibration, "calibration")
        itof_calibration = calibrate_data_cli._resolve(
            calibrate_data_cli.ITOF_CALIBRATIONS, args.itof_calibration, "itof calibration"
        )
        write_calibration_info(args.root, calibration, itof_calibration, args.symmetrize, args.ignore_itof)
        for job in jobs:
            calibrate_job(job, calibration, itof_calibration, args.symmetrize, args.ignore_itof)


if __name__ == "__main__":
    main()
