"""Command-line interface for converting raw TPX3/VMI runs.

Wraps the conversion pipelines used by clustering.py (CV4, UV4, and Stony
Brook cluster conversion) behind an argparse CLI, so runs can be converted
from a terminal without hand-editing the hardcoded paths in that script.

Examples:
    python clustering_cli.py cv4 "J:\\ctgroup\\Edward\\DATA\\VMI\\20260722\\run1"
    python clustering_cli.py cv4 "J:\\...\\run1" --cluster-processes 4 --cluster-class dbscan
    python clustering_cli.py uv4 "C:\\DATA\\20250123\\run1\\0"
    python clustering_cli.py sb "C:\\DATA\\StonyBrookCollab\\run1.tpx3"
    python clustering_cli.py bulk-cv4 "J:\\...\\20260722" --cluster-processes 4
"""

import argparse
import os
import sys

# The repo root (for the `vmi_analysis` package) and `sb_pipelines`/
# `uconn_pipelines` (and everything they import) live outside this
# directory. Add them to sys.path so this script also works when run
# directly from a terminal, not just from an IDE with those roots configured.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_REPO_ROOT, "Examples")
for _path in (_REPO_ROOT, os.path.join(_EXAMPLES, "uconn_vmi"), os.path.join(_EXAMPLES, "stony_brook_vmi")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sb_pipelines import StonyBrookClusterPipeline
from uconn_pipelines import CV4ConverterPipeline, VMIConverterPipeline
from vmi_analysis.processing import processes
from vmi_analysis.processing.pipelines import run_pipeline

CLUSTER_CLASSES = {
    "custom": processes.CustomClusterer,
    "dbscan": processes.DBSCANClusterer,
}


def cluster_to_cv4(input_path, output_path, cluster_processes=1, converter_processes=1, cluster_class="custom"):
    pipeline = CV4ConverterPipeline(
        input_path,
        output_path,
        cluster_processes=cluster_processes,
        converter_processes=converter_processes,
        cluster_class=CLUSTER_CLASSES[cluster_class],
    )
    run_pipeline(pipeline)


def convert_cv4(fname, cluster_processes=1, converter_processes=1, cluster_class="custom"):
    cluster_to_cv4(
        fname,
        fname + ".cv4",
        cluster_processes=cluster_processes,
        converter_processes=converter_processes,
        cluster_class=cluster_class,
    )


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


def bulk_convert_cv4(dirname, cluster_processes=1, converter_processes=1, cluster_class="custom"):
    for f in os.listdir(dirname):
        fname = os.path.join(dirname, f)
        if not os.path.isdir(fname) or os.path.exists(fname + ".cv4"):
            continue
        print(f"Converting {fname}")
        convert_cv4(
            fname,
            cluster_processes=cluster_processes,
            converter_processes=converter_processes,
            cluster_class=cluster_class,
        )


def _add_cluster_args(subparser):
    subparser.add_argument(
        "--cluster-processes", type=int, default=1, help="Number of clustering worker processes (default: 1)."
    )
    subparser.add_argument(
        "--converter-processes", type=int, default=1, help="Number of converter worker processes (default: 1)."
    )
    subparser.add_argument(
        "--cluster-class",
        choices=sorted(CLUSTER_CLASSES),
        default="custom",
        help="Clustering algorithm to use (default: custom).",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert raw TPX3/VMI runs to .cv4/.uv4 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cv4 = subparsers.add_parser(
        "cv4", help="Convert a single run to a .cv4 file (clustered, correlated)."
    )
    cv4.add_argument("path", help="Path to the raw run file/directory.")
    _add_cluster_args(cv4)

    uv4 = subparsers.add_parser(
        "uv4", help="Convert a single run to a .uv4 file (unclustered, correlated)."
    )
    uv4.add_argument("path", help="Path to the raw run file/directory.")

    sb = subparsers.add_parser(
        "sb", help="Convert a single Stony Brook run to a .cv4 file."
    )
    sb.add_argument("path", help="Path to the raw .tpx3 file.")

    bulk = subparsers.add_parser(
        "bulk-cv4",
        help="Convert every not-yet-converted run directory inside a folder to .cv4.",
    )
    bulk.add_argument("path", help="Path to the folder containing run directories.")
    _add_cluster_args(bulk)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "cv4":
        convert_cv4(
            args.path,
            cluster_processes=args.cluster_processes,
            converter_processes=args.converter_processes,
            cluster_class=args.cluster_class,
        )
    elif args.command == "uv4":
        convert_uv4(args.path)
    elif args.command == "sb":
        convert_stonybrook(args.path)
    elif args.command == "bulk-cv4":
        bulk_convert_cv4(
            args.path,
            cluster_processes=args.cluster_processes,
            converter_processes=args.converter_processes,
            cluster_class=args.cluster_class,
        )


if __name__ == "__main__":
    main()
