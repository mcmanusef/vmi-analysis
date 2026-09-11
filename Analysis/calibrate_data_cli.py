"""Command-line interface for calibrating .cv4 files into momentum-space .h5 files.

Wraps the same logic as "Calibrate Data.ipynb" (process_file/process_folder)
behind an argparse CLI, so a run or a folder of runs can be calibrated from a
terminal without hand-editing the notebook.

Examples:
    python calibrate_data_cli.py list-calibrations
    python calibrate_data_cli.py file "J:\\...\\PO- P Sample.cv4" --calibration calibration_20260820 --ignore-itof
    python calibrate_data_cli.py folder "J:\\...\\20260820\\PO- P" --calibration calibration_20260820 --symmetrize
"""

import argparse
import inspect
import os
import sys

import numpy as np
import pandas as pd

# The repo root (for the `vmi_analysis` package, imported transitively) and
# `coincidence_v4`/`calibrations` live outside this directory. Add them to
# sys.path so this script also works when run directly from a terminal, not
# just from an IDE with those roots configured.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UCONN_VMI = os.path.join(_REPO_ROOT, "Examples", "uconn_vmi")
for _path in (_REPO_ROOT, _UCONN_VMI):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import calibrations
import coincidence_v4

CALIBRATIONS = {
    name: fn
    for name, fn in inspect.getmembers(calibrations, inspect.isfunction)
    if name.startswith("calibration_")
}
ITOF_CALIBRATIONS = {
    name: fn
    for name, fn in inspect.getmembers(calibrations, inspect.isfunction)
    if name.startswith("itof_calibration")
}


def process_file(
    file,
    save_path,
    calibration,
    symmetrize=False,
    itof_calibration=None,
    center=None,
    angle=None,
    ignore_itof=False,
):
    x, y, t, etof, itof = coincidence_v4.load_file(file, coincidence=not ignore_itof)
    etof = etof + 0.26 * np.random.random_sample(len(etof))

    kwargs = {"symmetrize": symmetrize}
    if center is not None:
        kwargs["center"] = center
    if angle is not None:
        kwargs["angle"] = angle
    px, py, pz = calibration(x, y, etof, **kwargs)

    if not ignore_itof:
        data = pd.DataFrame(
            {
                "px": px,
                "py": py,
                "pz": pz,
                "m/q": itof_calibration(itof),
                "raw_x": x,
                "raw_y": y,
                "raw_t": t,
                "raw_etof": etof,
                "raw_itof": itof,
            }
        )
    else:
        data = pd.DataFrame(
            {"px": px, "py": py, "pz": pz, "raw_x": x, "raw_y": y, "raw_t": t, "raw_etof": etof}
        )
    data.to_hdf(save_path, key="data", mode="w")


def process_folder(
    folder,
    save_folder=None,
    calibration=None,
    symmetrize=False,
    itof_calibration=None,
    center=None,
    angle=None,
    ignore_itof=False,
):
    if save_folder is None:
        save_folder = os.path.join(folder, "calibrated")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for file in os.listdir(folder):
        if not file.endswith(".cv4"):
            continue
        save_path = os.path.join(save_folder, file.replace(".cv4", "_calibrated.h5"))
        if os.path.exists(save_path):
            print(f"Skipping {file}, already processed.")
            continue
        try:
            print(f"Processing {file}")
            process_file(
                os.path.join(folder, file),
                save_path,
                calibration,
                symmetrize,
                itof_calibration,
                center,
                angle,
                ignore_itof,
            )
            print(f"Saved to {save_path}")
        except OSError as e:
            print(f"Error processing {file}: {e}")


def _resolve(registry, name, kind):
    if name not in registry:
        choices = ", ".join(sorted(registry))
        raise SystemExit(f"Unknown {kind} '{name}'. Available: {choices}")
    return registry[name]


def _add_common_args(subparser):
    subparser.add_argument(
        "--calibration", required=True, help="Name of the calibration function in calibrations.py to apply."
    )
    subparser.add_argument(
        "--itof-calibration",
        default="itof_calibration",
        help="Name of the itof calibration function to apply (default: itof_calibration).",
    )
    subparser.add_argument("--symmetrize", action="store_true", help="Symmetrize the momentum distribution.")
    subparser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "T"),
        default=None,
        help="Override the calibration's default detector center (x, y, t).",
    )
    subparser.add_argument(
        "--angle", type=float, default=None, help="Override the calibration's default rotation angle."
    )
    subparser.add_argument(
        "--ignore-itof",
        action="store_true",
        help="Skip loading/calibrating the ion-ToF channel (electron-only data, no coincidence).",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate .cv4 files into momentum-space .h5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-calibrations", help="List available calibration and itof-calibration names.")

    file_parser = subparsers.add_parser("file", help="Calibrate a single .cv4 file.")
    file_parser.add_argument("path", help="Path to the input .cv4 file.")
    file_parser.add_argument(
        "-o", "--output", default=None, help="Output .h5 path (default: <path without .cv4>_calibrated.h5)."
    )
    _add_common_args(file_parser)

    folder_parser = subparsers.add_parser("folder", help="Calibrate every .cv4 file in a folder.")
    folder_parser.add_argument("path", help="Path to the folder of .cv4 files.")
    folder_parser.add_argument(
        "-o", "--output-folder", default=None, help="Output folder (default: <path>/calibrated)."
    )
    _add_common_args(folder_parser)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "list-calibrations":
        print("Calibrations:")
        for name in sorted(CALIBRATIONS):
            print(f"  {name}")
        print("Itof calibrations:")
        for name in sorted(ITOF_CALIBRATIONS):
            print(f"  {name}")
        return

    calibration = _resolve(CALIBRATIONS, args.calibration, "calibration")
    itof_calibration = _resolve(ITOF_CALIBRATIONS, args.itof_calibration, "itof calibration")
    center = tuple(args.center) if args.center is not None else None

    if args.command == "file":
        output = args.output or args.path.replace(".cv4", "_calibrated.h5")
        process_file(
            args.path,
            output,
            calibration=calibration,
            symmetrize=args.symmetrize,
            itof_calibration=itof_calibration,
            center=center,
            angle=args.angle,
            ignore_itof=args.ignore_itof,
        )
        print(f"Saved to {output}")
    elif args.command == "folder":
        process_folder(
            args.path,
            save_folder=args.output_folder,
            calibration=calibration,
            symmetrize=args.symmetrize,
            itof_calibration=itof_calibration,
            center=center,
            angle=args.angle,
            ignore_itof=args.ignore_itof,
        )


if __name__ == "__main__":
    main()
