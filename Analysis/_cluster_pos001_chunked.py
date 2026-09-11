r"""Chunked re-clustering of PO (-) e0.5 Long / pos_001_-26.5700.

The normal path clusters all 145 raw folders for this sweep position in ONE
pipeline run; the reader races ahead of the correlator/saver and the unbounded
in-memory queues exhaust RAM (see memory: batch-cluster-calibrate-notes).

Instead, this clusters each raw folder on its own (tiny, ~85 MB of tpx3 -> a
small .cv4 in a staging dir), then merges the per-folder .cv4 files into the
final one by concatenating every dataset and offsetting the per-folder
correlation indices (cluster_corr / etof_corr / tof_corr) by the running pulse
count, so coincidence matching (which is purely `corr == corr == corr` over
monotonic streams, see coincidence_v4.iter_first) still works across the join.

Each folder is clustered in its OWN fresh subprocess (`python _cluster_..._chunked.py
one <src> <out>`). `base_pipeline.stop()` calls `Process.join(timeout=5)` inside a
`try/except TimeoutError`, but `multiprocessing.Process.join` never raises TimeoutError,
so a pipeline whose workers don't stop cleanly is never `terminate()`d -- calling
`cluster_to_cv4` in a loop in one process leaked ~5 worker processes per folder and hit
MemoryError around folder 45. A one-shot subprocess per folder exits fully and the OS
reclaims everything.

Phases (resumable, each folder's staging .cv4 is skipped if already valid):
    python _cluster_pos001_chunked.py cluster   # per-folder clustering (spawns subprocesses)
    python _cluster_pos001_chunked.py merge     # staging -> final .cv4
    python _cluster_pos001_chunked.py           # cluster, then merge
    python _cluster_pos001_chunked.py one SRC OUT   # (internal) cluster one folder

Must stay __main__-guarded: spawn-based multiprocessing re-imports this module
in every child.
"""
import os
import subprocess
import sys
import traceback

import h5py
import numpy as np

PER_FOLDER_TIMEOUT = 900  # s; a folder clusters in ~90 s, 10x headroom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clustering_cli  # noqa: E402

ROOT = r"J:\ctgroup\Edward\DATA\VMI\20260820"
EXP = r"PO (-) e0.5 Long"
VALUE = "-26.5700"
RAW = os.path.join(ROOT, EXP, "raw")
STAGING = os.path.join(ROOT, "_cluster_staging", EXP, "pos_001_-26.5700_chunks")
FINAL = os.path.join(ROOT, EXP, "pos_001_-26.5700.cv4")

# (h5 dataset name, dtype) for a .cv4 written by CV4ConverterPipeline / SaveToH5.
CORR_KEYS = ("cluster_corr", "etof_corr", "tof_corr")
SPECS = {
    "cluster_corr": "i4", "t": "f8", "x": "f8", "y": "f8",
    "etof_corr": "i4", "t_etof": "f8",
    "tof_corr": "i4", "t_tof": "f8",
    "t_pulse": "f8",
}


def folders():
    out = []
    for d in sorted(os.listdir(RAW)):
        full = os.path.join(RAW, d)
        if not os.path.isdir(full) or not d.endswith(VALUE):
            continue
        if any(f.endswith(".tpx3") for f in os.listdir(full)):
            out.append((d, full))
    return out


def valid_cv4(path):
    if not os.path.isfile(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return all(k in f for k in SPECS)
    except Exception:
        return False


def cluster_one(src, out):
    if os.path.isfile(out):
        os.remove(out)
    clustering_cli.cluster_to_cv4(
        src, out, cluster_processes=1, converter_processes=1, cluster_class="custom"
    )
    if not valid_cv4(out):
        raise RuntimeError(f"clustering produced an invalid .cv4: {out}")


def cluster_all():
    os.makedirs(STAGING, exist_ok=True)
    fl = folders()
    print(f"{len(fl)} raw folder(s) at position {VALUE}", flush=True)
    for i, (name, full) in enumerate(fl, 1):
        out = os.path.join(STAGING, name + ".cv4")
        if valid_cv4(out):
            print(f"[skip {i}/{len(fl)}] {name}", flush=True)
            continue
        if os.path.isfile(out):
            os.remove(out)  # partial from a killed run
        print(f"[cluster {i}/{len(fl)}] {name}", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "one", full, out]
        try:
            r = subprocess.run(cmd, timeout=PER_FOLDER_TIMEOUT)
        except subprocess.TimeoutExpired:
            if os.path.isfile(out):
                os.remove(out)
            raise RuntimeError(f"{name}: clustering timed out after {PER_FOLDER_TIMEOUT}s")
        if r.returncode != 0 or not valid_cv4(out):
            if os.path.isfile(out):
                os.remove(out)
            raise RuntimeError(f"{name}: clustering subprocess failed (rc={r.returncode})")
    return fl


def merge(fl=None):
    if fl is None:
        fl = folders()
    missing = [n for n, _ in fl if not valid_cv4(os.path.join(STAGING, n + ".cv4"))]
    if missing:
        raise RuntimeError(f"{len(missing)} folder(s) not yet clustered, e.g. {missing[:3]}")

    tmp = FINAL + ".merging"
    if os.path.isfile(tmp):
        os.remove(tmp)

    offset = 0
    with h5py.File(tmp, "w") as out:
        ds = {
            k: out.create_dataset(k, (0,), maxshape=(None,), dtype=dt, chunks=(1 << 20,))
            for k, dt in SPECS.items()
        }

        def append(key, arr):
            n = ds[key].shape[0]
            ds[key].resize(n + len(arr), axis=0)
            if len(arr):
                ds[key][n:] = arr

        for i, (name, _full) in enumerate(fl, 1):
            with h5py.File(os.path.join(STAGING, name + ".cv4"), "r") as f:
                npulse = len(f["t_pulse"])
                corrs = {k: f[k][()] for k in CORR_KEYS}
                for k, arr in corrs.items():
                    if len(arr):
                        assert np.all(np.diff(arr) >= 0), f"{name}/{k} not sorted"
                        assert arr.min() >= 0, f"{name}/{k} negative index"
                        assert arr.max() < npulse, f"{name}/{k} index {arr.max()} >= npulse {npulse}"
                append("cluster_corr", corrs["cluster_corr"] + offset)
                append("t", f["t"][()])
                append("x", f["x"][()])
                append("y", f["y"][()])
                append("etof_corr", corrs["etof_corr"] + offset)
                append("t_etof", f["t_etof"][()])
                append("tof_corr", corrs["tof_corr"] + offset)
                append("t_tof", f["t_tof"][()])
                append("t_pulse", f["t_pulse"][()])
                offset += npulse
            print(
                f"[merge {i}/{len(fl)}] {name}: +{len(corrs['cluster_corr'])} clusters "
                f"(running pulses {offset}, out clusters {ds['cluster_corr'].shape[0]})",
                flush=True,
            )

    os.replace(tmp, FINAL)
    with h5py.File(FINAL, "r") as f:
        print(
            f"FINAL {FINAL}: {len(f['cluster_corr'])} clusters, {len(f['etof_corr'])} etofs, "
            f"{len(f['tof_corr'])} itofs, {len(f['t_pulse'])} pulses",
            flush=True,
        )


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "one":
        cluster_one(sys.argv[2], sys.argv[3])
        return
    if phase in ("cluster", "all"):
        fl = cluster_all()
    else:
        fl = None
    if phase in ("merge", "all"):
        merge(fl)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
