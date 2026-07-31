#!/usr/bin/env python3
"""
Rebuild missing out/presto_xy/healpix_<id>_float32.npz files from the dense
out/healpix_<id>_presto.npz files that are already on disk.

The xy file is a pure extraction from the dense file — the same pixels, with the
GEDI-invalid ones dropped (see save_xy_npz). So a tile that has its dense npz does NOT
need Presto, a GPU, or the materialized Sentinel-2 imagery re-run to get its xy file;
it only needs the mask re-applied. This is seconds per tile instead of minutes, and it
works even after the imagery has been deleted to reclaim /ocean quota.

It calls save_xy_npz() directly, so the selection rules (finite GEDI AND fully finite
embedding row) cannot drift from what the pipeline itself writes.

Usage:
    # rebuild every xy file that is missing
    python make_xy_from_dense.py --out-dir /ocean/.../prithvi_presto/2019/out

    # see what it would do first
    python make_xy_from_dense.py --out-dir ... --dry-run

    # also replace xy files that already exist
    python make_xy_from_dense.py --out-dir ... --overwrite

MEMORY: a dense npz holds X as (H, W, 128) float32 and np.load must decompress the whole
array — this can be several GB per tile. Run it on a compute node, not the login node.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing.unified_embedding_pipeline import save_xy_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="the pipeline --output-dir holding healpix_<id>_presto.npz")
    ap.add_argument("--model", default="presto", choices=["presto", "olmo", "tessera"],
                    help="which dense files to convert (default: presto). Reads "
                         "healpix_<id>_<model>.npz, writes <model>_xy/healpix_<id>_float32.npz")
    ap.add_argument("--healpix-id", type=int, default=None,
                    help="convert only this tile (used by run_make_xy.sh's array job). "
                         "Omit to convert every dense npz found in --out-dir.")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite xy files that already exist (default: skip them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be done, write nothing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        ap.error(f"--out-dir is not a directory: {out_dir}")
    xy_dir = out_dir / f"{args.model}_xy"

    if args.healpix_id is not None:
        one = out_dir / f"healpix_{args.healpix_id}_{args.model}.npz"
        if not one.exists():
            # Not an error: the array job covers a tile list that may include tiles whose
            # dense npz was never written. Exit 0 so the task does not report FAILED.
            print(f"No dense npz for tile {args.healpix_id} ({one.name}) — nothing to do.")
            return
        dense_files = [one]
    else:
        dense_files = sorted(out_dir.glob(f"healpix_*_{args.model}.npz"))
    if not dense_files:
        print(f"No healpix_*_{args.model}.npz found in {out_dir}")
        return

    print(f"Dense files : {len(dense_files)} in {out_dir}")
    print(f"xy target   : {xy_dir}")
    print(f"Mode        : {'DRY RUN' if args.dry_run else 'writing'}"
          f"{', overwriting existing' if args.overwrite else ', skipping existing'}\n")

    made = skipped = failed = 0
    for dense in dense_files:
        tid = int(dense.name.split("_")[1])
        xy_path = xy_dir / f"healpix_{tid}_float32.npz"
        if xy_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"  would build healpix_{tid}_float32.npz from {dense.name}")
            made += 1
            continue
        try:
            with np.load(dense, allow_pickle=False) as z:
                X = z["X"]          # (H, W, C) — the layout save_dense_npz wrote
                y = z["y"]          # (H, W), NaN where GEDI is invalid
            # hwc=True: X is already (H,W,C), so no transpose round-trip on a multi-GB array.
            n = save_xy_npz(xy_dir, tid, X, y, hwc=True)
            del X, y
            print(f"  healpix_{tid}: {n} samples -> {xy_path.name}")
            made += 1
        except Exception as exc:
            # A truncated dense npz (quota casualty) lands here — report and continue so
            # one bad tile does not stop the rebuild.
            print(f"  healpix_{tid}: FAILED reading {dense.name} — {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\nBuilt: {made}   Already present: {skipped}   Failed: {failed}")
    if failed:
        print("Failed tiles have an unreadable dense npz — those DO need a full re-run "
              "from the imagery (delete the bad dense file first).")


if __name__ == "__main__":
    main()
