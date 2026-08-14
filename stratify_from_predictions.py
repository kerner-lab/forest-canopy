#!/usr/bin/env python3
"""
Backfill stratified (height-binned) metrics into completed seed-sweep runs.

Runs finished before the stratified code was added have no "stratified" block in
their metrics.json — but every run already saved predictions_<split>.parquet
(y_true, y_pred), which is all the binned metrics need. So the numbers can be
recovered WITHOUT re-training anything.

It calls run_gedi_prediction_30m.stratified_metrics() directly, so the binning
rules cannot drift from what the pipeline itself computes.

For each <root>/<source>/seed_<n>/:
    read  predictions_<split>.parquet
    write metrics.json["<split>"]["stratified"]

Then re-run aggregate_seed_sweep.py to get the mean +/- std table.

Usage:
    # see what would change
    python stratify_from_predictions.py --root outputs_seed_sweep_2019 --dry-run

    # write the stratified block into every run's metrics.json
    python stratify_from_predictions.py --root outputs_seed_sweep_2019

    # both splits, then aggregate
    python stratify_from_predictions.py --root outputs_seed_sweep_2019 --splits train test
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --split test

The balancing resample is seeded per run from metrics.json's random_state (falling
back to the seed_<n> directory name), matching what an in-pipeline run would use.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_stratified import stratified_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="sweep root holding <source>/seed_<n>/")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--bin-size", type=float, default=10.0)
    ap.add_argument("--max-height", type=float, default=60.0)
    ap.add_argument("--min-bin-n", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if a stratified block already exists")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"ERROR: no such directory: {root}")

    run_dirs = sorted(p.parent for p in root.glob("*/seed_*/metrics.json"))
    if not run_dirs:
        raise SystemExit(f"No */seed_*/metrics.json found under {root}")
    print(f"{len(run_dirs)} run(s) under {root}\n")

    written = skipped = missing = 0
    for d in run_dirs:
        source, seed_dir = d.parent.name, d.name
        mpath = d / "metrics.json"
        try:
            metrics = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  {source}/{seed_dir}: UNREADABLE metrics.json ({e})")
            missing += 1
            continue

        seed = metrics.get("random_state")
        if seed is None:
            try:
                seed = int(seed_dir.replace("seed_", ""))
            except ValueError:
                seed = 42

        changed = False
        for split in args.splits:
            block = metrics.get(split)
            # train metrics are nested one level deeper in the saved json
            if split == "train" and isinstance(block, dict) and "train" in block:
                block = block["train"]
            if not isinstance(block, dict):
                print(f"  {source}/{seed_dir} [{split}]: no metrics block (null split)")
                continue
            if block.get("stratified") is not None and not args.overwrite:
                skipped += 1
                continue

            ppath = d / f"predictions_{split}.parquet"
            if not ppath.is_file():
                print(f"  {source}/{seed_dir} [{split}]: MISSING {ppath.name}")
                missing += 1
                continue

            df = pd.read_parquet(ppath, columns=["y_true", "y_pred"])
            s = stratified_metrics(
                df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                bin_size=args.bin_size, max_height=args.max_height,
                min_bin_n=args.min_bin_n, balance_seed=seed,
            )
            block["stratified"] = s
            changed = True
            print(f"  {source}/{seed_dir} [{split}]: n={len(df):,}  "
                  f"macro_rmse={s['macro_rmse']:.3f}  "
                  f"balanced_rmse={s.get('balanced_rmse', float('nan')):.3f}  "
                  f"balanced_r2={s.get('balanced_r2', float('nan')):.3f}")

        if changed and not args.dry_run:
            mpath.write_text(json.dumps(metrics, indent=2))
            written += 1

    print(f"\nwritten={written}  already-present={skipped}  missing={missing}")
    if args.dry_run:
        print("DRY RUN — nothing was modified.")
    else:
        print(f"\nNow aggregate:\n"
              f"  python aggregate_seed_sweep.py --root {root} "
              f"--split {args.splits[-1]} --csv results.csv")


if __name__ == "__main__":
    main()
