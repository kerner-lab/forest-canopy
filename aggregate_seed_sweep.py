#!/usr/bin/env python3
"""
Aggregate a multi-seed sweep into mean +/- std per embedding source.

Reads every  <root>/<source>/seed_<n>/metrics.json  written by
run_gedi_prediction_30m.py (via run_seed_sweep.sh) and reports, per source and
split, the mean and sample standard deviation (ddof=1) of RMSE, MAE and R^2.

Usage:
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --split test
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --csv summary.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("rmse", "mae", "r2")


def load_runs(root: Path):
    """{source: {seed: {split: {metric: value}}}} from the sweep tree."""
    runs = {}
    for mpath in sorted(root.glob("*/seed_*/metrics.json")):
        source = mpath.parent.parent.name
        try:
            d = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: unreadable {mpath}: {e}")
            continue
        # Prefer the recorded seed over the directory name — the directory is only
        # a convention, the json is what the run actually used.
        seed = d.get("random_state", mpath.parent.name.replace("seed_", ""))
        runs.setdefault(source, {})[seed] = d
    return runs


def summarize(values):
    """(mean, sample std) — std is 0.0 for a single run rather than an error."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="sweep root (OUT_ROOT from run_seed_sweep.sh)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"ERROR: no such directory: {root}")

    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No */seed_*/metrics.json found under {root}")

    rows = []
    for source in sorted(runs):
        per_seed = runs[source]
        cols = {}
        for m in METRICS:
            vals = []
            for seed in sorted(per_seed):
                block = per_seed[seed].get(args.split) or {}
                # train metrics are nested one level deeper in the saved json
                if args.split == "train" and "train" in block:
                    block = block["train"]
                if m in block:
                    vals.append(float(block[m]))
            cols[m] = summarize(vals)
            cols[f"n_{m}"] = len(vals)
        rows.append((source, len(per_seed), cols))

    n_label = f"{args.split} split"
    print(f"\n{n_label}: mean +/- std over seeds (ddof=1)\n")
    print(f"{'source':<10} {'n':>3}  {'RMSE (m)':>16} {'MAE (m)':>16} {'R2':>16}")
    print("-" * 68)
    for source, n_seeds, cols in rows:
        cells = []
        for m in METRICS:
            mean, std = cols[m]
            cells.append(f"{mean:.3f} +/- {std:.3f}")
        print(f"{source:<10} {n_seeds:>3}  {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    print()

    incomplete = [(s, n) for s, n, _ in rows if n != rows[0][1]]
    if incomplete:
        print("NOTE: seed counts differ across sources — "
              "means are not over the same number of runs: "
              + ", ".join(f"{s}={n}" for s, n in incomplete) + "\n")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "split", "n_seeds"]
                       + [c for m in METRICS for c in (f"{m}_mean", f"{m}_std")])
            for source, n_seeds, cols in rows:
                out = [source, args.split, n_seeds]
                for m in METRICS:
                    mean, std = cols[m]
                    out += [f"{mean:.6f}", f"{std:.6f}"]
                w.writerow(out)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
