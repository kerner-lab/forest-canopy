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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_stratified import mean_ci95  # noqa: E402

METRICS = ("rmse", "mae", "r2")

# Stratified summaries, read from metrics["<split>"]["stratified"].
# macro_*    = unweighted mean over height bins (every height class counts equally)
# balanced_* = ordinary metrics on a bin-balanced subsample
# macro_r2 is deliberately NOT included: within-bin R^2 is range-restricted and
# collapses toward 0 regardless of model quality. Use balanced_r2 instead.
STRAT_METRICS = ("macro_rmse", "macro_mae", "balanced_rmse", "balanced_r2")


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
    """(mean, sample std, 95% CI half-width of the mean).

    The CI uses t(0.975, n-1), not 1.96 — at the n=5 seed counts used here the
    normal approximation would understate the interval by roughly 40%.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0, 0.0
    mean, half, _n = mean_ci95(values)
    return mean, statistics.stdev(values), half


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
        for m in METRICS + STRAT_METRICS:
            vals = []
            for seed in sorted(per_seed):
                block = per_seed[seed].get(args.split) or {}
                # train metrics are nested one level deeper in the saved json
                if args.split == "train" and "train" in block:
                    block = block["train"]
                # stratified summaries live under the "stratified" sub-dict
                src = block.get("stratified") or {} if m in STRAT_METRICS else block
                if src.get(m) is not None:
                    vals.append(float(src[m]))
            cols[m] = summarize(vals)
            cols[f"n_{m}"] = len(vals)
        rows.append((source, len(per_seed), cols))

    n_label = f"{args.split} split"
    print(f"\n{n_label}: mean +/- 95% CI of the mean over seeds "
          f"(t(0.975, n-1))\n")
    print(f"{'source':<10} {'n':>3}  {'RMSE (m)':>16} {'MAE (m)':>16} {'R2':>16}")
    print("-" * 68)
    for source, n_seeds, cols in rows:
        cells = []
        for m in METRICS:
            mean, _std, half = cols[m]
            cells.append(f"{mean:.3f} +/- {half:.3f}")
        print(f"{source:<10} {n_seeds:>3}  {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    print()

    if any(cols[f"n_{m}"] for _s, _n, cols in rows for m in STRAT_METRICS):
        print(f"{n_label}: STRATIFIED over height bins\n")
        print(f"{'source':<10} {'n':>3}  {'macro RMSE':>16} {'macro MAE':>16} "
              f"{'bal. RMSE':>16} {'bal. R2':>16}")
        print("-" * 86)
        for source, n_seeds, cols in rows:
            cells = []
            for m in STRAT_METRICS:
                mean, _std, half = cols[m]
                cells.append("n/a" if cols[f"n_{m}"] == 0
                             else f"{mean:.3f} +/- {half:.3f}")
            print(f"{source:<10} {n_seeds:>3}  " + " ".join(f"{c:>16}" for c in cells))
        print("\n  macro_*    = unweighted mean over height bins; every height class counts\n"
              "               equally, so tall-canopy error is no longer drowned out by the\n"
              "               short-canopy pixels that dominate the overall number.\n"
              "  balanced_* = metrics on a bin-balanced subsample. Use balanced_r2, not the\n"
              "               mean of per-bin R^2 — within a single bin the true heights\n"
              "               barely vary, so per-bin R^2 is range-restricted and collapses\n"
              "               toward 0 even for a good model.\n"
              "  Expect macro RMSE > overall RMSE. That gap IS the result: it measures how\n"
              "  much of the headline score comes from the most common height class.\n")

    incomplete = [(s, n) for s, n, _ in rows if n != rows[0][1]]
    if incomplete:
        print("NOTE: seed counts differ across sources — "
              "means are not over the same number of runs: "
              + ", ".join(f"{s}={n}" for s, n in incomplete) + "\n")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "split", "n_seeds"]
                       + [c for m in METRICS + STRAT_METRICS
                          for c in (f"{m}_mean", f"{m}_std", f"{m}_ci95")])
            for source, n_seeds, cols in rows:
                out = [source, args.split, n_seeds]
                for m in METRICS + STRAT_METRICS:
                    mean, std, half = cols[m]
                    out += [f"{mean:.6f}", f"{std:.6f}", f"{half:.6f}"]
                w.writerow(out)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
