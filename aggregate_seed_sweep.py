#!/usr/bin/env python3
"""
Aggregate a multi-seed sweep into mean +/- std per embedding source.

Reads every  <root>/<source>/seed_<n>/metrics.json  written by
run_gedi_prediction_30m.py (via run_seed_sweep.sh) and reports, per source and
split, the mean and sample standard deviation (ddof=1) of RMSE, MAE, R^2 and bias.

Bias is mean(pred - true) in metres: positive = the model over-predicts height.
Runs written before bias was recorded in metrics.json get it computed from the
predictions_<split>.parquet saved next to their metrics.json (see fill_bias()).

Seed-averaged predictions (--out-dir)
-------------------------------------
With --out-dir, also writes per source

    <out-dir>/<source>/predictions_<split>.parquet   y_true, y_pred (mean over seeds),
                                                     y_pred_std (across seeds)

in the same row order as the per-seed files, so it drops straight into
data_processing/reconstruct_predictions_to_raster.py --parquet ... and from there
plot_track_comparison.py. Averaging is row-by-row, which is only valid when every
seed predicted the same pixels in the same order; the test rows depend on
subset_seed / stratified_sample_seed, not random_state, so a random_state sweep
lines up. That is CHECKED, not assumed: y_true must be identical across seeds or the
source is skipped.

Note the averaged prediction is an ensemble: its RMSE is typically LOWER than the
mean of the per-seed RMSEs in the table. Its own metrics are recorded in
<out-dir>/manifest.json; quote the table's mean +/- CI for per-model performance.

Usage:
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --split test
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --csv summary.csv

    # table CSV + seed-averaged parquets in one folder, ready to scp
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2019 --out-dir seed_avg_2019

    # add baselines as extra rows (single run each, so no CI)
    python aggregate_seed_sweep.py --root outputs_seed_sweep_2020 \
        --baseline echosat=/ocean/.../echosat/test_2020/metrics.json \
        --baseline eth=/ocean/.../eth_baseline_2020/metrics.json

Baselines (--baseline NAME=PATH[:BLOCK])
---------------------------------------
Baseline metrics.json files (echosat_baseline.py / potapov_baseline.py /
eth_canopy_baseline.py) hold several result blocks, e.g. ECHOSAT 'echosat_valid'
and 'masked_filled_zero'. Without :BLOCK the block with the LARGEST n_samples is
used: that is the one keeping every valid GEDI pixel (nodata filled, not dropped),
i.e. the same pixel set the GFMs are scored on. A warning is printed if its n
differs from the GFMs' test n.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_stratified import mean_ci95, stratified_metrics  # noqa: E402

METRICS = ("rmse", "mae", "r2", "bias")

# Stratified summaries, read from metrics["<split>"]["stratified"].
# macro_*    = unweighted mean over height bins (every height class counts equally)
# balanced_* = ordinary metrics on a bin-balanced subsample
# macro_r2 is deliberately NOT included: within-bin R^2 is range-restricted and
# collapses toward 0 regardless of model quality. Use balanced_r2 instead.
# There is no macro bias either: per-bin biases are signed and cancel (short canopy
# over-predicted, tall under-predicted), so their mean reads near 0 for a model
# that is badly biased in every bin. balanced_bias has no such cancellation.
STRAT_METRICS = ("macro_rmse", "macro_mae", "balanced_rmse", "balanced_r2",
                 "balanced_bias")

# Tolerance for accepting a balanced_bias recomputed from the parquet: the
# recomputed balanced_rmse must match the stored one this closely (metres). The
# parquet holds float32 predictions, so an identical subsample still differs by
# ~1e-6 m; a different subsample differs by centimetres or more.
BALANCED_MATCH_ATOL = 1e-3


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
        runs.setdefault(source, {})[seed] = (d, mpath.parent)
    return runs


def split_block(d, split):
    """The metrics dict for `split` (train metrics may be nested one level deeper)."""
    block = d.get(split) or {}
    if split == "train" and "train" in block:
        block = block["train"]
    return block


def fill_bias(block, seed, seed_dir: Path, split, pq: Path = None):
    """Add bias / balanced_bias to `block` from predictions_<split>.parquet if absent.

    Returns a short note when something could not be filled, else None.

    bias is exact (the parquet is the same y_true/y_pred the metrics came from).
    balanced_bias needs the same bin-balanced subsample the run drew, so it is
    recomputed with the run's seed and bins and accepted ONLY if the recomputed
    balanced_rmse reproduces the stored one — proof the subsample is identical
    (min_bin_n is not recorded in metrics.json; this check catches a non-default).
    """
    strat = block.get("stratified") or {}
    need_bias = block.get("bias") is None
    need_bal = strat.get("balanced_rmse") is not None and strat.get("balanced_bias") is None
    if not (need_bias or need_bal):
        return None
    pq = pq or seed_dir / f"predictions_{split}.parquet"
    if not pq.is_file():
        return f"no {pq.name}, bias unavailable"
    df = pd.read_parquet(pq, columns=["y_true", "y_pred"])
    yt = df["y_true"].to_numpy(np.float64)
    yp = df["y_pred"].to_numpy(np.float64)
    if need_bias:
        block["bias"] = float(np.mean(yp - yt))
    if need_bal:
        bins = strat.get("bins") or []
        try:
            balance_seed = int(seed)
        except (TypeError, ValueError):
            return "seed not an int, balanced_bias unavailable"
        re = stratified_metrics(yt, yp, bin_size=strat["bin_size"],
                                max_height=bins[-1]["hi"] if bins else 60.0,
                                balance_seed=balance_seed)
        if (re.get("balanced_rmse") is not None
                and abs(re["balanced_rmse"] - strat["balanced_rmse"]) <= BALANCED_MATCH_ATOL):
            strat["balanced_bias"] = re["balanced_bias"]
        else:
            return "could not reproduce the balanced subsample, balanced_bias unavailable"
    return None


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


def fmt_cell(cols, m):
    """'mean +/- ci' for metric m, signed for bias, 'n/a' when no seed reported it."""
    if cols[f"n_{m}"] == 0:
        return "n/a"
    mean, _std, half = cols[m]
    sign = "+" if m.endswith("bias") else ""
    if cols[f"n_{m}"] == 1:          # single run (a baseline): no CI to show
        return f"{mean:{sign}.3f}"
    return f"{mean:{sign}.3f} +/- {half:.3f}"


def load_baseline(name, spec, split):
    """(row, note) for one --baseline NAME=PATH[:BLOCK]; row shaped like a model row."""
    path, _, block_name = spec.partition(":")
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.json"
    d = json.loads(path.read_text())
    blocks = {k: v for k, v in d.items()
              if isinstance(v, dict) and "rmse" in v and "stratified" in v}
    if not blocks:
        raise SystemExit(f"ERROR: --baseline {name}: no result blocks in {path}")
    if block_name:
        if block_name not in blocks:
            raise SystemExit(f"ERROR: --baseline {name}: no block '{block_name}' in "
                             f"{path} (have: {', '.join(blocks)})")
    else:
        block_name = max(blocks, key=lambda k: blocks[k].get("n_samples", 0))
    block = blocks[block_name]
    # Baselines predate balanced_bias. Recover it from whichever of the baseline's
    # parquets reproduces this block's balanced_rmse (fill_bias verifies that).
    note = None
    if (block["stratified"].get("balanced_rmse") is not None
            and block["stratified"].get("balanced_bias") is None):
        note = "no parquet reproduces this block, balanced_bias unavailable"
        for pq in sorted(path.parent.glob(f"predictions_{split}*.parquet")):
            if fill_bias(block, d.get("metric_balance_seed", 42), path.parent,
                         split, pq=pq) is None:
                note = None
                break
    cols = {}
    for m in METRICS + STRAT_METRICS:
        src = block["stratified"] if m in STRAT_METRICS else block
        v = src.get(m)
        cols[m] = summarize([float(v)] if v is not None else [])
        cols[f"n_{m}"] = int(v is not None)
    return (name, block_name, int(block.get("n_samples", 0)), cols), note


def average_predictions(per_seed, split):
    """Row-wise mean/std of y_pred over seeds. Returns (df, seeds, note).

    df is None (with the reason in note) when a seed lacks its parquet or the seeds
    do not share y_true row-for-row — averaging misaligned rows would silently mix
    different pixels.
    """
    seeds, y_true, preds = [], None, []
    for seed in sorted(per_seed):
        _d, seed_dir = per_seed[seed]
        pq = seed_dir / f"predictions_{split}.parquet"
        if not pq.is_file():
            return None, seeds, f"seed {seed} has no {pq.name}"
        df = pd.read_parquet(pq, columns=["y_true", "y_pred"])
        yt = df["y_true"].to_numpy(np.float32)
        if y_true is None:
            y_true = yt
        elif len(yt) != len(y_true) or not np.array_equal(yt, y_true):
            n = f"{len(yt):,} vs {len(y_true):,} rows" if len(yt) != len(y_true) \
                else "same length, different y_true"
            return None, seeds, (f"seed {seed} rows do not match seed {seeds[0]} ({n}) — "
                                 f"seeds sampled different pixels, cannot average")
        preds.append(df["y_pred"].to_numpy(np.float64))
        seeds.append(seed)
    P = np.vstack(preds)                                   # (n_seeds, N)
    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": P.mean(axis=0).astype(np.float32),
        "y_pred_std": (P.std(axis=0, ddof=1) if len(seeds) > 1
                       else np.zeros(P.shape[1])).astype(np.float32),
    })
    return out, seeds, None


def write_summary_csv(path, rows, split, base_rows=()):
    """Model rows, then baseline rows (kind=baseline; std/ci left blank)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "n_seeds"]
                   + [c for m in METRICS + STRAT_METRICS
                      for c in (f"{m}_mean", f"{m}_std", f"{m}_ci95")]
                   + ["kind", "baseline_block"])
        for source, n_seeds, cols in rows:
            out = [source, split, n_seeds]
            for m in METRICS + STRAT_METRICS:
                mean, std, half = cols[m]
                out += [f"{mean:.6f}", f"{std:.6f}", f"{half:.6f}"]
            w.writerow(out + ["model", ""])
        for name, block_name, _n, cols in base_rows:
            out = [name, split, 1]
            for m in METRICS + STRAT_METRICS:
                mean, _std, _half = cols[m]
                out += [f"{mean:.6f}" if cols[f"n_{m}"] else "", "", ""]
            w.writerow(out + ["baseline", block_name])
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="sweep root (OUT_ROOT from run_seed_sweep.sh)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--csv", default=None, help="also write the table here")
    ap.add_argument("--out-dir", default=None, metavar="DIR",
                    help="write the table CSV (summary_<split>.csv, unless --csv is "
                         "given) and per-source seed-averaged predictions_<split>.parquet "
                         "+ manifest.json here — one folder to scp")
    ap.add_argument("--baseline", action="append", default=[], metavar="NAME=PATH[:BLOCK]",
                    help="add a baseline row from its metrics.json (or its directory). "
                         "BLOCK defaults to the block with the most pixels (every valid "
                         "GEDI pixel, nodata filled). Repeat per baseline.")
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
        blocks = {}
        for seed in sorted(per_seed):
            d, seed_dir = per_seed[seed]
            block = split_block(d, args.split)
            if block:
                note = fill_bias(block, seed, seed_dir, args.split)
                if note:
                    print(f"NOTE: {source} seed {seed}: {note}")
            blocks[seed] = block
        cols = {}
        for m in METRICS + STRAT_METRICS:
            vals = []
            for seed in sorted(blocks):
                block = blocks[seed]
                # stratified summaries live under the "stratified" sub-dict
                src = block.get("stratified") or {} if m in STRAT_METRICS else block
                if src.get(m) is not None:
                    vals.append(float(src[m]))
            cols[m] = summarize(vals)
            cols[f"n_{m}"] = len(vals)
        rows.append((source, len(per_seed), cols))

    # GFM test pixel count, to check each baseline is scored on the same pixels.
    gfm_n = {int(split_block(d, args.split).get("n_samples", -1))
             for per_seed in runs.values() for d, _dir in per_seed.values()
             if split_block(d, args.split).get("n_samples") is not None}
    base_rows = []
    for spec in args.baseline:
        name, sep, rest = spec.partition("=")
        if not sep:
            raise SystemExit(f"ERROR: --baseline expects NAME=PATH[:BLOCK], got {spec!r}")
        row, note = load_baseline(name, rest, args.split)
        base_rows.append(row)
        print(f"baseline {name}: block '{row[1]}' (n={row[2]:,})")
        if note:
            print(f"NOTE: baseline {name}: {note}")
        if gfm_n and row[2] not in gfm_n:
            print(f"WARNING: baseline {name} n={row[2]:,} but the GFMs' {args.split} n is "
                  f"{', '.join(f'{n:,}' for n in sorted(gfm_n))} — NOT the same pixels "
                  f"(check tile list, GEDI year, and the block).")

    n_label = f"{args.split} split"
    print(f"\n{n_label}: mean +/- 95% CI of the mean over seeds "
          f"(t(0.975, n-1))\n")
    print(f"{'source':<10} {'n':>3}  {'RMSE (m)':>16} {'MAE (m)':>16} {'R2':>16} "
          f"{'bias (m)':>17}")
    print("-" * 86)
    for source, n_seeds, cols in rows:
        cells = [fmt_cell(cols, m) for m in METRICS]
        print(f"{source:<10} {n_seeds:>3}  {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} "
              f"{cells[3]:>17}")
    if base_rows:
        print("-" * 86)
        for name, _b, _n, cols in base_rows:
            cells = [fmt_cell(cols, m) for m in METRICS]
            print(f"{name:<10} {'-':>3}  {cells[0]:>16} {cells[1]:>16} {cells[2]:>16} "
                  f"{cells[3]:>17}")
    print("\n  bias = mean(pred - true); positive = over-predicts canopy height.\n")

    if any(cols[f"n_{m}"] for _s, _n, cols in rows for m in STRAT_METRICS):
        print(f"{n_label}: STRATIFIED over height bins\n")
        print(f"{'source':<10} {'n':>3}  {'macro RMSE':>16} {'macro MAE':>16} "
              f"{'bal. RMSE':>16} {'bal. R2':>16} {'bal. bias':>17}")
        print("-" * 104)
        for source, n_seeds, cols in rows:
            cells = [fmt_cell(cols, m) for m in STRAT_METRICS]
            print(f"{source:<10} {n_seeds:>3}  "
                  + " ".join(f"{c:>16}" for c in cells[:-1]) + f" {cells[-1]:>17}")
        if base_rows:
            print("-" * 104)
            for name, _b, _n, cols in base_rows:
                cells = [fmt_cell(cols, m) for m in STRAT_METRICS]
                print(f"{name:<10} {'-':>3}  "
                      + " ".join(f"{c:>16}" for c in cells[:-1]) + f" {cells[-1]:>17}")
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

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv or (out_dir / f"summary_{args.split}.csv" if out_dir else None)
    if csv_path:
        write_summary_csv(csv_path, rows, args.split, base_rows)

    if out_dir:
        print(f"\nSeed-averaged predictions ({args.split}) -> {out_dir}")
        manifest = {"root": str(root.resolve()), "split": args.split, "sources": {}}
        for source in sorted(runs):
            df, seeds, note = average_predictions(runs[source], args.split)
            if df is None:
                print(f"  {source:<10} SKIPPED — {note}")
                manifest["sources"][source] = {"skipped": note}
                continue
            dst = out_dir / source / f"predictions_{args.split}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False, compression="snappy")
            yt = df["y_true"].to_numpy(np.float64)
            yp = df["y_pred"].to_numpy(np.float64)
            ens = {"rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                   "mae": float(mean_absolute_error(yt, yp)),
                   "r2": float(r2_score(yt, yp)),
                   "bias": float(np.mean(yp - yt))}
            manifest["sources"][source] = {
                "seeds": [str(s) for s in seeds], "n_rows": int(len(df)),
                "parquet": str(dst.relative_to(out_dir)),
                "ensemble_metrics": ens,
                "median_pred_std_m": float(np.median(df["y_pred_std"])),
            }
            print(f"  {source:<10} {len(seeds)} seeds, {len(df):,} rows  "
                  f"ensemble RMSE={ens['rmse']:.3f} bias={ens['bias']:+.3f}  "
                  f"median seed-to-seed std={np.median(df['y_pred_std']):.3f} m  -> {dst}")
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  manifest -> {out_dir / 'manifest.json'}")
        print("\n  Ensemble metrics are of the AVERAGED prediction and are usually better\n"
              "  than the per-seed mean in the table — quote the table for performance.")


if __name__ == "__main__":
    main()
