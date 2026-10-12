#!/usr/bin/env python3
"""
Verify that two or more baseline prediction parquets cover the SAME tiles and the
SAME pixels, so their metrics are comparable.

Why this is needed
------------------
echosat_baseline.py, eth_canopy_baseline.py and potapov_baseline.py all apply the
identical GEDI validity rule (!= nodata, > 0, < 12000 cm) and all emit
healpix_id/y_true/y_pred. But each then drops its OWN unusable pixels, and each
names its variants differently:

  potapov  predictions_test.parquet          all valid GEDI px  (0 m == nodata, KEPT)
           predictions_test_nozero.parquet   baseline 0 m dropped
  eth      predictions_test.parquet          masked (water) px dropped, real 0 m kept
           predictions_test_maskfill0.parquet  full px set, masked filled with 0
  echosat  predictions_test.parquet          nodata px dropped, real 0 m kept
           predictions_test_maskfill0.parquet  full px set, nodata filled with 0

So the files called "predictions_test.parquet" are NOT the same pixel set across
baselines. The full-set variants are:  potapov predictions_test.parquet,
eth predictions_test_maskfill0.parquet, echosat predictions_test_maskfill0.parquet.

How alignment is established
----------------------------
Identical y_true (the GEDI label) at the same position within a healpix_id means the
same pixel -- the same argument count_measurements.py uses for the embedding caches.
Rows within a tile are in np.where(valid) order over the same GEDI raster, so when no
rows have been dropped, row k of tile T is the same pixel in every baseline.

A row-count mismatch therefore means the pixel sets differ and the metrics are not
comparable without intersecting first.

Usage
-----
  # are the full-set variants pixel-identical?
  python check_baseline_alignment.py \
      --baseline potapov=/path/potapov_baseline_2019/predictions_test.parquet \
      --baseline eth=/path/eth_baseline_2019/predictions_test_maskfill0.parquet \
      --baseline echosat=/path/echosat/test_2019/predictions_test_maskfill0.parquet

  # write an intersected set (common tiles, common pixels) for a fair comparison
  ... --out-dir /path/aligned_baselines_2019
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(spec):
    if "=" not in spec:
        raise SystemExit(f"--baseline needs NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{name}: not found: {p}")
    df = pd.read_parquet(p)
    missing = {"healpix_id", "y_true", "y_pred"} - set(df.columns)
    if missing:
        raise SystemExit(f"{name}: parquet missing columns {sorted(missing)}")
    return name, p, df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="append", required=True, metavar="NAME=PARQUET",
                    help="repeatable; NAME=path/to/predictions_*.parquet")
    ap.add_argument("--reference", default=None,
                    help="baseline everything is compared against (default: the first)")
    ap.add_argument("--atol", type=float, default=1e-4,
                    help="tolerance when comparing y_true (default 1e-4 m)")
    ap.add_argument("--out-dir", default=None,
                    help="write intersected parquets (common tiles AND common pixels) here")
    ap.add_argument("--max-report", type=int, default=15)
    args = ap.parse_args()

    loaded = [load(s) for s in args.baseline]
    if len(loaded) < 2:
        raise SystemExit("give at least two --baseline entries")
    names = [n for n, _, _ in loaded]
    ref = args.reference or names[0]
    if ref not in names:
        raise SystemExit(f"--reference {ref!r} is not one of {names}")
    dfs = {n: d for n, _, d in loaded}

    print("=" * 74)
    print("BASELINE ALIGNMENT CHECK")
    print("=" * 74)
    for n, p, d in loaded:
        print(f"  {n:10s} {len(d):>10,} rows  {d.healpix_id.nunique():>3d} tiles  {p}")
    print(f"\n  reference: {ref}")

    # ---- tile sets ----------------------------------------------------------
    tilesets = {n: set(d.healpix_id.unique()) for n, d in dfs.items()}
    common_tiles = set.intersection(*tilesets.values())
    union_tiles = set.union(*tilesets.values())
    print(f"\nTILES  common={len(common_tiles)}  union={len(union_tiles)}")
    problems = []
    for n in names:
        extra = tilesets[n] - common_tiles
        if extra:
            problems.append(f"{n} has {len(extra)} tile(s) the others lack")
            print(f"  {n}: {len(extra)} tile(s) not in all baselines -> "
                  f"{sorted(extra)[:args.max_report]}")
    if len(common_tiles) == len(union_tiles):
        print("  identical tile sets")

    # ---- per-tile row counts ------------------------------------------------
    print("\nPIXELS per tile (row counts must match to be pixel-aligned)")
    counts = pd.DataFrame({n: dfs[n].groupby("healpix_id").size() for n in names})
    counts = counts.loc[sorted(common_tiles)]
    same = (counts.nunique(axis=1) == 1)
    print(f"  tiles where all baselines agree on row count: {int(same.sum())}/{len(counts)}")
    if not same.all():
        bad = counts[~same]
        problems.append(f"{len(bad)} tile(s) disagree on pixel count")
        print(f"  first {min(args.max_report, len(bad))} disagreements:")
        print(bad.head(args.max_report).to_string())

    # ---- y_true agreement on the tiles that DO match ------------------------
    print("\nGEDI LABEL AGREEMENT (identical y_true == identical pixels)")
    aligned_tiles = [t for t in counts.index[same]]
    mismatched = []
    for n in names:
        if n == ref:
            continue
        nbad = 0
        worst = 0.0
        for t in aligned_tiles:
            a = dfs[ref].loc[dfs[ref].healpix_id == t, "y_true"].to_numpy()
            b = dfs[n].loc[dfs[n].healpix_id == t, "y_true"].to_numpy()
            d = np.abs(a - b)
            if d.size and d.max() > args.atol:
                nbad += 1
                worst = max(worst, float(d.max()))
        if nbad:
            mismatched.append(n)
            problems.append(f"{n} y_true differs from {ref} on {nbad} tile(s)")
            print(f"  {n:10s} differs on {nbad}/{len(aligned_tiles)} tiles "
                  f"(max |dy_true| = {worst:.4f} m) -> DIFFERENT PIXELS")
        else:
            print(f"  {n:10s} identical to {ref} on all {len(aligned_tiles)} "
                  f"row-count-matching tiles")

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    if not problems:
        n_px = int(counts[ref].sum())
        print(f"ALIGNED: {len(common_tiles)} tiles, {n_px:,} pixels, identical across "
              f"all {len(names)} baselines.")
        print("Metrics from these files are directly comparable.")
    else:
        print("NOT ALIGNED:")
        for p in problems:
            print(f"  - {p}")
        print("\nMetrics from these files are NOT directly comparable as-is.")
        print("Either re-run with a shared --tile-list, use each baseline's FULL-SET")
        print("variant (see the module docstring), or pass --out-dir to write the")
        print("intersection.")
    print("=" * 74)

    # ---- optional intersection ---------------------------------------------
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Only tiles where every baseline agrees on the row count can be intersected
        # positionally; anything else would silently pair up different pixels.
        keep = aligned_tiles
        if mismatched:
            print(f"\nNOT writing an intersection: {mismatched} disagree on y_true, so "
                  f"rows cannot be paired positionally. Fix the pixel sets first.")
        else:
            for n in names:
                sub = dfs[n][dfs[n].healpix_id.isin(keep)].reset_index(drop=True)
                path = out / f"predictions_test_{n}_aligned.parquet"
                sub.to_parquet(path, index=False, compression="snappy")
                print(f"wrote {path}  ({len(sub):,} rows, {len(keep)} tiles)")
            print(f"\nEvaluate these with the same command you use for a model run, e.g.")
            print(f"  python results-plotting/evaluate.py --parquet "
                  f"{out / f'predictions_test_{names[0]}_aligned.parquet'}")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
