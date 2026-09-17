#!/usr/bin/env python3
"""Build evaluate.py-compatible parquets for several methods on ONE common sample set.

Why this exists
---------------
evaluate.py does no masking of its own beyond --ignore_above: it trusts the parquet to
already hold the right samples. So the only way to compare a baseline (Potapov, ETH)
against the GFM encoders honestly is to feed every method a parquet built from the SAME
pixels. Comparing a baseline scored on its own coverage against encoders scored on
theirs is not a comparison -- whichever method drops the hardest pixels wins.

This script takes per-tile prediction rasters that are already on the GEDI grid (what
reconstruct_predictions_to_raster.py writes for the encoders, and what
potapov_baseline.py / eth_canopy_baseline.py write for the baselines), computes the
INTERSECTION of valid pixels across every method plus GEDI, and writes one parquet per
method over exactly those pixels.

Valid GEDI pixel (matches every embedding pipeline):
    (gedi != nodata) & (gedi > 0) & (gedi < 12000)   [cm]   ->  y_true = gedi / 100

Per-method validity: finite, not equal to the raster's own nodata, not equal to any
--method-declared nodata, and -- if the method sets dropzero -- not exactly 0.

Usage
-----
    python make_comparison_parquets.py \\
        --year 2019 \\
        --gedi-dir  /path/gedi_tiles_2019_healpix \\
        --tile-ids  /path/test_tile_ids.txt \\
        --out-dir   /path/comparison_2019 \\
        --method 'potapov|/path/potapov_tiles/healpix_{gid}_potapov.tif|0|dropzero' \\
        --method 'olmo|/path/olmo_pred_rasters/2019/healpix_{gid}_olmo.tif|-9999|keepzero' \\
        --method 'tessera|/path/tessera_pred_rasters/2019/healpix_{gid}_tessera.tif|-9999|keepzero'

Then, for each method:
    python evaluate.py --ignore_above 120 --parquet <out-dir>/<method>/predictions_test.parquet

Spec format: 'label|path_template|nodata|zeropolicy'
  path_template may use {gid} and {year}
  nodata       a float, or 'none'
  zeropolicy   'dropzero' (0 is invalid) or 'keepzero' (0 is a real prediction)

A note on dropzero
------------------
Dropping 0 removes pixels where the method predicted no canopy but GEDI measured some --
i.e. some of that method's worst misses -- which lowers its RMSE. Under --intersect those
pixels are removed for EVERY method, so the ranking stays internally consistent, but the
reported errors are over a subset that excludes those locations. Say so in the paper.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

GEDI_MAX_CM = 12000


def parse_method(spec):
    parts = spec.split("|")
    if len(parts) != 4:
        sys.exit(f"bad --method spec (need label|template|nodata|zeropolicy): {spec}")
    label, tmpl, nod, zero = parts
    nodata = None if nod.strip().lower() in ("none", "") else float(nod)
    if zero not in ("dropzero", "keepzero"):
        sys.exit(f"zeropolicy must be dropzero or keepzero, got {zero!r}")
    return {"label": label, "tmpl": tmpl, "nodata": nodata, "dropzero": zero == "dropzero"}


def gedi_valid(gedi_path):
    """(y_true_metres, flat_valid_mask, shape) for one tile, or None."""
    with rasterio.open(gedi_path) as s:
        g = s.read(1)
        nod = s.nodata if s.nodata is not None else -9999
    valid = (g != nod) & (g > 0) & (g < GEDI_MAX_CM)
    if not valid.any():
        return None
    return (g[valid] / 100.0).astype(np.float32), valid, g.shape


def method_values(path, shape, gedi_mask, method):
    """Values at the GEDI-valid pixels plus a per-pixel validity mask, or None."""
    with rasterio.open(path) as s:
        arr = s.read(1).astype(np.float64)
        rnod = s.nodata
    if arr.shape != shape:
        print(f"    {method['label']}: shape {arr.shape} != GEDI {shape} -- skipping tile",
              file=sys.stderr)
        return None
    vals = arr[gedi_mask]
    ok = np.isfinite(vals)
    for nd in (rnod, method["nodata"]):
        if nd is not None:
            ok &= vals != nd
    if method["dropzero"]:
        ok &= vals != 0
    return vals, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gedi-dir", required=True)
    ap.add_argument("--tile-ids", required=True,
                    help="one healpix id per line (e.g. test_tile_ids.txt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", action="append", required=True, metavar="SPEC",
                    help="'label|template|nodata|dropzero|keepzero'; repeat per method")
    ap.add_argument("--intersect", action=argparse.BooleanOptionalAction, default=True,
                    help="score every method on the pixels valid for ALL of them "
                         "(default: on). --no-intersect scores each on its own coverage.")
    ap.add_argument("--split-name", default="test",
                    help="written as predictions_<split-name>.parquet (default: test)")
    args = ap.parse_args()

    methods = [parse_method(s) for s in args.method]
    gedi_dir = Path(args.gedi_dir)
    out_dir = Path(args.out_dir)
    tiles = [t.strip() for t in Path(args.tile_ids).read_text().split() if t.strip()]

    print(f"year        : {args.year}")
    print(f"tiles       : {len(tiles)} from {args.tile_ids}")
    print(f"methods     : {', '.join(m['label'] for m in methods)}")
    print(f"intersect   : {args.intersect}")
    for m in methods:
        print(f"  {m['label']:10s} nodata={m['nodata']}  "
              f"zero={'dropped' if m['dropzero'] else 'kept'}")
    print()

    rows = {m["label"]: {"y_true": [], "y_pred": [], "tile": []} for m in methods}
    n_gedi = n_kept = 0
    skipped = []

    for tid in tiles:
        gpath = next((gedi_dir / f"healpix_{tid}_gedi{e}" for e in (".tif", ".tiff", "")
                      if (gedi_dir / f"healpix_{tid}_gedi{e}").exists()), None)
        if gpath is None:
            skipped.append((tid, "no GEDI raster"))
            continue
        g = gedi_valid(gpath)
        if g is None:
            skipped.append((tid, "no valid GEDI pixels"))
            continue
        y_true, gmask, shape = g

        per_method, missing = {}, []
        for m in methods:
            p = Path(m["tmpl"].format(gid=tid, year=args.year))
            if not p.exists():
                missing.append(m["label"])
                continue
            got = method_values(p, shape, gmask, m)
            if got is None:
                missing.append(m["label"])
                continue
            per_method[m["label"]] = got

        if missing:
            # A tile absent for one method cannot be part of a common sample set.
            skipped.append((tid, f"missing: {', '.join(missing)}"))
            continue

        n_gedi += len(y_true)
        if args.intersect:
            keep = np.ones(len(y_true), dtype=bool)
            for _, ok in per_method.values():
                keep &= ok
        else:
            keep = None

        for label, (vals, ok) in per_method.items():
            sel = keep if keep is not None else ok
            rows[label]["y_true"].append(y_true[sel])
            rows[label]["y_pred"].append(vals[sel].astype(np.float32))
            rows[label]["tile"].append(np.full(int(sel.sum()), int(tid), dtype=np.int32))
        if keep is not None:
            n_kept += int(keep.sum())

    if skipped:
        print(f"skipped {len(skipped)} tile(s):")
        for tid, why in skipped[:20]:
            print(f"  {tid}: {why}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
        print()

    print(f"{'method':<12} {'n':>10} {'RMSE':>8} {'bias':>8} {'MAE':>8}")
    print("-" * 50)
    for m in methods:
        d = rows[m["label"]]
        if not d["y_true"]:
            print(f"{m['label']:<12} {'0':>10}   no samples")
            continue
        y_true = np.concatenate(d["y_true"])
        y_pred = np.concatenate(d["y_pred"])
        tile = np.concatenate(d["tile"])
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        bias = float(np.mean(y_pred - y_true))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        print(f"{m['label']:<12} {len(y_true):>10,} {rmse:>8.3f} {bias:>+8.3f} {mae:>8.3f}")

        d_out = out_dir / m["label"]
        d_out.mkdir(parents=True, exist_ok=True)
        p = d_out / f"predictions_{args.split_name}.parquet"
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "tile": tile}).to_parquet(
            p, index=False, compression="snappy")
        print(f"{'':<12} -> {p}")

    if args.intersect and n_gedi:
        print(f"\ncommon sample set: {n_kept:,} of {n_gedi:,} GEDI pixels "
              f"({100 * n_kept / n_gedi:.1f}%) valid across all {len(methods)} method(s)")
    print("\nThe RMSE above is a convenience check; evaluate.py --ignore_above 120 on "
          "each parquet is the reportable version (it also applies the height cut).")


if __name__ == "__main__":
    main()
