#!/usr/bin/env python3
"""
Re-attach geospatial information to a coord-less predictions parquet and write
per-tile prediction RASTERS on the GEDI tile grid.

The prediction parquets produced by run_gedi_prediction_30m.py contain only
`y_true, y_pred` with no pixel coordinates. But the rows are in a fully
deterministic order:

  * tiles are concatenated in shapefile order for the split (skipping tiles with
    no valid GEDI pixels for that year), and
  * within a tile, samples are in `np.where(valid_mask)` (row-major) order, where
      valid_mask = (gedi != nodata) & (gedi > 0 [or >=0 if include_zeros])
                   & (gedi < 12000 cm)

So the flattening can be retraced: recompute `valid_mask` from the GEDI tifs,
which reproduces the exact (row, col) sequence, and slice `y_pred` back onto it.
This script auto-detects the GEDI *year* and the include_zeros setting by finding
the (year, include_zeros) whose recomputed `y_true` matches the parquet exactly,
then writes healpix_<id>_<model>.tif (single-band float32, y_pred at valid
pixels, nodata elsewhere) on each tile's GEDI grid.

Those rasters are on the GEDI grid, so they drop straight into
plot_track_comparison.py (add a DEFAULT_MODELS entry) or any pixel-aligned
analysis.

Example:
  python reconstruct_predictions_to_raster.py \
      --parquet .../aef/outputs_aef_rf_30m_float32_.../predictions_test.parquet \
      --model-name aef --out-dir .../aef_pred_rasters --tile 31330
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

GEDI_MAX_CM = 12000
SENTINEL = -9999.0


def tile_valid(gedi_path, include_zeros):
    """Return (rows, cols, y_m, profile) for one GEDI tile's valid pixels, in
    np.where row-major order — the exact order used when building (X, y)."""
    with rasterio.open(gedi_path) as s:
        g = s.read(1)
        nod = s.nodata if s.nodata is not None else -9999
        profile = s.profile.copy()
    m = (g != nod) & (g >= 0 if include_zeros else g > 0) & (g < GEDI_MAX_CM)
    rows, cols = np.where(m)
    return rows, cols, (g[rows, cols] / 100.0).astype(np.float32), profile


def find_gedi_dirs(gedi_base, years):
    out = {}
    for y in years:
        d = Path(gedi_base) / f"gedi_tiles_{y}_healpix"
        if d.exists():
            out[y] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--shapefile",
                    default="/Users/kdoerkse/Documents/ASU/JPL/data/shapefiles/healpix_shapefile.shp")
    ap.add_argument("--gedi-base", default="/Users/kdoerkse/Documents/ASU/JPL/data",
                    help="Directory containing gedi_tiles_<year>_healpix/ folders")
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022],
                    help="Candidate GEDI years to auto-detect from")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-name", required=True,
                    help="Used in output filename healpix_<id>_<model>.tif")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=None,
                    help="Only write this healpix id (still auto-detects year on full split)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yt = df["y_true"].values.astype(np.float32)
    yp = df["y_pred"].values.astype(np.float32)
    print(f"parquet: {len(df)} rows")

    gdf = gpd.read_file(args.shapefile)
    tile_ids = gdf[gdf["split"] == args.split]["healpix_id"].values
    print(f"{len(tile_ids)} '{args.split}' tiles in shapefile order")

    gedi_dirs = find_gedi_dirs(args.gedi_base, args.years)
    if not gedi_dirs:
        raise SystemExit(f"No gedi_tiles_<year>_healpix dirs under {args.gedi_base}")

    # --- auto-detect (year, include_zeros) by matching y_true exactly ---
    match = None
    for iz in (False, True):
        for year, gd in gedi_dirs.items():
            ys, per_tile = [], {}
            for hid in tile_ids:
                p = gd / f"healpix_{hid}_gedi.tif"
                if not p.exists():
                    continue
                _, _, y, _ = tile_valid(p, iz)
                if len(y) == 0:
                    continue
                per_tile[hid] = len(y)
                ys.append(y)
            yfull = np.concatenate(ys) if ys else np.array([], np.float32)
            if len(yfull) == len(yt) and np.allclose(yfull, yt, atol=1e-2):
                match = (year, iz, gd, per_tile)
                break
        if match:
            break

    if not match:
        raise SystemExit(
            "Could not match parquet y_true to any (year, include_zeros). "
            "Check --years / --shapefile / --split (the run may have skipped "
            "tiles or used a different shapefile order).")
    year, include_zeros, gd, per_tile = match
    print(f"MATCH: GEDI year={year}  include_zeros={include_zeros}  dir={gd}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- write per-tile prediction rasters, retracing the flattening ---
    offset = 0
    written = 0
    for hid in tile_ids:
        if hid not in per_tile:
            continue
        n = per_tile[hid]
        sl = slice(offset, offset + n)
        offset += n
        if args.tile is not None and hid != args.tile:
            continue

        rows, cols, y_recompute, profile = tile_valid(
            gd / f"healpix_{hid}_gedi.tif", include_zeros)
        assert len(rows) == n, f"tile {hid}: {len(rows)} != {n}"
        if not np.allclose(y_recompute, yt[sl], atol=1e-2):
            raise SystemExit(f"tile {hid}: y_true mismatch — reconstruction unsafe")

        arr = np.full((profile["height"], profile["width"]), SENTINEL, np.float32)
        arr[rows, cols] = yp[sl]
        profile.update(dtype="float32", count=1, nodata=SENTINEL, compress="lzw")
        out_p = out_dir / f"healpix_{hid}_{args.model_name}.tif"
        with rasterio.open(out_p, "w", **profile) as dst:
            dst.write(arr, 1)
        written += 1
        print(f"  wrote {out_p.name}: {n} px, "
              f"pred [{yp[sl].min():.1f}, {yp[sl].max():.1f}] m")

    print(f"\ndone: {written} raster(s) in {out_dir}  (year={year})")


if __name__ == "__main__":
    main()
