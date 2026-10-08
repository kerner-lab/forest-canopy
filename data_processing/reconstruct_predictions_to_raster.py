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


def align_tile(y_full, yt, offset, atol=1e-2, n_expected=None):
    """Order-preserving alignment of a tile's parquet rows onto its GEDI pixels.

    Some pipelines drop GEDI samples that fall on uncovered embedding pixels --
    tessera_processing.py does this for all-zero (no TESSERA coverage) rows. The kept
    rows are a boolean mask of the np.where() order, so ORDER IS PRESERVED and the
    surviving pixels can be recovered by walking the tile's full y sequence and
    consuming a parquet row whenever the values agree.

    Returns (keep_idx, n_ties). `keep_idx` indexes into the tile's full (rows, cols).
    `n_ties` counts matches whose neighbour has the same height, i.e. places where the
    alignment could equally have chosen the adjacent pixel: the VALUE is right either
    way, the pixel may be off by one position. Reported so the ambiguity is visible.
    """
    keep, ties = [], 0
    i, j = 0, offset
    while i < len(y_full) and j < len(yt):
        if abs(float(y_full[i]) - float(yt[j])) <= atol:
            keep.append(i)
            if i + 1 < len(y_full) and abs(float(y_full[i + 1]) - float(y_full[i])) <= atol:
                ties += 1
            j += 1
        i += 1
    if n_expected is not None and len(keep) != n_expected:
        return None, ties
    return np.asarray(keep, dtype=np.int64), ties


def cache_counts(cache_dir, tile_ids, model_name):
    """Per-tile row counts read from the embedding caches, if available.

    Pins the tile boundaries exactly, so alignment cannot over-consume across a tile
    edge when the last height of one tile equals the first of the next.
    """
    if not cache_dir:
        return {}
    d = Path(cache_dir)
    counts = {}
    for hid in tile_ids:
        for name in (f"healpix_{hid}_float32.npz", f"healpix_{hid}_{model_name}.npz"):
            p = d / name
            if p.exists():
                try:
                    with np.load(p) as z:
                        counts[hid] = int(z["y"].shape[0])
                except Exception:
                    pass
                break
    return counts


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
    ap.add_argument("--align", action="store_true",
                    help="Allow per-tile subsequence alignment when the cache dropped "
                         "GEDI samples on uncovered embedding pixels (tessera does "
                         "this). Slower and slightly ambiguous where adjacent pixels "
                         "share a height; only used if the exact matchers fail.")
    ap.add_argument("--cache-dir", default=None,
                    help="Embedding cache dir (healpix_<id>_float32.npz). With --align, "
                         "pins each tile's row count exactly so the alignment cannot "
                         "run past a tile boundary.")
    ap.add_argument("--tile-list", default=None, metavar="PATH",
                    help="healpix ids (one per line) the run was restricted to, i.e. the "
                         "--tile-list given to run_gedi_prediction_30m.py. Limits the "
                         "retrace to those tiles; needed with --align when no complete "
                         "--cache-dir is available, or alignment can spill rows into "
                         "tiles the run never saw.")
    ap.add_argument("--tile", type=int, default=None,
                    help="Only write this healpix id (still auto-detects year on full split)")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    yt = df["y_true"].values.astype(np.float32)
    yp = df["y_pred"].values.astype(np.float32)
    print(f"parquet: {len(df)} rows")

    gdf = gpd.read_file(args.shapefile)
    tile_ids = gdf[gdf["split"] == args.split]["healpix_id"].values
    if args.tile_list:
        # Same filter as run_gedi_prediction_30m.py: keep shapefile order.
        keep = {int(t) for t in Path(args.tile_list).read_text().split()
                if t.strip() and not t.strip().startswith("#")}
        tile_ids = np.array([t for t in tile_ids if int(t) in keep])
    print(f"{len(tile_ids)} '{args.split}' tiles in shapefile order")

    gedi_dirs = find_gedi_dirs(args.gedi_base, args.years)
    if not gedi_dirs:
        raise SystemExit(f"No gedi_tiles_<year>_healpix dirs under {args.gedi_base}")

    # --- auto-detect (year, include_zeros) by matching y_true exactly ---
    #
    # Two passes. STRICT expects every split tile to be present, which is the case
    # when the run had a cache for all of them. If a run skipped tiles (a missing
    # embedding cache -> see missing_tiles.json in the run dir), the totals cannot
    # line up, so SUBSET walks the tiles in shapefile order and consumes a tile only
    # when its y block matches the next rows of the parquet; a tile that does not
    # match was skipped by the run. Order is preserved either way, which is what
    # makes the retrace valid.
    def strict(iz, gd):
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
            return per_tile, []
        return None, None

    def subset(iz, gd):
        per_tile, skipped, offset = {}, [], 0
        for hid in tile_ids:
            p = gd / f"healpix_{hid}_gedi.tif"
            if not p.exists():
                continue
            _, _, y, _ = tile_valid(p, iz)
            n = len(y)
            if n == 0:
                continue
            if offset + n <= len(yt) and np.allclose(yt[offset:offset + n], y, atol=1e-2):
                per_tile[hid] = n
                offset += n
            else:
                skipped.append(hid)
        if offset == len(yt) and per_tile:
            return per_tile, skipped
        return None, None

    def aligned(iz, gd):
        """Per-tile subsequence alignment, for caches that dropped uncovered pixels."""
        counts = cache_counts(args.cache_dir, tile_ids, args.model_name)
        per_tile, keep_map, offset, total_ties = {}, {}, 0, 0
        for hid in tile_ids:
            p = gd / f"healpix_{hid}_gedi.tif"
            if not p.exists():
                continue
            _, _, y_full, _ = tile_valid(p, iz)
            if len(y_full) == 0:
                continue
            if counts and hid not in counts:
                # A cache dir was supplied and this tile has no cache: the run never
                # saw it, so it contributes NO parquet rows. Aligning it anyway lets
                # the greedy match steal rows from the next tile and silently shifts
                # everything downstream.
                continue
            keep, ties = align_tile(y_full, yt, offset, n_expected=counts.get(hid))
            if keep is None or len(keep) == 0:
                continue
            per_tile[hid] = len(keep)
            keep_map[hid] = keep
            offset += len(keep)
            total_ties += ties
        if offset == len(yt) and per_tile:
            aligned.keep_map = keep_map
            aligned.ties = total_ties
            return per_tile, []
        return None, None

    finders = (strict, subset, aligned) if args.align else (strict, subset)
    match = None
    for finder in finders:
        for iz in (False, True):
            for year, gd in gedi_dirs.items():
                per_tile, skipped = finder(iz, gd)
                if per_tile is not None:
                    match = (year, iz, gd, per_tile, skipped, finder.__name__)
                    break
            if match:
                break
        if match:
            break

    if not match:
        raise SystemExit(
            "Could not match parquet y_true to any (year, include_zeros). If this "
            "cache drops GEDI samples on uncovered embedding pixels (tessera does), "
            "re-run with --align (and --cache-dir for an exact per-tile row count). "
            "Otherwise check --years / --shapefile / --split, and that the GEDI "
            "rasters are the ones the run actually used.")
    year, include_zeros, gd, per_tile, skipped, how = match
    print(f"MATCH ({how}): GEDI year={year}  include_zeros={include_zeros}  dir={gd}")
    print(f"  {len(per_tile)} tile(s) matched, {sum(per_tile.values()):,} rows")
    if skipped:
        print(f"  {len(skipped)} tile(s) absent from the parquet (skipped by the run): "
              f"{', '.join(str(h) for h in skipped)}")
    if how == "aligned":
        ties = getattr(aligned, "ties", 0)
        print(f"  ALIGNED mode: pixel positions recovered by subsequence matching. "
              f"{ties:,} tie(s) where a neighbouring pixel shares the same height "
              f"({100 * ties / max(1, len(yt)):.2f}% — value correct, position may be "
              f"off by one).")

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
        keep = getattr(aligned, "keep_map", {}).get(hid) if how == "aligned" else None
        if keep is not None:
            rows, cols, y_recompute = rows[keep], cols[keep], y_recompute[keep]
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
