#!/usr/bin/env python3
"""
Quantify how many GEDI reference pixels fall per independent Prithvi embedding.

Prithvi-EO-2.0 is run at its native 30 m scale with a 16-pixel patch size, so the
encoder emits ONE token per 480 m cell. Those tokens are then reprojected onto the
30 m GEDI grid with bilinear resampling (unified_embedding_pipeline.py:377), so
neighbouring GEDI pixels do not receive byte-identical vectors — they receive
interpolated blends. The embedding field nevertheless carries only one independent
value per 480 m cell; the 30 m variation is interpolation, not new information.

What this reports, per tile and in total:
  n_valid      GEDI pixels used for training/eval (the rows of the (X, y) table)
  n_cells      distinct 480 m cells those pixels fall in
  ratio        n_valid / n_cells — reference pixels per independent embedding

The ratio is the number to quote. Its theoretical maximum is (480/30)^2 = 256, but
GEDI is track-sparse rather than wall-to-wall, so the realised value is far lower
and must be measured rather than assumed.

Usage:
    python quantify_prithvi_support.py \\
        --shapefile /ocean/.../healpix_shapefile.shp \\
        --gedi-dir /ocean/.../2019/gedi --split test

    # restrict to the master tile list used for the comparison
    python quantify_prithvi_support.py ... --tile-list /ocean/.../master_tile_ids.txt
"""

import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio

GEDI_MAX_CM = 12000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--gedi-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile-list", default=None,
                    help="restrict to these healpix ids (one per line)")
    ap.add_argument("--token-m", type=float, default=480.0,
                    help="Prithvi token footprint in m (patch 16 x 30 m; default 480)")
    ap.add_argument("--pixel-m", type=float, default=30.0,
                    help="GEDI grid resolution in m (default 30)")
    ap.add_argument("--gedi-nodata", type=float, default=-9999.0)
    ap.add_argument("--include-zeros", action="store_true",
                    help="match a run that kept GEDI height == 0")
    ap.add_argument("--per-tile", action="store_true", help="print every tile")
    args = ap.parse_args()

    factor = int(round(args.token_m / args.pixel_m))
    if factor < 1:
        raise SystemExit("ERROR: token-m must be >= pixel-m")

    gdf = gpd.read_file(args.shapefile)
    tiles = gdf[gdf["split"] == args.split]["healpix_id"].astype(int).tolist()
    if args.tile_list:
        keep = {int(t) for t in Path(args.tile_list).read_text().split() if t.strip()}
        tiles = [t for t in tiles if t in keep]
    if not tiles:
        raise SystemExit(f"No '{args.split}' tiles (after --tile-list filtering)")

    gedi_dir = Path(args.gedi_dir)
    tot_valid = tot_cells = 0
    per_cell_counts = []
    n_missing = 0

    print(f"{len(tiles)} '{args.split}' tiles; token = {args.token_m:.0f} m "
          f"= {factor}x{factor} GEDI pixels (max {factor**2} px/token)\n")

    for tid in sorted(tiles):
        path = next((gedi_dir / f"healpix_{tid}_gedi{e}"
                     for e in (".tif", ".tiff")
                     if (gedi_dir / f"healpix_{tid}_gedi{e}").exists()), None)
        if path is None:
            n_missing += 1
            continue
        with rasterio.open(path) as src:
            g = src.read(1).astype(np.float32)

        valid = (g != args.gedi_nodata) & (g < GEDI_MAX_CM)
        valid &= (g >= 0) if args.include_zeros else (g > 0)
        rows, cols = np.where(valid)
        if rows.size == 0:
            continue

        # Which 480 m cell each valid pixel belongs to. Cell origin is taken from
        # the tile grid origin; a half-cell offset relative to the true token
        # lattice changes individual assignments but not the ratio materially.
        cell = (rows // factor).astype(np.int64) * (1 << 32) + (cols // factor)
        _uniq, counts = np.unique(cell, return_counts=True)

        tot_valid += rows.size
        tot_cells += counts.size
        per_cell_counts.append(counts)
        if args.per_tile:
            print(f"  healpix_{tid:<6} n_valid={rows.size:>9,}  "
                  f"cells={counts.size:>8,}  ratio={rows.size / counts.size:6.2f}")

    if tot_cells == 0:
        raise SystemExit("No valid GEDI pixels found.")

    allc = np.concatenate(per_cell_counts)
    print(f"\n{'TOTAL':<14} valid GEDI pixels : {tot_valid:>12,}")
    print(f"{'':<14} distinct {args.token_m:.0f} m cells: {tot_cells:>12,}")
    print(f"{'':<14} pixels per cell    : mean {allc.mean():.2f}   "
          f"median {np.median(allc):.0f}   max {allc.max()}   "
          f"(theoretical max {factor**2})")
    print(f"\n  Effective reduction: {tot_valid:,} reference pixels are supported by "
          f"{tot_cells:,}\n  independent embeddings — a factor of "
          f"{tot_valid / tot_cells:.1f}.")
    share = 100.0 * allc[allc > 1].sum() / tot_valid
    print(f"  {share:.1f}% of reference pixels lie in a cell containing at least one "
          f"other pixel.")
    if n_missing:
        print(f"\n  ({n_missing} tile(s) had no GEDI raster and were skipped.)")


if __name__ == "__main__":
    main()
