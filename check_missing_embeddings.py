#!/usr/bin/env python3
"""
Report which healpix tiles are missing their Presto embedding .npz in a directory.

Reads valid healpix_ids from the shapefile (optionally filtered to one or more
splits) and checks whether healpix_<ID>_float32.npz exists in --dir.

Usage:
    python check_missing_embeddings.py \\
        --shapefile /path/healpix.shp \\
        --dir /ocean/.../data/presto/2021

    # restrict to certain splits, e.g. just train
    python check_missing_embeddings.py --shapefile ... --dir ... --splits train

    # check the --keep-zeros cache instead (healpix_<ID>_float32_withzeros.npz)
    python check_missing_embeddings.py --shapefile ... --dir ... --keep-zeros

    # write the missing IDs to a file (one per line) for a re-run
    python check_missing_embeddings.py --shapefile ... --dir ... --out missing.txt
"""

import argparse
from pathlib import Path

import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapefile", required=True,
                        help="Healpix shapefile (.shp) with healpix_id (and split) columns")
    parser.add_argument("--dir", required=True,
                        help="Directory containing healpix_<ID>_float32.npz files")
    parser.add_argument("--splits", nargs="+", default=None, metavar="SPLIT",
                        help="Only check tiles in these splits (default: all tiles)")
    parser.add_argument("--embedding-format", default="float32",
                        help="Format suffix in the filename (default: float32)")
    parser.add_argument("--keep-zeros", action="store_true",
                        help="Check the _withzeros cache (healpix_<ID>_<fmt>_withzeros.npz)")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the missing healpix_ids (one per line)")
    args = parser.parse_args()

    emb_dir = Path(args.dir)
    if not emb_dir.is_dir():
        parser.error(f"--dir is not a directory: {emb_dir}")

    gdf = gpd.read_file(args.shapefile)
    if "healpix_id" not in gdf.columns:
        parser.error(f"'healpix_id' column not found in {args.shapefile}")

    if args.splits:
        if "split" not in gdf.columns:
            parser.error("--splits given but shapefile has no 'split' column")
        gdf = gdf[gdf["split"].isin(args.splits)]

    tile_ids = [int(t) for t in gdf["healpix_id"].values]

    suffix = "_withzeros" if args.keep_zeros else ""
    present, missing = [], []
    for tid in tile_ids:
        fname = f"healpix_{tid}_{args.embedding_format}{suffix}.npz"
        if (emb_dir / fname).exists():
            present.append(tid)
        else:
            missing.append(tid)

    scope = f"splits={args.splits}" if args.splits else "all splits"
    print(f"Shapefile : {args.shapefile}  ({scope})")
    print(f"Directory : {emb_dir}")
    print(f"Filename  : healpix_<ID>_{args.embedding_format}{suffix}.npz")
    print(f"Present   : {len(present)}/{len(tile_ids)}")
    print(f"Missing   : {len(missing)}")
    if missing:
        preview = missing[:50]
        print(f"  {preview}{' ...' if len(missing) > 50 else ''}")

    if args.out and missing:
        Path(args.out).write_text("\n".join(str(t) for t in missing) + "\n")
        print(f"Wrote {len(missing)} missing IDs -> {args.out}")


if __name__ == "__main__":
    main()
