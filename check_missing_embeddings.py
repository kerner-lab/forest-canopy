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

import numpy as np

# NOTE: geopandas is imported inside main(), not here. run_prithvi_embed.sh imports
# validate_npz directly to check its resume guard, and that must stay a fast, dependency-
# light import (geopandas costs seconds and is only needed to read the shapefile).


def validate_npz(path: Path) -> str:
    """Return an error string if `path` is not a fully readable .npz, else ""."""
    # np.load is LAZY for .npz — it only opens the zip directory, so a truncated
    # member does not raise until the array is actually read. Touch every array.
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:                      # BadZipFile, EOFError, zlib error, ...
        return f"unreadable ({type(exc).__name__}: {exc})"
    if not arrays:
        return "no arrays"
    # X/y row mismatch means the write was cut between members even if the zip parsed.
    X, y = arrays.get("X"), arrays.get("y")
    if X is not None and y is not None and len(X) != len(y):
        return f"X/y length mismatch ({len(X)} vs {len(y)})"
    return ""


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
    parser.add_argument("--validate", action="store_true",
                        help="Open each present .npz and read every array, instead of just "
                             "checking that the path exists. Corrupt files are reported "
                             "separately and included in --out. Use this after a run that "
                             "hit a disk quota / full filesystem: np.savez leaves a "
                             "TRUNCATED but existing .npz, which a plain existence check "
                             "(and the sbatch resume guard) will silently accept forever.")
    args = parser.parse_args()

    import geopandas as gpd

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
    present, missing, corrupt = [], [], []
    for tid in tile_ids:
        fname = f"healpix_{tid}_{args.embedding_format}{suffix}.npz"
        path = emb_dir / fname
        if not path.exists():
            missing.append(tid)
            continue
        if args.validate:
            err = validate_npz(path)
            if err:
                corrupt.append((tid, err))
                continue
        present.append(tid)

    scope = f"splits={args.splits}" if args.splits else "all splits"
    print(f"Shapefile : {args.shapefile}  ({scope})")
    print(f"Directory : {emb_dir}")
    print(f"Filename  : healpix_<ID>_{args.embedding_format}{suffix}.npz")
    print(f"Mode      : {'existence + validate' if args.validate else 'existence only'}")
    print(f"Present   : {len(present)}/{len(tile_ids)}")
    print(f"Missing   : {len(missing)}")
    if missing:
        preview = missing[:50]
        print(f"  {preview}{' ...' if len(missing) > 50 else ''}")
    if args.validate:
        print(f"Corrupt   : {len(corrupt)}")
        for tid, err in corrupt[:50]:
            print(f"  {tid}: {err}")
        if len(corrupt) > 50:
            print(f"  ... and {len(corrupt) - 50} more")

    # A corrupt file must be re-run just like a missing one, so both go to --out.
    rerun = sorted(missing + [tid for tid, _ in corrupt])
    if args.out and rerun:
        Path(args.out).write_text("\n".join(str(t) for t in rerun) + "\n")
        print(f"Wrote {len(rerun)} IDs to re-run -> {args.out}")


if __name__ == "__main__":
    main()
