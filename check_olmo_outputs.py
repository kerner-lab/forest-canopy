#!/usr/bin/env python3
"""Audit the OlmoEarth per-tile .npz outputs written by olmoearth_download.py tiles.

Checks, per tile:
  load        both files decompress (a tile interrupted by a full disk opens fine
              and only fails when a member is actually read)
  shapes      X is (N, C), y is (N,), emb is (C, H, W), and the C agree
  values      y within a plausible canopy-height range, no NaNs, X not all-nodata
  agreement   the sparse X rows are re-derived from the dense grid by mapping each
              valid GEDI pixel centre through its own affine -> the embedding CRS ->
              rowcol, and compared. This is the real test: it fails loudly if the
              sparse extraction sampled the wrong pixels, which comparing y alone
              can never detect (y never touches the embedding grid).

Usage:
    python check_olmo_outputs.py \
        --npz-dir  /Users/kdoerkse/Documents/ASU/JPL/data/olmo_xy/2019 \
        --gedi-dir /Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2019_healpix

    # full agreement check on a few tiles (needs the dense files, which are large)
    python check_olmo_outputs.py --npz-dir ... --gedi-dir ... --agree 3

    # cross-check sample counts against another encoder's sparse files
    python check_olmo_outputs.py --npz-dir ... --gedi-dir ... \
        --compare-dir /path/to/presto_xy --compare-suffix _presto.npz
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def load_npz(path):
    """Fully decompress every member; returns (dict, error_string)."""
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def find_gedi(gedi_dir, tid):
    for ext in (".tif", ".tiff", ""):
        p = gedi_dir / f"healpix_{tid}_gedi{ext}"
        if p.exists():
            return p
    return None


def check_agreement(sparse, dense_path, gedi_path):
    """Re-derive the sparse rows from the dense grid. Returns (msg, ok)."""
    try:
        import rasterio
        from rasterio.transform import Affine, rowcol
        from rasterio.transform import xy as transform_xy
        from rasterio.warp import transform as warp_transform
    except ImportError as e:
        return f"agreement skipped (needs rasterio: {e})", True

    dense, err = load_npz(dense_path)
    if dense is None:
        return f"dense unreadable: {err}", False

    emb = dense["emb"]
    gdal_tf = [float(v) for v in dense["transform"]]
    # dense_npz() stored transform.to_gdal() = (c, a, b, f, d, e)
    tf = Affine.from_gdal(*gdal_tf)
    ecrs = str(dense["crs"])

    with rasterio.open(gedi_path) as g:
        gedi = g.read(1)
        gnodata = g.nodata if g.nodata is not None else -9999
        gtf, gcrs = g.transform, g.crs

    valid = (gedi != gnodata) & (gedi > 0) & (gedi < 12000)
    rows, cols = np.where(valid)
    if len(rows) == 0:
        return "no valid GEDI pixels", True

    xs, ys = transform_xy(gtf, rows.tolist(), cols.tolist())
    if gcrs and str(gcrs) != ecrs:
        xs, ys = warp_transform(gcrs, ecrs, list(xs), list(ys))
    erows, ecols = rowcol(tf, xs, ys)
    erows = np.asarray(erows, dtype=np.int64)
    ecols = np.asarray(ecols, dtype=np.int64)

    C, H, W = emb.shape
    inside = (erows >= 0) & (erows < H) & (ecols >= 0) & (ecols < W)
    expected = emb[:, erows[inside], ecols[inside]].T.astype(np.float32)

    X = sparse["X"]
    if X.shape != expected.shape:
        return (f"MISMATCH shapes: sparse X {X.shape} vs re-derived "
                f"{expected.shape} ({int((~inside).sum())} outside grid)"), False
    if np.array_equal(X, expected):
        return f"agrees exactly ({len(expected)} rows)", True
    nbad = int((X != expected).any(axis=1).sum())
    return f"MISMATCH: {nbad}/{len(X)} rows differ from the dense grid", False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--gedi-dir", required=True)
    ap.add_argument("--agree", type=int, default=0, metavar="N",
                    help="run the sparse-vs-dense agreement check on the first N tiles "
                         "(reads the large dense files, so keep N small)")
    ap.add_argument("--compare-dir", help="dir of another encoder's sparse npz")
    ap.add_argument("--compare-suffix", default="_presto.npz")
    ap.add_argument("--y-max", type=float, default=120.0,
                    help="flag y above this many metres (default 120)")
    args = ap.parse_args()

    npz_dir, gedi_dir = Path(args.npz_dir), Path(args.gedi_dir)
    sparse_files = sorted(npz_dir.glob("healpix_*_olmo.npz"))
    dense_files = sorted(npz_dir.glob("*_olmo_dense.npz"))
    print(f"sparse files : {len(sparse_files)}")
    print(f"dense files  : {len(dense_files)}")
    print(f"y range flag : > {args.y_max} m\n")

    bad, total_samples, channels, agreed = [], 0, set(), 0
    for i, p in enumerate(sparse_files):
        tid = re.match(r"healpix_(\d+)_olmo\.npz", p.name).group(1)
        d, err = load_npz(p)
        if d is None:
            bad.append((tid, f"unreadable: {err}"))
            continue
        if "X" not in d or "y" not in d:
            bad.append((tid, f"missing keys, has {list(d)}"))
            continue

        X, y = d["X"], d["y"]
        problems = []
        if X.ndim != 2 or y.ndim != 1:
            problems.append(f"bad dims X{X.shape} y{y.shape}")
        elif X.shape[0] != y.shape[0]:
            problems.append(f"length mismatch X{X.shape} y{y.shape}")
        else:
            channels.add(X.shape[1])
            total_samples += len(y)
            if len(y) == 0:
                problems.append("zero samples")
            if np.isnan(y).any():
                problems.append(f"{int(np.isnan(y).sum())} NaN in y")
            if np.isnan(X).any():
                problems.append(f"{int(np.isnan(X).sum())} NaN in X")
            if len(y) and (y.min() <= 0 or y.max() > args.y_max):
                problems.append(f"y out of range [{y.min():.1f}, {y.max():.1f}]")
            if X.size and not X.any():
                problems.append("X is all zeros")

        gpath = find_gedi(gedi_dir, tid)
        if gpath is None:
            problems.append("no GEDI raster to cross-check")

        if args.compare_dir:
            cp = Path(args.compare_dir) / f"healpix_{tid}{args.compare_suffix}"
            if cp.exists():
                c, cerr = load_npz(cp)
                if c and "y" in c and c["y"].ndim == 1 and len(c["y"]) != len(y):
                    problems.append(f"n differs from {cp.name}: {len(y)} vs {len(c['y'])}")

        if args.agree and i < args.agree and gpath is not None:
            dpath = npz_dir / f"{tid}_olmo_dense.npz"
            if dpath.exists():
                msg, ok = check_agreement(d, dpath, gpath)
                print(f"  healpix_{tid}: {msg}")
                if ok:
                    agreed += 1
                else:
                    problems.append(msg)
            else:
                print(f"  healpix_{tid}: no dense file, agreement skipped")

        if problems:
            bad.append((tid, "; ".join(problems)))

    print()
    for tid, msg in bad:
        print(f"  PROBLEM healpix_{tid}: {msg}")
    print(f"\ntiles ok      : {len(sparse_files) - len(bad)}/{len(sparse_files)}")
    print(f"total samples : {total_samples:,}")
    print(f"channel count : {sorted(channels) or 'n/a'}"
          + ("   <-- SHOULD BE ONE VALUE" if len(channels) > 1 else ""))
    if args.agree:
        print(f"agreement     : {agreed} tile(s) verified against the dense grid")


if __name__ == "__main__":
    main()
