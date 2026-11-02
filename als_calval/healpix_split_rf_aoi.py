#!/usr/bin/env python3
"""
Single-AOI ALS experiment: split ONE ALS AOI into train/val/test by HEALPix cell, train an
RF on the train cells, score it on val and test.

Every valid pixel (ALS RH98 finite AND every embedding band finite) is assigned to the
HEALPix cell (NESTED ordering, --nside) containing its centre. Whole cells go to one split,
so train and test pixels are never interleaved (dense ALS pixels are strongly spatially
autocorrelated; a random pixel split would score near-duplicates). Cells are visited in
random order and each goes to the split furthest below its target PIXEL fraction
(--fractions), so the splits hit their pixel shares while staying spatially random.

    nside   cell size   ~cells over NorthernCA (~105 x 83 km)
    512     ~12.7 km    ~55
    1024    ~6.4 km     ~210    (default)
    2048    ~3.2 km     ~850

Optional --buffer-px drops TRAIN pixels within N px of a val/test pixel (guards against
leakage across cell edges through the embeddings' spatial context and resampling).

Validation set: always scored. With --tune, a small RF grid (min_samples_leaf x
max_features) is fit on train, the best by val balanced RMSE is kept, and only that model
is scored on test.

Outputs in --out-dir (prefix <als stem>_<emb tag>_nside<N>):
  _split.tif / _split.png   split per pixel on the ALS grid (0 none, 1 train, 2 val, 3 test)
  _cells.csv                healpix_id, split, n_px, height stats (before --buffer-px)
  _cells.shp                the same cells as EPSG:4326 polygons (for QGIS)
  _metrics.csv / .json      val + test, overall + stratified (quote balanced_rmse/_r2)
  _test_scatter.png         test predictions vs ALS
  _rf.joblib                model bundle (model + meta incl. cell lists)
  _predictions.npz          row, col, y, pred, split for every evaluated pixel

Usage (Bridges-2):
    python als_calval/healpix_split_rf_aoi.py \\
        --als-tif /ocean/projects/agr260002p/kdoerkse/data/als/CA_NorthernCA_1_B22_4326.tif \\
        --emb-tif /ocean/projects/agr260002p/kdoerkse/data/aef/als_aois/CA_NorthernCA_1_B22_4326_aef_2022.tif \\
        --out-dir /ocean/projects/agr260002p/kdoerkse/results/als_healpix/NorthernCA_aef

Requires healpy (pip install healpy). The embedding tif must be on the ALS grid
(check_wall_to_wall_tifs.py).
"""

import argparse
import contextlib
import itertools
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
import geopandas as gpd
from scipy.ndimage import binary_dilation
from shapely.geometry import Polygon

try:
    import healpy as hp
except ImportError:
    sys.exit("ERROR: healpy is not installed: pip install healpy")

import matplotlib
matplotlib.use("Agg")                    # headless (compute nodes); before pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SPLITS = ("train", "val", "test")
CODE = {"train": 1, "val": 2, "test": 3}


def scan_valid(als_tif, emb_tif, args):
    """Flat indices + ALS heights of pixels valid in ALS (band, range) and ALL emb bands.
    With emb_tif None (--split-only), validity is from the ALS alone."""
    idx, ys = [], []
    emb_ctx = rasterio.open(emb_tif) if emb_tif else contextlib.nullcontext()
    with rasterio.open(als_tif) as a, emb_ctx as e:
        mism = C._grid_mismatches(e, a, 1e-3) if e is not None else []
        if mism:
            sys.exit("ERROR: embedding is not on the ALS grid:\n  - " + "\n  - ".join(mism)
                     + f"\nRegenerate it with --like {als_tif}")
        if not 1 <= args.als_band <= a.count:
            sys.exit(f"ERROR: --als-band {args.als_band} but the ALS tif has {a.count} bands")
        d = a.descriptions[args.als_band - 1]
        if args.als_band_name and d and d.lower() != args.als_band_name.lower():
            sys.exit(f"ERROR: ALS band {args.als_band} is {d!r}, not {args.als_band_name!r}")
        H, W = a.height, a.width
        for r0 in range(0, H, args.block_rows):
            bh = min(args.block_rows, H - r0)
            win = Window(0, r0, W, bh)
            y = a.read(args.als_band, window=win, masked=True).astype(np.float32)
            y = y.filled(np.nan).ravel()
            v = np.isfinite(y)
            if args.als_min is not None:
                v &= ~(y < args.als_min)
            if args.als_max is not None:
                v &= ~(y > args.als_max)
            if not v.any():
                continue
            if e is not None:
                v &= np.isfinite(e.read(window=win)).all(axis=0).ravel()
            k = np.flatnonzero(v)
            idx.append(k + r0 * W)
            ys.append(y[k])
        info = {"H": H, "W": W, "crs": a.crs, "transform": a.transform,
                "n_features": e.count if e is not None else None}
    if not idx:
        sys.exit("ERROR: no pixel is valid in both ALS and the embedding")
    idx, ys = np.concatenate(idx), np.concatenate(ys)
    n_neg = int((ys < 0).sum())
    if n_neg:   # an undeclared nodata value (e.g. -9999) would show up here
        print(f"WARNING: {n_neg:,} valid ALS pixels are < 0 m (min {ys.min():.2f}); pass "
              f"--als-min 0 if these are nodata", file=sys.stderr)
    return idx, ys, info


def healpix_ids(flat_idx, info, nside):
    W, T = info["W"], info["transform"]
    rows, cols = flat_idx // W, flat_idx % W
    x = T.c + (cols + 0.5) * T.a + (rows + 0.5) * T.b
    y = T.f + (cols + 0.5) * T.d + (rows + 0.5) * T.e
    crs = CRS.from_user_input(info["crs"])
    if not (crs.is_geographic and crs.to_epsg() == 4326):
        x, y = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return hp.ang2pix(nside, x, y, nest=True, lonlat=True).astype(np.int64)


def assign_cells(cells, counts, fractions, rng):
    """Random cell order; each cell to the split with the largest pixel deficit."""
    target = np.asarray(fractions) / np.sum(fractions) * counts.sum()
    got = np.zeros(3)
    split = {}
    for i in rng.permutation(len(cells)):
        s = int(np.argmax(target - got))
        split[int(cells[i])] = SPLITS[s]
        got[s] += counts[i]
    return split


def write_cells_shapefile(rows, nside, path, step=8):
    """One EPSG:4326 polygon per HEALPix cell (edges densified with `step` points per
    side, so the curved cell edges draw correctly), attributes from the cells table."""
    geoms = []
    for r in rows:
        vec = hp.boundaries(nside, r["healpix_id"], step=step, nest=True)   # (3, 4*step)
        lon, lat = hp.vec2ang(vec.T, lonlat=True)
        # healpy gives lon in [0, 360); wrap to [-180, 180) relative to the first vertex
        # so a cell never splits across the wrap.
        lon0 = ((lon[0] + 180) % 360) - 180
        lon = lon0 + ((lon - lon[0] + 180) % 360) - 180
        geoms.append(Polygon(zip(lon, lat)))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(path)
    print(f"Shapefile: {path}  ({len(gdf)} cells)")


def gather(tif, flat_idx, W, block_rows):
    """Features (n, C) at SORTED flat indices, strip by strip."""
    with rasterio.open(tif) as e:
        out = np.empty((len(flat_idx), e.count), np.float32)
        rows = flat_idx // W
        for r0 in range(0, e.height, block_rows):
            bh = min(block_rows, e.height - r0)
            lo, hi = np.searchsorted(rows, [r0, r0 + bh])
            if lo == hi:
                continue
            strip = e.read(window=Window(0, r0, W, bh)).astype(np.float32)
            out[lo:hi] = strip.reshape(e.count, -1)[:, flat_idx[lo:hi] - r0 * W].T
    return out


def plot_split(label, out_png, title):
    step = max(1, max(label.shape) // 2000)          # keep the png light
    cols = ["#ffffff", "#4c72b0", "#dd8452", "#c44e52"]
    fig, ax = plt.subplots(figsize=(10, 10 * label.shape[0] / label.shape[1] + 0.8))
    ax.imshow(label[::step, ::step], cmap=ListedColormap(cols), vmin=0, vmax=3,
              interpolation="nearest")
    ax.legend(handles=[Patch(color=cols[CODE[s]], label=s) for s in SPLITS],
              loc="lower right")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Figure: {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-tif", required=True)
    ap.add_argument("--emb-tif", default=None, help="wall-to-wall embedding on the ALS grid")
    ap.add_argument("--split-only", action="store_true",
                    help="only build the HEALPix split (tif, png, cells csv/shp) and stop; "
                         "--emb-tif optional (without it, validity is from the ALS alone)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None,
                    help="name for outputs (default: encoder from the emb filename, "
                         "e.g. aef / olmo / tessera)")

    g = ap.add_argument_group("ALS target")
    g.add_argument("--als-band", type=int, default=2, help="1-based RH98 band (default 2)")
    g.add_argument("--als-band-name", default="rh98", help="expected band name ('' to skip)")
    g.add_argument("--als-min", type=float, default=None, help="drop ALS heights below (m)")
    g.add_argument("--als-max", type=float, default=None, help="drop ALS heights above (m)")

    g = ap.add_argument_group("HEALPix split")
    g.add_argument("--nside", type=int, default=1024, help="HEALPix nside (power of 2)")
    g.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="target PIXEL fractions")
    g.add_argument("--min-cell-px", type=int, default=100,
                   help="drop cells with fewer valid pixels (edge slivers)")
    g.add_argument("--buffer-px", type=int, default=0,
                   help="drop train pixels within N px of a val/test pixel")

    g = ap.add_argument_group("sampling")
    g.add_argument("--max-train-px", type=int, default=1_000_000,
                   help="random train pixels used to fit (0 = all)")
    g.add_argument("--max-eval-px", type=int, default=0,
                   help="random val/test pixels scored per split (0 = all)")

    C.add_rf_args(ap)
    ap.add_argument("--tune", action="store_true",
                    help="pick min_samples_leaf x max_features on val (grid below)")
    ap.add_argument("--grid-min-samples-leaf", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--grid-max-features", nargs="+", default=["sqrt", "0.33"])
    C.add_metric_args(ap)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-rows", type=int, default=128)
    args = ap.parse_args()

    if args.nside < 1 or args.nside & (args.nside - 1):
        sys.exit("ERROR: --nside must be a power of 2")

    if not args.emb_tif and not args.split_only:
        sys.exit("ERROR: --emb-tif is required unless --split-only")
    als_tif = Path(args.als_tif)
    emb_tif = Path(args.emb_tif) if args.emb_tif else None
    m = re.search(r"_(aef|olmo|tessera)_\d{4}$", emb_tif.stem) if emb_tif else None
    tag = args.tag or (m.group(1) if m else emb_tif.stem if emb_tif else "split")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"{als_tif.stem}_{tag}_nside{args.nside}"
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # 1. Valid pixels + HEALPix cell per pixel.
    print(f"Scanning {als_tif.name}" + (f" + {emb_tif.name}" if emb_tif else " (ALS only)") + " ...")
    idx, y, info = scan_valid(als_tif, emb_tif, args)
    H, W = info["H"], info["W"]
    print(f"  {len(idx):,} valid px ({info['n_features'] or 'no'} features)  "
          f"ALS mean {y.mean():.2f} m  ({time.time() - t0:.0f}s)")
    hpx = healpix_ids(idx, info, args.nside)

    # 2. Split cells.
    cells, inv, counts = np.unique(hpx, return_inverse=True, return_counts=True)
    keep = counts >= args.min_cell_px
    print(f"  {len(cells)} HEALPix cells (nside {args.nside}); {int((~keep).sum())} with "
          f"< {args.min_cell_px} px dropped ({int(counts[~keep].sum()):,} px)")
    split_of = assign_cells(cells[keep], counts[keep], args.fractions, rng)
    cell_split = np.array([CODE.get(split_of.get(int(c)), 0) for c in cells], np.uint8)
    lab = cell_split[inv]                                   # per valid pixel

    label = np.zeros((H, W), np.uint8)
    label.ravel()[idx] = lab
    n_buf = 0
    if args.buffer_px:
        # 3x3 structure: "within N px" in every direction, diagonals included.
        near = binary_dilation(label >= 2, structure=np.ones((3, 3), bool),
                               iterations=args.buffer_px)
        drop = (label == 1) & near
        n_buf = int(drop.sum())
        label[drop] = 0
        lab = label.ravel()[idx]
        print(f"  buffer {args.buffer_px} px: dropped {n_buf:,} train px next to val/test")

    empty = [s for s in SPLITS if not (lab == CODE[s]).any()]
    if empty:
        sys.exit(f"ERROR: no pixels in {', '.join(empty)} — only {int(keep.sum())} usable "
                 f"cells at nside {args.nside}. Use a larger --nside (smaller cells) or a "
                 f"lower --min-cell-px / --buffer-px.")

    rows = []
    for s in SPLITS:
        m = lab == CODE[s]
        cs = cells[(cell_split == CODE[s])]
        print(f"  {s:5s} {len(cs):4d} cells  {m.sum():>11,} px ({100 * m.mean():5.1f}%)  "
              f"ALS mean {y[m].mean():6.2f}  p95 {np.percentile(y[m], 95):6.2f} m")
    for i, c in enumerate(cells):
        m = inv == i
        rows.append({"healpix_id": int(c), "split": split_of.get(int(c), "dropped"),
                     "n_px": int(counts[i]), "als_mean": float(y[m].mean()),
                     "als_p95": float(np.percentile(y[m], 95))})
    C.write_csv(f"{prefix}_cells.csv", rows)
    write_cells_shapefile(rows, args.nside, f"{prefix}_cells.shp")

    prof = dict(driver="GTiff", width=W, height=H, count=1, dtype="uint8", nodata=0,
                crs=info["crs"], transform=info["transform"], compress="lzw",
                tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(f"{prefix}_split.tif", "w", **prof) as d:
        d.write(label, 1)
        d.write_colormap(1, {0: (0, 0, 0, 0), 1: (76, 114, 176, 255),
                             2: (221, 132, 82, 255), 3: (196, 78, 82, 255)})
    plot_split(label, f"{prefix}_split.png",
               f"{als_tif.stem} — HEALPix nside {args.nside} split "
               f"(train/val/test = {' / '.join(f'{100 * (lab == CODE[s]).mean():.0f}%' for s in SPLITS)})")
    del label
    if args.split_only:
        print(f"\nSplit only — done in {time.time() - t0:.0f}s")
        return

    # 3. Sample + gather features.
    sel = {}
    for s, cap in (("train", args.max_train_px), ("val", args.max_eval_px),
                   ("test", args.max_eval_px)):
        k = np.flatnonzero(lab == CODE[s])
        if cap and len(k) > cap:
            k = np.sort(rng.choice(k, cap, replace=False))
        sel[s] = k
    allk = np.sort(np.concatenate(list(sel.values())))
    print(f"\nGathering features for {len(allk):,} px "
          + "(" + ", ".join(f"{s} {len(k):,}" for s, k in sel.items()) + ") ...")
    X_all = gather(emb_tif, idx[allk], W, args.block_rows)
    pos = {s: np.searchsorted(allk, k) for s, k in sel.items()}
    X = {s: X_all[p] for s, p in pos.items()}
    Y = {s: y[k] for s, k in sel.items()}
    del X_all

    # 4. RF (optionally tuned on val).
    base = C.rf_params_from_args(args)
    if args.tune:
        grid, rf, best_score = [], None, np.inf
        for msl, mf in itertools.product(args.grid_min_samples_leaf, args.grid_max_features):
            p = base | {"min_samples_leaf": msl,
                        "max_features": mf if mf in ("sqrt", "log2") else float(mf)}
            cand = C.fit_rf(X["train"], Y["train"], p, n_jobs=args.n_jobs,
                            random_state=args.seed, label="RF (tune)")
            r = C.evaluate(Y["val"], cand.predict(X["val"]), f"val {p}", args, args.seed)
            bal = r["stratified"].get("balanced_rmse")
            score, metric = (bal, "balanced_rmse") if bal is not None else \
                (r["overall"]["rmse"], "rmse")
            grid.append({"params": p, "val_score": score, "val_metric": metric})
            if score < best_score:            # keep the best fitted model; no refit
                rf, best, best_score = cand, p, score
            else:
                del cand
        print(f"\nBest on val: {best}")
    else:
        grid, best = None, base
        rf = C.fit_rf(X["train"], Y["train"], best, n_jobs=args.n_jobs,
                      random_state=args.seed, label="RF")

    results, preds, summary = {}, {}, []
    for s in ("val", "test"):
        preds[s] = rf.predict(X[s])
        results[s] = C.evaluate(Y[s], preds[s], s, args, args.seed)
        summary.append({"split": s, **C.summary_row(s, results[s])})

    # 5. Save.
    C.write_csv(f"{prefix}_metrics.csv", summary)
    meta = {
        "als_tif": str(als_tif), "emb_tif": str(emb_tif), "tag": tag,
        "nside": args.nside, "nest": True, "fractions": args.fractions,
        "cells": {s: sorted(c for c, v in split_of.items() if v == s) for s in SPLITS},
        "rf_params": best, "n_features": info["n_features"], "seed": args.seed,
    }
    C.write_json(f"{prefix}_metrics.json", {
        "args": vars(args), **meta, "tuning": grid, "buffer_dropped_px": n_buf,
        "n_px": {s: int(len(k)) for s, k in sel.items()}, "results": results,
    })
    C.save_model_bundle(f"{prefix}_rf.joblib", rf, meta)
    np.savez_compressed(
        f"{prefix}_predictions.npz",
        **{f"{s}_{k}": v for s in ("val", "test") for k, v in
           (("row", (idx[sel[s]] // W).astype(np.int32)),
            ("col", (idx[sel[s]] % W).astype(np.int32)),
            ("y", Y[s]), ("pred", preds[s].astype(np.float32)))})
    C.plot_scatter_panels(f"{prefix}_test_scatter.png", Y["test"], {tag: preds["test"]},
                          {tag: results["test"]}, "ALS RH98 (m)", args.bin_size)

    print(f"\nDone in {time.time() - t0:.0f}s")
    for s in ("val", "test"):
        o, st = results[s]["overall"], results[s]["stratified"]
        bal = (f"  bal.RMSE {st['balanced_rmse']:.2f}  bal.R2 {st['balanced_r2']:.3f}"
               if st.get("balanced_rmse") is not None else "")
        print(f"  {s:4s} n={o['n_samples']:,}  RMSE {o['rmse']:.2f}  R2 {o['r2']:.3f}  "
              f"bias {o['bias']:+.2f}{bal}")


if __name__ == "__main__":
    main()
