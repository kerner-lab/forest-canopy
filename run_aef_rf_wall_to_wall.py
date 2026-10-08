#!/usr/bin/env python3
"""
Train a Random Forest on Alpha Earth (AEF) embeddings at GEDI train pixels, then predict
GEDI canopy height wall-to-wall over an AOI AEF GeoTIFF, save the prediction as a
georeferenced .tif, and plot a map with a colorbar.

Pipeline
--------
0. SELECT TILES: take the healpix tiles of the requested --split, then DROP every tile
   whose footprint overlaps the prediction AOI, so the wall-to-wall map is not predicted
   from pixels the model was trained on (no spatial data leakage).
1. TRAIN: load the per-tile (X, y) .npz produced by alphaearth_processing.py for the
   remaining TRAIN-split healpix tiles, across one or more year directories (e.g.
   2019-2022), and fit a RandomForestRegressor.  X = (N, 64) AEF channels, y = canopy
   height in METERS.
2. PREDICT: read the wall-to-wall AOI GeoTIFF from alphaearth_wall_to_wall.py
   (64 bands `aef_ch_001..064`, float32, NaN nodata) — band order matches X — flatten to
   (pixels, 64), predict on the valid (finite) pixels, and reshape back to the grid.
3. SAVE: write a single-band float32 GeoTIFF of predicted height, preserving the AOI's
   CRS/transform, with NaN nodata (so a predicted height of 0 stays a VALID value).
4. PLOT: render the prediction as a map with a colorbar (height in m, 0 valid).

Usage
-----
    python run_aef_rf_wall_to_wall.py \
        --shapefile /path/healpix.shp \
        --aoi-shapefile /path/oregon_aoi.shp \
        --train-dirs /data/aef/2019 /data/aef/2020 /data/aef/2021 /data/aef/2022 \
        --aoi-tif    /path/oregon_validation.tif \
        --output-tif /path/oregon_canopy_height_rf.tif \
        --output-plot /path/oregon_canopy_height_rf.png

Notes
-----
- Each --train-dirs entry is a directory of healpix_<id>_float32[_withzeros].npz (one AEF
  year). Train tiles are gdf[gdf.split == --split] minus the AOI-overlapping tiles; tiles
  missing in a given year are skipped.
- AOI leakage guard: pass --aoi-shapefile (the same boundary shapefile given to
  alphaearth_wall_to_wall.py). Without it the AOI footprint falls back to the bounding box
  of --aoi-tif, which is a superset of the real polygon and therefore still leak-safe (it
  may drop a few extra tiles). --aoi-buffer-m adds a guard band around the AOI;
  --no-exclude-aoi-overlap turns the guard off entirely (leaky — diagnostics only).
- Overlap is measured as intersected area / tile area in an equal-area CRS, so a healpix
  tile that merely shares an edge with the AOI (zero-area touch) is NOT dropped.
- --include-zeros loads the *_withzeros.npz caches (0 m heights kept). 0 m is treated as a
  valid prediction throughout (it is NOT used as nodata).
- The AOI .tif and the training X must have the same 64 channels in the same order — both
  come from the same AEF dequantization, so this holds; the script asserts it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.transform import array_bounds
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# 0. Tile selection (AOI leakage guard)
# ---------------------------------------------------------------------------

# Equal-area CRS used only to measure overlap areas/fractions (not for output).
EQUAL_AREA_CRS = "EPSG:6933"


def load_aoi_geometry(aoi_shapefile, aoi_tif):
    """Return (AOI footprint as one shapely geometry in EPSG:4326, description).

    Prefers the boundary shapefile (the true polygon). Falls back to the bounding box of
    the AOI GeoTIFF, which encloses the polygon — a superset, so still leak-safe.
    """
    if aoi_shapefile:
        if Path(aoi_shapefile).suffix.lower() in {".tif", ".tiff", ".vrt"}:
            sys.exit(f"ERROR: --aoi-shapefile expects a VECTOR boundary (.shp/.geojson/.gpkg), "
                     f"but got a raster: {aoi_shapefile}\n"
                     f"       Omit --aoi-shapefile entirely — the AOI footprint then comes "
                     f"from --aoi-tif, which is what you want here.")
        g = gpd.read_file(aoi_shapefile)
        if g.crs is None:
            print("Warning: --aoi-shapefile has no CRS; assuming EPSG:4326", file=sys.stderr)
            g = g.set_crs(4326)
        return g.to_crs(4326).union_all(), f"{aoi_shapefile} ({len(g)} feature(s))"

    with rasterio.open(aoi_tif) as src:
        if src.crs is None:
            sys.exit("ERROR: --aoi-tif has no CRS; pass --aoi-shapefile for the AOI footprint")
        left, bottom, right, top = rasterio.warp.transform_bounds(
            src.crs, "EPSG:4326", *src.bounds)
    return box(left, bottom, right, top), f"bounding box of {aoi_tif}"


def select_train_tiles(gdf, split, aoi_geom, buffer_m, max_overlap_frac, n_print=15):
    """Tile ids of `split`, minus every tile overlapping the AOI.

    Overlap is intersected area / tile area in an equal-area CRS, so healpix neighbours
    that only share an edge with the AOI (zero-area touch) survive. Returns
    (kept_ids, dropped) where `dropped` is a list of {tile_id, overlap_frac} dicts.
    """
    if "split" not in gdf.columns:
        sys.exit("ERROR: shapefile has no 'split' column")
    if gdf.crs is None:
        print("Warning: healpix shapefile has no CRS; assuming EPSG:4326", file=sys.stderr)
        gdf = gdf.set_crs(4326)

    tiles = gdf[gdf["split"] == split].copy()
    ids = tiles["healpix_id"]
    if np.issubdtype(ids.dtype, np.floating):  # some shapefiles store the id as a float
        ids = ids.astype("int64")
    tiles["tile_id"] = ids.astype(str)
    print(f"Split '{split}': {len(tiles)} tiles")

    if aoi_geom is None:
        print("  AOI overlap guard DISABLED — training tiles may overlap the prediction AOI")
        return tiles["tile_id"].tolist(), []

    aoi = gpd.GeoSeries([aoi_geom], crs=4326)
    if buffer_m:
        aoi = aoi.to_crs(aoi.estimate_utm_crs()).buffer(buffer_m).to_crs(4326)
    aoi_ea = aoi.to_crs(EQUAL_AREA_CRS).union_all()

    tiles_ea = tiles.to_crs(EQUAL_AREA_CRS)
    tile_area = tiles_ea.geometry.area.to_numpy()
    inter_area = tiles_ea.geometry.intersection(aoi_ea).area.to_numpy()
    frac = np.divide(inter_area, tile_area, out=np.zeros_like(inter_area),
                     where=tile_area > 0)

    drop = frac > max_overlap_frac
    dropped = [{"tile_id": t, "overlap_frac": float(f)}
               for t, f in zip(tiles["tile_id"].to_numpy()[drop], frac[drop])]
    dropped.sort(key=lambda d: -d["overlap_frac"])
    kept = tiles["tile_id"].to_numpy()[~drop].tolist()

    print(f"  AOI overlap guard: dropped {len(dropped)} tile(s), kept {len(kept)}"
          f"  (buffer={buffer_m:g} m, max_overlap_frac={max_overlap_frac:g})")
    for d in dropped[:n_print]:
        print(f"    - healpix {d['tile_id']}: {d['overlap_frac']*100:.1f}% inside AOI")
    if len(dropped) > n_print:
        print(f"    ... and {len(dropped)-n_print} more (full list in the output .json)")
    if not kept:
        sys.exit("ERROR: every training tile overlaps the AOI — nothing left to train on")
    return kept, dropped


# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------

def load_training_data(tile_ids, train_dirs, include_zeros, embedding_format,
                       max_samples, seed):
    """Concatenate (X, y) over the selected tiles across all train_dirs (years)."""
    print(f"Loading {len(tile_ids)} tiles x {len(train_dirs)} year-dir(s)")

    suffix = "_withzeros" if include_zeros else ""
    Xs, ys = [], []
    n_files = n_missing = 0
    for d in train_dirs:
        d = Path(d)
        for tid in tile_ids:
            p = d / f"healpix_{tid}_{embedding_format}{suffix}.npz"
            if not p.exists():
                n_missing += 1
                continue
            with np.load(p) as npz:
                X, y = npz["X"], npz["y"]
            if X.size == 0:
                continue
            Xs.append(X.astype(np.float32))
            ys.append(y.astype(np.float32))
            n_files += 1

    if not Xs:
        sys.exit("ERROR: no training .npz found — check --train-dirs / --split / --include-zeros")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"Loaded {n_files} tile-files ({n_missing} missing) -> X={X.shape} y={y.shape}")
    print(f"  height (m): min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}")

    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"  subsampled to {max_samples:,} training pixels (seed={seed})")
    return X, y


def train_rf(X, y, args):
    print(f"\nTraining RandomForest (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}, "
          f"n_jobs={args.n_jobs}) on {len(y):,} pixels...")
    t = time.time()
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    rf.fit(X, y)
    pred = rf.predict(X)
    metrics = {
        "n_train": int(len(y)),
        "train_r2": float(r2_score(y, pred)),
        "train_rmse_m": float(np.sqrt(mean_squared_error(y, pred))),
        "train_mae_m": float(mean_absolute_error(y, pred)),
    }
    print(f"  fit in {time.time()-t:.1f}s  | train R2={metrics['train_r2']:.3f} "
          f"RMSE={metrics['train_rmse_m']:.2f} m  MAE={metrics['train_mae_m']:.2f} m")
    return rf, metrics


# ---------------------------------------------------------------------------
# 2-3. Wall-to-wall prediction -> GeoTIFF
# ---------------------------------------------------------------------------

def predict_wall_to_wall(rf, aoi_tif, output_tif, n_features, block_rows,
                         compress="lzw", predictor=0):
    """Predict over the AOI AEF GeoTIFF, writing a single-band float32 height raster.

    Reads in row strips to bound memory. Valid pixels = finite across all bands (the AOI
    tif uses NaN for outside-polygon / no-coverage). Output nodata = NaN, so a predicted
    height of 0 remains a valid value.

    The output is one band (~tens of MB), so compression barely matters — default to LZW
    (no predictor), which is stable everywhere. `deflate`+`predictor=3` (float predictor)
    can abort with "double free / corruption" at GDAL close on some libtiff builds, so it
    is NOT the default; pass --compress deflate --predictor 3 only if you know it's safe.
    """
    with rasterio.open(aoi_tif) as src:
        if src.count != n_features:
            sys.exit(f"ERROR: AOI tif has {src.count} bands but the model expects "
                     f"{n_features} features — band/feature mismatch.")
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.float32("nan"),
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
                       num_threads=1)
        # Don't inherit the source AOI tif's deflate+predictor=3 (the likely crash-at-close
        # culprit). Reset and apply our own, stable choice.
        profile.pop("compress", None)
        profile.pop("predictor", None)
        if compress and compress.lower() != "none":
            profile["compress"] = compress
            if predictor:
                profile["predictor"] = predictor
        H, W = src.height, src.width

        n_valid = 0
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.set_band_description(1, "predicted_canopy_height_m")
            rows = range(0, H, block_rows)
            it = tqdm(list(rows), desc="predict") if _HAS_TQDM else rows
            for r0 in it:
                bh = min(block_rows, H - r0)
                win = rasterio.windows.Window(0, r0, W, bh)
                block = src.read(window=win).astype(np.float32)  # (C, bh, W)
                feats = block.reshape(block.shape[0], -1).T       # (bh*W, C)
                valid = np.all(np.isfinite(feats), axis=1)
                out = np.full(feats.shape[0], np.nan, dtype=np.float32)
                if valid.any():
                    out[valid] = rf.predict(feats[valid]).astype(np.float32)
                    n_valid += int(valid.sum())
                dst.write(out.reshape(bh, W), 1, window=win)

    print(f"Predicted {n_valid:,} valid pixels -> {output_tif}")
    return n_valid


# ---------------------------------------------------------------------------
# 4. Map plot
# ---------------------------------------------------------------------------

def plot_prediction(output_tif, output_plot, cmap, vmin, vmax, title):
    matplotlib.use("Agg")

    with rasterio.open(output_tif) as src:
        arr = src.read(1)
        left, bottom, right, top = array_bounds(src.height, src.width, src.transform)
        crs = src.crs

    arr = np.ma.masked_invalid(arr)  # NaN nodata -> transparent
    if vmax is None and arr.count() > 0:
        vmax = float(np.percentile(arr.compressed(), 98))  # robust upper bound

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)  # nodata transparent

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   extent=[left, right, bottom, top], origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted canopy height (m)")
    ax.set_title(title)
    ax.set_xlabel(f"Easting ({crs.to_string()})" if crs else "Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map: {output_plot}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile", required=True, help="healpix shapefile with a 'split' column")
    ap.add_argument("--train-dirs", nargs="+", required=True,
                    help="one or more dirs of per-tile AEF .npz (e.g. the 2019..2022 dirs)")
    ap.add_argument("--aoi-tif", required=True,
                    help="wall-to-wall AEF GeoTIFF from alphaearth_wall_to_wall.py (64 bands)")
    ap.add_argument("--output-tif", required=True, help="output prediction GeoTIFF path")
    ap.add_argument("--output-plot", default=None,
                    help="output map PNG path (default: <output-tif>.png)")
    ap.add_argument("--split", default="train", help="shapefile split to train on")
    # AOI leakage guard
    ap.add_argument("--aoi-shapefile", default=None,
                    help="boundary shapefile of the prediction AOI (the one passed to "
                         "alphaearth_wall_to_wall.py). Training tiles overlapping it are "
                         "dropped. Default: fall back to the --aoi-tif bounding box")
    ap.add_argument("--aoi-buffer-m", type=float, default=0.0,
                    help="expand the AOI by this many metres before testing overlap, to add "
                         "a spatial guard band around the prediction region")
    ap.add_argument("--max-overlap-frac", type=float, default=0.0,
                    help="drop a train tile when (tile ∩ AOI area)/(tile area) exceeds this. "
                         "0.0 (default) drops any tile with real overlap; edge-touching tiles "
                         "(zero area) are kept")
    ap.add_argument("--no-exclude-aoi-overlap", action="store_true",
                    help="do NOT drop AOI-overlapping train tiles (leaky — diagnostics only)")
    ap.add_argument("--include-zeros", action="store_true",
                    help="load *_withzeros.npz (keep 0 m GEDI heights as valid training samples)")
    ap.add_argument("--embedding-format", default="float32",
                    help="npz format token in the filename (matches alphaearth_processing.py)")
    ap.add_argument("--max-train-samples", type=int, default=3_000_000,
                    help="cap total training pixels (random subsample) for memory; 0 = no cap")
    # RF hyperparameters
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=5)
    ap.add_argument("--max-features", default="sqrt", help="RF max_features (sqrt, 1.0, etc.)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--random-state", type=int, default=42)
    # Prediction / plot
    ap.add_argument("--block-rows", type=int, default=256, help="rows per prediction strip")
    ap.add_argument("--compress", default="lzw",
                    help="output GeoTIFF compression: lzw (default, stable), deflate, zstd, none")
    ap.add_argument("--predictor", type=int, default=0,
                    help="TIFF predictor (0=none). 3=float predictor needs deflate/zstd and "
                         "can crash at close on some libtiff builds — leave 0 unless sure.")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap for the height map")
    ap.add_argument("--vmin", type=float, default=0.0, help="colorbar min (0 m is valid)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="colorbar max (default: 98th percentile of predictions)")
    ap.add_argument("--title", default="Predicted GEDI canopy height (Random Forest, AEF)")
    ap.add_argument("--model-out", default=None, help="optional path to joblib-dump the RF")
    args = ap.parse_args()

    output_tif = Path(args.output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    output_plot = Path(args.output_plot) if args.output_plot else output_tif.with_suffix(".png")

    # 0. Select training tiles, excluding any that overlap the prediction AOI
    if args.no_exclude_aoi_overlap:
        aoi_geom, aoi_desc = None, "disabled"
    else:
        aoi_geom, aoi_desc = load_aoi_geometry(args.aoi_shapefile, args.aoi_tif)
        print(f"AOI footprint from: {aoi_desc}")
    gdf = gpd.read_file(args.shapefile)
    tile_ids, dropped = select_train_tiles(
        gdf, args.split, aoi_geom, args.aoi_buffer_m, args.max_overlap_frac)

    # 1. Train
    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    X, y = load_training_data(
        tile_ids, args.train_dirs, args.include_zeros,
        args.embedding_format, max_samples, args.random_state)
    rf, metrics = train_rf(X, y, args)
    n_features = X.shape[1]

    if args.model_out:
        import joblib
        joblib.dump(rf, args.model_out)
        print(f"Saved model: {args.model_out}")

    # 2-3. Predict wall-to-wall -> GeoTIFF
    print(f"\nPredicting wall-to-wall over {args.aoi_tif} ...")
    n_valid = predict_wall_to_wall(rf, args.aoi_tif, str(output_tif), n_features,
                                   args.block_rows, compress=args.compress, predictor=args.predictor)

    # 4. Map
    plot_prediction(str(output_tif), str(output_plot), args.cmap, args.vmin, args.vmax, args.title)

    # Sidecar metrics/metadata
    meta = {
        "shapefile": args.shapefile, "split": args.split, "train_dirs": args.train_dirs,
        "include_zeros": args.include_zeros, "n_features": int(n_features),
        "aoi_exclusion": {
            "enabled": not args.no_exclude_aoi_overlap,
            "aoi_source": aoi_desc,
            "aoi_shapefile": args.aoi_shapefile,
            "buffer_m": args.aoi_buffer_m,
            "max_overlap_frac": args.max_overlap_frac,
            "n_tiles_kept": len(tile_ids),
            "n_tiles_dropped": len(dropped),
            "tiles_kept": tile_ids,
            "tiles_dropped": dropped,
        },
        "aoi_tif": args.aoi_tif, "output_tif": str(output_tif), "output_plot": str(output_plot),
        "n_valid_pixels": int(n_valid),
        "rf": {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
               "min_samples_leaf": args.min_samples_leaf, "max_features": args.max_features},
        "metrics": metrics,
    }
    with open(output_tif.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Sidecar: {output_tif.with_suffix('.json')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
