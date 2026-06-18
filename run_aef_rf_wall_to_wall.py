#!/usr/bin/env python3
"""
Train a Random Forest on Alpha Earth (AEF) embeddings at GEDI train pixels, then predict
GEDI canopy height wall-to-wall over an AOI AEF GeoTIFF, save the prediction as a
georeferenced .tif, and plot a map with a colorbar.

Pipeline
--------
1. TRAIN: load the per-tile (X, y) .npz produced by alphaearth_processing.py for the
   TRAIN-split healpix tiles, across one or more year directories (e.g. 2019-2022), and
   fit a RandomForestRegressor.  X = (N, 64) AEF channels, y = canopy height in METERS.
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
        --train-dirs /data/aef/2019 /data/aef/2020 /data/aef/2021 /data/aef/2022 \
        --aoi-tif    /path/oregon_validation.tif \
        --output-tif /path/oregon_canopy_height_rf.tif \
        --output-plot /path/oregon_canopy_height_rf.png

Notes
-----
- Each --train-dirs entry is a directory of healpix_<id>_float32[_withzeros].npz (one AEF
  year). Train tiles are gdf[gdf.split == --split]; tiles missing in a given year are skipped.
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
from rasterio.transform import array_bounds
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------

def load_training_data(shapefile, train_dirs, split, include_zeros, embedding_format,
                       max_samples, seed):
    """Concatenate (X, y) over the split's tiles across all train_dirs (years)."""
    gdf = gpd.read_file(shapefile)
    if "split" not in gdf.columns:
        sys.exit("ERROR: shapefile has no 'split' column")
    tile_ids = gdf[gdf["split"] == split]["healpix_id"].tolist()
    print(f"Split '{split}': {len(tile_ids)} tiles  x {len(train_dirs)} year-dir(s)")

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
    return X, y, gdf


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

def predict_wall_to_wall(rf, aoi_tif, output_tif, n_features, block_rows):
    """Predict over the AOI AEF GeoTIFF, writing a single-band float32 height raster.

    Reads in row strips to bound memory. Valid pixels = finite across all bands (the AOI
    tif uses NaN for outside-polygon / no-coverage). Output nodata = NaN, so a predicted
    height of 0 remains a valid value.
    """
    with rasterio.open(aoi_tif) as src:
        if src.count != n_features:
            sys.exit(f"ERROR: AOI tif has {src.count} bands but the model expects "
                     f"{n_features} features — band/feature mismatch.")
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.float32("nan"),
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # 1. Train
    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    X, y, _ = load_training_data(
        args.shapefile, args.train_dirs, args.split, args.include_zeros,
        args.embedding_format, max_samples, args.random_state)
    rf, metrics = train_rf(X, y, args)
    n_features = X.shape[1]

    if args.model_out:
        import joblib
        joblib.dump(rf, args.model_out)
        print(f"Saved model: {args.model_out}")

    # 2-3. Predict wall-to-wall -> GeoTIFF
    print(f"\nPredicting wall-to-wall over {args.aoi_tif} ...")
    n_valid = predict_wall_to_wall(rf, args.aoi_tif, str(output_tif), n_features, args.block_rows)

    # 4. Map
    plot_prediction(str(output_tif), str(output_plot), args.cmap, args.vmin, args.vmax, args.title)

    # Sidecar metrics/metadata
    meta = {
        "shapefile": args.shapefile, "split": args.split, "train_dirs": args.train_dirs,
        "include_zeros": args.include_zeros, "n_features": int(n_features),
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
