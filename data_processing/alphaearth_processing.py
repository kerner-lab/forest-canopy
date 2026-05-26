#!/usr/bin/env python3
"""
Process Alpha Earth Foundation (aef-mosaic) embeddings per healpix tile.

For each tile:
  1. Read tile bounds from the healpix shapefile.
  2. Subset aef-mosaic embeddings to tile bounds and the requested year.
  3. Dequantize int8 → float32 using the signed-square formula stored in the dataset.
  4. Reproject / resample to 30 m resolution, matching the GEDI raster for that tile
     (same CRS, affine transform, height, width).
  5. Extract (X, y) pixel pairs at valid non-zero GEDI locations.
  6. Save as healpix_{id}_float32.npz — identical format to the output of
     preprocess_olmo_to_gedi_tiles.py, so the directory can be passed directly to
     run_gedi_prediction_30m.py via --preprocessed-dir --embedding-format float32.

Single-tile usage (HPC array job):
    python alphaearth_processing.py \\
        --healpix-id 12345 --year 2020 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/alphaearth_cache

Batch usage (all tiles in a shapefile):
    python alphaearth_processing.py \\
        --year 2020 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/alphaearth_cache

Notes:
  - aef-mosaic CRS is EPSG:4326 (~10 m GSD). Shapefile must also be in WGS84 or
    have a CRS that pyproj can transform from.
  - Peak memory per tile scales with the spatial extent of the tile at ~10 m (source)
    resolution times 64 channels. Expect ~1–4 GB for a typical healpix tile.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import rasterio.warp
from rasterio.warp import Resampling
import xarray as xr
import zarr
import zarr.storage
from zarr.errors import ZarrUserWarning
from affine import Affine
from pyproj import CRS, Transformer

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

warnings.filterwarnings("ignore", category=ZarrUserWarning)

STORE_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"
GEDI_NODATA_DEFAULT = -9999
GEDI_MAX_HEIGHT_CM = 12000
# int8 nodata value defined in the aef-mosaic zarr_conventions metadata
AEF_INT8_NODATA = -128


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def open_aef_mosaic() -> xr.Dataset:
    tic = time.time()
    store = zarr.storage.FsspecStore.from_url(
        STORE_URL,
        storage_options={"anon": True},
        read_only=True,
    )
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False)
    print(f"Opened aef-mosaic in {time.time() - tic:.2f}s")
    return ds


def get_aef_crs(ds: xr.Dataset) -> CRS:
    """Return the CRS of the aef-mosaic (stored in ds.attrs, not ds['embeddings'].attrs)."""
    code = ds.attrs.get("proj:code")
    if code is None:
        raise ValueError("Could not find 'proj:code' in ds.attrs — check dataset metadata")
    return CRS.from_user_input(code)


# ---------------------------------------------------------------------------
# Tile bounds
# ---------------------------------------------------------------------------

def get_tile_bounds(healpix_id: int, gdf: gpd.GeoDataFrame) -> tuple:
    """Return (minx, miny, maxx, maxy) of the tile in the shapefile's CRS."""
    row = gdf[gdf["healpix_id"] == healpix_id]
    if len(row) == 0:
        raise ValueError(f"Tile {healpix_id} not found in shapefile")
    return row.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)


def bounds_to_crs(minx, miny, maxx, maxy, src_crs, dst_crs) -> tuple:
    """Transform a bounding box from src_crs to dst_crs by reprojecting all four corners."""
    if CRS.from_user_input(src_crs) == CRS.from_user_input(dst_crs):
        return minx, miny, maxx, maxy
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs, ys = t.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Spatial subset
# ---------------------------------------------------------------------------

def subset_aef(
    ds: xr.Dataset,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    year: int,
    buffer: float = 0.01,
) -> xr.DataArray:
    """
    Return a lazy spatial + temporal slice of the embeddings DataArray.

    Args:
        buffer: Extra degrees to pad around the tile bounds before slicing,
                ensuring the reprojection has coverage at tile edges (default: 0.01 °).
    """
    emb = ds["embeddings"]
    bx0, bx1 = minx - buffer, maxx + buffer
    by0, by1 = miny - buffer, maxy + buffer

    # aef-mosaic y is north-up (decreasing), so slice from north (high) to south (low)
    subset = emb.sel(
        x=slice(bx0, bx1),
        y=slice(by1, by0),
        time=year,
    )

    if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
        # Fallback: try south-up y ordering
        subset = emb.sel(x=slice(bx0, bx1), y=slice(by0, by1), time=year)

    return subset


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

def dequantize_aef(data: np.ndarray) -> np.ndarray:
    """
    Convert aef-mosaic int8 values to float32 using the signed-square formula:
        f = (x / 127.5)^2 * sign(x)
    nodata pixels (value == -128) are set to 0.0.
    """
    nodata_mask = data == AEF_INT8_NODATA
    x = data.astype(np.float32)
    out = (x / 127.5) ** 2 * np.sign(x)
    out[nodata_mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Source transform from coordinate arrays
# ---------------------------------------------------------------------------

def src_transform_from_coords(x_vals: np.ndarray, y_vals: np.ndarray) -> Affine:
    """
    Reconstruct an affine transform from pixel-centre coordinate arrays.
    Follows rasterio convention: transform maps (col, row) to the top-left corner
    of each pixel.
    """
    if len(x_vals) < 2 or len(y_vals) < 2:
        raise ValueError("Need ≥ 2 pixels in each spatial dimension to infer transform")
    dx = float(x_vals[1] - x_vals[0])
    dy = float(y_vals[1] - y_vals[0])
    # Top-left corner of top-left pixel = centre - half pixel
    return Affine(dx, 0.0, float(x_vals[0]) - dx / 2.0,
                  0.0, dy, float(y_vals[0]) - dy / 2.0)


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

def reproject_to_gedi(
    data: np.ndarray,
    src_crs: CRS,
    src_transform: Affine,
    gedi_transform: Affine,
    gedi_crs,
    gedi_height: int,
    gedi_width: int,
) -> np.ndarray:
    """
    Reproject a (C, H_src, W_src) float32 array to match the GEDI raster geometry.
    Uses bilinear resampling (appropriate for continuous embedding values).

    Returns (C, gedi_height, gedi_width) float32 array.
    """
    n_channels = data.shape[0]
    dst = np.zeros((n_channels, gedi_height, gedi_width), dtype=np.float32)
    rasterio.warp.reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=gedi_transform,
        dst_crs=gedi_crs,
        resampling=Resampling.bilinear,
        src_nodata=0.0,
        dst_nodata=0.0,
    )
    return dst


# ---------------------------------------------------------------------------
# GEDI loader
# ---------------------------------------------------------------------------

def load_gedi_tile(healpix_id: int, gedi_dir: Path):
    """Return (data, nodata, transform, crs, height, width) or raise FileNotFoundError."""
    for ext in (".tif", ".tiff", ""):
        p = gedi_dir / f"healpix_{healpix_id}_gedi{ext}"
        if p.exists():
            with rasterio.open(p) as src:
                return (
                    src.read(1),
                    src.nodata if src.nodata is not None else GEDI_NODATA_DEFAULT,
                    src.transform,
                    src.crs,
                    src.height,
                    src.width,
                )
    raise FileNotFoundError(f"No GEDI file for tile {healpix_id} in {gedi_dir}")


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def run_qa(
    healpix_id: int,
    reproj: np.ndarray,       # (C, H, W) reprojected embeddings at GEDI resolution
    gedi_data: np.ndarray,    # (H, W) raw GEDI heights in cm
    gedi_nodata: float,
    gedi_transform: Affine,
    gedi_crs,
    X: np.ndarray,            # (N, C) embeddings at valid GEDI pixels
    y: np.ndarray,            # (N,) GEDI heights in metres
    qa_dir: Path,
    logger: logging.Logger,
) -> dict:
    """
    Run alignment and coverage checks on one processed tile.

    Saves:
      tile_{id}_qa.png  — 4-panel diagnostic figure
      tile_{id}_qa.json — machine-readable stats

    Checks performed:
      1. Shape match between reprojected embeddings and GEDI raster.
      2. Pixel size extracted from the GEDI transform.
      3. Coverage: fraction of valid GEDI pixels that received a non-zero embedding
         after reprojection (gaps indicate the source extent didn't fully cover the tile).
      4. Embedding value range (expect roughly [-1, 1] after dequantization).
      5. Visual overlay: GEDI height map vs mean embedding map at the same 30 m grid.
      6. Scatter of embedding L2 norm vs GEDI height.
    """
    qa_dir.mkdir(parents=True, exist_ok=True)
    C, H, W = reproj.shape

    if H != gedi_data.shape[0] or W != gedi_data.shape[1]:
        raise ValueError(
            f"Shape mismatch: reproj ({H}, {W}) vs GEDI {gedi_data.shape}"
        )

    px_size_x = abs(float(gedi_transform.a))
    px_size_y = abs(float(gedi_transform.e))

    valid_mask = (
        (gedi_data != gedi_nodata) &
        (gedi_data > 0) &
        (gedi_data < GEDI_MAX_HEIGHT_CM)
    )
    n_valid = int(valid_mask.sum())

    emb_mean_map = np.mean(reproj, axis=0)          # (H, W) mean across channels
    nonzero_emb = emb_mean_map != 0.0
    n_covered = int(np.sum(nonzero_emb[valid_mask])  )
    coverage = n_covered / max(1, n_valid)

    emb_norm = np.linalg.norm(X, axis=1)             # (N,) per-pixel L2 norm

    stats = {
        "healpix_id": int(healpix_id),
        "gedi_shape": [H, W],
        "reproj_shape": [C, H, W],
        "gedi_crs": str(gedi_crs),
        "pixel_size_m": [round(px_size_x, 4), round(px_size_y, 4)],
        "n_valid_gedi_pixels": n_valid,
        "n_covered": n_covered,
        "coverage_fraction": round(coverage, 4),
        "n_samples": int(len(y)),
        "embedding_value_range": [round(float(X.min()), 4), round(float(X.max()), 4)],
        "embedding_mean": round(float(X.mean()), 4),
        "embedding_l2_norm_mean": round(float(emb_norm.mean()), 4),
        "gedi_height_m_min": round(float(y.min()), 2),
        "gedi_height_m_max": round(float(y.max()), 2),
        "gedi_height_m_mean": round(float(y.mean()), 2),
    }

    qa_json = qa_dir / f"tile_{healpix_id}_qa.json"
    with open(qa_json, "w") as f:
        json.dump(stats, f, indent=2)

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"QA — Tile {healpix_id}  |  {H}×{W} px @ {px_size_x:.1f} m  |  "
        f"CRS: {gedi_crs}",
        fontsize=11,
    )

    # Panel 1: GEDI height map
    ax = axes[0, 0]
    gedi_m = np.where(valid_mask, gedi_data / 100.0, np.nan)
    im = ax.imshow(gedi_m, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Height (m)", fraction=0.046)
    ax.set_title(f"GEDI height (m)  — {n_valid:,} valid pixels")
    ax.axis("off")

    # Panel 2: Mean embedding map at 30 m (same grid as GEDI)
    ax = axes[0, 1]
    emb_display = np.where(nonzero_emb, emb_mean_map, np.nan)
    im = ax.imshow(emb_display, cmap="plasma", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean embedding value", fraction=0.046)
    ax.set_title(f"Reprojected embedding (mean of {C} ch)")
    ax.axis("off")

    # Panel 3: Coverage map — which valid GEDI pixels got embeddings
    ax = axes[1, 0]
    cov_map = np.full((H, W), np.nan)
    cov_map[valid_mask & nonzero_emb]  = 1.0   # covered
    cov_map[valid_mask & ~nonzero_emb] = 0.0   # missing
    im = ax.imshow(cov_map, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(
        f"Coverage: {coverage:.1%} of valid GEDI pixels\n"
        f"green = has embedding, red = missing"
    )
    ax.axis("off")

    # Panel 4: Scatter — embedding L2 norm vs GEDI height
    ax = axes[1, 1]
    ax.scatter(y, emb_norm, alpha=0.2, s=1, rasterized=True, color="steelblue")
    ax.set_xlabel("GEDI height (m)")
    ax.set_ylabel("Embedding L2 norm")
    ax.set_title(f"Embedding norm vs GEDI height  (N={len(y):,})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    qa_png = qa_dir / f"tile_{healpix_id}_qa.png"
    plt.savefig(qa_png, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Tile {healpix_id} QA: shape=({H},{W}) px={px_size_x:.1f}m "
        f"coverage={coverage:.1%} emb=[{X.min():.3f},{X.max():.3f}]"
    )
    print(
        f"  QA: shape=({H}×{W})  px={px_size_x:.1f}m  "
        f"coverage={coverage:.1%}  "
        f"emb=[{X.min():.3f}, {X.max():.3f}]  → {qa_png.name}"
    )
    return stats


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(
    healpix_id: int,
    year: int,
    gdf: gpd.GeoDataFrame,
    aef_crs: CRS,
    ds: xr.Dataset,
    gedi_dir: Path,
    output_dir: Path,
    overwrite: bool,
    buffer: float,
    logger: logging.Logger,
    do_qa: bool = False,
    qa_dir: Path = None,
    test_fraction: float = None,
) -> dict:
    """Process one tile end-to-end; return a status dict.

    When test_fraction is set (e.g. 0.1), the tile's bounding box is cropped to
    that fraction of its linear dimensions (so 0.1 → ~1% of the area), drastically
    reducing the S3 fetch. Output is saved as healpix_{id}_float32_test.npz and is
    NOT suitable for training — use it only to verify alignment and QA plots quickly.
    """
    is_test = test_fraction is not None
    suffix = "_test" if is_test else ""
    out_path = output_dir / f"healpix_{healpix_id}_float32{suffix}.npz"
    if not overwrite and out_path.exists():
        logger.info(f"Tile {healpix_id}: cached, skipping")
        return {"status": "skipped", "healpix_id": int(healpix_id)}

    t_start = time.time()
    try:
        # 1. Tile bounds (shapefile CRS → aef CRS)
        shp_minx, shp_miny, shp_maxx, shp_maxy = get_tile_bounds(healpix_id, gdf)
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        minx, miny, maxx, maxy = bounds_to_crs(
            shp_minx, shp_miny, shp_maxx, shp_maxy, shp_crs, aef_crs
        )

        # Crop to a fraction of the tile for fast local testing.
        # We keep the NW corner so the subset still overlaps with real GEDI data.
        if is_test:
            f = float(test_fraction)
            maxx = minx + (maxx - minx) * f
            miny = maxy - (maxy - miny) * f
            print(
                f"  [test mode] fraction={f}  cropped bounds: "
                f"({minx:.5f}, {miny:.5f}, {maxx:.5f}, {maxy:.5f})"
            )

        logger.info(
            f"Tile {healpix_id}{'[test]' if is_test else ''}: bounds in aef CRS "
            f"({minx:.5f}, {miny:.5f}, {maxx:.5f}, {maxy:.5f})"
        )

        # 2. Spatial + temporal subset (lazy)
        subset = subset_aef(ds, minx, miny, maxx, maxy, year, buffer=buffer)
        if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
            logger.warning(f"Tile {healpix_id}: empty spatial subset")
            return {"status": "empty_subset", "healpix_id": int(healpix_id)}

        logger.info(
            f"Tile {healpix_id}: subset shape before compute: "
            f"{dict(subset.sizes)}"
        )

        # 3. Compute (triggers actual S3 download of the relevant chunks)
        t_fetch = time.time()
        raw = subset.compute().values  # (band, y, x) after time selection
        print(
            f"  Tile {healpix_id}: fetched {raw.shape} int8 in "
            f"{time.time() - t_fetch:.1f}s"
        )

        # Ensure (C, H, W) — time dim may survive as size-1 if year not in index
        if raw.ndim == 4:
            raw = raw.squeeze(axis=0)
        if raw.ndim != 3:
            raise ValueError(f"Unexpected embedding ndim={raw.ndim}, shape={raw.shape}")

        # band dim should be smallest (64 << spatial dims)
        if raw.shape[0] > raw.shape[-1]:
            raw = np.transpose(raw, (2, 0, 1))  # (H, W, C) → (C, H, W)

        # 4. Dequantize int8 → float32
        data = dequantize_aef(raw)  # (C, H, W) float32

        # 5. Source affine transform from coordinate arrays
        x_vals = subset.x.values
        y_vals = subset.y.values
        src_transform = src_transform_from_coords(x_vals, y_vals)

        # 6. Load GEDI target geometry
        gedi_data, gedi_nodata, gedi_transform, gedi_crs, gedi_h, gedi_w = \
            load_gedi_tile(healpix_id, gedi_dir)

        # In test mode crop the GEDI raster to the same NW sub-window so the
        # reprojection target is small and the QA coverage map is meaningful.
        if is_test:
            crop_h = max(1, int(round(gedi_h * test_fraction)))
            crop_w = max(1, int(round(gedi_w * test_fraction)))
            gedi_data = gedi_data[:crop_h, :crop_w]
            gedi_h, gedi_w = crop_h, crop_w

        # 7. Reproject embeddings to GEDI 30 m grid
        t_reproj = time.time()
        reproj = reproject_to_gedi(
            data, aef_crs, src_transform,
            gedi_transform, gedi_crs, gedi_h, gedi_w,
        )
        logger.info(
            f"Tile {healpix_id}: reprojected {data.shape} → {reproj.shape} "
            f"in {time.time() - t_reproj:.1f}s"
        )

        # 8. Extract valid non-zero GEDI pixels
        valid_mask = (
            (gedi_data != gedi_nodata) &
            (gedi_data > 0) &
            (gedi_data < GEDI_MAX_HEIGHT_CM)
        )
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            logger.info(f"Tile {healpix_id}: no valid non-zero GEDI pixels")
            return {"status": "no_valid_pixels", "healpix_id": int(healpix_id)}

        y = (gedi_data[rows, cols] / 100.0).astype(np.float32)  # cm → m
        X = reproj[:, rows, cols].T  # (N, C)

        # 9. QA (optional)
        if do_qa:
            run_qa(
                healpix_id=healpix_id,
                reproj=reproj,
                gedi_data=gedi_data,
                gedi_nodata=gedi_nodata,
                gedi_transform=gedi_transform,
                gedi_crs=gedi_crs,
                X=X,
                y=y,
                qa_dir=qa_dir,
                logger=logger,
            )

        # 10. Save
        np.savez_compressed(out_path, X=X, y=y)
        elapsed = time.time() - t_start

        if _HAS_PSUTIL:
            mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
            mem_str = f"  mem={mem_gb:.1f}GB"
        else:
            mem_str = ""

        logger.info(
            f"Tile {healpix_id}: {len(y):,} samples, "
            f"height [{y.min():.1f}, {y.max():.1f}] m, "
            f"emb {X.shape[1]} ch, saved in {elapsed:.1f}s{mem_str}"
        )
        print(
            f"  Tile {healpix_id}: {len(y):,} samples  "
            f"{X.shape[1]} ch  {elapsed:.1f}s{mem_str}  → {out_path.name}"
        )
        return {
            "status": "ok",
            "healpix_id": int(healpix_id),
            "n_samples": int(len(y)),
            "n_channels": int(X.shape[1]),
            "elapsed_s": round(elapsed, 1),
        }

    except FileNotFoundError as exc:
        logger.warning(f"Tile {healpix_id}: {exc}")
        return {"status": "missing_gedi", "healpix_id": int(healpix_id), "error": str(exc)}
    except Exception as exc:
        logger.error(f"Tile {healpix_id}: unexpected error — {exc}", exc_info=True)
        return {"status": "error", "healpix_id": int(healpix_id), "error": str(exc)}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("alphaearth")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_file = output_dir / f"alphaearth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Alpha Earth Foundation embeddings per healpix tile and save "
            "as (X, y) .npz files for use with run_gedi_prediction_30m.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--healpix-id", type=int, default=None,
        help="Single tile to process (omit for batch mode over all tiles in --splits)",
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Year of embeddings to extract (e.g. 2020)")
    parser.add_argument("--shapefile", required=True,
                        help="Healpix shapefile (.shp) with healpix_id and split columns")
    parser.add_argument("--gedi-dir", required=True,
                        help="Directory with GEDI tif files (healpix_*_gedi.tif)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-tile healpix_*_float32.npz files")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to include in batch mode",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process tiles that already have a cached .npz")
    parser.add_argument(
        "--buffer", type=float, default=0.01,
        help="Degrees to pad around tile bounds before spatial subsetting",
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Report cache coverage and exit without processing")
    parser.add_argument("--qa", action="store_true",
                        help="Save per-tile diagnostic plots and stats after processing")
    parser.add_argument("--qa-dir", default=None,
                        help="Directory for QA outputs (default: {output-dir}/qa/)")
    parser.add_argument(
        "--test-fraction", type=float, default=None, metavar="F",
        help=(
            "Crop each tile's bounding box to F of its linear dimensions before "
            "fetching from S3 (e.g. 0.1 → ~1%% of area). Output saved as "
            "healpix_*_float32_test.npz. Use with --qa to verify alignment quickly."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    qa_dir = Path(args.qa_dir) if args.qa_dir else output_dir / "qa"
    if args.qa:
        qa_dir.mkdir(parents=True, exist_ok=True)
        print(f"QA outputs will be saved to: {qa_dir}")

    # Load shapefile
    gdf = gpd.read_file(args.shapefile)
    print(f"Loaded shapefile: {len(gdf)} tiles")
    if "split" in gdf.columns:
        print(f"Split distribution:\n{gdf['split'].value_counts().to_string()}")

    # Determine tile list
    if args.healpix_id is not None:
        if int(args.healpix_id) not in gdf["healpix_id"].values:
            print(f"Error: healpix_id {args.healpix_id} not found in shapefile", file=sys.stderr)
            sys.exit(1)
        tile_ids = [args.healpix_id]
        print(f"Single-tile mode: tile {args.healpix_id}  year={args.year}")
    else:
        if "split" in gdf.columns:
            tile_ids = gdf[gdf["split"].isin(args.splits)]["healpix_id"].tolist()
        else:
            tile_ids = gdf["healpix_id"].tolist()
        print(f"Batch mode: {len(tile_ids)} tiles  splits={args.splits}  year={args.year}")

    if args.check_only:
        cached = [t for t in tile_ids if (output_dir / f"healpix_{t}_float32.npz").exists()]
        missing = [t for t in tile_ids if t not in set(cached)]
        print(f"Cached : {len(cached)}/{len(tile_ids)}")
        if missing:
            print(f"Missing: {len(missing)} — first 20: {missing[:20]}")
        return

    # Open aef-mosaic once (lazy; actual data loaded per-tile inside process_tile)
    print("\nOpening aef-mosaic (lazy)...")
    ds = open_aef_mosaic()
    aef_crs = get_aef_crs(ds)
    print(f"aef-mosaic CRS  : {aef_crs.to_epsg() or aef_crs.name}")
    print(f"aef-mosaic dims : {dict(ds['embeddings'].sizes)}")
    print(f"Available years : {ds['embeddings'].coords['time'].values.tolist()}")
    if year := args.year:
        available = ds["embeddings"].coords["time"].values.tolist()
        if year not in available:
            print(
                f"Warning: year {year} not in available years {available}",
                file=sys.stderr,
            )

    # Process tiles
    gedi_dir = Path(args.gedi_dir)
    results = []
    iterator = tqdm(tile_ids, desc="tiles") if _HAS_TQDM else tile_ids

    for healpix_id in iterator:
        result = process_tile(
            healpix_id=healpix_id,
            year=args.year,
            gdf=gdf,
            aef_crs=aef_crs,
            ds=ds,
            gedi_dir=gedi_dir,
            output_dir=output_dir,
            overwrite=args.overwrite,
            buffer=args.buffer,
            logger=logger,
            do_qa=args.qa,
            qa_dir=qa_dir,
            test_fraction=args.test_fraction,
        )
        results.append(result)

    # Summary
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total_samples = sum(r.get("n_samples", 0) for r in results)

    print("\n" + "=" * 60)
    print(f"Done  year={args.year}  tiles={len(tile_ids)}")
    for status, count in sorted(counts.items()):
        print(f"  {status:25s}: {count}")
    print(f"  {'total samples':25s}: {total_samples:,}")
    print(f"\nOutput: {output_dir}")
    print(
        f"Use with run_gedi_prediction_30m.py:\n"
        f"  --preprocessed-dir {output_dir} --embedding-format float32"
    )

    summary = {
        "year": args.year,
        "n_tiles": len(tile_ids),
        "total_samples": total_samples,
        "status_counts": counts,
        "tiles": results,
    }
    summary_path = output_dir / f"alphaearth_summary_{args.year}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
