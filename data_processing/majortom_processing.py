#!/usr/bin/env python3
"""
Process Major TOM precomputed embeddings (Core-S2L1C-SSL4EO) per healpix tile.

This is the Major TOM analogue of tessera_processing.py / alphaearth_processing.py.
It produces the exact same output format — healpix_{id}_float32.npz containing
(X, y) — so the output directory can be passed straight to
run_gedi_prediction_30m.py via --preprocessed-dir --embedding-format float32.

  Dataset: https://huggingface.co/datasets/Major-TOM/Core-S2L1C-SSL4EO

IMPORTANT — how Major TOM differs from TESSERA / Alpha Earth
------------------------------------------------------------
TESSERA and Alpha Earth are DENSE PER-PIXEL embeddings (128-D / 64-D at ~10 m).
You reproject/mosaic them onto the GEDI grid and sample a vector at every pixel.

Major TOM Core-S2L1C-SSL4EO is different: each row is ONE 2048-D vector for a
whole 224x224-pixel patch (~2.24 km x 2.24 km at 10 m), produced by the
SSL4EO ResNet50-DINO image encoder. There is no per-pixel structure inside a
patch. Building a full (2048, H, W) reprojected raster for a healpix tile would
need ~100 GB of RAM, so instead we do a SPATIAL JOIN:

  - Every valid GEDI pixel is assigned the 2048-D vector of the Major TOM patch
    whose footprint contains it (nearest patch centroid breaks overlaps and fills
    small gaps, up to --max-dist-m).

The resulting X is therefore "blocky": every 30 m GEDI pixel inside a given
~2.24 km patch shares the same 2048-D vector. That is inherent to this dataset.

Temporal coverage
-----------------
Core-S2L1C-SSL4EO is effectively SINGLE-TEMPORAL: Major TOM Core-S2L1C stores one
least-cloudy Sentinel-2 L1C scene per grid point (chosen within a random 4-month
window), so each location has exactly one acquisition. Acquisition timestamps span
the Sentinel-2 archive (roughly 2017-2023). Run `--list-years` to print the actual
year histogram from the data. The --year flag *filters* to a single acquisition
year; because coverage is single-temporal, filtering by year will drop most
locations, so the default is to keep all years (one patch per location).

For each tile:
  1. Read tile bounds from the healpix shapefile (assumed WGS84 / EPSG:4326).
  2. Query the Major TOM parquet for patches whose centre falls in those bounds
     (+ buffer). lat/lon predicate pushdown prunes the row groups (the parquet is
     ordered south->north), so a single tile reads only a small slice.
  3. Optionally filter patches to --year.
  4. Reproject the patch footprints into the GEDI tile CRS and spatially join:
     assign each valid GEDI pixel the embedding of the containing / nearest patch.
  5. Extract (X, y) pixel pairs at valid GEDI locations.
  6. Save as healpix_{id}_float32.npz.

Single-tile usage (HPC array job):
    python majortom_processing.py \\
        --healpix-id 12345 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/majortom_cache

Batch usage (all tiles in a shapefile):
    python majortom_processing.py \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/majortom_cache

Report the year distribution and exit:
    python majortom_processing.py --list-years --year-scan-files 20

Notes:
  - Requires: pip install pyarrow huggingface_hub geopandas shapely rasterio pyproj
  - The first query streams parquet directly from the Hugging Face hub. Set
    HF_TOKEN in the environment if you hit rate limits. Use --local-parquet to
    point at an already-downloaded copy of the embeddings/ folder instead.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
import shapely
from shapely import wkb
from shapely.strtree import STRtree

import pyarrow.dataset as pads
import pyarrow.compute as pc

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

GEDI_NODATA_DEFAULT = -9999
GEDI_MAX_HEIGHT_CM = 12000
MAJORTOM_N_CHANNELS = 2048
# ~2.24 km patch (224 px x 10 m). Default nearest-patch cap is one patch diagonal.
MAJORTOM_PATCH_M = 2240.0
DEFAULT_MAX_DIST_M = MAJORTOM_PATCH_M  # cap for assigning a pixel to a non-containing patch

HF_REPO = "Major-TOM/Core-S2L1C-SSL4EO"
HF_EMBED_GLOB = "embeddings"  # folder of part_*.parquet inside the repo

# Columns we actually need (NOT the full row — keeps the scan light).
SCAN_COLUMNS = ["embedding", "centre_lat", "centre_lon", "geometry", "timestamp"]


# ---------------------------------------------------------------------------
# Major TOM parquet source
# ---------------------------------------------------------------------------

def open_majortom_dataset(local_parquet: str = None):
    """
    Open the Major TOM embeddings parquet as a pyarrow Dataset.

    If --local-parquet is given, read that directory of part_*.parquet files.
    Otherwise stream directly from the Hugging Face hub via HfFileSystem.
    """
    tic = time.time()
    if local_parquet:
        ds = pads.dataset(local_parquet, format="parquet")
        print(f"Opened local Major TOM parquet ({local_parquet}) in {time.time()-tic:.2f}s")
        return ds

    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to stream Major TOM from the hub. "
            "Install it (`pip install huggingface_hub`) or pass --local-parquet."
        ) from exc

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    root = f"datasets/{HF_REPO}/{HF_EMBED_GLOB}"
    files = sorted(fs.glob(f"{root}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under hf://{root}")
    ds = pads.dataset(files, filesystem=fs, format="parquet")
    print(f"Opened Major TOM hub dataset: {len(files)} parquet files "
          f"in {time.time()-tic:.2f}s")
    return ds


# ---------------------------------------------------------------------------
# Tile bounds (shared with the TESSERA / Alpha Earth scripts)
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
# Fetch covering Major TOM patches
# ---------------------------------------------------------------------------

def fetch_covering_patches(ds, lon_min, lat_min, lon_max, lat_max, year, buffer, logger):
    """
    Return (embeddings, gdf) for every Major TOM patch whose centre lies inside
    the (buffered) WGS84 bounds.

      embeddings : (M, 2048) float32
      gdf        : GeoDataFrame (EPSG:4326) of M patch footprint polygons, with a
                   'year' column.

    The bounds filter is pushed down to the parquet scan (the files are ordered by
    latitude, so this prunes most row groups). A patch is kept if its CENTRE is in
    bounds; the buffer (in degrees) widens the catch so patches straddling the tile
    edge are included.
    """
    bx0, by0 = lon_min - buffer, lat_min - buffer
    bx1, by1 = lon_max + buffer, lat_max + buffer

    filt = (
        (pc.field("centre_lon") >= bx0) & (pc.field("centre_lon") <= bx1) &
        (pc.field("centre_lat") >= by0) & (pc.field("centre_lat") <= by1)
    )
    table = ds.to_table(columns=SCAN_COLUMNS, filter=filt)
    if table.num_rows == 0:
        return np.empty((0, MAJORTOM_N_CHANNELS), dtype=np.float32), None

    raw_emb = table.column("embedding").to_pylist()
    geom_wkb = table.column("geometry").to_pylist()
    ts = table.column("timestamp").to_pylist()

    # Keep only rows with a valid geometry and a full-length 2048-D embedding.
    # (Lengths can differ if a row is malformed, so filter row-by-row instead of
    # relying on np.asarray producing a clean 2-D array.)
    keep_rows, geoms, years = [], [], []
    for i, (e, g, t) in enumerate(zip(raw_emb, geom_wkb, ts)):
        if g is None or e is None or len(e) != MAJORTOM_N_CHANNELS:
            continue
        keep_rows.append(i)
        geoms.append(wkb.loads(g))
        years.append(_year_of(t))
    n_bad = len(raw_emb) - len(keep_rows)
    if n_bad:
        logger.warning("Dropped %d patch(es) with missing geometry or bad embedding length", n_bad)
    if not keep_rows:
        return np.empty((0, MAJORTOM_N_CHANNELS), dtype=np.float32), None

    emb = np.asarray([raw_emb[i] for i in keep_rows], dtype=np.float32)
    gdf = gpd.GeoDataFrame(
        {"year": np.asarray(years, dtype=np.int32)}, geometry=geoms, crs="EPSG:4326"
    )

    if year is not None:
        keep = gdf["year"].values == int(year)
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info("Year filter %s: kept %d / %d patches", year, int(keep.sum()), len(gdf))
        emb = emb[keep]
        gdf = gdf[keep].reset_index(drop=True)

    return emb.astype(np.float32, copy=False), gdf


def _year_of(timestamp: str) -> int:
    """Major TOM timestamps are 'YYYYMMDDTHHMMSS'."""
    try:
        return int(str(timestamp)[:4])
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Spatial join: assign each GEDI pixel a patch embedding
# ---------------------------------------------------------------------------

def assign_pixels_to_patches(
    rows: np.ndarray,
    cols: np.ndarray,
    gedi_transform: Affine,
    gedi_crs,
    patch_gdf: gpd.GeoDataFrame,
    embeddings: np.ndarray,
    max_dist_m: float,
    logger: logging.Logger,
):
    """
    Map valid GEDI pixels -> Major TOM patch embeddings via nearest patch.

    Args:
        rows, cols     : (N,) pixel indices of valid GEDI pixels.
        gedi_transform : affine of the GEDI raster.
        gedi_crs       : CRS of the GEDI raster.
        patch_gdf      : (M,) patch footprints in EPSG:4326.
        embeddings     : (M, 2048) float32 patch vectors (aligned with patch_gdf).
        max_dist_m     : pixels farther than this from any patch are dropped.

    Returns:
        X        : (K, 2048) float32 embeddings for the K pixels that matched.
        keep_idx : (K,) indices into rows/cols that matched (subset, order preserved).
    """
    # Pixel centres in GEDI CRS.
    px = gedi_transform * (cols + 0.5, rows + 0.5)
    xs = np.asarray(px[0], dtype=np.float64)
    ys = np.asarray(px[1], dtype=np.float64)

    # Reproject patch footprints into the GEDI CRS so distances are in its units.
    patches = patch_gdf.to_crs(gedi_crs)
    polys = patches.geometry.values

    tree = STRtree(polys)
    pts = gpd.points_from_xy(xs, ys)

    # Nearest patch polygon for each pixel (shapely 2.x).
    nearest_idx = tree.nearest(pts)  # (N,) index into polys for each point

    # Enforce the distance cap. If the GEDI CRS is geographic, convert the cap from
    # metres to degrees at the tile-centre latitude; otherwise units are metres.
    if CRS.from_user_input(gedi_crs).is_geographic:
        center_lat = float(np.mean(ys))
        m_per_deg = 111320.0 * max(np.cos(np.radians(center_lat)), 1e-3)
        cap = max_dist_m / m_per_deg
    else:
        cap = max_dist_m

    # Vectorised point->nearest-polygon distance (shapely 2.x).
    dist = shapely.distance(pts, polys[nearest_idx])
    keep = dist <= cap
    keep_idx = np.where(keep)[0]
    if len(keep_idx) == 0:
        return np.empty((0, embeddings.shape[1]), dtype=np.float32), keep_idx

    X = embeddings[nearest_idx[keep_idx]].astype(np.float32, copy=False)
    logger.info(
        "Spatial join: %d/%d valid pixels within %.0f m of a patch (dropped %d)",
        len(keep_idx), len(rows), max_dist_m, len(rows) - len(keep_idx),
    )
    return X, keep_idx


# ---------------------------------------------------------------------------
# GEDI loader (shared)
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
    gedi_data: np.ndarray,
    gedi_nodata: float,
    gedi_transform: Affine,
    gedi_crs,
    valid_rows: np.ndarray,
    valid_cols: np.ndarray,
    keep_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_patches: int,
    qa_dir: Path,
    logger: logging.Logger,
    keep_zeros: bool = False,
) -> dict:
    """Alignment + coverage checks for one processed tile (4-panel PNG + JSON)."""
    qa_dir.mkdir(parents=True, exist_ok=True)
    H, W = gedi_data.shape

    px_x_native = abs(float(gedi_transform.a))
    px_y_native = abs(float(gedi_transform.e))
    if CRS.from_user_input(gedi_crs).is_geographic:
        center_lat = float(gedi_transform.f) + float(gedi_transform.e) * H / 2.0
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
        px_size_x = px_x_native * m_per_deg_lon
        px_size_y = px_y_native * m_per_deg_lat
    else:
        px_size_x, px_size_y = px_x_native, px_y_native

    n_valid = len(valid_rows)
    n_covered = len(keep_idx)
    coverage = n_covered / max(1, n_valid)
    emb_norm = np.linalg.norm(X, axis=1) if len(X) else np.array([0.0])

    stats = {
        "healpix_id": int(healpix_id),
        "gedi_shape": [H, W],
        "gedi_crs": str(gedi_crs),
        "pixel_size_m": [round(px_size_x, 2), round(px_size_y, 2)],
        "n_majortom_patches": int(n_patches),
        "n_valid_gedi_pixels": int(n_valid),
        "n_covered": int(n_covered),
        "coverage_fraction": round(coverage, 4),
        "n_samples": int(len(y)),
        "n_channels": int(X.shape[1]) if X.ndim == 2 and len(X) else MAJORTOM_N_CHANNELS,
        "embedding_value_range": [round(float(X.min()), 4), round(float(X.max()), 4)] if len(X) else [0, 0],
        "embedding_l2_norm_mean": round(float(emb_norm.mean()), 4),
        "gedi_height_m_min": round(float(y.min()), 2) if len(y) else 0,
        "gedi_height_m_max": round(float(y.max()), 2) if len(y) else 0,
        "gedi_height_m_mean": round(float(y.mean()), 2) if len(y) else 0,
    }
    with open(qa_dir / f"tile_{healpix_id}_qa.json", "w") as f:
        json.dump(stats, f, indent=2)

    valid_mask = (
        (gedi_data != gedi_nodata) &
        (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
        (gedi_data < GEDI_MAX_HEIGHT_CM)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"QA — Tile {healpix_id} (Major TOM SSL4EO)  |  {H}×{W} px @ {px_size_x:.1f} m  |  "
        f"CRS: {gedi_crs}", fontsize=11,
    )

    # Panel 1: GEDI height
    ax = axes[0, 0]
    gedi_m = np.where(valid_mask, gedi_data / 100.0, np.nan)
    im = ax.imshow(gedi_m, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Height (m)", fraction=0.046)
    ax.set_title(f"GEDI height (m) — {n_valid:,} valid pixels")
    ax.axis("off")

    # Panel 2: which patch each covered pixel got (block structure should be visible)
    ax = axes[0, 1]
    patch_map = np.full((H, W), np.nan)
    covered_rows = valid_rows[keep_idx]
    covered_cols = valid_cols[keep_idx]
    # use mean embedding value as a per-patch-ish colour
    patch_map[covered_rows, covered_cols] = np.mean(X, axis=1) if len(X) else np.nan
    im = ax.imshow(patch_map, cmap="plasma", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean embedding value", fraction=0.046)
    ax.set_title(f"Assigned patch embedding (mean of {stats['n_channels']} ch)")
    ax.axis("off")

    # Panel 3: coverage
    ax = axes[1, 0]
    cov_map = np.full((H, W), np.nan)
    cov_map[valid_rows, valid_cols] = 0.0
    cov_map[covered_rows, covered_cols] = 1.0
    im = ax.imshow(cov_map, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Coverage: {coverage:.1%} of valid GEDI pixels\n"
                 f"green = has embedding, red = no nearby patch")
    ax.axis("off")

    # Panel 4: norm vs height
    ax = axes[1, 1]
    if len(y):
        ax.scatter(y, emb_norm, alpha=0.2, s=1, rasterized=True, color="steelblue")
    ax.set_xlabel("GEDI height (m)")
    ax.set_ylabel("Embedding L2 norm")
    ax.set_title(f"Embedding norm vs GEDI height (N={len(y):,})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    qa_png = qa_dir / f"tile_{healpix_id}_qa.png"
    plt.savefig(qa_png, dpi=120, bbox_inches="tight")
    plt.close()

    logger.info("Tile %s QA: coverage=%.1f%% patches=%d samples=%d",
                healpix_id, 100 * coverage, n_patches, len(y))
    print(f"  QA: shape=({H}×{W})  px={px_size_x:.1f}m  patches={n_patches}  "
          f"coverage={coverage:.1%}  → {qa_png.name}")
    return stats


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(
    healpix_id: int,
    year,
    gdf: gpd.GeoDataFrame,
    ds,
    gedi_dir: Path,
    output_dir: Path,
    overwrite: bool,
    buffer: float,
    max_dist_m: float,
    logger: logging.Logger,
    do_qa: bool = False,
    qa_dir: Path = None,
    test_fraction: float = None,
    keep_zeros: bool = False,
) -> dict:
    """Process one tile end-to-end; return a status dict."""
    is_test = test_fraction is not None
    suffix = ("_withzeros" if keep_zeros else "") + ("_test" if is_test else "")
    out_path = output_dir / f"healpix_{healpix_id}_float32{suffix}.npz"
    if not overwrite and out_path.exists():
        logger.info(f"Tile {healpix_id}: cached, skipping")
        return {"status": "skipped", "healpix_id": int(healpix_id)}

    t_start = time.time()
    try:
        # 1. Tile bounds in WGS84.
        shp_minx, shp_miny, shp_maxx, shp_maxy = get_tile_bounds(healpix_id, gdf)
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = bounds_to_crs(
            shp_minx, shp_miny, shp_maxx, shp_maxy, shp_crs, CRS.from_epsg(4326)
        )
        if is_test:
            f = float(test_fraction)
            lon_max = lon_min + (lon_max - lon_min) * f
            lat_min = lat_max - (lat_max - lat_min) * f
            print(f"  [test mode] fraction={f}  cropped WGS84 bounds: "
                  f"({lon_min:.5f}, {lat_min:.5f}, {lon_max:.5f}, {lat_max:.5f})")

        logger.info("Tile %s%s: WGS84 bounds (%.5f, %.5f, %.5f, %.5f)",
                    healpix_id, "[test]" if is_test else "",
                    lon_min, lat_min, lon_max, lat_max)

        # 2/3. Fetch + filter covering Major TOM patches.
        t_fetch = time.time()
        embeddings, patch_gdf = fetch_covering_patches(
            ds, lon_min, lat_min, lon_max, lat_max, year, buffer, logger
        )
        if patch_gdf is None or len(patch_gdf) == 0:
            logger.warning(f"Tile {healpix_id}: no Major TOM patches cover bounds")
            return {"status": "empty_subset", "healpix_id": int(healpix_id)}
        n_patches = len(patch_gdf)
        print(f"  Tile {healpix_id}: fetched {n_patches} Major TOM patch(es) in "
              f"{time.time()-t_fetch:.1f}s")

        # 4. Load GEDI target geometry.
        gedi_data, gedi_nodata, gedi_transform, gedi_crs, gedi_h, gedi_w = \
            load_gedi_tile(healpix_id, gedi_dir)
        if is_test:
            crop_h = max(1, int(round(gedi_h * test_fraction)))
            crop_w = max(1, int(round(gedi_w * test_fraction)))
            gedi_data = gedi_data[:crop_h, :crop_w]
            gedi_h, gedi_w = crop_h, crop_w

        # 5. Valid GEDI pixels.
        valid_mask = (
            (gedi_data != gedi_nodata) &
            (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
            (gedi_data < GEDI_MAX_HEIGHT_CM)
        )
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            logger.info(f"Tile {healpix_id}: no valid GEDI pixels")
            return {"status": "no_valid_pixels", "healpix_id": int(healpix_id)}

        # 6. Spatial join: pixel -> nearest patch embedding.
        t_join = time.time()
        X, keep_idx = assign_pixels_to_patches(
            rows, cols, gedi_transform, gedi_crs,
            patch_gdf, embeddings, max_dist_m, logger,
        )
        logger.info("Tile %s: spatial join in %.1fs", healpix_id, time.time() - t_join)
        if len(keep_idx) == 0:
            logger.warning(f"Tile {healpix_id}: no GEDI pixels within {max_dist_m} m of a patch")
            return {"status": "no_coverage", "healpix_id": int(healpix_id)}

        y = (gedi_data[rows[keep_idx], cols[keep_idx]] / 100.0).astype(np.float32)  # cm -> m
        n_total = len(rows)

        # 7. QA (optional)
        if do_qa:
            run_qa(
                healpix_id=healpix_id, gedi_data=gedi_data, gedi_nodata=gedi_nodata,
                gedi_transform=gedi_transform, gedi_crs=gedi_crs,
                valid_rows=rows, valid_cols=cols, keep_idx=keep_idx,
                X=X, y=y, n_patches=n_patches, qa_dir=qa_dir, logger=logger,
                keep_zeros=keep_zeros,
            )

        # 8. Save
        np.savez_compressed(out_path, X=X, y=y)
        elapsed = time.time() - t_start
        mem_str = (f"  mem={psutil.Process(os.getpid()).memory_info().rss/1e9:.1f}GB"
                   if _HAS_PSUTIL else "")
        logger.info(
            "Tile %s: %d samples, height [%.1f, %.1f] m, emb %d ch, %d patches, %.1fs%s",
            healpix_id, len(y), y.min(), y.max(), X.shape[1], n_patches, elapsed, mem_str,
        )
        print(f"  Tile {healpix_id}: {len(y):,} samples  {X.shape[1]} ch  "
              f"{n_patches} patches  {elapsed:.1f}s{mem_str}  → {out_path.name}")
        return {
            "status": "ok",
            "healpix_id": int(healpix_id),
            "n_samples": int(len(y)),
            "n_valid_gedi": int(n_total),
            "n_dropped_no_coverage": int(n_total - len(keep_idx)),
            "coverage_fraction": round(len(keep_idx) / max(1, n_total), 4),
            "n_patches": int(n_patches),
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
# --list-years utility
# ---------------------------------------------------------------------------

def list_years(ds, scan_files: int, logger: logging.Logger):
    """
    Print the acquisition-year histogram from the timestamp column.

    Scans the first `scan_files` parquet fragments (each ~50 grid rows). Pass a
    large number to scan the whole dataset (slow / lots of network). Only the
    timestamp column is read, so this is far cheaper than reading embeddings.
    """
    frags = list(ds.get_fragments())
    if scan_files and scan_files > 0:
        frags = frags[:scan_files]
    print(f"Scanning timestamps in {len(frags)} parquet fragment(s)...")

    counts: dict = {}
    for i, frag in enumerate(frags):
        tbl = frag.to_table(columns=["timestamp"])
        for ts in tbl.column("timestamp").to_pylist():
            yr = _year_of(ts)
            counts[yr] = counts.get(yr, 0) + 1
        if _HAS_TQDM is False:
            print(f"  scanned fragment {i+1}/{len(frags)}", end="\r")
    print()

    total = sum(counts.values())
    print("\nAcquisition-year distribution (Core-S2L1C-SSL4EO):")
    for yr in sorted(counts):
        n = counts[yr]
        print(f"  {yr}: {n:>12,}  ({100*n/max(1,total):5.1f}%)")
    yrs = [y for y in counts if y > 0]
    if yrs:
        print(f"\nRange: {min(yrs)}–{max(yrs)}  ({max(yrs)-min(yrs)+1} calendar years)  "
              f"over {total:,} patches scanned")
    print("Note: this dataset is single-temporal — one acquisition per location.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("majortom")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"majortom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            "Extract Major TOM (Core-S2L1C-SSL4EO) embeddings per healpix tile and "
            "save as (X, y) .npz for use with run_gedi_prediction_30m.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--healpix-id", type=int, default=None,
                        help="Single tile to process (omit for batch over --splits)")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter patches to this acquisition year (default: keep all; "
                             "dataset is single-temporal so a year filter drops most coverage)")
    parser.add_argument("--shapefile",
                        help="Healpix shapefile (.shp) with healpix_id and split columns")
    parser.add_argument("--gedi-dir",
                        help="Directory with GEDI tif files (healpix_*_gedi.tif)")
    parser.add_argument("--output-dir",
                        help="Directory to write per-tile healpix_*_float32.npz files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to include in batch mode")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process tiles that already have a cached .npz")
    parser.add_argument("--keep-zeros", action="store_true",
                        help="Keep GEDI height == 0 measurements (output *_float32_withzeros.npz)")
    parser.add_argument("--buffer", type=float, default=0.02,
                        help="Degrees to pad around tile bounds when selecting patches")
    parser.add_argument("--max-dist-m", type=float, default=DEFAULT_MAX_DIST_M,
                        help="Max distance (m) from a patch to assign its embedding to a pixel")
    parser.add_argument("--local-parquet", default=None,
                        help="Path to a local embeddings/ folder of part_*.parquet "
                             "(default: stream from the Hugging Face hub)")
    parser.add_argument("--check-only", action="store_true",
                        help="Report cache coverage and exit without processing")
    parser.add_argument("--list-years", action="store_true",
                        help="Print the acquisition-year histogram and exit")
    parser.add_argument("--year-scan-files", type=int, default=20,
                        help="With --list-years: number of parquet fragments to scan "
                             "(0 or large = whole dataset, slow)")
    parser.add_argument("--qa", action="store_true",
                        help="Save per-tile diagnostic plots and stats after processing")
    parser.add_argument("--qa-dir", default=None,
                        help="Directory for QA outputs (default: {output-dir}/qa/)")
    parser.add_argument("--test-fraction", type=float, default=None, metavar="F",
                        help="Crop each tile to F of its linear dims for fast testing "
                             "(output *_float32_test.npz). Use with --qa.")
    args = parser.parse_args()

    # --list-years can run without shapefile/gedi/output.
    if args.list_years:
        logger = logging.getLogger("majortom")
        logging.basicConfig(level=logging.WARNING)
        ds = open_majortom_dataset(local_parquet=args.local_parquet)
        list_years(ds, args.year_scan_files, logger)
        return

    # Otherwise these are required.
    missing = [a for a in ("shapefile", "gedi_dir", "output_dir")
               if getattr(args, a) is None]
    if missing:
        parser.error("the following arguments are required: "
                     + ", ".join("--" + m.replace("_", "-") for m in missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    qa_dir = Path(args.qa_dir) if args.qa_dir else output_dir / "qa"
    if args.qa:
        qa_dir.mkdir(parents=True, exist_ok=True)
        print(f"QA outputs will be saved to: {qa_dir}")

    gdf = gpd.read_file(args.shapefile)
    print(f"Loaded shapefile: {len(gdf)} tiles")
    if "split" in gdf.columns:
        print(f"Split distribution:\n{gdf['split'].value_counts().to_string()}")

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
        cache_suffix = "_withzeros" if args.keep_zeros else ""
        cached = [t for t in tile_ids
                  if (output_dir / f"healpix_{t}_float32{cache_suffix}.npz").exists()]
        missing = [t for t in tile_ids if t not in set(cached)]
        print(f"Cached : {len(cached)}/{len(tile_ids)}")
        if missing:
            print(f"Missing: {len(missing)} — first 20: {missing[:20]}")
        return

    print("\nOpening Major TOM dataset...")
    ds = open_majortom_dataset(local_parquet=args.local_parquet)

    gedi_dir = Path(args.gedi_dir)
    results = []
    iterator = tqdm(tile_ids, desc="tiles") if _HAS_TQDM else tile_ids
    for healpix_id in iterator:
        results.append(process_tile(
            healpix_id=healpix_id, year=args.year, gdf=gdf, ds=ds,
            gedi_dir=gedi_dir, output_dir=output_dir, overwrite=args.overwrite,
            buffer=args.buffer, max_dist_m=args.max_dist_m, logger=logger,
            do_qa=args.qa, qa_dir=qa_dir, test_fraction=args.test_fraction,
            keep_zeros=args.keep_zeros,
        ))

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
    print(f"Use with run_gedi_prediction_30m.py:\n"
          f"  --preprocessed-dir {output_dir} --embedding-format float32")

    summary = {
        "dataset": HF_REPO,
        "year": args.year,
        "n_tiles": len(tile_ids),
        "total_samples": total_samples,
        "status_counts": counts,
        "tiles": results,
    }
    summary_path = output_dir / f"majortom_summary_{args.year}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
