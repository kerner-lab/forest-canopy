#!/usr/bin/env python3
"""
Process TESSERA precomputed embeddings per healpix tile.

This is the TESSERA analogue of alphaearth_processing.py. It produces the exact
same output format — healpix_{id}_float32.npz containing (X, y) — so the output
directory can be passed straight to run_gedi_prediction_30m.py via
--preprocessed-dir --embedding-format float32.

For each tile:
  1. Read tile bounds from the healpix shapefile (assumed WGS84 / EPSG:4326).
  2. Ask the GeoTessera registry which precomputed 0.1deg x 0.1deg embedding
     tiles cover those bounds for the requested year.
  3. Fetch + dequantize each covering tile to float32 (int8 * scales, 128 ch).
     Each TESSERA tile arrives in its OWN native UTM CRS at 10 m.
  4. Reproject / resample every covering tile onto the GEDI raster grid for this
     healpix tile (same CRS, affine transform, height, width) and mosaic them.
  5. Extract (X, y) pixel pairs at valid non-zero GEDI locations.
  6. Save as healpix_{id}_float32.npz.

Single-tile usage (HPC array job):
    python tessera_processing.py \\
        --healpix-id 12345 --year 2024 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/tessera_cache

Batch usage (all tiles in a shapefile):
    python tessera_processing.py \\
        --year 2024 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/tessera_cache

Notes:
  - Requires the GeoTessera library:  pip install geotessera
    (see https://github.com/ucam-eo/geotessera). The first fetch downloads tiles
    into the GeoTessera cache (~/.cache/geotessera by default; override with
    --cache-dir).
  - TESSERA global coverage is currently 2024 at 10 m, 128-D embeddings. Earlier
    years are being back-filled; pass --year accordingly.
  - Embeddings are dequantized to float32 as final = int8 * scales. We request
    the high-level (already-dequantized) float arrays from GeoTessera and cast to
    float32; a runtime warning is logged if an integer dtype slips through (which
    would mean scales were not applied).
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
import rasterio.warp
from rasterio.warp import Resampling
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

GEDI_NODATA_DEFAULT = -9999
GEDI_MAX_HEIGHT_CM = 12000
TESSERA_N_CHANNELS = 128
# Year with current TESSERA global coverage at the time of writing.
TESSERA_DEFAULT_YEAR = 2024


# ---------------------------------------------------------------------------
# GeoTessera client
# ---------------------------------------------------------------------------

def open_geotessera(cache_dir: str = None, dataset_version: str = None):
    """Instantiate a GeoTessera client (lazy; tiles fetched on demand)."""
    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise ImportError(
            "geotessera is not installed. Install it with `pip install geotessera` "
            "(see https://github.com/ucam-eo/geotessera)."
        ) from exc

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if dataset_version:
        kwargs["dataset_version"] = dataset_version
    tic = time.time()
    gt = GeoTessera(**kwargs)
    print(f"Opened GeoTessera client in {time.time() - tic:.2f}s")
    return gt


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
# Fetch covering TESSERA tiles
# ---------------------------------------------------------------------------

def fetch_covering_tiles(gt, lon_min, lat_min, lon_max, lat_max, year, buffer, logger):
    """
    Return a list of (embedding, crs, transform) for every precomputed TESSERA
    tile that overlaps the (buffered) WGS84 bounds for `year`.

    embedding is cast to float32 (C, H, W) ready for reprojection.
    """
    blocks = list_covering_blocks(gt, lon_min, lat_min, lon_max, lat_max, year, buffer)
    return list(iter_covering_tiles(gt, blocks, logger))


def list_covering_blocks(gt, lon_min, lat_min, lon_max, lat_max, year, buffer):
    """Registry-only lookup of the blocks covering the buffered bounds. No downloads."""
    bounds = (lon_min - buffer, lat_min - buffer,
              lon_max + buffer, lat_max + buffer)
    return gt.registry.load_blocks_for_region(bounds=bounds, year=year) or []


def iter_covering_tiles(gt, blocks, logger):
    """Yield (embedding(C,H,W) float32, crs, transform) ONE BLOCK AT A TIME.

    Streaming matters: a single 0.1deg block is ~485 MB as float32 (1131 x 838 x 128),
    so materialising every block of a full healpix tile in a list first costs ~44 GB for
    a 90-block tile and gets the process oom-killed. Consuming this generator keeps only
    one block live at a time, so peak memory is the reprojection accumulators plus ~0.5 GB.
    """
    if not blocks:
        return
    for record in gt.fetch_embeddings(blocks):
        # fetch_embeddings yields (year, tile_lon, tile_lat, embedding, crs, transform)
        _, _, _, emb, crs, transform = record
        yield _to_float32_chw(emb, logger), CRS.from_user_input(crs), transform


def _to_float32_chw(emb: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Normalise a fetched embedding to (C, H, W) float32.

    GeoTessera returns (H, W, 128). The high-level fetch path is expected to
    return already-dequantized float values; if an integer dtype arrives we still
    cast it but warn loudly, because that means scales were not applied.
    """
    if np.issubdtype(emb.dtype, np.integer):
        logger.warning(
            "Fetched embedding has integer dtype %s — expected dequantized "
            "float. Values may be un-scaled (missing int8 * scales step).",
            emb.dtype,
        )
    arr = emb.astype(np.float32)
    # (H, W, C) -> (C, H, W); band dim is 128, far smaller than spatial dims
    if arr.ndim == 3 and arr.shape[-1] == TESSERA_N_CHANNELS:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 3 and arr.shape[0] != TESSERA_N_CHANNELS:
        # last resort: put the smallest axis first
        c_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, c_axis, 0)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Reprojection / mosaic onto the GEDI grid
# ---------------------------------------------------------------------------

def reproject_tiles_to_gedi(
    tiles: list,
    gedi_transform: Affine,
    gedi_crs,
    gedi_height: int,
    gedi_width: int,
) -> np.ndarray:
    """
    Reproject + mosaic a list of (embedding(C,H,W) float32, src_crs, src_transform)
    tiles onto the GEDI raster geometry using a FEATHERED WEIGHTED AVERAGE.

    Each TESSERA tile is reprojected (bilinear) independently, then blended by a
    per-tile coverage weight: the weight is 1 across a tile's interior and ramps
    to 0 at its edges (a reprojected unit mask). Overlapping tiles are combined as

        out = sum(value_i * weight_i) / sum(weight_i)

    so that near any tile's edge — where bilinear resampling has bled the 0 nodata
    fill into the data and the weight is small — the neighbouring tile (well
    inside its own interior, weight ~1) dominates. This eliminates the rectangular
    seams that a "first tile wins" merge leaves at every 0.1 deg tile boundary.

    0.0 is treated as nodata. Returns (C, gedi_height, gedi_width) float32.
    """
    # `tiles` may be a LIST or a GENERATOR (see iter_covering_tiles). Nothing below
    # indexes or re-iterates it, so streaming works unchanged — the accumulators are
    # allocated lazily from the first block's channel count.
    acc = wsum = None
    n_channels = TESSERA_N_CHANNELS

    for emb, src_crs, src_transform in tiles:
        if acc is None:
            n_channels = emb.shape[0]
            acc = np.zeros((n_channels, gedi_height, gedi_width), dtype=np.float32)
            wsum = np.zeros((gedi_height, gedi_width), dtype=np.float32)
        dst = np.zeros((n_channels, gedi_height, gedi_width), dtype=np.float32)
        rasterio.warp.reproject(
            source=emb,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=gedi_transform,
            dst_crs=gedi_crs,
            resampling=Resampling.bilinear,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

        # Feather weight: reproject a unit mask the same way. 1 in the tile's
        # interior, ramping to 0 at the tile edges (and 0 outside its extent).
        weight = np.zeros((gedi_height, gedi_width), dtype=np.float32)
        rasterio.warp.reproject(
            source=np.ones(emb.shape[1:], dtype=np.float32),
            destination=weight,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=gedi_transform,
            dst_crs=gedi_crs,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=0.0,
        )
        # Only weight pixels that actually received embedding data.
        weight *= np.any(dst != 0.0, axis=0)

        acc += dst * weight[None, :, :]
        wsum += weight
        del dst, weight, emb          # free before the next block is fetched

    if acc is None:                   # no blocks at all
        return np.zeros((TESSERA_N_CHANNELS, gedi_height, gedi_width), dtype=np.float32)

    mosaic = np.zeros((n_channels, gedi_height, gedi_width), dtype=np.float32)
    nz = wsum > 0
    mosaic[:, nz] = acc[:, nz] / wsum[nz]
    return mosaic


# ---------------------------------------------------------------------------
# GeoTIFF export (QGIS sanity check)
# ---------------------------------------------------------------------------

def save_embedding_geotiff(
    reproj: np.ndarray,        # (C, H, W) reprojected embeddings on the GEDI grid
    gedi_transform: Affine,
    gedi_crs,
    out_path: Path,
    bands=None,                # list of channel indices, or None for all 128
    logger: logging.Logger = None,
) -> Path:
    """
    Write the reprojected embedding (on the GEDI grid) as a georeferenced
    multiband float32 GeoTIFF for visual inspection in QGIS.

    This is the SAME array that gets sampled into X, carrying the GEDI tile's
    CRS + transform, so overlaying it on healpix_{id}_gedi.tif in QGIS confirms
    pixel-for-pixel alignment. nodata = 0 (uncovered pixels). In QGIS, render
    bands 1/2/3 as RGB for a quick look.
    """
    arr = reproj if bands is None else reproj[list(bands)]
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "crs": gedi_crs,
        "transform": gedi_transform,
        "nodata": 0.0,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))
        for i in range(count):
            ch = list(bands)[i] if bands is not None else i
            dst.set_band_description(i + 1, f"emb_{ch}")
    if logger:
        logger.info(f"Wrote embedding GeoTIFF ({count} bands) → {out_path}")
    print(f"  GeoTIFF: {count} bands  → {out_path.name}")
    return out_path


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
    keep_zeros: bool = False,
) -> dict:
    """
    Alignment and coverage checks for one processed tile.

    Saves tile_{id}_qa.png (4-panel figure) and tile_{id}_qa.json (stats).
    """
    qa_dir.mkdir(parents=True, exist_ok=True)
    C, H, W = reproj.shape

    if H != gedi_data.shape[0] or W != gedi_data.shape[1]:
        raise ValueError(
            f"Shape mismatch: reproj ({H}, {W}) vs GEDI {gedi_data.shape}"
        )

    # Pixel size in the GEDI CRS units. If geographic (EPSG:4326), convert the
    # degree spacing to approximate metres at the tile-centre latitude so the QA
    # reads a real ground resolution (~30 m) instead of a tiny degree value.
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

    valid_mask = (
        (gedi_data != gedi_nodata) &
        (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
        (gedi_data < GEDI_MAX_HEIGHT_CM)
    )
    n_valid = int(valid_mask.sum())

    emb_mean_map = np.mean(reproj, axis=0)          # (H, W) mean across channels
    nonzero_emb = emb_mean_map != 0.0
    n_covered = int(np.sum(nonzero_emb[valid_mask]))
    coverage = n_covered / max(1, n_valid)

    emb_norm = np.linalg.norm(X, axis=1)             # (N,) per-pixel L2 norm

    stats = {
        "healpix_id": int(healpix_id),
        "gedi_shape": [H, W],
        "reproj_shape": [C, H, W],
        "gedi_crs": str(gedi_crs),
        "pixel_size_m": [round(px_size_x, 2), round(px_size_y, 2)],
        "pixel_size_native": [round(px_x_native, 6), round(px_y_native, 6)],
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
    cov_map[valid_mask & nonzero_emb] = 1.0   # covered
    cov_map[valid_mask & ~nonzero_emb] = 0.0  # missing
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
    gt,
    gedi_dir: Path,
    output_dir: Path,
    overwrite: bool,
    buffer: float,
    logger: logging.Logger,
    do_qa: bool = False,
    qa_dir: Path = None,
    test_fraction: float = None,
    keep_zeros: bool = False,
    save_geotiff: bool = False,
    geotiff_bands=None,
    dense: bool = False,
) -> dict:
    """Process one tile end-to-end; return a status dict.

    When test_fraction is set (e.g. 0.1), the tile's bounding box is cropped to
    that fraction of its linear dimensions (so 0.1 → ~1% of the area), drastically
    reducing the number of TESSERA tiles fetched. Output is saved as
    healpix_{id}_float32_test.npz and is NOT suitable for training — use it only
    to verify alignment and QA plots quickly.
    """
    is_test = test_fraction is not None
    suffix = ("_withzeros" if keep_zeros else "") + ("_test" if is_test else "")
    if dense:
        # Dense mode writes the whole reprojected grid, GEDI-unfiltered, under the same
        # name/format the rest of the pipeline uses for dense embeddings — so
        # make_xy_from_dense.py --model tessera can do the (X,y) filtering later.
        out_path = output_dir / f"healpix_{healpix_id}_tessera{suffix}.npz"
    else:
        out_path = output_dir / f"healpix_{healpix_id}_float32{suffix}.npz"
    if not overwrite and out_path.exists():
        logger.info(f"Tile {healpix_id}: cached, skipping")
        return {"status": "skipped", "healpix_id": int(healpix_id)}

    t_start = time.time()
    try:
        # 1. Tile bounds in WGS84 (GeoTessera region queries are lon/lat).
        shp_minx, shp_miny, shp_maxx, shp_maxy = get_tile_bounds(healpix_id, gdf)
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = bounds_to_crs(
            shp_minx, shp_miny, shp_maxx, shp_maxy, shp_crs, CRS.from_epsg(4326)
        )

        # Crop to a fraction of the tile for fast local testing, CENTERED on the tile.
        # This used to keep the NW corner, which is the worst choice for a coastal tile:
        # TESSERA only covers land, so on a tile straddling the coast the NW corner is
        # often pure ocean and the test returns "no TESSERA tiles cover bounds" — looking
        # like a coverage failure when the tile is actually fine. The centre is far more
        # likely to be land and to carry GEDI shots.
        if is_test:
            f = float(test_fraction)
            lon_c, lat_c = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
            half_lon, half_lat = (lon_max - lon_min) * f / 2, (lat_max - lat_min) * f / 2
            lon_min, lon_max = lon_c - half_lon, lon_c + half_lon
            lat_min, lat_max = lat_c - half_lat, lat_c + half_lat
            print(
                f"  [test mode] fraction={f} (centred)  cropped WGS84 bounds: "
                f"({lon_min:.5f}, {lat_min:.5f}, {lon_max:.5f}, {lat_max:.5f})"
            )

        logger.info(
            f"Tile {healpix_id}{'[test]' if is_test else ''}: WGS84 bounds "
            f"({lon_min:.5f}, {lat_min:.5f}, {lon_max:.5f}, {lat_max:.5f})"
        )

        # 2/3. Find + fetch + dequantize covering TESSERA tiles (each in its UTM CRS).
        t_fetch = time.time()
        # Registry lookup first (cheap, no downloads) so an uncovered tile costs nothing,
        # then STREAM the blocks into the reprojection instead of listing them. At ~485 MB
        # per block as float32, listing a 90-block tile would need ~44 GB of RAM.
        blocks = list_covering_blocks(
            gt, lon_min, lat_min, lon_max, lat_max, year, buffer
        )
        if not blocks:
            logger.warning(f"Tile {healpix_id}: no TESSERA tiles cover bounds")
            return {"status": "empty_subset", "healpix_id": int(healpix_id)}
        n_blocks = len(blocks)
        print(f"  Tile {healpix_id}: streaming {n_blocks} TESSERA block(s)")
        tiles = iter_covering_tiles(gt, blocks, logger)   # generator, one block live

        # 4. Load GEDI target geometry.
        gedi_data, gedi_nodata, gedi_transform, gedi_crs, gedi_h, gedi_w = \
            load_gedi_tile(healpix_id, gedi_dir)

        # In test mode crop the GEDI raster to the same CENTRED sub-window used for the
        # lon/lat crop above. This must stay centred to match: cropping the NW corner
        # here while fetching TESSERA for the centre gives a reprojection target that
        # does not overlap the fetched data at all, and the tile reports
        # "no_valid_pixels" no matter how much GEDI it actually contains.
        # The transform has to be offset too, or the cropped grid is georeferenced as
        # though it were still the tile's NW corner.
        if is_test:
            crop_h = max(1, int(round(gedi_h * test_fraction)))
            crop_w = max(1, int(round(gedi_w * test_fraction)))
            r0 = max(0, (gedi_h - crop_h) // 2)
            c0 = max(0, (gedi_w - crop_w) // 2)
            gedi_data = gedi_data[r0:r0 + crop_h, c0:c0 + crop_w]
            gedi_transform = gedi_transform * Affine.translation(c0, r0)
            gedi_h, gedi_w = gedi_data.shape

        # 5. Reproject + mosaic embeddings onto the GEDI 30 m grid.
        t_reproj = time.time()
        reproj = reproject_tiles_to_gedi(
            tiles, gedi_transform, gedi_crs, gedi_h, gedi_w,
        )
        logger.info(
            f"Tile {healpix_id}: reprojected {n_blocks} tile(s) → {reproj.shape} "
            f"in {time.time() - t_reproj:.1f}s"
        )

        # Optional: write the reprojected embedding as a GeoTIFF for QGIS.
        if save_geotiff:
            tif_path = output_dir / f"healpix_{healpix_id}_tessera_emb{suffix}.tif"
            save_embedding_geotiff(
                reproj, gedi_transform, gedi_crs, tif_path,
                bands=geotiff_bands, logger=logger,
            )

        # 6a. Dense mode: save the whole grid, GEDI-unfiltered, and stop. Same layout as
        #     unified_embedding_pipeline.save_dense_npz — X (H,W,C), y (H,W) with NaN
        #     where GEDI is invalid, plus the geo metadata needed to map it back — so
        #     make_xy_from_dense.py --model tessera can extract (X,y) later without
        #     re-fetching anything from GeoTessera.
        if dense:
            y_grid = np.full(gedi_data.shape, np.nan, dtype=np.float32)
            gedi_valid = (
                (gedi_data != gedi_nodata) &
                (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
                (gedi_data < GEDI_MAX_HEIGHT_CM)
            )
            y_grid[gedi_valid] = gedi_data[gedi_valid] / 100.0     # cm -> m
            np.savez_compressed(
                out_path,
                X=np.ascontiguousarray(reproj.transpose(1, 2, 0)),  # (H, W, C)
                y=y_grid,
                grid_transform=np.asarray(tuple(gedi_transform)[:6], dtype=np.float64),
                grid_crs=np.asarray(CRS.from_user_input(gedi_crs).to_wkt()),
                grid_shape=np.asarray(y_grid.shape, dtype=np.int32),
            )
            mb = out_path.stat().st_size / 1e6
            covered = float(np.any(reproj != 0.0, axis=0).mean())
            print(f"  Tile {healpix_id}: dense -> {out_path.name}  "
                  f"X={(gedi_h, gedi_w, reproj.shape[0])}  {mb:.1f} MB  "
                  f"embedding coverage {100*covered:.1f}%  GEDI px {int(gedi_valid.sum()):,}")
            return {"status": "dense_ok", "healpix_id": int(healpix_id),
                    "shape": [gedi_h, gedi_w, int(reproj.shape[0])],
                    "size_mb": round(mb, 1),
                    "embedding_coverage": round(covered, 4),
                    "n_gedi_px": int(gedi_valid.sum()),
                    "elapsed_s": round(time.time() - t_start, 1)}

        # 6. Extract valid GEDI pixels. With keep_zeros, 0-height measurements are
        #    retained (gedi_data >= 0); otherwise only strictly-positive heights.
        valid_mask = (
            (gedi_data != gedi_nodata) &
            (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
            (gedi_data < GEDI_MAX_HEIGHT_CM)
        )
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            logger.info(f"Tile {healpix_id}: no valid GEDI pixels")
            return {"status": "no_valid_pixels", "healpix_id": int(healpix_id)}

        y = (gedi_data[rows, cols] / 100.0).astype(np.float32)  # cm → m
        X = reproj[:, rows, cols].T.astype(np.float32)  # (N, C)

        # Drop GEDI samples that fell on no-coverage (all-zero) embedding pixels.
        # A real 128-D embedding is never exactly zero in every channel, so an
        # all-zero row means that pixel had no TESSERA coverage. Keeping these
        # would feed zero-vector rows into the RF.
        n_total = len(y)
        covered_rows = np.any(X != 0.0, axis=1)
        n_dropped = int((~covered_rows).sum())
        if n_dropped:
            X = X[covered_rows]
            y = y[covered_rows]
            logger.info(
                f"Tile {healpix_id}: dropped {n_dropped}/{n_total} samples with "
                f"no embedding coverage ({100 * n_dropped / n_total:.1f}%)"
            )
        if len(y) == 0:
            logger.warning(
                f"Tile {healpix_id}: all {n_total} GEDI samples lack embedding "
                f"coverage (likely no TESSERA tiles for year {year} here)"
            )
            return {"status": "no_coverage", "healpix_id": int(healpix_id)}

        # 7. QA (optional)
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
                keep_zeros=keep_zeros,
            )

        # 8. Save
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
            "n_valid_gedi": int(n_total),
            "n_dropped_no_coverage": int(n_dropped),
            "coverage_fraction": round(len(y) / max(1, n_total), 4),
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
    logger = logging.getLogger("tessera")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_file = output_dir / f"tessera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            "Extract TESSERA precomputed embeddings per healpix tile and save as "
            "(X, y) .npz files for use with run_gedi_prediction_30m.py"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--healpix-id", type=int, default=None,
        help="Single tile to process (omit for batch mode over all tiles in --splits)",
    )
    parser.add_argument("--year", type=int, default=TESSERA_DEFAULT_YEAR,
                        help="Year of embeddings to extract (TESSERA global coverage: 2024)")
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
        "--keep-zeros", action="store_true",
        help=(
            "Keep GEDI height == 0 measurements in the output (gedi_data >= 0) "
            "instead of discarding them (gedi_data > 0). Output saved as "
            "healpix_*_float32_withzeros.npz so it does not collide with the "
            "default zero-filtered cache."
        ),
    )
    parser.add_argument(
        "--buffer", type=float, default=0.01,
        help="Degrees to pad around tile bounds when selecting covering TESSERA tiles",
    )
    parser.add_argument("--cache-dir", default=None,
                        help="GeoTessera download cache directory (default: library default)")
    parser.add_argument("--dataset-version", default=None,
                        help="GeoTessera dataset version (e.g. v1.1); default: library default")
    parser.add_argument("--check-only", action="store_true",
                        help="Report which output .npz files already exist in --output-dir "
                             "and exit. This is a LOCAL cache check only — it never contacts "
                             "GeoTessera. Use --check-coverage to ask whether TESSERA has "
                             "data for a year.")
    parser.add_argument("--check-coverage", action="store_true",
                        help="Ask the GeoTessera registry how many precomputed tiles cover "
                             "each healpix tile for --year, and exit without downloading "
                             "embeddings. This is the real answer to 'does TESSERA have "
                             "2019 over my AOI'. Registry query only; no embedding fetches.")
    parser.add_argument(
        "--dense", action="store_true",
        help="Save the FULL reprojected grid (GEDI-unfiltered) as "
             "healpix_<id>_tessera.npz instead of the (X,y) pairs. Same dense layout "
             "as the rest of the pipeline, so make_xy_from_dense.py --model tessera can "
             "do the filtering later. WARNING: 128 channels x the full GEDI grid is "
             "GB-scale per tile — check the reported size on one tile before batching.")
    parser.add_argument("--qa", action="store_true",
                        help="Save per-tile diagnostic plots and stats after processing")
    parser.add_argument("--qa-dir", default=None,
                        help="Directory for QA outputs (default: {output-dir}/qa/)")
    parser.add_argument(
        "--save-geotiff", action="store_true",
        help=(
            "Also write the reprojected embedding (on the GEDI grid) as a "
            "georeferenced multiband GeoTIFF, healpix_*_tessera_emb*.tif, for "
            "sanity-checking in QGIS (overlay it on the GEDI tif; render bands "
            "1/2/3 as RGB)."
        ),
    )
    parser.add_argument(
        "--geotiff-bands", nargs="+", type=int, default=None, metavar="B",
        help=(
            "Channel indices to include in the GeoTIFF (e.g. --geotiff-bands 0 1 2 "
            "for a 3-band RGB-able file). Default: all 128 bands."
        ),
    )
    parser.add_argument(
        "--test-fraction", type=float, default=None, metavar="F",
        help=(
            "Crop each tile's bounding box to F of its linear dimensions before "
            "fetching (e.g. 0.1 → ~1%% of area). Output saved as "
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
        cache_suffix = "_withzeros" if args.keep_zeros else ""
        cached = [t for t in tile_ids if (output_dir / f"healpix_{t}_float32{cache_suffix}.npz").exists()]
        missing = [t for t in tile_ids if t not in set(cached)]
        print(f"Cached : {len(cached)}/{len(tile_ids)}")
        if missing:
            print(f"Missing: {len(missing)} — first 20: {missing[:20]}")
        return

    # Open GeoTessera once (lazy; tiles fetched per-tile inside process_tile)
    print("\nOpening GeoTessera (lazy)...")
    gt = open_geotessera(cache_dir=args.cache_dir, dataset_version=args.dataset_version)

    if args.check_coverage:
        # Registry lookup only — load_blocks_for_region returns the identifiers of the
        # 0.1deg precomputed tiles covering the bounds, without fetching any embeddings.
        # A tile with 0 blocks has no TESSERA data for that year, full stop.
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        covered, uncovered = [], []
        print(f"\nQuerying TESSERA registry for year {args.year} over {len(tile_ids)} tiles...")
        for i, tid in enumerate(tile_ids, 1):
            lon_min, lat_min, lon_max, lat_max = bounds_to_crs(
                *get_tile_bounds(tid, gdf), shp_crs, CRS.from_epsg(4326))
            bounds = (lon_min - args.buffer, lat_min - args.buffer,
                      lon_max + args.buffer, lat_max + args.buffer)
            try:
                blocks = gt.registry.load_blocks_for_region(bounds=bounds, year=args.year)
                n = len(blocks) if blocks else 0
            except Exception as exc:
                print(f"  healpix_{tid}: registry error — {type(exc).__name__}: {exc}")
                n = 0
            (covered if n else uncovered).append((tid, n))
            print(f"[{i}/{len(tile_ids)}] healpix_{tid}: {n} TESSERA block(s)",
                  file=sys.stderr, flush=True)
        print(f"\nYear {args.year}: {len(covered)}/{len(tile_ids)} tiles have coverage")
        if covered:
            ns = [n for _t, n in covered]
            print(f"  blocks per covered tile: min={min(ns)} max={max(ns)} "
                  f"mean={sum(ns)/len(ns):.1f}")
        if uncovered:
            ids = [t for t, _n in uncovered]
            print(f"  NO coverage ({len(ids)}): {ids[:20]}"
                  f"{' ...' if len(ids) > 20 else ''}")
        return

    # Process tiles
    gedi_dir = Path(args.gedi_dir)
    results = []
    iterator = tqdm(tile_ids, desc="tiles") if _HAS_TQDM else tile_ids

    for healpix_id in iterator:
        result = process_tile(
            healpix_id=healpix_id,
            year=args.year,
            gdf=gdf,
            gt=gt,
            gedi_dir=gedi_dir,
            output_dir=output_dir,
            overwrite=args.overwrite,
            buffer=args.buffer,
            logger=logger,
            dense=args.dense,
            do_qa=args.qa,
            qa_dir=qa_dir,
            test_fraction=args.test_fraction,
            keep_zeros=args.keep_zeros,
            save_geotiff=args.save_geotiff,
            geotiff_bands=args.geotiff_bands,
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
    summary_path = output_dir / f"tessera_summary_{args.year}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
