#!/usr/bin/env python3
"""
Process Presto embeddings per healpix tile, with imagery fetched for free via rslearn.

This is the Presto analogue of tessera_processing.py / alphaearth_processing.py.
 It produces the exact same output format —
healpix_{id}_float32.npz containing (X, y) — so the output directory can be passed
straight to run_gedi_prediction_30m.py via --preprocessed-dir --embedding-format float32.

  Presto: https://github.com/nasaharvest/presto  (pip install presto-classification
          OR pip install -e . from the repo; importable as `presto`)
  rslearn: https://github.com/allenai/rslearn  (free Sentinel-2 via
          rslearn.data_sources.gcp_public_data.Sentinel2 — no Earth Engine account)

How Presto differs from TESSERA / Alpha Earth
---------------------------------------------------------
Presto is NOT a precomputed embedding archive — it is a *model* you run yourself.
It consumes a PER-PIXEL monthly TIME SERIES (Presto's native input: 12 monthly
Sentinel-2 observations of one pixel, optionally + S1 / ERA5 / SRTM) and outputs a
128-D vector for that pixel. So the pipeline here is:

  1. rslearn fetches a 12-month Sentinel-2 mosaic stack over each healpix tile
     (UTM, 10 m), one monthly composite per timestep. Periods with no usable scene
     are placed by date (not by position) and masked, so a gap never shifts the
     months after it — see read_period_slots.
  2. For each VALID GEDI pixel we read its 12-month S2 time series (nearest 10 m
     S2 pixel), build the Presto input with the official
     presto.construct_single_presto_input, and run the encoder -> 128-D embedding.
  3. Extract (X, y) and save healpix_{id}_float32.npz.

We embed only at valid GEDI pixel locations (not the whole 10 m grid), which keeps
the number of Presto forward passes tractable.

Temporal coverage
-----------------
Presto consumes whatever year you ask rslearn to fetch — Sentinel-2 L1C is available
globally from ~2017 onward, so pick --year to match the GEDI epoch (GEDI mission:
2019-2023). The embedding always covers a 12-month window starting at --start-month
of --year.

Single-tile usage (HPC array job):
    python presto_processing.py \\
        --healpix-id 12345 --year 2021 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/presto_cache \\
        --rslearn-root /scratch/presto_rslearn

Use already-materialized imagery (skip rslearn):
    python presto_processing.py --healpix-id 12345 --year 2021 \\
        --shapefile ... --gedi-dir ... --output-dir ... \\
        --imagery-dir /path/with/healpix_12345_s2_{month}.tif

Notes:
  - Requires: pip install presto-classification torch rslearn rasterio geopandas pyproj
  - rslearn data-source auth: the GCP public Sentinel-2 bucket is anonymous; set
    GOOGLE_APPLICATION_CREDENTIALS only if rslearn asks for it in your environment.
  - The first Presto.load_pretrained() downloads the ~7 MB default checkpoint.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine
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
PRESTO_N_CHANNELS = 128
PRESTO_NUM_TIMESTEPS = 12  # Presto's native monthly cadence (1 year)

# Sentinel-2 bands to materialize via rslearn, paired with the names Presto expects.
# (rslearn gcp_public_data.Sentinel2 names them B01..B12, B8A; Presto wants B1..B12, B8A.)
# B1/B9/B10 are dropped by Presto internally (REMOVED_BANDS / B9 unused), so we skip them.
RSLEARN_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
PRESTO_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_BANDSET_NAME = "_".join(RSLEARN_S2_BANDS)

RSLEARN_LAYER = "sentinel2"
RSLEARN_GROUP = "presto"

# Sentinel-1 (optional, --with-s1): VV/VH from Planetary Computer, materialized as 12
# monthly mosaics alongside S2 on the SAME window grid. Presto's built-in "S1" channel
# group expects VV/VH in decibels; PC's Sentinel-1 RTC is linear gamma0, so we convert
# (see s1_to_presto_db). Element84 has no S1, so S1 always comes from PC even when
# --s2-source element84 (mixed per-layer sources are fine in one rslearn dataset).
RSLEARN_S1_LAYER = "2"
RSLEARN_S1_BANDS = ["vv", "vh"]
S1_BANDSET_NAME = "_".join(RSLEARN_S1_BANDS)
PRESTO_S1_BANDS = ["VV", "VH"]  # names construct_single_presto_input expects


# ---------------------------------------------------------------------------
# Tile bounds (shared with the TESSERA / Alpha Earth scripts)
# ---------------------------------------------------------------------------

def get_tile_bounds(healpix_id: int, gdf: gpd.GeoDataFrame) -> tuple:
    """Return (minx, miny, maxx, maxy) of the tile in the shapefile's CRS."""
    row = gdf[gdf["healpix_id"] == healpix_id]
    if len(row) == 0:
        raise ValueError(f"Tile {healpix_id} not found in shapefile")
    return row.geometry.iloc[0].bounds


def bounds_to_crs(minx, miny, maxx, maxy, src_crs, dst_crs) -> tuple:
    """Transform a bounding box from src_crs to dst_crs by reprojecting all four corners."""
    if CRS.from_user_input(src_crs) == CRS.from_user_input(dst_crs):
        return minx, miny, maxx, maxy
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs, ys = t.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    return min(xs), min(ys), max(xs), max(ys)


def crop_bounds_centered(minx, miny, maxx, maxy, fraction: float) -> tuple:
    """Shrink a bbox to a centered `fraction` of its width and height.

    fraction=0.1 keeps the central 10% of each axis -> ~1% of the area. Used by
    --subset-fraction to test the full pipeline locally over a small sub-window.
    """
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    hw = (maxx - minx) * fraction / 2.0
    hh = (maxy - miny) * fraction / 2.0
    return cx - hw, cy - hh, cx + hw, cy + hh


# ---------------------------------------------------------------------------
# Tile-level subsetting (mirror of run_gedi_prediction_30m.py)
# ---------------------------------------------------------------------------

def subset_tiles_for_split(tiles: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministically pick `fraction` of a split's tiles.

    This is a byte-for-byte mirror of the subsetting in
    run_gedi_prediction_30m.py::prepare_split_data so the SAME tiles are selected
    for a given (split, fraction, seed). Because np.random.default_rng(seed).choice
    draws positional indices that depend only on len(tiles) and the seed, identical
    selections require identical input arrays — i.e. the same shapefile, filtered
    with gdf[gdf["split"] == split_name]["healpix_id"].values (same order). Do NOT
    pre-sort or otherwise reorder `tiles` before calling this.

    Returns the chosen healpix_ids sorted ascending (matching the predictor).
    """
    rng = np.random.default_rng(seed)
    n_subset = max(1, int(round(len(tiles) * fraction)))
    chosen = rng.choice(tiles, size=n_subset, replace=False)
    return np.sort(chosen)


# ---------------------------------------------------------------------------
# rslearn: fetch a 12-month Sentinel-2 stack for a tile
# ---------------------------------------------------------------------------

def write_rslearn_config(root: Path, source: str = "pc", with_s1: bool = False) -> Path:
    """
    Write a minimal rslearn dataset config.json that pulls a 12-month Sentinel-2
    mosaic for each tile.

    The window's 12-month time range (set by `add_windows --start/--end`) is split into
    PRESTO_NUM_TIMESTEPS 30-day sub-periods via period_duration="30d" + max_matches=12,
    yielding 12 monthly composites materialized as sentinel2, sentinel2.1, ... sentinel2.11.

    NOTE on durations: rslearn parses these strings with pytimeparse, which reads "M" as
    MINUTES, not months ("1M" -> 60s, "12M" -> 12min). Use day units ("30d"). Also do NOT
    set the data_source "duration" key — it would override the request end to start+duration
    and, with a months-style string, silently collapse the search to minutes (no matches).

    source:
      "pc" (default) — Microsoft Planetary Computer Sentinel-2 L2A. Free, no billing,
        anonymous (needs the `planetary_computer` pip package). Direct-materialize.
      "element84" — Sentinel-2 L2A COGs on AWS Open Data via Element84 Earth Search.
        Anonymous, no account/billing, no BigQuery. Direct-materialize.
      "gcp" — gcp_public_data.Sentinel2 (L1C). Imagery is free, but the scene index is
        queried via BigQuery, which requires a GCP project with billing enabled.
    """
    query_config = {
        # MOSAIC + period_duration = one mosaic per 30-day period -> 12 monthly layer
        # groups. (This is the modern equivalent of the now-deprecated PER_PERIOD_MOSAIC;
        # rslearn warns "Use SpaceMode.MOSAIC with period_duration instead".)
        "space_mode": "MOSAIC",
        "max_matches": PRESTO_NUM_TIMESTEPS,
        "period_duration": "30d",  # ~1 month; pytimeparse has no month unit
        # Return the 12 monthly groups oldest-first (chronological). rslearn still
        # defaults this to True (most-recent-first); without it, find_monthly_geotiffs
        # would build the stack reversed while embed_pixels labels timestep t as
        # month0+t -> every embedding's months would be inverted (Jan imagery told to
        # Presto as Dec, etc.). False fixes the order AND silences the deprecation warning.
        "per_period_mosaic_reverse_time_order": False,
    }
    if source == "pc":
        data_source = {
            "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
            # sort_by=eo:cloud_cover -> each period picks its least-cloudy scene
            # (matches the OLMo config that pulls S2 over every tile).
            "init_args": {"harmonize": True, "sort_by": "eo:cloud_cover"},
            "query_config": query_config,
            "ingest": False,  # COGs are read directly during materialize
        }
    elif source == "element84":
        data_source = {
            "class_path": "rslearn.data_sources.aws_sentinel2_element84.Sentinel2",
            # No cache_dir: it's deprecated/unused in current rslearn (item data is
            # passed straight to materialization) and warns/errors if set. COGs on
            # s3://sentinel-cogs are read directly during materialize (ingest=False).
            # Bands map via ASSET_BANDS (blue->B02 ... swir22->B12, nir08->B8A) and
            # are already harmonized, so no harmonize arg is needed.
            "init_args": {"sort_by": "eo:cloud_cover"},
            "query_config": query_config,
            "ingest": False,
        }
    elif source == "gcp":
        data_source = {
            "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
            "init_args": {"index_cache_dir": str(root / "cache" / "s2_index"),
                          "harmonize": True},
            "query_config": query_config,
            "ingest": True,
        }
    else:
        raise ValueError(f"Unknown --s2-source {source!r} (expected pc, element84, or gcp)")

    layers = {
        RSLEARN_LAYER: {
            "type": "raster",
            "band_sets": [
                {
                    "dtype": "uint16",
                    "bands": RSLEARN_S2_BANDS,
                }
            ],
            "data_source": data_source,
            "resampling_method": "bilinear",
            "compositing_method": "FIRST_VALID",
        }
    }
    if with_s1:
        # Sentinel-1 VV/VH from Planetary Computer (IW mode, dual-pol), 12 monthly
        # mosaics on the same chronological schedule as S2 (same query_config, so the
        # per_period_mosaic_reverse_time_order=False fix applies to S1 too).
        layers[RSLEARN_S1_LAYER] = {
            "type": "raster",
            "band_sets": [
                {
                    "dtype": "float32",
                    "bands": RSLEARN_S1_BANDS,
                    "nodata_vals": [-32768, -32768],
                }
            ],
            "data_source": {
                "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
                "init_args": {
                    "query": {
                        "sar:instrument_mode": {"eq": "IW"},
                        "sar:polarizations": {"eq": ["VV", "VH"]},
                    }
                },
                "query_config": query_config,
                "ingest": False,
            },
            "resampling_method": "bilinear",
            "compositing_method": "FIRST_VALID",
        }
    config = {"layers": layers}
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "config.json"
    new_text = json.dumps(config, indent=2)
    # Write only when the content differs, so we never reuse a stale config (e.g. a
    # previous --s2-source or an old period_duration) but also avoid pointless rewrites
    # that could race across concurrent HPC array jobs. Replace atomically.
    if cfg_path.exists() and cfg_path.read_text() == new_text:
        return cfg_path
    tmp = cfg_path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(new_text)
    os.replace(tmp, cfg_path)
    return cfg_path


# Set by main() from --verbose. When True, _run streams rslearn's stdout/stderr
# live (so prepare/materialize progress shows in the job log) instead of capturing
# it silently and only surfacing it on failure.
_STREAM_RSLEARN = False


def _run(cmd: list, logger: logging.Logger):
    logger.info("rslearn: %s", " ".join(cmd))
    if not _STREAM_RSLEARN:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
            )
        return proc.stdout

    # Verbose: stream output line-by-line to our stdout (the job .out) and the log,
    # while keeping a tail for the error message if the command fails.
    print(f"  $ {' '.join(cmd)}", flush=True)
    from collections import deque
    tail = deque(maxlen=40)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(f"  [rslearn] {line}", flush=True)
        logger.info("rslearn> %s", line)
        tail.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Command failed ({rc}): {' '.join(cmd)}\n"
            f"OUTPUT (tail):\n" + "\n".join(tail)
        )
    return "\n".join(tail)


def materialize_tile_rslearn(
    healpix_id: int,
    lon_min, lat_min, lon_max, lat_max,
    year: int,
    start_month: int,
    root: Path,
    logger: logging.Logger,
    workers: int = 4,
    source: str = "pc",
    with_s1: bool = False,
) -> Path:
    """
    Add a window for the tile, then prepare/(ingest)/materialize a 12-month S2 stack
    (and, with with_s1, a 12-month S1 stack on the same grid).

    Returns the window directory: root/windows/<group>/healpix_<id>/
    """
    # Always refresh the config (it only rewrites when content changed) so a stale
    # config from a previous --s2-source / --with-s1 / period_duration is never reused.
    write_rslearn_config(root, source=source, with_s1=with_s1)

    win_name = f"healpix_{healpix_id}"
    win_dir = root / "windows" / RSLEARN_GROUP / win_name
    start = f"{year}-{start_month:02d}-01T00:00:00+00:00"
    # 12 months from start: start of the same month one year later.
    end = f"{year + 1}-{start_month:02d}-01T00:00:00+00:00"

    add_cmd = [
        "rslearn", "dataset", "add_windows",
        "--root", str(root),
        "--group", RSLEARN_GROUP,
        # Use --box=... (not "--box", value): negative longitudes start with '-' and
        # argparse would otherwise treat the value as a flag ("expected one argument").
        f"--box={lon_min},{lat_min},{lon_max},{lat_max}",
        "--src_crs", "EPSG:4326",
        "--utm",
        "--resolution", "10",
        "--start", start,
        "--end", end,
        "--name", win_name,
    ]
    _run(add_cmd, logger)

    # Planetary Computer's STAC search and blob downloads return transient 5xx
    # (e.g. "504 Gateway Timeout"), especially when several array jobs hit the
    # endpoint at once. Retry with backoff. AND pass --no-ignore-errors so that a
    # failure surviving the retries makes the step exit non-zero: otherwise
    # rslearn's default --ignore-errors logs the 504, marks the window failed, and
    # exits 0, so the run silently falls through to a false "no imagery" with a
    # clean exit code (the tile then looks done and never gets resubmitted).
    retry_args = ["--retry-max-attempts", "5", "--retry-backoff-seconds", "5",
                  "--no-ignore-errors"]

    # PC and Element84 read COGs directly during materialize -> no ingest step.
    steps = ("prepare", "ingest", "materialize") if source == "gcp" \
        else ("prepare", "materialize")
    for step in steps:
        _run(["rslearn", "dataset", step, "--root", str(root),
              "--group", RSLEARN_GROUP, "--workers", str(workers),
              *retry_args], logger)

    if not win_dir.exists():
        raise FileNotFoundError(f"rslearn window not found after materialize: {win_dir}")
    return win_dir


def read_period_slots(win_dir: Path, layer_name: str, logger: logging.Logger = None,
                      n_slots: int = PRESTO_NUM_TIMESTEPS, period_days: int = 30) -> dict:
    """
    Map each materialized layer-group index -> its chronological slot (0..n_slots-1).

    rslearn drops periods that matched no scene instead of leaving a
    hole — in data_sources/utils.py the per-period loop does `if period_groups:
    groups.append(...)`. So on a tile where (say) February had no usable scene you get
    11 groups numbered sentinel2, sentinel2.1 ... sentinel2.10 with the gap compacted
    out, and layer group i is NOT month i. Feeding that compacted run to Presto as a
    consecutive series (embed_pixels passes month=month0, which Presto expands to
    month0, month0+1, ...) mislabels every timestep after the gap.

    The true period is recoverable: items.json records the scene timestamps of each
    group, and metadata.json records the window time range. rslearn anchors periods at
    the window END and walks backwards (utils.py: `period_end = geometry.time_range[1]`,
    then `period_start = period_end - period_duration`), so the period covering
    [end-(k+1)*P, end-k*P) is the k-th from the end and its chronological slot is
    n_slots-1-k.

    Returns {} if the metadata isn't readable, so callers can fall back to positional
    indexing (the old behaviour) rather than fail.
    """
    try:
        meta = json.loads((win_dir / "metadata.json").read_text())
        win_end = datetime.fromisoformat(meta["time_range"][1])
        layer_datas = json.loads((win_dir / "items.json").read_text())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if logger:
            logger.warning("Could not read period metadata in %s (%s); falling back to "
                           "positional month indexing", win_dir, exc)
        return {}

    period = period_days * 86400.0
    slots = {}
    for layer_data in layer_datas:
        if layer_data.get("layer_name") != layer_name:
            continue
        for i, group in enumerate(layer_data.get("serialized_item_groups", [])):
            times = []
            for item in group:
                try:
                    times.append(datetime.fromisoformat(
                        item["geometry"]["time_range"][0]))
                except (KeyError, TypeError, ValueError):
                    continue
            if not times:
                continue
            # Any scene in the group falls inside that group's period; use the earliest.
            k = int((win_end - min(times)).total_seconds() // period)
            slot = n_slots - 1 - k
            if 0 <= slot < n_slots:
                slots[i] = slot
            elif logger:
                logger.warning("Layer %s group %d resolves to out-of-range slot %d; "
                               "leaving it positional", layer_name, i, slot)
    return slots


def find_monthly_geotiffs(win_dir: Path, layer_name: str = RSLEARN_LAYER,
                          bandset: str = S2_BANDSET_NAME,
                          logger: logging.Logger = None) -> tuple:
    """
    Return (paths, slots) for a materialized window: the monthly GeoTIFFs that exist and,
    for each, its chronological slot (0..11) inside the 12-month window.

    rslearn writes layer groups <layer>, <layer>.1, ... each as
    layers/<layer>/<bandset>/geotiff.tif. Group index != month on tiles with partial
    coverage, so the slot comes from read_period_slots (see its docstring); when that
    metadata is unavailable we fall back to the group index.
    """
    slot_map = read_period_slots(win_dir, layer_name, logger)
    paths, slots = [], []
    for i in range(PRESTO_NUM_TIMESTEPS):
        layer = layer_name if i == 0 else f"{layer_name}.{i}"
        p = win_dir / "layers" / layer / bandset / "geotiff.tif"
        if p.exists():
            paths.append(p)
            slots.append(slot_map.get(i, i))
    # Two groups resolving to one slot means the period maths disagrees with what was
    # materialized; positional order is the safer fallback there.
    if len(set(slots)) != len(slots):
        if logger:
            logger.warning("Duplicate period slots for layer %s in %s; falling back to "
                           "positional month indexing", layer_name, win_dir)
        slots = list(range(len(paths)))
    return paths, slots


def find_s1_monthly_geotiffs(win_dir: Path, logger: logging.Logger = None) -> tuple:
    """(paths, slots) for the monthly Sentinel-1 VV/VH geotiffs — see find_monthly_geotiffs.

    S1 is slotted independently of S2, so a month missing from one but not the other no
    longer shifts either series.
    """
    return find_monthly_geotiffs(win_dir, RSLEARN_S1_LAYER, S1_BANDSET_NAME, logger)


def s1_to_presto_db(s1_stack: np.ndarray, units: str = "linear") -> np.ndarray:
    """Prepare a raw S1 stack (T, 2, H, W) for Presto's dB-scale S1 channel group.

    PC Sentinel-1 RTC is linear gamma0 (units="linear") -> convert to dB via 10*log10.
    If your source already provides dB, pass units="db" (pass-through). Nodata (the
    -32768 fill, and non-positive values that can't be log'd) are replaced with the
    per-band mean so they normalize to ~0 (neutral) inside Presto rather than corrupting
    the embedding. NOTE: verify your S1 units by value range — linear gamma0 ~ 0..1,
    dB ~ -30..+5.
    """
    s1 = s1_stack.astype(np.float32)
    s1 = np.where(s1 <= -1000.0, np.nan, s1)  # nodata fill
    if units == "linear":
        with np.errstate(divide="ignore", invalid="ignore"):
            s1 = np.where(s1 > 0, 10.0 * np.log10(s1), np.nan)
    for b in range(s1.shape[1]):
        band = s1[:, b]
        m = np.nanmean(band)
        band[~np.isfinite(band)] = m if np.isfinite(m) else 0.0
        s1[:, b] = band
    return s1


def find_imagery_dir_geotiffs(imagery_dir: Path, healpix_id: int) -> tuple:
    """Fallback: read pre-materialized healpix_<id>_s2_<month>.tif (month 0..11).

    Returns (paths, slots). Here the month is in the filename, so the slot is that
    month index directly — a missing month leaves its slot empty rather than shifting
    the ones after it.
    """
    paths, slots = [], []
    for i in range(PRESTO_NUM_TIMESTEPS):
        for pat in (f"healpix_{healpix_id}_s2_{i}.tif", f"healpix_{healpix_id}_s2_{i:02d}.tif"):
            p = imagery_dir / pat
            if p.exists():
                paths.append(p)
                slots.append(i)
                break
    return paths, slots


# ---------------------------------------------------------------------------
# Load the monthly S2 stack
# ---------------------------------------------------------------------------

def load_s2_stack(geotiff_paths: list, slots: list, n_bands_expected: int = None,
                  n_slots: int = PRESTO_NUM_TIMESTEPS):
    """
    Read monthly GeoTIFFs into a fixed (n_slots, n_bands, H, W) float32 stack, placing
    each file at its own slot rather than at its position in the list.

    `slots[i]` is the 0..n_slots-1 month of `geotiff_paths[i]`. Slots with no file stay
    zero and are reported False in `present`, so the caller can mask them for Presto
    instead of silently passing a compacted, month-shifted series.

    All months must share the same grid (rslearn guarantees this within one window).
    Returns (stack, present, transform, crs, height, width).
    """
    # Preallocate and fill in place. The obvious version — collect each month into a list
    # and np.stack() it — holds the list AND the stacked copy live at the same moment, so
    # peak RSS is 2x the final array. At 12 months x 12 bands x a 10 m healpix window that
    # doubling is tens of GB and gets the job oom-killed by the slurm cgroup. Filling a
    # preallocated array keeps the peak at 1x plus one month's read buffer.
    stack, transform, crs, H, W = None, None, None, None, None
    present = np.zeros(n_slots, dtype=bool)
    for p, slot in zip(geotiff_paths, slots):
        with rasterio.open(p) as src:
            if stack is None:
                transform, crs, H, W = src.transform, src.crs, src.height, src.width
                if n_bands_expected is not None and src.count != n_bands_expected:
                    raise ValueError(f"{p.name}: expected {n_bands_expected} bands, "
                                     f"got {src.count}")
                # Zero-filled: an unmaterialized month reads as 0 reflectance, which
                # normalizes to 0 and is masked out via `present` anyway.
                stack = np.zeros((n_slots, src.count, H, W), dtype=np.float32)
            elif (src.height, src.width) != (H, W):
                raise ValueError(f"Month grid mismatch in {p.name}: "
                                 f"{(src.height, src.width)} vs {(H, W)}")
            elif src.count != stack.shape[1]:
                raise ValueError(f"Month band-count mismatch in {p.name}: "
                                 f"{src.count} vs {stack.shape[1]}")
            stack[slot] = src.read()       # (bands, H, W), cast to float32 on assignment
            present[slot] = True
    return stack, present, transform, crs, H, W


# ---------------------------------------------------------------------------
# Presto encoding
# ---------------------------------------------------------------------------

def load_presto():
    """Load the pretrained Presto encoder (eval mode, CPU/GPU)."""
    try:
        import torch
        from presto import Presto
    except ImportError as exc:
        raise ImportError(
            "presto and torch are required. Install with "
            "`pip install presto-classification torch` (or pip install -e . from the "
            "nasaharvest/presto repo)."
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Presto.load_pretrained()
    encoder = model.encoder.to(device).eval()
    return encoder, device


def embed_pixels(
    s2_stack: np.ndarray,      # (T, n_bands, H, W) float32 reflectance DN (0..1e4)
    s2_band_names: list,       # Presto band names aligned with axis 1 of s2_stack
    rows: np.ndarray,          # (N,) S2-grid row indices of pixels to embed
    cols: np.ndarray,          # (N,) S2-grid col indices
    latlons: np.ndarray,       # (N, 2) lat, lon of each pixel
    start_month: int,          # calendar month of timestep 0 (1..12)
    encoder,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    s1_stack: np.ndarray = None,   # (T, 2, H, W) dB-scale VV/VH, or None for S2-only
    s1_band_names: list = None,    # e.g. PRESTO_S1_BANDS (["VV","VH"])
    s2_present: np.ndarray = None, # (T,) bool: which timesteps have real S2 (None = all)
    s1_present: np.ndarray = None, # (T,) bool: same for S1
) -> np.ndarray:
    """Run Presto over N pixels -> (N, 128) float32 embeddings.

    If s1_stack is given, each pixel's VV/VH time series is fed alongside S2 so Presto's
    S1 channel group is active; otherwise S1 is masked (S2-only) exactly as before.

    Timesteps flagged False in s2_present / s1_present are zero in the stack and are
    masked out per channel group. This matters because construct_single_presto_input
    sets mask=0 for EVERY timestep of any band you hand it, so without the override an
    unmaterialized month would be presented to Presto as a real all-zero observation.
    S1 and S2 are masked independently, so a month missing from only one of them costs
    just that channel group.
    """
    import torch
    from presto import construct_single_presto_input
    # Column layout of the (T, 17) x/mask that construct_single_presto_input returns.
    from presto.dataops.pipelines.s1_s2_era5_srtm import NORMED_BANDS

    N = len(rows)
    out = np.zeros((N, PRESTO_N_CHANNELS), dtype=np.float32)
    month0 = (start_month - 1) % 12  # Presto month index is 0-based (Jan=0)
    T = s2_stack.shape[0]

    # Build the mask override once — it is identical for every pixel in the tile.
    normed = list(NORMED_BANDS)
    s2_cols = [normed.index(b) for b in s2_band_names if b in normed]
    if "B8" in s2_band_names and "B4" in s2_band_names:
        # construct_single_presto_input unmasks the derived NDVI when B8 and B4 are
        # both supplied; a missing month has no NDVI either.
        s2_cols.append(normed.index("NDVI"))
    s1_cols = [normed.index(b) for b in (s1_band_names or []) if b in normed]

    mask_override = torch.zeros(T, len(normed))
    for t in range(T):
        if s2_present is not None and not s2_present[t]:
            mask_override[t, s2_cols] = 1.0
        if s1_stack is not None and s1_present is not None and not s1_present[t]:
            mask_override[t, s1_cols] = 1.0
    n_masked = int((mask_override.sum(dim=1) > 0).sum())
    if n_masked:
        logger.info("Presto: masking %d/%d timesteps with no imagery", n_masked, T)

    # Per-pixel time series: (N, T, n_bands)
    ts = np.transpose(s2_stack[:, :, rows, cols], (2, 0, 1)).astype(np.float32)
    ts_s1 = None
    if s1_stack is not None:
        ts_s1 = np.transpose(s1_stack[:, :, rows, cols], (2, 0, 1)).astype(np.float32)  # (N,T,2)

    iterator = range(0, N, batch_size)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc=f"  presto ({N:,}px)", leave=False)

    for start in iterator:
        sl = slice(start, min(start + batch_size, N))
        xs, masks, dws = [], [], []
        for k in range(sl.start, sl.stop):
            kwargs = dict(
                s2=torch.from_numpy(ts[k]),       # (T, n_bands)
                s2_bands=s2_band_names,
                normalize=True,
            )
            if ts_s1 is not None:
                kwargs["s1"] = torch.from_numpy(ts_s1[k])   # (T, 2)
                kwargs["s1_bands"] = s1_band_names
            x, mask, dw = construct_single_presto_input(**kwargs)
            xs.append(x)
            masks.append(torch.maximum(mask, mask_override))
            dws.append(dw)
        x_b = torch.stack(xs).to(device)
        mask_b = torch.stack(masks).to(device)
        dw_b = torch.stack(dws).to(device)
        ll_b = torch.from_numpy(latlons[sl].astype(np.float32)).to(device)
        with torch.no_grad():
            emb = encoder(
                x_b, dynamic_world=dw_b.long(), latlons=ll_b,
                mask=mask_b, month=month0, eval_task=True,
            )
        out[sl] = emb.cpu().numpy().astype(np.float32)
    return out


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
                    src.transform, src.crs, src.height, src.width,
                )
    raise FileNotFoundError(f"No GEDI file for tile {healpix_id} in {gedi_dir}")


# ---------------------------------------------------------------------------
# Map GEDI pixels -> nearest S2 pixel
# ---------------------------------------------------------------------------

def gedi_pixels_to_s2(rows_g, cols_g, gedi_transform, gedi_crs,
                      s2_transform, s2_crs, s2_h, s2_w):
    """
    For each valid GEDI pixel return its (row, col) in the S2 grid (nearest pixel)
    and its (lat, lon). Pixels falling outside the S2 grid are flagged invalid.

    Returns (s2_rows, s2_cols, latlons, valid_mask) all length len(rows_g).
    """
    # GEDI pixel centres in GEDI CRS.
    gx, gy = gedi_transform * (cols_g + 0.5, rows_g + 0.5)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)

    # -> lon/lat (for Presto latlons)
    to_ll = Transformer.from_crs(gedi_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_ll.transform(gx, gy)
    latlons = np.stack([lat, lon], axis=1)

    # -> S2 CRS, then to S2 pixel indices via the inverse affine.
    if CRS.from_user_input(gedi_crs) == CRS.from_user_input(s2_crs):
        sx, sy = gx, gy
    else:
        to_s2 = Transformer.from_crs(gedi_crs, s2_crs, always_xy=True)
        sx, sy = to_s2.transform(gx, gy)
    inv = ~s2_transform
    fc, fr = inv * (np.asarray(sx), np.asarray(sy))
    s2_cols = np.floor(fc).astype(int)
    s2_rows = np.floor(fr).astype(int)
    valid = (s2_rows >= 0) & (s2_rows < s2_h) & (s2_cols >= 0) & (s2_cols < s2_w)
    return s2_rows, s2_cols, latlons, valid


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def run_qa(healpix_id, gedi_data, gedi_nodata, gedi_transform, gedi_crs,
           valid_rows, valid_cols, keep_idx, X, y, qa_dir, logger,
           keep_zeros=False) -> dict:
    qa_dir.mkdir(parents=True, exist_ok=True)
    H, W = gedi_data.shape
    n_valid = len(valid_rows)
    n_covered = len(keep_idx)
    coverage = n_covered / max(1, n_valid)
    emb_norm = np.linalg.norm(X, axis=1) if len(X) else np.array([0.0])

    stats = {
        "healpix_id": int(healpix_id),
        "gedi_shape": [H, W],
        "gedi_crs": str(gedi_crs),
        "n_valid_gedi_pixels": int(n_valid),
        "n_covered": int(n_covered),
        "coverage_fraction": round(coverage, 4),
        "n_samples": int(len(y)),
        "n_channels": int(X.shape[1]) if len(X) else PRESTO_N_CHANNELS,
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
    fig.suptitle(f"QA — Tile {healpix_id} (Presto)  |  {H}×{W} px  |  CRS: {gedi_crs}", fontsize=11)

    ax = axes[0, 0]
    gedi_m = np.where(valid_mask, gedi_data / 100.0, np.nan)
    im = ax.imshow(gedi_m, cmap="viridis"); plt.colorbar(im, ax=ax, fraction=0.046, label="Height (m)")
    ax.set_title(f"GEDI height (m) — {n_valid:,} valid"); ax.axis("off")

    ax = axes[0, 1]
    emb_map = np.full((H, W), np.nan)
    emb_map[valid_rows[keep_idx], valid_cols[keep_idx]] = np.mean(X, axis=1) if len(X) else np.nan
    im = ax.imshow(emb_map, cmap="plasma"); plt.colorbar(im, ax=ax, fraction=0.046, label="Mean emb")
    ax.set_title(f"Presto embedding (mean of {stats['n_channels']} ch)"); ax.axis("off")

    ax = axes[1, 0]
    cov = np.full((H, W), np.nan)
    cov[valid_rows, valid_cols] = 0.0
    cov[valid_rows[keep_idx], valid_cols[keep_idx]] = 1.0
    im = ax.imshow(cov, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title(f"Coverage: {coverage:.1%}\ngreen=embedded, red=no S2"); ax.axis("off")

    ax = axes[1, 1]
    if len(y):
        ax.scatter(y, emb_norm, alpha=0.2, s=1, rasterized=True, color="steelblue")
    ax.set_xlabel("GEDI height (m)"); ax.set_ylabel("Embedding L2 norm")
    ax.set_title(f"Norm vs height (N={len(y):,})"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    qa_png = qa_dir / f"tile_{healpix_id}_qa.png"
    plt.savefig(qa_png, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  QA: coverage={coverage:.1%}  → {qa_png.name}")
    return stats


# ---------------------------------------------------------------------------
# Save embeddings as a georeferenced GeoTIFF (for QGIS)
# ---------------------------------------------------------------------------

def save_embedding_tif(healpix_id, X, rows, cols, transform, crs, height, width,
                       out_dir, suffix="") -> Path:
    """
    Scatter the (N, 128) embeddings back onto the GEDI grid and write a georeferenced
    multi-band GeoTIFF (one band per Presto channel) so it can be opened in QGIS.

    rows/cols are the GEDI-grid indices of the embedded pixels (length N). Pixels with
    no embedding are NaN (set as the GeoTIFF nodata value).
    """
    n_ch = X.shape[1]
    grid = np.full((n_ch, height, width), np.nan, dtype=np.float32)
    grid[:, rows, cols] = X.T  # (n_ch, N) scattered onto (n_ch, H, W)

    tif_path = out_dir / f"healpix_{healpix_id}_embedding{suffix}.tif"
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=height, width=width, count=n_ch,
        dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        compress="deflate", predictor=3, tiled=True,
    ) as dst:
        dst.write(grid)
        for b in range(n_ch):
            dst.set_band_description(b + 1, f"presto_ch_{b + 1:03d}")
    return tif_path


def tif_from_cached_npz(healpix_id, npz_path: Path, out_dir: Path, suffix: str) -> Path:
    """
    Rebuild the embedding GeoTIFF from an already-saved .npz without re-fetching imagery
    or re-running Presto. Relies on the pixel positions / geo-metadata stored alongside
    (X, y) by process_tile (emb_rows, emb_cols, grid_transform, grid_crs, grid_shape).
    """
    d = np.load(npz_path, allow_pickle=False)
    if "emb_rows" not in d.files:
        raise ValueError(
            f"{npz_path.name} has no pixel positions (written before --save-tif support); "
            "re-run that tile with --overwrite to regenerate it with positions.")
    X = d["X"]
    transform = Affine(*d["grid_transform"].tolist())
    crs = RioCRS.from_wkt(str(d["grid_crs"]))
    h, w = int(d["grid_shape"][0]), int(d["grid_shape"][1])
    return save_embedding_tif(healpix_id, X, d["emb_rows"], d["emb_cols"],
                              transform, crs, h, w, out_dir, suffix)


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(
    healpix_id, year, start_month, gdf, gedi_dir, output_dir, overwrite,
    encoder, device, batch_size, rslearn_root, imagery_dir, rslearn_workers,
    logger, do_qa=False, qa_dir=None, keep_zeros=False, subset_fraction=None,
    s2_source="pc", save_tif=False, with_s1=False, s1_units="linear",
) -> dict:
    suffix = "_withzeros" if keep_zeros else ""
    out_path = output_dir / f"healpix_{healpix_id}_float32{suffix}.npz"
    if not overwrite and out_path.exists():
        # Embedding already computed. If --save-tif is set and the GeoTIFF is missing,
        # rebuild it straight from the cached .npz (no imagery fetch, no Presto re-run).
        if save_tif:
            tif_path = output_dir / f"healpix_{healpix_id}_embedding{suffix}.tif"
            if not tif_path.exists():
                try:
                    tif_path = tif_from_cached_npz(healpix_id, out_path, output_dir, suffix)
                    print(f"  Tile {healpix_id}: embedding GeoTIFF from cached npz "
                          f"→ {tif_path.name}")
                    return {"status": "tif_from_cache", "healpix_id": int(healpix_id)}
                except Exception as exc:
                    logger.warning(f"Tile {healpix_id}: could not rebuild tif from cache — {exc}")
                    return {"status": "tif_failed", "healpix_id": int(healpix_id),
                            "error": str(exc)}
        logger.info(f"Tile {healpix_id}: cached, skipping")
        return {"status": "skipped", "healpix_id": int(healpix_id)}

    t_start = time.time()
    try:
        # 1. Tile bounds in WGS84.
        shp_minx, shp_miny, shp_maxx, shp_maxy = get_tile_bounds(healpix_id, gdf)
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = bounds_to_crs(
            shp_minx, shp_miny, shp_maxx, shp_maxy, shp_crs, CRS.from_epsg(4326))

        # Optional: crop to a centered sub-window for fast local testing. Shrinks the
        # rslearn fetch and restricts the GEDI pixels embedded (step 4) to this window.
        subset_bbox = None
        if subset_fraction is not None:
            lon_min, lat_min, lon_max, lat_max = crop_bounds_centered(
                lon_min, lat_min, lon_max, lat_max, subset_fraction)
            subset_bbox = (lon_min, lat_min, lon_max, lat_max)
            print(f"  Tile {healpix_id}: subset to {subset_fraction:.2%} of bbox "
                  f"-> lon[{lon_min:.4f},{lon_max:.4f}] lat[{lat_min:.4f},{lat_max:.4f}]")

        # 2. Get the monthly S2 stack (rslearn, or pre-materialized imagery).
        if imagery_dir is not None:
            paths, slots = find_imagery_dir_geotiffs(Path(imagery_dir), healpix_id)
        else:
            win_dir = materialize_tile_rslearn(
                healpix_id, lon_min, lat_min, lon_max, lat_max, year,
                start_month, Path(rslearn_root), logger, workers=rslearn_workers,
                source=s2_source, with_s1=with_s1)
            paths, slots = find_monthly_geotiffs(win_dir, logger=logger)
        if len(paths) == 0:
            logger.warning(f"Tile {healpix_id}: no S2 monthly imagery found")
            return {"status": "no_imagery", "healpix_id": int(healpix_id)}

        s2_stack, s2_present, s2_transform, s2_crs, s2_h, s2_w = load_s2_stack(
            paths, slots, n_bands_expected=len(PRESTO_S2_BANDS))
        n_s2_months = int(s2_present.sum())
        if n_s2_months < PRESTO_NUM_TIMESTEPS:
            logger.warning("Tile %s: only %d/%d months available (slots %s); the rest "
                           "are masked for Presto", healpix_id, n_s2_months,
                           PRESTO_NUM_TIMESTEPS,
                           np.where(~s2_present)[0].tolist())
        print(f"  Tile {healpix_id}: S2 stack {s2_stack.shape} ({n_s2_months} months)")

        # Optional Sentinel-1. S1 is slotted into the same 12-month calendar as S2 by
        # its own timestamps, so the two can no longer drift apart — a month present in
        # one and missing from the other just masks that channel group for that month.
        # S1 shares the S2 window grid, so the same s2_rows/s2_cols index it.
        # (Not available via --imagery-dir.)
        s1_stack = None
        s1_present = None
        if with_s1 and imagery_dir is None:
            s1_paths, s1_slots = find_s1_monthly_geotiffs(win_dir, logger=logger)
            if len(s1_paths) > 0:
                s1_raw, s1_present, _, _, s1_h, s1_w = load_s2_stack(
                    s1_paths, s1_slots, n_bands_expected=len(RSLEARN_S1_BANDS))
                if (s1_h, s1_w) != (s2_h, s2_w):
                    # s2_rows/s2_cols index both stacks, so a different grid would read
                    # the wrong pixels rather than fail.
                    logger.warning("Tile %s: S1 grid %s != S2 grid %s; using S2-only",
                                   healpix_id, (s1_h, s1_w), (s2_h, s2_w))
                    s1_stack, s1_present = None, None
                else:
                    s1_stack = s1_to_presto_db(s1_raw, units=s1_units)
                    print(f"  Tile {healpix_id}: S1 stack {s1_stack.shape} (dB, "
                          f"{int(s1_present.sum())} months)")
            else:
                logger.warning("Tile %s: --with-s1 set but no S1 imagery; using S2-only",
                               healpix_id)

        # 3. GEDI target + valid pixels.
        gedi_data, gedi_nodata, gedi_transform, gedi_crs, gedi_h, gedi_w = \
            load_gedi_tile(healpix_id, gedi_dir)
        valid_mask = (
            (gedi_data != gedi_nodata) &
            (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
            (gedi_data < GEDI_MAX_HEIGHT_CM)
        )
        rows_g, cols_g = np.where(valid_mask)
        if len(rows_g) == 0:
            logger.info(f"Tile {healpix_id}: no valid GEDI pixels")
            return {"status": "no_valid_pixels", "healpix_id": int(healpix_id)}

        # 4. Map GEDI pixels -> S2 pixels; drop those outside the S2 grid.
        s2_rows, s2_cols, latlons, covered = gedi_pixels_to_s2(
            rows_g, cols_g, gedi_transform, gedi_crs, s2_transform, s2_crs, s2_h, s2_w)
        if subset_bbox is not None:
            s_lon_min, s_lat_min, s_lon_max, s_lat_max = subset_bbox
            in_window = (
                (latlons[:, 1] >= s_lon_min) & (latlons[:, 1] <= s_lon_max) &
                (latlons[:, 0] >= s_lat_min) & (latlons[:, 0] <= s_lat_max)
            )
            covered = covered & in_window
        keep_idx = np.where(covered)[0]
        if len(keep_idx) == 0:
            logger.warning(f"Tile {healpix_id}: no GEDI pixels fall inside the S2 grid")
            return {"status": "no_coverage", "healpix_id": int(healpix_id)}

        # 5. Presto encode at the covered GEDI pixels.
        t_emb = time.time()
        X = embed_pixels(
            s2_stack, PRESTO_S2_BANDS,
            s2_rows[keep_idx], s2_cols[keep_idx], latlons[keep_idx],
            start_month, encoder, device, batch_size, logger,
            s1_stack=s1_stack, s1_band_names=PRESTO_S1_BANDS,
            s2_present=s2_present, s1_present=s1_present)
        logger.info("Tile %s: Presto embedded %d px in %.1fs",
                    healpix_id, len(keep_idx), time.time() - t_emb)

        y = (gedi_data[rows_g[keep_idx], cols_g[keep_idx]] / 100.0).astype(np.float32)
        n_total = len(rows_g)

        if do_qa:
            run_qa(healpix_id, gedi_data, gedi_nodata, gedi_transform, gedi_crs,
                   rows_g, cols_g, keep_idx, X, y, qa_dir, logger, keep_zeros)

        # Store pixel positions + GEDI geo-metadata alongside (X, y) so an embedding
        # GeoTIFF can be rebuilt later (a --save-tif rerun) without re-fetching imagery
        # or re-running Presto. run_gedi_prediction_30m.py reads X/y by key, so the
        # extra arrays are harmless to it.
        np.savez_compressed(
            out_path, X=X, y=y,
            emb_rows=rows_g[keep_idx].astype(np.int32),
            emb_cols=cols_g[keep_idx].astype(np.int32),
            grid_transform=np.asarray(tuple(gedi_transform)[:6], dtype=np.float64),
            grid_crs=np.asarray(gedi_crs.to_wkt()),
            grid_shape=np.asarray([gedi_h, gedi_w], dtype=np.int32),
        )
        if save_tif:
            tif_path = save_embedding_tif(
                healpix_id, X, rows_g[keep_idx], cols_g[keep_idx],
                gedi_transform, gedi_crs, gedi_h, gedi_w, output_dir, suffix)
            print(f"  Tile {healpix_id}: embedding GeoTIFF ({X.shape[1]} bands) "
                  f"→ {tif_path.name}")
        elapsed = time.time() - t_start
        mem_str = (f"  mem={psutil.Process(os.getpid()).memory_info().rss/1e9:.1f}GB"
                   if _HAS_PSUTIL else "")
        print(f"  Tile {healpix_id}: {len(y):,} samples  {X.shape[1]} ch  "
              f"{elapsed:.1f}s{mem_str}  → {out_path.name}")
        return {
            "status": "ok", "healpix_id": int(healpix_id),
            "n_samples": int(len(y)), "n_valid_gedi": int(n_total),
            "coverage_fraction": round(len(keep_idx) / max(1, n_total), 4),
            "n_channels": int(X.shape[1]), "n_months": n_s2_months,
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

def setup_logging(output_dir: Path, console_level=logging.WARNING) -> logging.Logger:
    logger = logging.getLogger("presto")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"presto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(console_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=("Generate Presto embeddings per healpix tile (Sentinel-2 fetched "
                     "free via rslearn) and save (X, y) .npz for run_gedi_prediction_30m.py"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--healpix-id", type=int, default=None,
                        help="Single tile to process (omit for batch over --splits)")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the 12-month Sentinel-2 window to embed")
    parser.add_argument("--start-month", type=int, default=1,
                        help="Calendar month (1-12) of the first timestep")
    parser.add_argument("--shapefile", required=True,
                        help="Healpix shapefile (.shp) with healpix_id and split columns")
    parser.add_argument("--gedi-dir", required=True,
                        help="Directory with GEDI tif files (healpix_*_gedi.tif)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-tile healpix_*_float32.npz files")
    parser.add_argument("--rslearn-root", default=None,
                        help="Working dir for the rslearn dataset (required unless --imagery-dir)")
    parser.add_argument("--imagery-dir", default=None,
                        help="Use pre-materialized healpix_<id>_s2_<month>.tif here instead of rslearn")
    parser.add_argument("--rslearn-workers", type=int, default=4,
                        help="Parallel workers for rslearn prepare/ingest/materialize")
    parser.add_argument("--s2-source", choices=["pc", "element84", "gcp"], default="pc",
                        help="Sentinel-2 source. 'pc' = Microsoft Planetary Computer L2A "
                             "(free, no billing). 'element84' = free AWS L2A COGs (no account/"
                             "billing). 'gcp' = gcp_public_data L1C (free imagery but the "
                             "scene index uses BigQuery, which needs a billing-enabled project).")
    parser.add_argument("--with-s1", action=argparse.BooleanOptionalAction, default=False,
                        help="Also feed Sentinel-1 (VV/VH from Planetary Computer) to "
                             "Presto, materialized as 12 monthly mosaics alongside S2. "
                             "Activates Presto's S1 channel group (S2-only when omitted).")
    parser.add_argument("--s1-units", choices=["linear", "db"], default="linear",
                        help="Units of the materialized S1. PC Sentinel-1 RTC is linear "
                             "gamma0 ('linear', converted to dB for Presto); use 'db' if "
                             "your source already provides decibels.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to include in batch mode")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Pixels per Presto forward pass")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process tiles that already have a cached .npz")
    parser.add_argument("--keep-zeros", action="store_true",
                        help="Keep GEDI height == 0 (output *_float32_withzeros.npz)")
    parser.add_argument("--subset-fraction", type=float, default=None,
                        help="For local testing: crop each tile to this centered "
                             "fraction of its bbox before fetching imagery (e.g. 0.1 "
                             "= central 10%% per axis, ~1%% of area). Shrinks the rslearn "
                             "download and the number of Presto passes.")
    parser.add_argument("--subset-tile-fraction", type=float, default=None,
                        help="Embed only this fraction (0, 1] of a split's TILES, picking "
                             "the SAME tiles as run_gedi_prediction_30m.py --subset-fraction "
                             "(same seed/shapefile -> identical selection). Use to embed just "
                             "10%% of train tiles for fast experiments. Distinct from "
                             "--subset-fraction (a spatial bbox crop within a tile). Selected "
                             "tile IDs are written to subset_tiles.json.")
    parser.add_argument("--subset-tile-seed", type=int, default=42,
                        help="Seed for --subset-tile-fraction. Must match "
                             "run_gedi_prediction_30m.py --subset-seed (default 42).")
    parser.add_argument("--subset-tile-splits", nargs="+", default=["train"],
                        metavar="SPLIT",
                        help="Which splits --subset-tile-fraction applies to. Must match "
                             "run_gedi_prediction_30m.py --subset-splits (default: train).")
    parser.add_argument("--check-only", action="store_true",
                        help="Report cache coverage and exit")
    parser.add_argument("--save-tif", action="store_true",
                        help="Also write a georeferenced multi-band embedding GeoTIFF "
                             "(healpix_<id>_embedding.tif) on the GEDI grid for QGIS")
    parser.add_argument("--qa", action="store_true",
                        help="Save per-tile diagnostic plots and stats")
    parser.add_argument("--qa-dir", default=None,
                        help="Directory for QA outputs (default: {output-dir}/qa/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Send INFO logs to the console and stream rslearn's "
                             "prepare/materialize output live (so you can watch S2 "
                             "download/retry progress in the job log).")
    args = parser.parse_args()

    if args.imagery_dir is None and args.rslearn_root is None:
        parser.error("provide --rslearn-root (to fetch imagery) or --imagery-dir "
                     "(to use pre-materialized S2 GeoTIFFs)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global _STREAM_RSLEARN
    _STREAM_RSLEARN = args.verbose
    logger = setup_logging(output_dir,
                           console_level=logging.INFO if args.verbose else logging.WARNING)
    qa_dir = Path(args.qa_dir) if args.qa_dir else output_dir / "qa"
    if args.qa:
        qa_dir.mkdir(parents=True, exist_ok=True)
        print(f"QA outputs will be saved to: {qa_dir}")

    gdf = gpd.read_file(args.shapefile)
    print(f"Loaded shapefile: {len(gdf)} tiles")

    if args.healpix_id is not None:
        if int(args.healpix_id) not in gdf["healpix_id"].values:
            print(f"Error: healpix_id {args.healpix_id} not found", file=sys.stderr)
            sys.exit(1)
        tile_ids = [args.healpix_id]
        print(f"Single-tile mode: tile {args.healpix_id}  year={args.year}")
    else:
        subset_tile_splits = set(args.subset_tile_splits)
        do_subset = (args.subset_tile_fraction is not None
                     and args.subset_tile_fraction < 1.0)
        if do_subset and not (0 < args.subset_tile_fraction <= 1):
            parser.error("--subset-tile-fraction must be in (0, 1]")

        subset_selected = {}  # split -> chosen healpix_ids (for subset_tiles.json)
        if "split" in gdf.columns:
            # Build the tile list per split (not via isin) so subsetting reproduces
            # run_gedi_prediction_30m.py's per-split selection exactly.
            tile_ids = []
            for split_name in args.splits:
                split_tiles = gdf[gdf["split"] == split_name]["healpix_id"].values
                if do_subset and split_name in subset_tile_splits:
                    n_before = len(split_tiles)
                    split_tiles = subset_tiles_for_split(
                        split_tiles, args.subset_tile_fraction, args.subset_tile_seed)
                    subset_selected[split_name] = [int(t) for t in split_tiles]
                    print(f"  Subset '{split_name}': {len(split_tiles)}/{n_before} tiles "
                          f"(fraction={args.subset_tile_fraction}, seed={args.subset_tile_seed})")
                tile_ids.extend(int(t) for t in split_tiles)
        else:
            tile_ids = gdf["healpix_id"].tolist()
            if do_subset:
                print("Warning: --subset-tile-fraction ignored (shapefile has no 'split' column)")
        print(f"Batch mode: {len(tile_ids)} tiles  splits={args.splits}  year={args.year}")

        if subset_selected:
            subset_path = output_dir / "subset_tiles.json"
            with open(subset_path, "w") as f:
                json.dump({
                    "subset_fraction": args.subset_tile_fraction,
                    "subset_seed": args.subset_tile_seed,
                    "splits": subset_selected,
                }, f, indent=2)
            print(f"Saved subset tile IDs: {subset_path}")

    if args.check_only:
        cache_suffix = "_withzeros" if args.keep_zeros else ""
        cached = [t for t in tile_ids
                  if (output_dir / f"healpix_{t}_float32{cache_suffix}.npz").exists()]
        missing = [t for t in tile_ids if t not in set(cached)]
        print(f"Cached : {len(cached)}/{len(tile_ids)}")
        if missing:
            print(f"Missing: {len(missing)} — first 20: {missing[:20]}")
        return

    print("\nLoading pretrained Presto...")
    encoder, device = load_presto()
    print(f"Presto encoder ready on {device}")

    gedi_dir = Path(args.gedi_dir)
    results = []
    iterator = tqdm(tile_ids, desc="tiles") if _HAS_TQDM else tile_ids
    for healpix_id in iterator:
        results.append(process_tile(
            healpix_id=healpix_id, year=args.year, start_month=args.start_month,
            gdf=gdf, gedi_dir=gedi_dir, output_dir=output_dir, overwrite=args.overwrite,
            encoder=encoder, device=device, batch_size=args.batch_size,
            rslearn_root=args.rslearn_root, imagery_dir=args.imagery_dir,
            rslearn_workers=args.rslearn_workers, logger=logger,
            do_qa=args.qa, qa_dir=qa_dir, keep_zeros=args.keep_zeros,
            subset_fraction=args.subset_fraction, s2_source=args.s2_source,
            save_tif=args.save_tif, with_s1=args.with_s1, s1_units=args.s1_units,
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
        "model": "Presto", "year": args.year, "start_month": args.start_month,
        "n_tiles": len(tile_ids), "total_samples": total_samples,
        "status_counts": counts, "tiles": results,
    }
    summary_path = output_dir / f"presto_summary_{args.year}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
