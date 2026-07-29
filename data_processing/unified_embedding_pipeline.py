#!/usr/bin/env python3
"""
Unified OLMo + Presto embedding pipeline — download Sentinel-2 ONCE, embed with BOTH models.

Motivation
----------
The OLMo and Presto pipelines both consume the same Sentinel-2 (+S1) imagery, but were run
as two separate rslearn datasets — so the (slow, expensive) S2/S1 download happened TWICE.
This script materializes one shared rslearn dataset per run, then:

  1. runs OLMo via `rslearn model predict` (your existing model-gpu.yaml, UNCHANGED), and
  2. runs the Presto encoder over the SAME materialized S2,

and writes DENSE, per-model embeddings over the WHOLE tile (not just GEDI pixels), each
reprojected onto the GEDI tile grid (CRS/resolution/extent of healpix_<id>_gedi.tif), as
per-tile .npz:

    healpix_<id>_olmo.npz     X=(H, W, 128) float32, y=(H, W) float32, + grid metadata
    healpix_<id>_presto.npz   X=(H, W, 128) float32, y=(H, W) float32, + grid metadata

`y` is the GEDI height (m), with NaN where there is no GEDI; **X is kept everywhere** (nothing
is dropped for missing GEDI). To recover a training table downstream:  mask = np.isfinite(y).

Why this matches your existing setups
-------------------------------------
- Shared dataset uses your OLMo config.json verbatim (sentinel1 + sentinel2_l2a + embeddings;
  landsat is present but disabled via --disabled-layers landsat, exactly like your bash).
- Your OLMo config already sets S2 max_matches=12 / period_duration=30d, so 12 monthly mosaics
  are materialized — OLMo's model.yaml reads its usual subset, Presto reads all 12. No change
  to OLMo's output.
- add_windows / prepare / materialize / predict mirror your rslearn_olmo bash flags.

Resolution / CRS
----------------
- S2 materialized at 10 m (UTM), as in your OLMo bash.
- OLMo: 10 m input + patch_size 3 -> 30 m embeddings natively (your downsample_factor 3).
- Presto: per-pixel; produced on the 30 m grid (see --presto-sampling), then reprojected.
- Both outputs are reprojected onto the GEDI tile grid so they align with `y`.

Presto dense sampling (--presto-sampling)
----------------------------------------
  aggregate30        (cheapest, recommended for full tiles) — spatially average the 10 m S2
                     stack to 30 m (3x3 mean), then run Presto once per 30 m cell.
  nearest30          — take the center 10 m pixel of each 3x3 block, embed once per 30 m cell.
  embed10_downsample (most faithful, VERY slow) — embed every 10 m pixel, then average the
                     embeddings to 30 m. ~9x more Presto forward passes; processed block-wise.

Environments
------------
rslearn (add_windows/prepare/materialize/predict) runs as subprocesses — wrap them in your
mamba env with --rslearn-prefix "mamba run -n jpl-py313". The Presto encoder runs IN-PROCESS,
so the env you launch THIS script in must import `presto` + torch + rasterio + geopandas.
(If OLMo and Presto live in different envs, launch the script in the Presto env and point
--rslearn-prefix at the OLMo env.)

Single-run usage (mirrors your OLMo bash, but for healpix tiles):
    python unified_embedding_pipeline.py \\
        --year 2020 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --dataset-path /scratch/kdoerkse/unified/2020/dataset \\
        --olmo-model-config /data/.../olmo/model-gpu.yaml \\
        --output-dir /scratch/kdoerkse/unified/2020/embeddings \\
        --splits train val test \\
        --rslearn-prefix "mamba run -n jpl-py313" \\
        --presto-sampling aggregate30
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
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from pyproj import CRS, Transformer

# Reuse the proven Presto helpers (encoder load, per-pixel encoding, S2 stack reader,
# GEDI loader, tile-bounds helpers) instead of duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import presto_processing as pp  # noqa: E402

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Band / layer constants (must match the OLMo config.json the dataset is built from)
# ---------------------------------------------------------------------------

# OLMo sentinel2_l2a band order (from your config.json band_sets).
OLMO_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
OLMO_S2_LAYER = "sentinel2_l2a"
OLMO_S2_BANDSET = "_".join(OLMO_S2_BANDS)
# Indices of the 10 bands Presto consumes (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12)
# within OLMO_S2_BANDS. Presto drops B01/B09.
OLMO_TO_PRESTO_BAND_IDX = [OLMO_S2_BANDS.index(b) for b in
                           ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]]

PRESTO_DOWNSAMPLE = 3  # 10 m -> 30 m

# Sentinel-1 (optional, --with-s1): VV/VH materialized alongside S2 on the same
# window grid, 12 monthly mosaics. Only Presto uses it (Prithvi is optical-only).
# The dataset config must add a `sentinel1` layer (bands vv,vh) with the SAME 12-
# month query_config as S2 so the S1/S2 monthly stacks align. PC S1 is linear
# gamma0; Presto wants dB, so we convert via pp.s1_to_presto_db(units="linear").
UNIFIED_S1_LAYER = "sentinel1"
UNIFIED_S1_BANDSET = "_".join(["vv", "vh"])  # "vv_vh"

# Prithvi-EO-2.0 (rslearn.models.prithvi.PrithviV2) output layer + embedding dim.
# 300M -> see NOTE on the prithvi_embeddings layer below; set to what predict writes.
PRITHVI_EMB_LAYER = "prithvi_embeddings"
PRITHVI_EMB_DIM = 1024


# The shared dataset config.json (your OLMo config, verbatim). landsat stays here but is
# disabled at prepare/materialize time via --disabled-layers landsat, matching your bash.
OLMO_DATASET_CONFIG = {
    "layers": {
        "landsat": {
            "band_sets": [{"bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
                           "dtype": "uint16"}],
            "data_source": {
                "class_path": "rslearn.data_sources.aws_landsat.LandsatOliTirs",
                "init_args": {"metadata_cache_dir": "cache/landsat", "sort_by": "cloud_cover"},
                "ingest": False,
                "query_config": {"max_matches": 12, "period_duration": "30d", "space_mode": "PER_PERIOD_MOSAIC"},
            },
            "type": "raster",
        },
        "sentinel1": {
            "band_sets": [{"bands": ["vv", "vh"], "dtype": "float32", "nodata_vals": [-32768, -32768]}],
            "data_source": {
                "class_path": "rslearn.data_sources.planetary_computer.Sentinel1",
                "init_args": {"cache_dir": "cache/planetary_computer",
                              "query": {"sar:instrument_mode": {"eq": "IW"},
                                        "sar:polarizations": {"eq": ["VV", "VH"]}}},
                "ingest": False,
                "query_config": {"max_matches": 12, "period_duration": "30d", "space_mode": "PER_PERIOD_MOSAIC"},
            },
            "type": "raster",
        },
        "sentinel2_l2a": {
            "band_sets": [{"bands": OLMO_S2_BANDS, "dtype": "uint16"}],
            "data_source": {
                "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
                "init_args": {"cache_dir": "cache/planetary_computer", "harmonize": True,
                              "sort_by": "eo:cloud_cover"},
                "ingest": False,
                "query_config": {"max_matches": 12, "period_duration": "30d", "space_mode": "PER_PERIOD_MOSAIC"},
            },
            "type": "raster",
        },
        "embeddings": {"band_sets": [{"dtype": "float32", "num_bands": 128}], "type": "raster"},
        # Prithvi-EO-2.0 embeddings output layer (written by RslearnWriter in
        # prithvi_predict.yaml). NOTE: num_bands MUST equal what rslearn's PrithviV2
        # FeatureExtractor writes. Its forward concatenates per-block features
        # (docstring: [B, H/ps, W/ps, 11*1024]) while get_backbone_channels() reports
        # 1024 — these disagree across versions. If `model predict` errors on a band-count
        # mismatch, set this to the value rslearn reports (likely 1024 or 11264) and rerun.
        "prithvi_embeddings": {"band_sets": [{"dtype": "float32", "num_bands": PRITHVI_EMB_DIM}],
                               "type": "raster"},
    }
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("unified")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO if verbose else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


def run_cmd(cmd: list, logger: logging.Logger, env: dict = None):
    """Run a subprocess, streaming output live to stdout + the log."""
    def echo(text):
        """Print, but never let a dead stdout kill the run.

        On the HPC, stdout is the slurm .out file on a quota'd filesystem. If that
        filesystem fills, `print` raises OSError(EDQUOT) from inside the streaming loop,
        which masks the subprocess's ACTUAL failure and skips the `tail` diagnostic below.
        Swallow it: the child's exit code is what we care about. (logging's FileHandler
        already swallows its own IO errors, so the logger calls need no guard.)
        """
        try:
            print(text, flush=True)
        except OSError:
            pass

    logger.info("cmd: %s", " ".join(cmd))
    echo(f"  $ {' '.join(cmd)}")
    from collections import deque
    tail = deque(maxlen=40)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    for line in proc.stdout:
        line = line.rstrip("\n")
        echo(f"  | {line}")
        logger.info("> %s", line)
        tail.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n" + "\n".join(tail))


# ---------------------------------------------------------------------------
# rslearn orchestration (mirrors your rslearn_olmo bash)
# ---------------------------------------------------------------------------

def write_dataset_config(dataset_path: Path, config_src: str = None,
                         prithvi_emb_dim: int = None) -> Path:
    """Write the shared dataset config.json (your OLMo config, or a copy of config_src).

    When using the built-in config, `prithvi_emb_dim` (if given) sets the
    prithvi_embeddings output layer's num_bands. This MUST equal the channel count
    rslearn's PrithviV2 actually writes; if `rslearn model predict` errors on a
    band-count mismatch, rerun with --prithvi-emb-dim set to the value in the error
    (no source edit needed).
    """
    dataset_path.mkdir(parents=True, exist_ok=True)
    cfg_path = dataset_path / "config.json"
    if config_src:
        text = Path(config_src).read_text()
    else:
        cfg = json.loads(json.dumps(OLMO_DATASET_CONFIG))   # deep copy
        if prithvi_emb_dim is not None:
            cfg["layers"]["prithvi_embeddings"]["band_sets"][0]["num_bands"] = int(prithvi_emb_dim)
        text = json.dumps(cfg, indent=2)
    if cfg_path.exists() and cfg_path.read_text() == text:
        return cfg_path
    tmp = cfg_path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, cfg_path)
    return cfg_path


def window_dir(dataset_path: Path, healpix_id: int) -> Path:
    name = f"healpix_{healpix_id}"
    return dataset_path / "windows" / name / name


def add_window(dataset_path: Path, healpix_id: int, lon_min, lat_min, lon_max, lat_max,
               year: int, prefix: list, logger):
    name = f"healpix_{healpix_id}"
    cmd = prefix + [
        "rslearn", "dataset", "add_windows",
        "--root", str(dataset_path),
        "--group", name, "--name", name,
        "--utm", "--resolution", "10", "--src_crs", "EPSG:4326",
        f"--box={lon_min},{lat_min},{lon_max},{lat_max}",
        "--start", f"{year}-01-01T00:00:00+00:00",
        "--end", f"{year + 1}-01-01T00:00:00+00:00",
    ]
    run_cmd(cmd, logger)


def prepare_materialize(dataset_path: Path, workers: int, prefix: list, logger):
    base = ["--root", str(dataset_path), "--workers", str(workers),
            "--disabled-layers", "landsat",
            "--retry-max-attempts", "5", "--retry-backoff-seconds", "5"]
    run_cmd(prefix + ["rslearn", "dataset", "prepare"] + base, logger)
    run_cmd(prefix + ["rslearn", "dataset", "materialize", "--no-use-initial-job"] + base, logger)


def olmo_predict(dataset_path: Path, model_config: str, prefix: list, logger):
    """Run OLMo `rslearn model predict` against the shared dataset (DATASET_PATH env)."""
    env = dict(os.environ)
    env["DATASET_PATH"] = str(dataset_path)
    run_cmd(prefix + ["rslearn", "model", "predict", "--config", model_config], logger, env=env)


def prithvi_predict(dataset_path: Path, model_config: str, prefix: list, logger):
    """Run Prithvi-EO-2.0 `rslearn model predict` (writes the prithvi_embeddings layer).

    Identical mechanics to olmo_predict — rslearn's PrithviV2 encoder reads the same
    materialized Sentinel-2 (a 3-frame subset, 6 HLS bands) and the RslearnWriter in
    prithvi_predict.yaml writes embeddings into the `prithvi_embeddings` layer.

    Speed note: prithvi_predict.yaml reads 3x-wider (672 px) crops of the existing 10 m
    S2 so PrithviV2 downsamples them to 30 m in-memory (its native scale) — no second
    copy of the imagery is written. That yields ~9x fewer 300M-ViT forward passes per
    tile than 224 px crops at 10 m did (the reason Prithvi was slow). See the YAML header.
    """
    env = dict(os.environ)
    env["DATASET_PATH"] = str(dataset_path)
    logger.info("Prithvi predict: reading 672 px crops -> 30 m on-the-fly downsample "
                "(~9x fewer crops/tile). Watch the log for `GPU available: True` — a CPU "
                "fallback would be the other reason it runs slow.")
    logger.info("Prithvi predict: if this fails with a band-count / num_bands mismatch, "
                "note the expected value in the error and rerun with --prithvi-emb-dim <N> "
                "(and --overwrite).")
    run_cmd(prefix + ["rslearn", "model", "predict", "--config", model_config], logger, env=env)


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def find_olmo_s2_geotiffs(win_dir: Path) -> list:
    """The (up to 12) monthly sentinel2_l2a geotiffs for a window, in temporal order."""
    paths = []
    for i in range(pp.PRESTO_NUM_TIMESTEPS):
        layer = OLMO_S2_LAYER if i == 0 else f"{OLMO_S2_LAYER}.{i}"
        p = win_dir / "layers" / layer / OLMO_S2_BANDSET / "geotiff.tif"
        if p.exists():
            paths.append(p)
    return paths


def find_unified_s1_geotiffs(win_dir: Path) -> list:
    """The (up to 12) monthly sentinel1 VV/VH geotiffs for a window, temporal order."""
    paths = []
    for i in range(pp.PRESTO_NUM_TIMESTEPS):
        layer = UNIFIED_S1_LAYER if i == 0 else f"{UNIFIED_S1_LAYER}.{i}"
        p = win_dir / "layers" / layer / UNIFIED_S1_BANDSET / "geotiff.tif"
        if p.exists():
            paths.append(p)
    return paths


def find_olmo_embedding_geotiff(win_dir: Path) -> Path:
    """The OLMo embeddings geotiff written by RslearnWriter (128 bands, 30 m, UTM)."""
    emb_dir = win_dir / "layers" / "embeddings"
    hits = sorted(emb_dir.glob("*/geotiff.tif"))
    return hits[0] if hits else None


def find_prithvi_embedding_geotiff(win_dir: Path) -> Path:
    """The Prithvi embeddings geotiff written by RslearnWriter (PRITHVI_EMB_DIM bands, UTM)."""
    emb_dir = win_dir / "layers" / PRITHVI_EMB_LAYER
    hits = sorted(emb_dir.glob("*/geotiff.tif"))
    return hits[0] if hits else None


def pixel_latlons(rows: np.ndarray, cols: np.ndarray, transform, crs) -> np.ndarray:
    """(lat, lon) of pixel centers for the given row/col indices in `crs`."""
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_ll.transform(np.asarray(xs, float), np.asarray(ys, float))
    return np.stack([lat, lon], axis=1).astype(np.float32)


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Spatial f x f mean over the last two axes of a (..., H, W) array (trims to multiple of f)."""
    H, W = a.shape[-2], a.shape[-1]
    H2, W2 = (H // f) * f, (W // f) * f
    a = a[..., :H2, :W2]
    new = a.reshape(*a.shape[:-2], H2 // f, f, W2 // f, f)
    return new.mean(axis=(-3, -1))


def reproject_to_grid(src, src_transform, src_crs, dst_transform, dst_crs, dst_h, dst_w,
                      resampling=Resampling.bilinear) -> np.ndarray:
    """Reproject a (C, H, W) float32 array onto the destination grid. NaN where no data."""
    dst = np.full((src.shape[0], dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(src), destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling,
    )
    return dst


# ---------------------------------------------------------------------------
# Presto dense encoding on the 30 m grid (in S2/UTM space)
# ---------------------------------------------------------------------------

def presto_dense_30m(win_dir: Path, start_month: int, sampling: str,
                     encoder, device, batch_size: int, block_rows: int, logger,
                     with_s1: bool = False):
    """
    Returns (emb_30, transform_30, s2_crs) where emb_30 is (128, H30, W30) float32 in the S2
    UTM grid at 30 m, or (None, None, None) if no S2 imagery.

    When with_s1 is set, the sentinel1 VV/VH monthly stack (same window grid, same 12
    months) is loaded, converted to dB (pp.s1_to_presto_db), downsampled identically to
    S2, and fed to Presto's S1 channel group. If S1 is missing, has a different month
    count, or a different grid than S2, S1 is skipped for that tile (S2-only) with a
    warning rather than failing.
    """
    paths = find_olmo_s2_geotiffs(win_dir)
    if not paths:
        return None, None, None
    stack, s2_transform, s2_crs, H, W = pp.load_s2_stack(paths)  # (T, 12, H, W)
    stack = stack[:, OLMO_TO_PRESTO_BAND_IDX, :, :]              # (T, 10, H, W)
    f = PRESTO_DOWNSAMPLE
    transform_30 = s2_transform * Affine.scale(f)               # 10 m -> 30 m pixels

    # Optional Sentinel-1 stack (dB), aligned to the S2 grid + timesteps.
    s1_full = None
    if with_s1:
        s1_paths = find_unified_s1_geotiffs(win_dir)
        if len(s1_paths) != len(paths):
            logger.warning("S1 has %d monthly geotiffs vs %d for S2 — skipping S1 (S2-only)",
                           len(s1_paths), len(paths))
        else:
            s1_raw, _, _, s1H, s1W = pp.load_s2_stack(s1_paths)  # (T, 2, H, W)
            if (s1H, s1W) != (H, W):
                logger.warning("S1 grid (%d,%d) != S2 (%d,%d) — skipping S1", s1H, s1W, H, W)
            else:
                s1_full = pp.s1_to_presto_db(s1_raw, units="linear")  # (T, 2, H, W) dB
                logger.info("Presto: feeding S1 (VV/VH) alongside S2")
    s1_bands = pp.PRESTO_S1_BANDS if s1_full is not None else None

    if sampling in ("aggregate30", "nearest30"):
        if sampling == "aggregate30":
            s2_30 = block_mean(stack, f)                        # (T, 10, H30, W30)
            s1_30 = block_mean(s1_full, f) if s1_full is not None else None
        else:  # nearest30: center pixel of each f x f block
            s2_30 = stack[:, :, 1::f, 1::f]
            H30, W30 = stack.shape[2] // f, stack.shape[3] // f
            s2_30 = s2_30[:, :, :H30, :W30]
            s1_30 = (s1_full[:, :, 1::f, 1::f][:, :, :H30, :W30]
                     if s1_full is not None else None)
            transform_30 = s2_transform * Affine.translation(1, 1) * Affine.scale(f)
        T, _, H30, W30 = s2_30.shape
        emb_30 = np.zeros((H30 * W30, pp.PRESTO_N_CHANNELS), dtype=np.float32)
        rblocks = range(0, H30, block_rows)
        if _HAS_TQDM:
            rblocks = tqdm(list(rblocks), desc=f"  presto[{sampling}]", leave=False)
        for r0 in rblocks:
            r1 = min(r0 + block_rows, H30)
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(W30), indexing="ij")
            rr, cc = rr.ravel(), cc.ravel()
            lls = pixel_latlons(rr, cc, transform_30, s2_crs)
            emb = pp.embed_pixels(s2_30, pp.PRESTO_S2_BANDS, rr, cc, lls,
                                  start_month, encoder, device, batch_size, logger,
                                  s1_stack=s1_30, s1_band_names=s1_bands)
            emb_30[r0 * W30:r1 * W30] = emb
        emb_30 = emb_30.reshape(H30, W30, pp.PRESTO_N_CHANNELS).transpose(2, 0, 1)
        return emb_30, transform_30, s2_crs

    # embed10_downsample: embed every 10 m pixel, scatter-average into the 30 m grid.
    H2, W2 = (H // f) * f, (W // f) * f
    stack = stack[:, :, :H2, :W2]
    s1_10 = s1_full[:, :, :H2, :W2] if s1_full is not None else None
    H30, W30 = H2 // f, W2 // f
    acc = np.zeros((H30, W30, pp.PRESTO_N_CHANNELS), dtype=np.float64)
    cnt = np.zeros((H30, W30), dtype=np.int64)
    rblocks = range(0, H2, block_rows * f)
    if _HAS_TQDM:
        rblocks = tqdm(list(rblocks), desc="  presto[embed10]", leave=False)
    for r0 in rblocks:
        r1 = min(r0 + block_rows * f, H2)
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(W2), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        lls = pixel_latlons(rr, cc, s2_transform, s2_crs)
        emb = pp.embed_pixels(stack, pp.PRESTO_S2_BANDS, rr, cc, lls,
                              start_month, encoder, device, batch_size, logger,
                              s1_stack=s1_10, s1_band_names=s1_bands)
        r30, c30 = rr // f, cc // f
        np.add.at(acc, (r30, c30), emb)
        np.add.at(cnt, (r30, c30), 1)
    cnt = np.maximum(cnt, 1)[:, :, None]
    emb_30 = (acc / cnt).astype(np.float32).transpose(2, 0, 1)
    return emb_30, transform_30, s2_crs


# ---------------------------------------------------------------------------
# GEDI (dense, kept everywhere)
# ---------------------------------------------------------------------------

def load_gedi_dense(healpix_id: int, gedi_dir: Path):
    """Return (y_grid (H,W) m with NaN where invalid, transform, crs, H, W)."""
    data, nodata, transform, crs, H, W = pp.load_gedi_tile(healpix_id, gedi_dir)
    valid = (data != nodata) & (data > 0) & (data < pp.GEDI_MAX_HEIGHT_CM)
    y = np.where(valid, data.astype(np.float32) / 100.0, np.nan).astype(np.float32)
    return y, transform, crs, H, W


def save_dense_npz(out_path: Path, X_grid, y_grid, transform, crs):
    """X_grid: (C,H,W) -> stored as (H,W,C). y_grid: (H,W)."""
    np.savez_compressed(
        out_path,
        X=np.ascontiguousarray(X_grid.transpose(1, 2, 0)),
        y=y_grid,
        grid_transform=np.asarray(tuple(transform)[:6], dtype=np.float64),
        grid_crs=np.asarray(crs.to_wkt()),
        grid_shape=np.asarray(y_grid.shape, dtype=np.int32),
    )


def save_xy_npz(xy_dir: Path, healpix_id: int, X_grid, y_grid) -> int:
    """Write the valid-GEDI-pixel (X, y) format consumed by
    run_gedi_prediction_30m.py --preprocessed-dir --embedding-format float32.

    X_grid: (C,H,W); y_grid: (H,W) with NaN where GEDI is invalid. Keeps only
    pixels with finite GEDI AND a fully finite embedding vector (drops no-coverage
    rows). File is named healpix_<id>_float32.npz so the per-model xy_dir can be
    passed straight to --preprocessed-dir. Returns the number of samples kept.
    """
    X_hwc = X_grid.transpose(1, 2, 0)                 # (H, W, C)
    mask = np.isfinite(y_grid)
    Xv = X_hwc[mask]                                   # (N, C)
    yv = y_grid[mask]                                  # (N,)
    finite = np.all(np.isfinite(Xv), axis=1)
    Xv, yv = Xv[finite], yv[finite]
    xy_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(xy_dir / f"healpix_{healpix_id}_float32.npz",
                        X=Xv.astype(np.float32), y=yv.astype(np.float32))
    return int(len(yv))


def prithvi_valid_xy_lowmem(emb_tif: Path, y_grid: np.ndarray, g_transform, g_crs,
                            gH: int, gW: int, chunk_bands: int, logger,
                            reduce_blocks: str = "mean", block_dim: int = 1024):
    """Reproject a Prithvi embeddings geotiff onto the GEDI grid in BAND CHUNKS, keeping
    only GEDI-valid, fully-finite pixels — WITHOUT ever holding the full dense
    (C, gH, gW) grid in memory.

    The dense grid is C * gH * gW * 4 bytes (e.g. ~25 GB for C=1024 on a ~2500 px tile,
    ~275 GB for C=11264); allocating it is what OOM-killed export. GEDI coverage is sparse
    (footprint tracks, not wall-to-wall), so we reproject bands in chunks and immediately
    subset to the GEDI-valid mask.

    BLOCK REDUCTION: rslearn's PrithviV2 concatenates the 11 transformer-block feature maps,
    so the geotiff has C = n_blocks * block_dim bands (e.g. 11 * 1024 = 11264). Reprojecting
    all 11264 bands is ~11x the work (a 30-min single-tile export). `reduce_blocks` collapses
    the blocks to `block_dim` BEFORE the reproject (the source geotiff is small — read whole,
    reduce, then reproject block_dim bands):
      - "mean": average the n_blocks feature maps (default; standard multi-depth pooling)
      - "last": keep only the final block's features
      - "none": keep all C bands (slow/large)
    Reduction only applies when C is an exact multiple of block_dim and n_blocks > 1.

    Returns (Xv, yv, eff_C): Xv (N, eff_C) float32 embeddings, yv (N,) heights, eff_C the
    band count actually written (block_dim if reduced, else C).
    """
    mask = np.isfinite(y_grid)                         # (gH, gW) — sparse GEDI
    yv = y_grid[mask].astype(np.float32)               # (N,)
    with rasterio.open(emb_tif) as src:
        C, st, sc, h, w = src.count, src.transform, src.crs, src.height, src.width
        n_blocks = C // block_dim if (block_dim and C % block_dim == 0) else 1
        reduced = None
        if reduce_blocks in ("mean", "last") and n_blocks > 1:
            # Source is small (h,w ~ tile/48); read all bands once, then collapse blocks.
            all_bands = src.read().astype(np.float32).reshape(n_blocks, block_dim, h, w)
            reduced = all_bands.mean(axis=0) if reduce_blocks == "mean" else all_bands[-1]
            del all_bands
            eff_C = block_dim
            logger.info("Prithvi export: %s-reduced %d bands (%d blocks x %d) -> %d before reproject",
                        reduce_blocks, C, n_blocks, block_dim, eff_C)
        else:
            eff_C = C
            if reduce_blocks in ("mean", "last"):
                logger.info("Prithvi export: no block reduction (C=%d not a multiple of "
                            "block_dim=%d, or single block)", C, block_dim)

        Xv = np.empty((int(mask.sum()), eff_C), dtype=np.float32)
        for c0 in range(0, eff_C, chunk_bands):
            c1 = min(c0 + chunk_bands, eff_C)
            if reduced is not None:
                emb_chunk = reduced[c0:c1]
            else:
                # rasterio band indexes are 1-based; read only this chunk from disk.
                emb_chunk = src.read(list(range(c0 + 1, c1 + 1))).astype(np.float32)  # (chunk,h,w)
            grid = reproject_to_grid(emb_chunk, st, sc, g_transform, g_crs, gH, gW)  # (chunk,gH,gW)
            Xv[:, c0:c1] = grid[:, mask].T             # (N, chunk)
            del emb_chunk, grid
        logger.info("Prithvi export: %d bands reprojected in chunks of %d; %d GEDI-valid px",
                    eff_C, chunk_bands, len(yv))
    # Drop pixels that fell outside embedding coverage (NaN in any band).
    finite = np.all(np.isfinite(Xv), axis=1)
    return Xv[finite], yv[finite], eff_C


# ---------------------------------------------------------------------------
# Per-tile export (after the shared materialize + OLMo predict)
# ---------------------------------------------------------------------------

def export_tile(healpix_id, dataset_path, gedi_dir, output_dir, start_month, sampling,
                encoder, device, batch_size, block_rows, do_olmo, do_presto, overwrite, logger,
                do_prithvi=False, emit_xy=False, with_s1=False, prithvi_chunk_bands=64,
                prithvi_reduce_blocks="mean", prithvi_block_dim=1024):
    win_dir = window_dir(dataset_path, healpix_id)
    y, g_transform, g_crs, gH, gW = load_gedi_dense(healpix_id, gedi_dir)

    results = {"healpix_id": int(healpix_id)}

    if do_prithvi:
        # Prithvi's embedding dim (1024+) makes a dense (C, gH, gW) grid at GEDI
        # resolution enormous (~25 GB for C=1024 on a ~2500 px tile) — building it
        # OOM-killed export. Since GEDI is sparse, we go straight to the (X, y) table
        # via a band-chunked reproject and never form the dense grid. Output is the
        # xy npz consumed by run_gedi_prediction_30m --preprocessed-dir.
        xy_dir = output_dir / "prithvi_xy"
        xy_out = xy_dir / f"healpix_{healpix_id}_float32.npz"
        if overwrite or not xy_out.exists():
            emb_tif = find_prithvi_embedding_geotiff(win_dir)
            if emb_tif is None:
                logger.warning("Tile %s: no Prithvi embeddings geotiff (predict may have skipped it)",
                               healpix_id)
                results["prithvi"] = "missing_embeddings"
            else:
                Xv, yv, C = prithvi_valid_xy_lowmem(
                    emb_tif, y, g_transform, g_crs, gH, gW, prithvi_chunk_bands, logger,
                    reduce_blocks=prithvi_reduce_blocks, block_dim=prithvi_block_dim)
                xy_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(xy_out, X=Xv.astype(np.float32), y=yv.astype(np.float32))
                print(f"  Tile {healpix_id}: Prithvi (xy) -> {xy_out.name}  "
                      f"N={len(yv)} px  C={C}")
                results["prithvi"] = "ok"
        else:
            results["prithvi"] = "skipped"

    if do_olmo:
        out = output_dir / f"healpix_{healpix_id}_olmo.npz"
        # Same xy-aware skip as the Presto branch below — see the comment there.
        xy_out = output_dir / "olmo_xy" / f"healpix_{healpix_id}_float32.npz"
        if overwrite or not out.exists() or (emit_xy and not xy_out.exists()):
            emb_tif = find_olmo_embedding_geotiff(win_dir)
            if emb_tif is None:
                logger.warning("Tile %s: no OLMo embeddings geotiff (predict may have skipped it)", healpix_id)
                results["olmo"] = "missing_embeddings"
            else:
                with rasterio.open(emb_tif) as src:
                    emb = src.read().astype(np.float32)  # (128, h, w) UTM 30 m
                    emb_grid = reproject_to_grid(emb, src.transform, src.crs,
                                                 g_transform, g_crs, gH, gW)
                save_dense_npz(out, emb_grid, y, g_transform, g_crs)
                msg = f"  Tile {healpix_id}: OLMo -> {out.name}  X={emb_grid.shape}"
                if emit_xy:
                    n = save_xy_npz(output_dir / "olmo_xy", healpix_id, emb_grid, y)
                    msg += f"  (+xy: {n} px)"
                print(msg)
                results["olmo"] = "ok"
        else:
            print(f"  Tile {healpix_id}: OLMo SKIPPED — {out.name} already exists "
                  f"(pass --overwrite to recompute)")
            results["olmo"] = "skipped"

    if do_presto:
        out = output_dir / f"healpix_{healpix_id}_presto.npz"
        xy_out = output_dir / "presto_xy" / f"healpix_{healpix_id}_float32.npz"
        # Recompute if the dense npz is missing OR --emit-xy was asked for and the (X,y)
        # file is missing. save_xy_npz runs ONLY in the compute branch below, so keying
        # the skip on the dense npz alone means a tile first embedded without --emit-xy
        # (or one whose xy write failed, e.g. on a full disk) can never produce its (X,y):
        # every later run sees the dense file, skips, and exits 0 having written nothing.
        if overwrite or not out.exists() or (emit_xy and not xy_out.exists()):
            emb_30, t30, s2_crs = presto_dense_30m(
                win_dir, start_month, sampling, encoder, device, batch_size, block_rows, logger,
                with_s1=with_s1)
            if emb_30 is None:
                logger.warning("Tile %s: no S2 imagery for Presto", healpix_id)
                results["presto"] = "no_imagery"
            else:
                emb_grid = reproject_to_grid(emb_30, t30, s2_crs, g_transform, g_crs, gH, gW)
                save_dense_npz(out, emb_grid, y, g_transform, g_crs)
                msg = f"  Tile {healpix_id}: Presto[{sampling}] -> {out.name}  X={emb_grid.shape}"
                if emit_xy:
                    n = save_xy_npz(output_dir / "presto_xy", healpix_id, emb_grid, y)
                    msg += f"  (+xy: {n} px)"
                print(msg)
                results["presto"] = "ok"
        else:
            # Say so on STDOUT. A silent skip is indistinguishable in the slurm .out from
            # a tile that failed, since every logger/tqdm line goes to stderr (.err).
            print(f"  Tile {healpix_id}: Presto SKIPPED — {out.name} already exists "
                  f"(pass --overwrite to recompute)")
            results["presto"] = "skipped"

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--shapefile", required=True, help="healpix shapefile with healpix_id + split")
    ap.add_argument("--gedi-dir", required=True, help="dir with healpix_<id>_gedi.tif")
    ap.add_argument("--dataset-path", required=True, help="shared rslearn dataset root")
    ap.add_argument("--output-dir", required=True, help="dir for per-tile *_olmo.npz / *_presto.npz")
    ap.add_argument("--olmo-model-config", default=None,
                    help="path to your OLMo model-gpu.yaml (required unless --skip-olmo)")
    ap.add_argument("--prithvi-model-config", default=None,
                    help="path to prithvi_predict.yaml (required when --do-prithvi unless "
                         "--skip-download/--dry-run)")
    ap.add_argument("--dataset-config", default=None,
                    help="optional config.json to copy as the shared dataset config "
                         "(default: built-in OLMo config)")
    ap.add_argument("--healpix-id", type=int, default=None, help="single tile (omit for batch)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--start-month", type=int, default=1, help="calendar month of S2 timestep 0")
    ap.add_argument("--workers", type=int, default=32, help="rslearn prepare/materialize workers")
    ap.add_argument("--rslearn-prefix", default="",
                    help='wrapper for rslearn cmds, e.g. "mamba run -n jpl-py313"')
    ap.add_argument("--presto-sampling", choices=["aggregate30", "nearest30", "embed10_downsample"],
                    default="embed10_downsample",
                    help="how Presto samples 10 m S2 onto the 30 m grid (see module docstring)")
    ap.add_argument("--batch-size", type=int, default=4096, help="pixels per Presto forward pass")
    ap.add_argument("--block-rows", type=int, default=64,
                    help="30 m rows per Presto block (memory/throughput tradeoff)")
    ap.add_argument("--skip-olmo", action="store_true")
    ap.add_argument("--skip-presto", action="store_true")
    ap.add_argument("--with-s1", action="store_true",
                    help="feed Sentinel-1 VV/VH (dB) alongside S2 into Presto. Requires a "
                         "'sentinel1' layer (bands vv,vh, same 12-month query as S2) in the "
                         "dataset config. Prithvi is optical-only and ignores this.")
    ap.add_argument("--do-prithvi", action="store_true",
                    help="also run Prithvi-EO-2.0 predict + export the (X,y) table to "
                         "<output-dir>/prithvi_xy/healpix_<id>_float32.npz")
    ap.add_argument("--prithvi-chunk-bands", type=int, default=64,
                    help="Prithvi export: number of embedding bands to reproject onto the "
                         "GEDI grid at a time (default 64). Peak export RAM ~ this * gH * gW "
                         "* 4 bytes; lower it (e.g. 32/16) if export OOMs on a big tile.")
    ap.add_argument("--prithvi-reduce-blocks", choices=["mean", "last", "none"], default="mean",
                    help="rslearn's PrithviV2 stacks the 11 transformer-block feature maps "
                         "(C = 11 * block-dim = 11264). Collapse them to block-dim BEFORE the "
                         "reproject: 'mean' (default, ~11x faster export + smaller table), "
                         "'last' (final block only), or 'none' (keep all 11264, slow/large).")
    ap.add_argument("--prithvi-block-dim", type=int, default=PRITHVI_EMB_DIM,
                    help=f"per-block Prithvi feature width for --prithvi-reduce-blocks "
                         f"(default {PRITHVI_EMB_DIM}; VIT_300/600 use 1024). Reduction applies "
                         "only when the geotiff band count is an exact multiple of this.")
    ap.add_argument("--prithvi-emb-dim", type=int, default=PRITHVI_EMB_DIM,
                    help="num_bands for the prithvi_embeddings output layer in the built-in "
                         f"dataset config (default {PRITHVI_EMB_DIM}). Set to the value rslearn's "
                         "PrithviV2 actually writes if `model predict` errors on a band-count "
                         "mismatch. Ignored when --dataset-config is provided.")
    ap.add_argument("--crop-fraction", type=float, default=None, metavar="F",
                    help="Shrink each tile's window to the central F of its width/height "
                         "(F=0.1 -> ~1%% of the area) for a fast subset smoke test. The "
                         "embeddings still reproject onto the full GEDI grid (only the "
                         "cropped sub-window is covered). Not for production embeddings.")
    ap.add_argument("--emit-xy", action="store_true",
                    help="in addition to the dense healpix_<id>_<model>.npz, write the valid-"
                         "GEDI-pixel (X,y) format to <output-dir>/<model>_xy/healpix_<id>_"
                         "float32.npz, ready for run_gedi_prediction_30m.py --preprocessed-dir.")
    ap.add_argument("--materialize-only", action="store_true",
                    help="PHASE 1 (run on a CPU node): add_windows + prepare + materialize "
                         "the shared S2/S1, then STOP before any model predict/encode. No "
                         "GPU needed. Then run phase 2 on GPU with --skip-download.")
    ap.add_argument("--skip-download", action="store_true",
                    help="PHASE 2 (run on GPU): imagery already materialized; skip "
                         "add_windows/prepare/materialize and go straight to Prithvi/OLMo "
                         "predict + Presto encode + export, reusing the same --dataset-path.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the windows + expected input/output paths and exit "
                         "(no rslearn, no download, no encoding)")
    args = ap.parse_args()

    do_olmo, do_presto, do_prithvi = not args.skip_olmo, not args.skip_presto, args.do_prithvi
    if (do_olmo and not args.olmo_model_config and not args.skip_download
            and not args.materialize_only and not args.dry_run):
        ap.error("--olmo-model-config is required unless --skip-olmo, --skip-download, "
                 "--materialize-only, or --dry-run")
    if (do_prithvi and not args.prithvi_model_config and not args.skip_download
            and not args.materialize_only and not args.dry_run):
        ap.error("--prithvi-model-config is required with --do-prithvi unless "
                 "--skip-download, --materialize-only, or --dry-run")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    gedi_dir = Path(args.gedi_dir)
    logger = setup_logging(output_dir, args.verbose)
    prefix = args.rslearn_prefix.split() if args.rslearn_prefix else []

    gdf = gpd.read_file(args.shapefile)
    shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
    print(f"Loaded shapefile: {len(gdf)} tiles")

    if args.healpix_id is not None:
        tile_ids = [args.healpix_id]
    elif "split" in gdf.columns:
        tile_ids = gdf[gdf["split"].isin(args.splits)]["healpix_id"].tolist()
    else:
        tile_ids = gdf["healpix_id"].tolist()
    tile_ids = [int(t) for t in tile_ids]
    print(f"Tiles to process: {len(tile_ids)}  year={args.year}  presto={args.presto_sampling}")

    # ----- Dry run: report the layout and exit before doing any work -----
    if args.dry_run:
        print("\n=== DRY RUN — no rslearn, no download, no encoding ===")
        print(f"dataset-path : {dataset_path}")
        print(f"config       : {'(copy) ' + args.dataset_config if args.dataset_config else '(built-in OLMo config)'}")
        print(f"output-dir   : {output_dir}")
        print(f"rslearn cmds : {' '.join(prefix + ['rslearn', '...']) if prefix else 'rslearn ...'}")
        print(f"olmo         : {'predict --config ' + str(args.olmo_model_config) if do_olmo else 'SKIPPED'}")
        print(f"prithvi      : {'predict --config ' + str(args.prithvi_model_config) if do_prithvi else 'SKIPPED'}")
        print(f"presto       : {args.presto_sampling if do_presto else 'SKIPPED'}")
        print(f"emit (X,y)   : {'yes -> <output-dir>/<model>_xy/healpix_<id>_float32.npz' if args.emit_xy else 'no'}")
        if do_prithvi and not args.dataset_config:
            print(f"prithvi dim  : num_bands={args.prithvi_emb_dim} (override via --prithvi-emb-dim)")
        print(f"window range : {args.year}-01-01 .. {args.year + 1}-01-01  (UTM, 10 m)")
        print(f"\n{len(tile_ids)} window(s):")
        for tid in tile_ids:
            minx, miny, maxx, maxy = pp.get_tile_bounds(tid, gdf)
            lon_min, lat_min, lon_max, lat_max = pp.bounds_to_crs(
                minx, miny, maxx, maxy, shp_crs, CRS.from_epsg(4326))
            if args.crop_fraction:
                lon_min, lat_min, lon_max, lat_max = pp.crop_bounds_centered(
                    lon_min, lat_min, lon_max, lat_max, args.crop_fraction)
            wd = window_dir(dataset_path, tid)
            gpath = next((str(gedi_dir / f"healpix_{tid}_gedi{e}")
                          for e in (".tif", ".tiff", "")
                          if (gedi_dir / f"healpix_{tid}_gedi{e}").exists()),
                         f"{gedi_dir}/healpix_{tid}_gedi.tif (MISSING)")
            print(f"\n  healpix_{tid}")
            print(f"    box (4326)   : {lon_min:.5f},{lat_min:.5f},{lon_max:.5f},{lat_max:.5f}")
            print(f"    window dir   : {wd}")
            print(f"    S2 in        : {wd}/layers/{OLMO_S2_LAYER}[.1..11]/{OLMO_S2_BANDSET}/geotiff.tif")
            if do_olmo:
                print(f"    OLMo emb in  : {wd}/layers/embeddings/*/geotiff.tif")
                print(f"    OLMo out     : {output_dir}/healpix_{tid}_olmo.npz")
            if do_prithvi:
                print(f"    Prithvi emb  : {wd}/layers/{PRITHVI_EMB_LAYER}/*/geotiff.tif")
                print(f"    Prithvi out  : {output_dir}/healpix_{tid}_prithvi.npz")
            if do_presto:
                print(f"    Presto out   : {output_dir}/healpix_{tid}_presto.npz")
            print(f"    GEDI grid    : {gpath}")
        print(f"\nTotal windows: {len(tile_ids)}. Re-run without --dry-run to execute.")
        return

    # Ensure the shared dataset config.json is current — needed by BOTH the
    # materialize (phase 1) and the model-predict (phase 2) steps, so it is written
    # unconditionally (it is idempotent). This also makes --prithvi-emb-dim take
    # effect in a predict-only (--skip-download) GPU run.
    write_dataset_config(dataset_path, args.dataset_config,
                         prithvi_emb_dim=args.prithvi_emb_dim if do_prithvi else None)

    # ----- Phase 1: shared S2/S1 download (materialize ONCE) — CPU/IO bound -----
    # Run this on a cheap CPU node. Skip it on the GPU run with --skip-download.
    if not args.skip_download:
        print("\n=== Building shared dataset (add_windows) ===")
        for tid in tile_ids:
            minx, miny, maxx, maxy = pp.get_tile_bounds(tid, gdf)
            lon_min, lat_min, lon_max, lat_max = pp.bounds_to_crs(
                minx, miny, maxx, maxy, shp_crs, CRS.from_epsg(4326))
            if args.crop_fraction:
                lon_min, lat_min, lon_max, lat_max = pp.crop_bounds_centered(
                    lon_min, lat_min, lon_max, lat_max, args.crop_fraction)
            add_window(dataset_path, tid, lon_min, lat_min, lon_max, lat_max,
                       args.year, prefix, logger)
        print("\n=== prepare + materialize (download S2 + S1 ONCE) ===")
        prepare_materialize(dataset_path, args.workers, prefix, logger)

    # Stop here after the (CPU) download so no GPU hours are spent on imagery I/O.
    if args.materialize_only:
        print(f"\n--materialize-only: S2/S1 materialized to {dataset_path}.\n"
              f"Run the embedding phase on GPU with the SAME --dataset-path and "
              f"--skip-download (+ --do-prithvi / --prithvi-model-config as needed).")
        return

    # ----- Phase 2: model predicts (reuse materialized imagery) — GPU -----
    if do_olmo:
        print("\n=== OLMo model predict ===")
        olmo_predict(dataset_path, args.olmo_model_config, prefix, logger)
    if do_prithvi:
        print("\n=== Prithvi model predict ===")
        prithvi_predict(dataset_path, args.prithvi_model_config, prefix, logger)

    # ----- Presto encoder (in-process) -----
    encoder = device = None
    if do_presto:
        print("\nLoading Presto encoder...")
        encoder, device = pp.load_presto()
        print(f"Presto ready on {device}")

    # ----- Per-tile export (OLMo/Presto: dense (H,W,C); Prithvi: band-chunked (X,y) only) -----
    print("\n=== Exporting per-tile embeddings ===")
    results = []
    iterator = tqdm(tile_ids, desc="export") if _HAS_TQDM else tile_ids
    for tid in iterator:
        try:
            results.append(export_tile(
                tid, dataset_path, gedi_dir, output_dir, args.start_month,
                args.presto_sampling, encoder, device, args.batch_size, args.block_rows,
                do_olmo, do_presto, args.overwrite, logger, do_prithvi=do_prithvi,
                emit_xy=args.emit_xy, with_s1=args.with_s1,
                prithvi_chunk_bands=args.prithvi_chunk_bands,
                prithvi_reduce_blocks=args.prithvi_reduce_blocks,
                prithvi_block_dim=args.prithvi_block_dim))
        except FileNotFoundError as exc:
            logger.warning("Tile %s: %s", tid, exc)
            results.append({"healpix_id": tid, "error": str(exc)})
        except Exception as exc:
            logger.error("Tile %s: %s", tid, exc, exc_info=True)
            results.append({"healpix_id": tid, "error": str(exc)})

    summary = {"year": args.year, "n_tiles": len(tile_ids),
               "presto_sampling": args.presto_sampling, "tiles": results}
    with open(output_dir / f"unified_summary_{args.year}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
