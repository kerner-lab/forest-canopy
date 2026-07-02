#!/usr/bin/env python3
"""
Process Clay (v1.5) embeddings per healpix tile, reusing the Presto rslearn + GEDI
plumbing so the two are an apples-to-apples comparison (same Sentinel-2 imagery, same
GEDI pixels, different encoder).

This is the Clay analogue of presto_processing.py. It writes the exact same output
format — healpix_{id}_clay.npz containing (X, y) — so the output directory drops
straight into run_gedi_prediction_30m.py via --preprocessed-dir --embedding-format float32.

  Clay:   https://github.com/Clay-foundation/model  (pip install claymodel; importable
          as `claymodel`. Download the v1.5 checkpoint + configs/metadata.yaml.)
  rslearn: same free Sentinel-2 (Planetary Computer / Element84) used by Presto.

Why we run Clay ourselves (not the Source Cooperative embeddings)
-----------------------------------------------------------------
The Source Cooperative "clay-model-v0-embeddings" archive is the embeddings of Clay
v0's *training chips* (~36k chips, concentrated regions e.g. the SF Bay Area), at fixed
locations/dates — it does NOT cover arbitrary GEDI AOIs and would not align to the GEDI
grid. To compare encoders fairly we run Clay on the SAME imagery Presto sees.

How Clay differs from Presto (important)
----------------------------------------
Presto consumes a PER-PIXEL 12-month time series -> one 128-D vector per pixel.
Clay is a patch-based ViT: it consumes a single multi-band IMAGE (one timestamp) and
emits one ~1024-D vector per PATCH_SIZE x PATCH_SIZE patch (+ a CLS token). So here:

  1. rslearn fetches the same 12-month S2 mosaic stack per tile (UTM, 10 m, 10 bands;
     band order already == Clay's sentinel-2-l2a band_order). [reused from presto_processing]
  2. For the VALID GEDI pixels, map each to its S2 pixel, then to the Clay PATCH that
     contains it (patch = 8 px = 80 m at 10 m GSD).
  3. Run Clay over the chips that contain GEDI pixels, ONCE PER MONTH, take the patch
     tokens, and AVERAGE across the 12 months -> one vector per patch.
  4. Assign each GEDI pixel its patch's averaged embedding -> (X, y); save healpix_{id}_clay.npz.

Two things you MUST verify against your installed claymodel version (clearly marked
NOTE below): (a) how the v1.5 checkpoint is loaded with masking disabled, and (b) the
exact encoder call/return. Everything else (rslearn fetch, GEDI alignment, normalization
stats, datacube fields) is grounded in Clay's configs/metadata.yaml and the v1 wall-to-wall
tutorial.

Single-tile usage (HPC array job, mirrors presto_processing.py):
    python clay_processing.py \\
        --healpix-id 12345 --year 2021 \\
        --shapefile /path/healpix.shp \\
        --gedi-dir  /path/gedi_tifs \\
        --output-dir /path/clay_cache \\
        --rslearn-root /scratch/clay_rslearn \\
        --clay-checkpoint /path/clay-v1.5.ckpt \\
        --s2-source element84 --verbose

Notes:
  - Clay is far heavier than Presto (a ViT). Run it on the GPU partition if you can —
    it auto-uses CUDA when torch.cuda.is_available().
  - Requires: pip install claymodel torch + (already present) rslearn rasterio geopandas pyproj.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio  # noqa: F401  (kept for parity / future raster IO)
from pyproj import CRS

# Reuse the proven rslearn + GEDI plumbing from the Presto pipeline. presto_processing
# imports only numpy/geopandas/rasterio/pyproj/matplotlib at module load (NOT torch/presto),
# so importing it here is cheap and has no extra heavy deps.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import presto_processing as pp  # noqa: E402

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


# ---------------------------------------------------------------------------
# Clay sentinel-2-l2a constants (from Clay configs/metadata.yaml).
# Band order is IDENTICAL to presto_processing.RSLEARN_S2_BANDS, so the materialized
# S2 stack channels already line up with Clay's band_order — no reordering needed.
# ---------------------------------------------------------------------------
CLAY_BANDS = pp.RSLEARN_S2_BANDS  # [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]
#                  blue   green  red    rededge1 rededge2 rededge3 nir    nir08  swir16 swir22
CLAY_WAVELENGTHS_UM = [0.493, 0.56, 0.665, 0.704,  0.74,    0.783,  0.842, 0.865, 1.61,  2.19]
CLAY_MEAN = [1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0]
CLAY_STD  = [1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0]
CLAY_GSD = 10.0            # Sentinel-2 ground sample distance (m); matches our materialize res
CLAY_PATCH_SIZE = 8        # Clay v1.5 patch size (px). 8 px * 10 m = 80 m patches.
CLAY_EMB_DIM = 1024        # Clay v1.5 embedding dimension
CLAY_NUM_TIMESTEPS = 12    # 12 monthly mosaics (same window as Presto)


# ---------------------------------------------------------------------------
# Clay model loading + encode (the two version-sensitive spots)
# ---------------------------------------------------------------------------

def load_clay(checkpoint: str, device: str | None = None):
    """Load the Clay v1.5 encoder in eval mode with masking disabled.

    NOTE (verify against your claymodel version): masking MUST be off for embedding
    extraction (the training default mask_ratio=0.75 would drop 75%% of patches). The
    kwargs below match recent claymodel; if load_from_checkpoint rejects one, set the
    attribute after loading instead (e.g. model.model.mask_ratio = 0.0).
    """
    import torch
    from claymodel.module import ClayMAEModule

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ClayMAEModule.load_from_checkpoint(
        checkpoint,
        mask_ratio=0.0,        # no masking at inference
        shuffle=False,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model, device


def _clay_patch_tokens(model, datacube):
    """Run the Clay encoder on one datacube and return the PATCH tokens (CLS dropped).

    NOTE (verify against your claymodel version): based on the Clay v1 wall-to-wall
    tutorial, `model.model.encoder(datacube)` returns a tuple whose first element is
    `unmsk_patch` of shape (B, 1 + n_patches, dim); index 0 is the CLS token and 1: are
    the spatial patch tokens. If your version exposes the encoder as `model.encoder` or
    returns a bare tensor, adjust here.
    """
    encoder = model.model.encoder if hasattr(model, "model") else model.encoder
    out = encoder(datacube)
    unmsk_patch = out[0] if isinstance(out, (tuple, list)) else out
    return unmsk_patch[:, 1:, :]  # (B, n_patches, dim) — drop CLS


def normalize_timestamp(date: datetime):
    """Clay time encoding: (sin week, cos week, sin hour, cos hour)."""
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week), math.sin(hour), math.cos(hour))


def normalize_latlon(lat: float, lon: float):
    """Clay latlon encoding: (sin lat, cos lat, sin lon, cos lon), lat/lon in radians."""
    lat_r = lat * math.pi / 180
    lon_r = lon * math.pi / 180
    return (math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r))


def month_dates(year: int, start_month: int, n: int) -> list:
    """Mid-month dates for the n monthly timesteps starting at (year, start_month)."""
    out = []
    for t in range(n):
        m0 = (start_month - 1 + t)
        y = year + m0 // 12
        m = m0 % 12 + 1
        out.append(datetime(y, m, 15))
    return out


# ---------------------------------------------------------------------------
# Clay embedding at GEDI pixels (patch-token approach)
# ---------------------------------------------------------------------------

def clay_embed_pixels(
    s2_stack: np.ndarray,      # (T, C, H, W) float32 raw S2 DN, C in CLAY_BANDS order
    s2_rows: np.ndarray,       # (N,) S2-grid row of each GEDI pixel
    s2_cols: np.ndarray,       # (N,) S2-grid col
    latlons: np.ndarray,       # (N, 2) lat, lon per pixel
    dates: list,               # length-T list of datetimes (per monthly timestep)
    model,
    device: str,
    chip_size: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Return (N, CLAY_EMB_DIM) float32 — each GEDI pixel's Clay patch embedding,
    averaged over the T monthly mosaics.

    Strategy: only chips that contain GEDI pixels are embedded. For each such chip we run
    one forward per month (batched as B=T), take patch tokens, reshape to a
    (chip/patch, chip/patch) grid, average over months, then look up each pixel's patch.
    """
    import torch

    T, C, H, W = s2_stack.shape
    P = CLAY_PATCH_SIZE
    if chip_size % P != 0:
        raise ValueError(f"--chip-size {chip_size} must be a multiple of patch size {P}")
    n_pp = chip_size // P  # patches per chip side
    N = len(s2_rows)

    # Pixel -> chip (cr, cc) and within-chip patch (pr, pc).
    chip_r = (s2_rows // chip_size).astype(np.int64)
    chip_c = (s2_cols // chip_size).astype(np.int64)
    patch_r = ((s2_rows % chip_size) // P).astype(np.int64)
    patch_c = ((s2_cols % chip_size) // P).astype(np.int64)

    chips: dict = {}  # (cr, cc) -> list of pixel indices
    for i in range(N):
        chips.setdefault((int(chip_r[i]), int(chip_c[i])), []).append(i)

    mean = np.asarray(CLAY_MEAN, np.float32)[None, :, None, None]
    std = np.asarray(CLAY_STD, np.float32)[None, :, None, None]
    waves = torch.tensor([CLAY_WAVELENGTHS_UM] * T, dtype=torch.float32, device=device)  # (T, C)
    gsd = torch.tensor(float(CLAY_GSD), device=device)
    time_enc = torch.tensor([normalize_timestamp(d) for d in dates],
                            dtype=torch.float32, device=device)  # (T, 4)

    X = np.zeros((N, CLAY_EMB_DIM), np.float32)

    chip_items = list(chips.items())
    iterator = tqdm(chip_items, desc=f"  clay ({len(chip_items)} chips)", leave=False) \
        if _HAS_TQDM else chip_items

    for (cr, cc), idxs in iterator:
        r0, c0 = cr * chip_size, cc * chip_size
        r1, c1 = min(r0 + chip_size, H), min(c0 + chip_size, W)

        # Extract chip (zero-padded at the tile edge). NOTE: edge padding feeds zeros into
        # self-attention; GEDI pixels always fall in the valid region, but if you want to
        # be strict you could reflect-pad instead.
        chip = np.zeros((T, C, chip_size, chip_size), np.float32)
        chip[:, :, : r1 - r0, : c1 - c0] = s2_stack[:, :, r0:r1, c0:c1]
        chip = (chip - mean) / std

        # One chip's latlon ~ mean of its GEDI pixels' latlons (Clay uses one latlon/chip).
        ll = latlons[idxs].mean(axis=0)
        latlon_row = normalize_latlon(float(ll[0]), float(ll[1]))
        latlon_enc = torch.tensor([latlon_row] * T, dtype=torch.float32, device=device)  # (T,4)

        datacube = {
            "pixels": torch.from_numpy(chip).to(device),   # (T, C, chip, chip)
            "time": time_enc,                              # (T, 4)
            "latlon": latlon_enc,                          # (T, 4)
            "gsd": gsd,                                     # scalar
            "waves": waves,                                 # (T, C) micrometers
        }
        with torch.no_grad():
            patches = _clay_patch_tokens(model, datacube)   # (T, n_pp*n_pp, dim)
        # (T, n_pp, n_pp, dim) -> average over months -> (n_pp, n_pp, dim)
        grid = patches.reshape(T, n_pp, n_pp, CLAY_EMB_DIM).mean(dim=0).cpu().numpy()

        for i in idxs:
            X[i] = grid[patch_r[i], patch_c[i]]

    return X


# ---------------------------------------------------------------------------
# Per-tile processing (steps 1-4 reuse presto_processing; step 5 is Clay)
# ---------------------------------------------------------------------------

def process_tile(
    healpix_id, year, start_month, gdf, gedi_dir, output_dir, overwrite,
    model, device, chip_size, rslearn_root, imagery_dir, rslearn_workers,
    logger, keep_zeros=False, subset_fraction=None, s2_source="element84",
) -> dict:
    suffix = "_withzeros" if keep_zeros else ""
    out_path = output_dir / f"healpix_{healpix_id}_clay{suffix}.npz"
    if not overwrite and out_path.exists():
        logger.info(f"Tile {healpix_id}: cached, skipping")
        return {"status": "skipped", "healpix_id": int(healpix_id)}

    t_start = time.time()
    try:
        # 1. Tile bounds in WGS84  [reused]
        shp_minx, shp_miny, shp_maxx, shp_maxy = pp.get_tile_bounds(healpix_id, gdf)
        shp_crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
        lon_min, lat_min, lon_max, lat_max = pp.bounds_to_crs(
            shp_minx, shp_miny, shp_maxx, shp_maxy, shp_crs, CRS.from_epsg(4326))

        subset_bbox = None
        if subset_fraction is not None:
            lon_min, lat_min, lon_max, lat_max = pp.crop_bounds_centered(
                lon_min, lat_min, lon_max, lat_max, subset_fraction)
            subset_bbox = (lon_min, lat_min, lon_max, lat_max)

        # 2. Monthly S2 stack (same rslearn fetch as Presto, incl. chronological-order fix)
        if imagery_dir is not None:
            paths = pp.find_imagery_dir_geotiffs(Path(imagery_dir), healpix_id)
        else:
            win_dir = pp.materialize_tile_rslearn(
                healpix_id, lon_min, lat_min, lon_max, lat_max, year,
                start_month, Path(rslearn_root), logger, workers=rslearn_workers,
                source=s2_source)
            paths = pp.find_monthly_geotiffs(win_dir)
        if len(paths) == 0:
            logger.warning(f"Tile {healpix_id}: no S2 monthly imagery found")
            return {"status": "no_imagery", "healpix_id": int(healpix_id)}
        if len(paths) < CLAY_NUM_TIMESTEPS:
            logger.warning("Tile %s: only %d/%d months available", healpix_id, len(paths),
                           CLAY_NUM_TIMESTEPS)

        s2_stack, s2_transform, s2_crs, s2_h, s2_w = pp.load_s2_stack(paths)
        print(f"  Tile {healpix_id}: S2 stack {s2_stack.shape} ({len(paths)} months)")

        # 3. GEDI target + valid pixels  [reused]
        gedi_data, gedi_nodata, gedi_transform, gedi_crs, gedi_h, gedi_w = \
            pp.load_gedi_tile(healpix_id, gedi_dir)
        valid_mask = (
            (gedi_data != gedi_nodata) &
            (gedi_data >= 0 if keep_zeros else gedi_data > 0) &
            (gedi_data < pp.GEDI_MAX_HEIGHT_CM)
        )
        rows_g, cols_g = np.where(valid_mask)
        if len(rows_g) == 0:
            logger.info(f"Tile {healpix_id}: no valid GEDI pixels")
            return {"status": "no_valid_pixels", "healpix_id": int(healpix_id)}

        # 4. Map GEDI pixels -> S2 pixels; drop those outside the S2 grid  [reused]
        s2_rows, s2_cols, latlons, covered = pp.gedi_pixels_to_s2(
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

        # 5. Clay encode at the covered GEDI pixels (patch tokens, month-averaged).
        t_emb = time.time()
        dates = month_dates(year, start_month, s2_stack.shape[0])
        X = clay_embed_pixels(
            s2_stack, s2_rows[keep_idx], s2_cols[keep_idx], latlons[keep_idx],
            dates, model, device, chip_size, logger)
        logger.info("Tile %s: Clay embedded %d px in %.1fs",
                    healpix_id, len(keep_idx), time.time() - t_emb)

        y = (gedi_data[rows_g[keep_idx], cols_g[keep_idx]] / 100.0).astype(np.float32)
        n_total = len(rows_g)

        # Same npz schema as presto_processing.py (run_gedi_prediction reads X/y by key).
        np.savez_compressed(
            out_path, X=X, y=y,
            emb_rows=rows_g[keep_idx].astype(np.int32),
            emb_cols=cols_g[keep_idx].astype(np.int32),
            grid_transform=np.asarray(tuple(gedi_transform)[:6], dtype=np.float64),
            grid_crs=np.asarray(gedi_crs.to_wkt()),
            grid_shape=np.asarray([gedi_h, gedi_w], dtype=np.int32),
        )
        elapsed = time.time() - t_start
        mem_str = (f"  mem={psutil.Process(os.getpid()).memory_info().rss/1e9:.1f}GB"
                   if _HAS_PSUTIL else "")
        print(f"  Tile {healpix_id}: {len(y):,} samples  {X.shape[1]} ch  "
              f"{elapsed:.1f}s{mem_str}  → {out_path.name}")
        return {
            "status": "ok", "healpix_id": int(healpix_id),
            "n_samples": int(len(y)), "n_valid_gedi": int(n_total),
            "coverage_fraction": round(len(keep_idx) / max(1, n_total), 4),
            "n_channels": int(X.shape[1]), "n_months": len(paths),
            "elapsed_s": round(elapsed, 1),
        }

    except FileNotFoundError as exc:
        logger.warning(f"Tile {healpix_id}: {exc}")
        return {"status": "missing_gedi", "healpix_id": int(healpix_id), "error": str(exc)}
    except Exception as exc:
        logger.error(f"Tile {healpix_id}: unexpected error — {exc}", exc_info=True)
        return {"status": "error", "healpix_id": int(healpix_id), "error": str(exc)}


def setup_logging(output_dir: Path, console_level=logging.WARNING) -> logging.Logger:
    logger = logging.getLogger("clay")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"clay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(console_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(
        description=("Generate Clay v1.5 embeddings per healpix tile (same rslearn S2 as "
                     "Presto) and save (X, y) .npz for run_gedi_prediction_30m.py"),
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
                        help="Directory to write per-tile healpix_*_clay.npz files")
    parser.add_argument("--clay-checkpoint", required=True,
                        help="Path to the Clay v1.5 checkpoint (.ckpt)")
    parser.add_argument("--chip-size", type=int, default=256,
                        help="Clay input chip size in px (multiple of patch size 8)")
    parser.add_argument("--rslearn-root", default=None,
                        help="Working dir for the rslearn dataset (required unless --imagery-dir)")
    parser.add_argument("--imagery-dir", default=None,
                        help="Use pre-materialized healpix_<id>_s2_<month>.tif here instead of rslearn")
    parser.add_argument("--rslearn-workers", type=int, default=2,
                        help="Parallel workers for rslearn prepare/materialize")
    parser.add_argument("--s2-source", choices=["pc", "element84", "gcp"], default="element84",
                        help="Sentinel-2 source (same options as presto_processing.py)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to include in batch mode")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process tiles that already have a cached .npz")
    parser.add_argument("--keep-zeros", action="store_true",
                        help="Keep GEDI height == 0 (output *_clay_withzeros.npz)")
    parser.add_argument("--subset-fraction", type=float, default=None,
                        help="For local testing: crop each tile to this centered fraction "
                             "of its bbox before fetching imagery")
    parser.add_argument("--check-only", action="store_true",
                        help="Report cache coverage and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Console INFO logs + stream rslearn prepare/materialize output")
    args = parser.parse_args()

    if args.imagery_dir is None and args.rslearn_root is None:
        parser.error("provide --rslearn-root (to fetch imagery) or --imagery-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reused presto_processing functions stream rslearn output when this flag is set.
    pp._STREAM_RSLEARN = args.verbose
    logger = setup_logging(output_dir,
                           console_level=logging.INFO if args.verbose else logging.WARNING)

    gdf = gpd.read_file(args.shapefile)
    print(f"Loaded shapefile: {len(gdf)} tiles")

    if args.healpix_id is not None:
        if int(args.healpix_id) not in gdf["healpix_id"].values:
            print(f"Error: healpix_id {args.healpix_id} not found", file=sys.stderr)
            sys.exit(1)
        tile_ids = [args.healpix_id]
        print(f"Single-tile mode: tile {args.healpix_id}  year={args.year}")
    elif "split" in gdf.columns:
        tile_ids = []
        for split_name in args.splits:
            tile_ids.extend(int(t) for t in gdf[gdf["split"] == split_name]["healpix_id"].values)
        print(f"Batch mode: {len(tile_ids)} tiles  splits={args.splits}  year={args.year}")
    else:
        tile_ids = gdf["healpix_id"].tolist()
        print(f"Batch mode: {len(tile_ids)} tiles (no split column)  year={args.year}")

    if args.check_only:
        cache_suffix = "_withzeros" if args.keep_zeros else ""
        cached = [t for t in tile_ids
                  if (output_dir / f"healpix_{t}_clay{cache_suffix}.npz").exists()]
        missing = [t for t in tile_ids if t not in set(cached)]
        print(f"Cached : {len(cached)}/{len(tile_ids)}")
        if missing:
            print(f"Missing: {len(missing)} — first 20: {missing[:20]}")
        return

    print("\nLoading Clay v1.5...")
    model, device = load_clay(args.clay_checkpoint)
    print(f"Clay encoder ready on {device}")

    gedi_dir = Path(args.gedi_dir)
    results = []
    iterator = tqdm(tile_ids, desc="tiles") if _HAS_TQDM else tile_ids
    for healpix_id in iterator:
        results.append(process_tile(
            healpix_id=healpix_id, year=args.year, start_month=args.start_month,
            gdf=gdf, gedi_dir=gedi_dir, output_dir=output_dir, overwrite=args.overwrite,
            model=model, device=device, chip_size=args.chip_size,
            rslearn_root=args.rslearn_root, imagery_dir=args.imagery_dir,
            rslearn_workers=args.rslearn_workers, logger=logger,
            keep_zeros=args.keep_zeros, subset_fraction=args.subset_fraction,
            s2_source=args.s2_source,
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
        "model": "Clay-v1.5", "year": args.year, "start_month": args.start_month,
        "chip_size": args.chip_size, "n_tiles": len(tile_ids),
        "total_samples": total_samples, "status_counts": counts, "tiles": results,
    }
    with open(output_dir / f"clay_summary_{args.year}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {output_dir / f'clay_summary_{args.year}.json'}")


if __name__ == "__main__":
    main()
