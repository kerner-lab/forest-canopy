#!/usr/bin/env python3
"""
Build an ETH Global Sentinel-2 10 m Canopy Height (2020) baseline over the GEDI
*test* HEALPix tiles, so it can be evaluated with the exact same analysis scripts
(evaluate.py / evaluate_bins.py) used for the embedding maps and the Potapov
baseline.

ETH map (Lang et al. 2023, https://doi.org/10.1038/s41477-023-01454-5) is hosted
on GEE as a single global IMAGE (not an ImageCollection):
  Image id: users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1
  Band:     b1  = canopy top height in metres.
  Native:   EPSG:4326 @ 8.333333e-05 deg (~9.28 m), origin aligned to the global
            grid (edges = n * NATIVE_RES).
  Nodata:   pixels are MASKED (e.g. water). IMPORTANT: 0 m is a REAL prediction
            (bare ground), NOT nodata -- unlike Potapov where 0 == nodata.

For each test tile we:
  1. Open the matching GEDI tile (healpix_<id>_gedi.tif) to get the exact output
     grid (CRS=EPSG:4326, transform, width, height) -- the same grid the
     embeddings are reprojected onto, so pixel indices align directly.
  2. Download the ETH map on its NATIVE 10 m grid over the tile bounds
     (masked pixels unmasked to a sentinel so real 0 m survives), then reproject
     it locally onto the GEDI 30 m grid. Default resampling is AVERAGE (the map
     is 10 m and the target is 30 m, so each 30 m cell averages ~9 valid 10 m
     pixels; sentinel/masked pixels are excluded from the average).
  3. Sample ETH at the *same valid GEDI pixels* the model uses
     (gedi != nodata, gedi > 0, gedi < 12000 cm), giving (y_true, y_pred) pairs
     in metres.

Outputs (in --out-dir):
  eth_tiles/healpix_<id>_eth.tif            reprojected ETH per tile (cache),
                                            nodata = SENTINEL (masked pixels).
  predictions_test.parquet                  y_true, y_pred (m) at valid GEDI
                                            pixels where ETH is UNMASKED. Real
                                            0 m predictions are kept; only masked
                                            (water/nodata) pixels are dropped.
                                            This is the fair, primary file.
  predictions_test_maskfill0.parquet        same GEDI pixel set as the embedding
                                            maps: masked ETH pixels filled with
                                            0 m instead of dropped (for exact
                                            pixel-count parity if needed).

Then run, e.g. (evaluate.py auto-names the plot <parquet>_evaluate.png):
  python evaluate.py --parquet <out-dir>/predictions_test.parquet
  python evaluate_bins.py --parquet <out-dir>/predictions_test.parquet
"""

import argparse
import json
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import requests

import ee

# Same metric code the RF runs use, so baseline and model numbers are computed
# identically. This file lives in data_processing/, hence the parent on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_stratified import evaluate_and_report  # noqa: E402

# ETH canopy height native grid: EPSG:4326, 8.333333e-05 deg (~9.28 m), aligned
# to the global grid where every pixel edge is an integer multiple of NATIVE_RES.
NATIVE_RES = 1.0 / 12000.0  # = 8.333333e-05 deg

# Sentinel written where ETH is masked (water / no data). Chosen well outside the
# plausible 0..~120 m canopy range so it is trivially separable from real 0 m.
SENTINEL = -9999.0

_RESAMPLING = {
    'average': Resampling.average,
    'nearest': Resampling.nearest,
    'bilinear': Resampling.bilinear,
}


def init_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception as e:
        print(f"ee.Initialize failed ({e}); attempting ee.Authenticate()...")
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"Earth Engine initialised (project={project}).")


def build_eth_image(image_id, band):
    """Return the ETH canopy-height band as a single float image.

    The map is a single global Image, so its native EPSG:4326 projection is
    preserved (no mosaic()). We unmask masked pixels to SENTINEL *before*
    download so that genuine 0 m (bare ground) predictions are not confused with
    water/nodata, which would otherwise export as 0 or nan.
    """
    img = ee.Image(image_id)
    bands = img.bandNames().getInfo()
    if band not in bands:
        band = bands[0]
    print(f"ETH image '{image_id}' -> band '{band}' (bands available: {bands})")
    return img.select([band]).rename('canopy_height').toFloat().unmask(SENTINEL)


def _snap(value, res, func):
    return func(value / res) * res


# getDownloadURL caps a single request at ~48 MiB. Keep blocks comfortably under
# that (EE counts a few bytes/pixel for the request) by limiting pixels/block.
MAX_PX_PER_REQUEST = 8_000_000


def _download_window(img, left, top, w, h, retries=4):
    """Download a single native-grid window; returns float32 array (h, w).

    Masked ETH pixels arrive as SENTINEL (from the server-side unmask); any
    non-finite value that slips through is also mapped to SENTINEL. Real 0 m
    values are preserved.
    """
    crs_transform = [NATIVE_RES, 0, left, 0, -NATIVE_RES, top]
    right = left + w * NATIVE_RES
    bottom = top - h * NATIVE_RES
    region = ee.Geometry.Rectangle([left, bottom, right, top],
                                   proj='EPSG:4326', geodesic=False)
    params = {
        'crs': 'EPSG:4326',
        'crsTransform': crs_transform,
        'dimensions': f'{w}x{h}',
        'region': region,
        'format': 'GEO_TIFF',
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            url = img.getDownloadURL(params)
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            content = r.content
            if content[:2] == b'PK':  # zip fallback
                zf = zipfile.ZipFile(io.BytesIO(content))
                tif_name = [n for n in zf.namelist() if n.lower().endswith('.tif')][0]
                content = zf.read(tif_name)
            with MemoryFile(content) as mem, mem.open() as ds:
                arr = ds.read(1).astype(np.float32)
            return np.where(np.isfinite(arr), arr, SENTINEL).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"    download attempt {attempt}/{retries} failed: "
                  f"{str(e)[:160]}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed window after {retries} attempts: {last_err}")


def download_native(img, bounds, retries=4):
    """Download `img` on its NATIVE grid (EPSG:4326 @ NATIVE_RES) over `bounds`.

    The grid is snapped to the global grid (edges = n * NATIVE_RES). Downloading
    on the native grid preserves the real 10 m data; we reproject locally
    afterwards so we control the 10 m -> 30 m resampling. Large tiles exceed EE's
    per-request size cap, so the grid is fetched in horizontal row blocks and
    stitched.

    Returns (array, Affine transform) for the native-grid raster.
    """
    left = _snap(bounds.left, NATIVE_RES, np.floor)
    right = _snap(bounds.right, NATIVE_RES, np.ceil)
    top = _snap(bounds.top, NATIVE_RES, np.ceil)
    bottom = _snap(bounds.bottom, NATIVE_RES, np.floor)
    w = int(round((right - left) / NATIVE_RES))
    h = int(round((top - bottom) / NATIVE_RES))

    rows_per_block = max(1, MAX_PX_PER_REQUEST // w)
    arr = np.zeros((h, w), dtype=np.float32)
    n_blocks = (h + rows_per_block - 1) // rows_per_block
    if n_blocks > 1:
        print(f"    native grid {w}x{h} -> {n_blocks} row blocks "
              f"({rows_per_block} rows each)")
    for r0 in range(0, h, rows_per_block):
        bh = min(rows_per_block, h - r0)
        block_top = top - r0 * NATIVE_RES
        arr[r0:r0 + bh, :] = _download_window(img, left, block_top, w, bh,
                                              retries=retries)
    return arr, Affine(NATIVE_RES, 0, left, 0, -NATIVE_RES, top)


def download_tile(img, transform, width, height, bounds, resampling,
                  crs_epsg=4326, retries=4):
    """Return ETH resampled to the GEDI tile grid (transform/width/height).

    Downloads the native 10 m ETH over `bounds`, then reprojects onto the target
    30 m grid so pixel indices align with the GEDI tile. SENTINEL (masked) pixels
    are treated as nodata, so with `average` resampling each 30 m cell is the mean
    of only its valid 10 m pixels; cells with no valid source stay SENTINEL.
    """
    native, native_transform = download_native(img, bounds, retries=retries)
    dst = np.full((height, width), SENTINEL, dtype=np.float32)
    reproject(
        source=native,
        destination=dst,
        src_transform=native_transform,
        src_crs=f'EPSG:{crs_epsg}',
        dst_transform=transform,
        dst_crs=f'EPSG:{crs_epsg}',
        resampling=resampling,
        src_nodata=SENTINEL,
        dst_nodata=SENTINEL,
    )
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-only', action='store_true',
                    help='Never contact Earth Engine: skip any tile without a cached '
                         'eth tif instead of downloading it.')
    ap.add_argument('--ee-project', default='ee-kelseydoerksen',
                    help='Google Cloud project registered for Earth Engine.')
    ap.add_argument('--shapefile',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/shapefiles/healpix_shapefile.shp')
    ap.add_argument('--gedi-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2021_healpix',
                    help='GEDI tiles to sample against (default: 2021 tiles).')
    ap.add_argument('--out-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/eth_canopy_baseline_2020')
    ap.add_argument('--tiles-dir', default=None,
                    help='Directory for the cached per-tile ETH rasters (default: '
                         '<out-dir>/eth_tiles). The rasters are the 2020 ETH map on '
                         'the fixed tile grid, so the same cache can be re-sampled '
                         'against any GEDI year without re-downloading.')
    ap.add_argument('--split', default='test')
    ap.add_argument('--image-id',
                    default='users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
    ap.add_argument('--band', default='b1')
    ap.add_argument('--resampling', default='average', choices=list(_RESAMPLING),
                    help='10 m -> 30 m resampling (default: average, best for '
                         'downsampling continuous canopy height).')
    ap.add_argument('--tile-id-column', default='healpix_id')
    ap.add_argument('--gedi-nodata', type=float, default=-9999.0)
    # Keep these identical to the RF runs or the stratified numbers are not
    # comparable (run_gedi_prediction_30m.py uses the same defaults).
    ap.add_argument('--metric-bin-size', type=float, default=10.0,
                    help='Height bin width in m for stratified metrics (default: 10)')
    ap.add_argument('--metric-max-height', type=float, default=60.0,
                    help='Top bin edge in m; taller pixels fall in the last bin')
    ap.add_argument('--metric-min-bin-n', type=int, default=100,
                    help='Bins smaller than this are excluded from the summaries')
    ap.add_argument('--metric-balance-seed', type=int, default=42,
                    help='Seed for the bin-balancing resample (default: 42)')
    ap.add_argument('--tile-list', default=None, metavar='PATH',
                    help='File of healpix ids (one per line) restricting the run to '
                         'those tiles. Use the master intersection list so the '
                         'baseline is evaluated on exactly the tiles the models were.')
    ap.add_argument('--max-tiles', type=int, default=None,
                    help='Optional cap on number of tiles (for quick tests).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-download ETH tiles even if cached tif exists.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else out_dir / 'eth_tiles'
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Earth Engine is initialised on demand, so a fully-cached run needs no
    # credentials and no network at all.
    ee_state = {'img': None}

    def get_eth():
        if ee_state['img'] is None:
            init_ee(args.ee_project)
            ee_state['img'] = build_eth_image(args.image_id, args.band)
        return ee_state['img']
    resampling = _RESAMPLING[args.resampling]

    gdf = gpd.read_file(args.shapefile)
    test_ids = gdf[gdf['split'] == args.split][args.tile_id_column].tolist()
    if args.tile_list:
        keep = {int(t) for t in Path(args.tile_list).read_text().split()
                if t.strip() and not t.strip().startswith('#')}
        before = len(test_ids)
        test_ids = [t for t in test_ids if int(t) in keep]
        missing = sorted(keep - {int(t) for t in test_ids})
        print(f"Tile list: {len(test_ids)}/{before} '{args.split}' tiles kept "
              f"({args.tile_list})")
        if missing:
            print(f"  WARNING: {len(missing)} id(s) in the list are not in this "
                  f"split: {missing}")
        if not test_ids:
            sys.exit("ERROR: tile list removed every tile.")
    if args.max_tiles:
        test_ids = test_ids[:args.max_tiles]
    print(f"{len(test_ids)} '{args.split}' tiles to process. "
          f"resampling={args.resampling}")

    rows = []
    skipped = []
    for i, hid in enumerate(test_ids):
        gedi_path = Path(args.gedi_dir) / f'healpix_{hid}_gedi.tif'
        if not gedi_path.exists():
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: missing GEDI tif -- skip")
            skipped.append(hid)
            continue

        with rasterio.open(gedi_path) as gsrc:
            gedi = gsrc.read(1)
            transform = gsrc.transform
            width, height = gsrc.width, gsrc.height
            bounds = gsrc.bounds
            profile = gsrc.profile.copy()

        nodata = args.gedi_nodata
        valid = (gedi != nodata) & (gedi > 0) & (gedi < 12000)
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: no valid GEDI pixels -- skip")
            skipped.append(hid)
            continue

        eth_tif = tiles_dir / f'healpix_{hid}_eth.tif'
        if eth_tif.exists() and not args.overwrite:
            with rasterio.open(eth_tif) as psrc:
                pred = psrc.read(1).astype(np.float32)
            src_note = 'cached'
        elif args.cache_only:
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: no cached tif and "
                  f"--cache-only -- skip")
            skipped.append(hid)
            continue
        else:
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: downloading ETH "
                  f"({width}x{height}, {n_valid} valid GEDI px)...")
            pred = download_tile(get_eth(), transform, width, height, bounds, resampling)
            profile.update(dtype='float32', count=1, nodata=SENTINEL, compress='lzw')
            with rasterio.open(eth_tif, 'w', **profile) as dst:
                dst.write(pred.astype(np.float32), 1)
            src_note = 'downloaded'

        if pred.shape != gedi.shape:
            print(f"  WARNING tile {hid}: shape mismatch eth{pred.shape} vs "
                  f"gedi{gedi.shape} -- skip")
            skipped.append(hid)
            continue

        # ETH is valid (unmasked) where it is finite and not the sentinel. Real
        # 0 m (bare ground) counts as valid; only masked/water pixels are dropped.
        eth_valid = np.isfinite(pred) & (pred != SENTINEL)

        y_true_cm = gedi[valid].astype(np.float64)          # cm
        y_pred_full = pred[valid].astype(np.float64)         # m, SENTINEL where masked
        masked = ~eth_valid[valid]
        n_masked = int(masked.sum())

        # Primary: drop masked ETH pixels, keep real 0 m.
        keep = ~masked
        tile_df = pd.DataFrame({
            'healpix_id': hid,
            'y_true': (y_true_cm[keep] / 100.0).astype(np.float32),
            'y_pred': y_pred_full[keep].astype(np.float32),
        })
        # Parity variant: same GEDI pixel set, masked ETH filled with 0 m.
        y_pred_fill = np.where(masked, 0.0, y_pred_full)
        tile_df_fill = pd.DataFrame({
            'healpix_id': hid,
            'y_true': (y_true_cm / 100.0).astype(np.float32),
            'y_pred': y_pred_fill.astype(np.float32),
        })
        rows.append((tile_df, tile_df_fill))
        n_zero = int((tile_df['y_pred'].values == 0).sum())
        print(f"  tile {hid} [{src_note}]: {n_valid} px, masked(dropped)="
              f"{n_masked} ({100*n_masked/n_valid:.1f}%), real 0 m kept={n_zero}")

    if not rows:
        print("No data collected; nothing written.")
        sys.exit(1)

    full = pd.concat([r[0] for r in rows], ignore_index=True)
    full_path = out_dir / 'predictions_test.parquet'
    full.to_parquet(full_path, index=False, compression='snappy')

    fill = pd.concat([r[1] for r in rows], ignore_index=True)
    fill_path = out_dir / 'predictions_test_maskfill0.parquet'
    fill.to_parquet(fill_path, index=False, compression='snappy')

    print("\n" + "=" * 78)
    print("ETH 2020 CANOPY HEIGHT BASELINE vs GEDI (test tiles)")
    print("=" * 78)
    print(f"tiles used: {len(rows)}  skipped: {len(skipped)}  "
          f"resampling: {args.resampling}\n")

    kw = dict(bin_size=args.metric_bin_size, max_height=args.metric_max_height,
              min_bin_n=args.metric_min_bin_n, balance_seed=args.metric_balance_seed)
    metrics = {
        'baseline': 'eth_2020',
        'split': args.split,
        'resampling': args.resampling,
        'n_tiles_used': len(rows),
        'n_tiles_skipped': len(skipped),
        'metric_bin_size': args.metric_bin_size,
        'metric_max_height': args.metric_max_height,
        'eth_unmasked': evaluate_and_report(
            full['y_true'].values, full['y_pred'].values,
            'ETH unmasked (primary)', **kw),
    }
    print()
    metrics['masked_filled_zero'] = evaluate_and_report(
        fill['y_true'].values, fill['y_pred'].values,
        'masked filled with 0 m', **kw)

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nwrote {full_path}")
    print(f"wrote {fill_path}")
    print(f"wrote {metrics_path}")
    print(f"eth tiles cached in {tiles_dir}")
    print("\nNOTE: R2 is the number comparable to the RF models; Pearson r is shown\n"
          "      alongside it because r ignores bias and scale, so a systematically\n"
          "      short baseline can post a high r and a negative R2.")


if __name__ == '__main__':
    main()
