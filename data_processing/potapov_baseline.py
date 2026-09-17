#!/usr/bin/env python3
"""
Build a Potapov et al. (2019) global forest canopy height baseline over the
GEDI *test* HEALPix tiles, so it can be evaluated with the exact same analysis
scripts (evaluate.py / evaluate_bins.py) used for the embedding maps.

Potapov 2019 map (https://doi.org/10.1016/j.rse.2020.112165) is hosted on GEE:
  ImageCollection id: users/potapovpeter/GEDI_V27
  Value: forest canopy height in metres. No data = 0.

For each test tile we:
  1. Open the matching GEDI tile (healpix_<id>_gedi.tif) to get the exact
     output grid (CRS=EPSG:4326, transform, width, height) -- the same grid the
     embeddings are reprojected onto, so pixel indices align directly.
  2. Download the Potapov mosaic reprojected onto that grid from Earth Engine.
  3. Sample Potapov at the *same valid GEDI pixels* the model uses
     (gedi != nodata, gedi > 0, gedi < 12000 cm), giving (y_true, y_pred) pairs
     in metres.

Outputs (in --out-dir):
  potapov_tiles/healpix_<id>_potapov.tif   reprojected Potapov per tile (cache)
  predictions_test.parquet                 y_true, y_pred (m) at every valid GEDI
                                           pixel -- baseline 0s kept as 0 m so the
                                           pixel set matches the embedding eval.
  predictions_test_nozero.parquet          same but baseline nodata (0) dropped.

Then run, e.g.:
  python evaluate.py --parquet <out-dir>/predictions_test.parquet \
      --out_plot <out-dir>/potapov_evaluate.png
  python evaluate_bins.py --parquet <out-dir>/predictions_test.parquet \
      --out_plot <out-dir>/potapov_evaluate_bins.png
"""

import argparse
import io
import json
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

# Potapov GEDI_V27 native grid: EPSG:4326, 0.00025 deg (~28 m), aligned to the
# global grid where every pixel edge is an integer multiple of NATIVE_RES.
NATIVE_RES = 0.00025


def init_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception as e:
        print(f"ee.Initialize failed ({e}); attempting ee.Authenticate()...")
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"Earth Engine initialised (project={project}).")


def build_potapov_image(collection_id):
    """Mosaic the Potapov collection to a single-band canopy-height image."""
    ic = ee.ImageCollection(collection_id)
    img = ic.mosaic()
    band = img.bandNames().getInfo()[0]
    print(f"Potapov collection '{collection_id}' -> mosaic band '{band}'")
    return img.select([band]).rename('canopy_height').toFloat()


def _snap(value, res, func):
    return func(value / res) * res


# getDownloadURL caps a single request at ~48 MiB. Keep blocks comfortably under
# that (EE counts a few bytes/pixel for the request) by limiting pixels/block.
MAX_PX_PER_REQUEST = 8_000_000


def _download_window(img, left, top, w, h, retries=4):
    """Download a single native-grid window; returns float32 array (h, w)."""
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
            # Masked pixels export as non-finite; Potapov spec is "no data = 0".
            return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"    download attempt {attempt}/{retries} failed: "
                  f"{str(e)[:160]}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed window after {retries} attempts: {last_err}")


def download_native(img, bounds, retries=4):
    """Download `img` on its NATIVE grid (EPSG:4326 @ NATIVE_RES) over `bounds`.

    NOTE: do NOT pass a non-native crsTransform to getDownloadURL -- mosaic()
    drops the native projection and EE will reproject from a 1-degree default
    grid, producing garbage. Downloading on the native grid (snapped to the
    global grid) preserves the real data; we reproject locally afterwards.

    Large tiles exceed EE's per-request size cap, so the grid is fetched in
    horizontal row blocks and stitched.

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


def download_tile(img, transform, width, height, bounds, crs_epsg=4326, retries=4):
    """Return Potapov resampled to the GEDI tile grid (transform/width/height).

    Downloads the native-grid Potapov over `bounds`, then reprojects (nearest
    neighbour) onto the target grid so pixel indices align with the GEDI tile.
    """
    native, native_transform = download_native(img, bounds, retries=retries)
    dst = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=native,
        destination=dst,
        src_transform=native_transform,
        src_crs=f'EPSG:{crs_epsg}',
        dst_transform=transform,
        dst_crs=f'EPSG:{crs_epsg}',
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ee-project', default=None,
                    help='Google Cloud project registered for Earth Engine. Only '
                         'needed if some tile is NOT already cached in '
                         '<out-dir>/potapov_tiles/ — Earth Engine is initialised '
                         'lazily, on the first tile that actually needs a download.')
    ap.add_argument('--cache-only', action='store_true',
                    help='Never contact Earth Engine: skip any tile without a cached '
                         'potapov tif instead of downloading it.')
    ap.add_argument('--shapefile',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/shapefiles/healpix_shapefile.shp')
    ap.add_argument('--gedi-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2019_healpix')
    ap.add_argument('--out-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/potapov_baseline_2019')
    ap.add_argument('--split', default='test')
    ap.add_argument('--collection-id', default='users/potapovpeter/GEDI_V27')
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
                    help='Re-download Potapov tiles even if cached tif exists.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tiles_dir = out_dir / 'potapov_tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Earth Engine is initialised on demand so a fully-cached run needs no
    # credentials and no network at all.
    ee_state = {'img': None}

    def get_potapov():
        if ee_state['img'] is None:
            if not args.ee_project:
                sys.exit("ERROR: a tile needs downloading but --ee-project was not "
                         "given. Pass it, or use --cache-only to skip uncached tiles.")
            init_ee(args.ee_project)
            ee_state['img'] = build_potapov_image(args.collection_id)
        return ee_state['img']

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
    print(f"{len(test_ids)} '{args.split}' tiles to process.")

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

        pot_tif = tiles_dir / f'healpix_{hid}_potapov.tif'
        if pot_tif.exists() and not args.overwrite:
            with rasterio.open(pot_tif) as psrc:
                pot = psrc.read(1).astype(np.float32)
            src_note = 'cached'
        elif args.cache_only:
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: no cached tif and "
                  f"--cache-only -- skip")
            skipped.append(hid)
            continue
        else:
            print(f"[{i+1}/{len(test_ids)}] tile {hid}: downloading Potapov "
                  f"({width}x{height}, {n_valid} valid GEDI px)...")
            pot = download_tile(get_potapov(), transform, width, height, bounds)
            profile.update(dtype='float32', count=1, nodata=0, compress='lzw')
            with rasterio.open(pot_tif, 'w', **profile) as dst:
                dst.write(pot.astype(np.float32), 1)
            src_note = 'downloaded'

        if pot.shape != gedi.shape:
            print(f"  WARNING tile {hid}: shape mismatch pot{pot.shape} vs "
                  f"gedi{gedi.shape} -- skip")
            skipped.append(hid)
            continue

        y_true = gedi[valid].astype(np.float64) / 100.0   # cm -> m
        y_pred = pot[valid].astype(np.float64)            # already m
        tile_df = pd.DataFrame({
            'healpix_id': hid,
            'y_true': y_true.astype(np.float32),
            'y_pred': y_pred.astype(np.float32),
        })
        rows.append(tile_df)
        n_zero = int((y_pred == 0).sum())
        print(f"  tile {hid} [{src_note}]: {n_valid} px, baseline nodata(0)="
              f"{n_zero} ({100*n_zero/n_valid:.1f}%)")

    if not rows:
        print("No data collected; nothing written.")
        sys.exit(1)

    full = pd.concat(rows, ignore_index=True)
    full_path = out_dir / 'predictions_test.parquet'
    full.to_parquet(full_path, index=False, compression='snappy')

    nozero = full[full['y_pred'] != 0].reset_index(drop=True)
    nozero_path = out_dir / 'predictions_test_nozero.parquet'
    nozero.to_parquet(nozero_path, index=False, compression='snappy')

    print("\n" + "=" * 78)
    print("POTAPOV 2019 BASELINE vs GEDI (test tiles)")
    print("=" * 78)
    print(f"tiles used: {len(rows)}  skipped: {len(skipped)}\n")

    kw = dict(bin_size=args.metric_bin_size, max_height=args.metric_max_height,
              min_bin_n=args.metric_min_bin_n, balance_seed=args.metric_balance_seed)
    metrics = {
        'baseline': 'potapov_2019',
        'split': args.split,
        'n_tiles_used': len(rows),
        'n_tiles_skipped': len(skipped),
        'metric_bin_size': args.metric_bin_size,
        'metric_max_height': args.metric_max_height,
        'all_valid': evaluate_and_report(
            full['y_true'].values, full['y_pred'].values, 'all valid GEDI px', **kw),
    }
    print()
    metrics['baseline_zero_dropped'] = evaluate_and_report(
        nozero['y_true'].values, nozero['y_pred'].values, 'baseline 0 dropped', **kw)

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nwrote {full_path}")
    print(f"wrote {nozero_path}")
    print(f"wrote {metrics_path}")
    print(f"potapov tiles cached in {tiles_dir}")
    print("\nNOTE: R2 is the number comparable to the RF models; Pearson r is shown\n"
          "      alongside it because r ignores bias and scale, so a systematically\n"
          "      short baseline can post a high r and a negative R2.")


if __name__ == '__main__':
    main()
