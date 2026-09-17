#!/usr/bin/env python3
"""
Build an ECHOSAT (Pauls et al.) 10 m canopy-height baseline over the GEDI HEALPix
tiles, so it can be evaluated with the exact same analysis scripts
(evaluate.py / evaluate_bins.py / metrics_stratified.py) used for the embedding
maps, the Potapov baseline and the ETH baseline.

ECHOSAT map: https://janpauls.org/projects/echosat/   (data CC-BY 4.0)
Downloads  : https://echosat.uni-muenster.de/ECHOSAT_<TILE>.tif
Code/paper : https://github.com/AI4Forest/ECHOSAT , arXiv:2602.21421

What the server actually serves (verified against the live files)
----------------------------------------------------------------
  Naming   : ECHOSAT_T<zone:02d><latband>.tif -- the Sentinel-2 MGRS grid
             AGGREGATED to zone + latitude band (T32TXP, T32TXQ -> T32T). Each
             file is therefore ~6 deg x 8 deg and 51-195 GB. DO NOT MIRROR THEM.
  CRS      : per-tile UTM (e.g. T10T = EPSG:32610). NOT EPSG:4326.
  Grid     : 10 m, Int16, nodata -9999, units CENTIMETRES.
  Bands    : 7, one per year. Band 1 = --base-year (2018), band 7 = 2024.
  Internal : tiled 256x256, LZW + PREDICTOR=2, NO overviews.

-------------------------------
The GeoTIFFs are INTERLEAVE=PIXEL, so one 256x256 block holds all 7 years

Note on band -> year mapping
----------------------------------
The GeoTIFFs carry no band descriptions and the README documents only the
training .npz layout ("(7, 12, 12, H, W)  years, months, bands, H, W"). Band 1 =
2018 is inferred from the stated 2018-2024 span, not documented. --base-year
exists so this is one flag to change, and --all-years reports every band against
your GEDI so you can see which one actually fits best.

Tiles that straddle an ECHOSAT boundary
---------------------------------------
A HEALPix tile can span two UTM zones or two latitude bands (62 of 311 do in the
CA/OR/WA AOI -- the zone 10/11 seam at 120W and the S/T seam at 40N cut straight
through it). Every ECHOSAT tile touching the AOI is read and reprojected into the
output grid; the first source with a valid value wins for a given pixel. Reading
only one of them would silently truncate those tiles.

For each tile we:
  1. Open the matching GEDI tile (healpix_<id>_gedi.tif) for the exact output
     grid (CRS/transform/width/height) -- the same grid the embeddings are
     reprojected onto, so pixel indices align directly.
  2. Windowed-read ECHOSAT over the tile bounds via /vsicurl/ convert cm -> m,
     and reproject onto the GEDI 30 m grid. Default resampling is AVERAGE (10 m
     source, 30 m target, so each cell averages ~9 valid source pixels; nodata is
     excluded from the average).
  3. Sample ECHOSAT at the *same valid GEDI pixels* the models use
     (gedi != nodata, gedi > 0, gedi < 12000 cm), giving (y_true, y_pred) in m.

Outputs (in --out-dir):
  echosat_tiles/healpix_<id>_echosat.tif   7-band (one per year) reprojected
                                           cache in METRES, nodata = SENTINEL.
                                           Year-independent: re-sample against
                                           any GEDI year without re-downloading.
  predictions_test.parquet                 y_true, y_pred (m) at valid GEDI
                                           pixels where ECHOSAT is valid.
  predictions_test_maskfill0.parquet       same GEDI pixel set, nodata ECHOSAT
                                           filled with 0 m (pixel-count parity).
  metrics.json                             stratified metrics, same code path as
                                           the RF runs.

Typical HPC use (cache first, evaluate after):
  export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
  python echosat_baseline.py --split test --year 2020 \\
      --gedi-dir /path/gedi_tiles_2020_healpix \\
      --out-dir  /path/echosat_baseline

Then, e.g.:
  python evaluate.py --parquet <out-dir>/predictions_test.parquet
"""

import argparse
import math
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.errors import WindowError, RasterioIOError
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds, intersection

# Same metric code the RF runs use, so baseline and model numbers are computed
# identically. This file lives in data_processing/, hence the parent on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_stratified import evaluate_and_report  # noqa: E402

ECHOSAT_URL = "https://echosat.uni-muenster.de/ECHOSAT_{tile}.tif"
ECHOSAT_NODATA = -9999          # as stored on the server (centimetres)
# GEDI validity cap, shared with run_gedi_prediction_30m.py / presto_processing.py /
# eth_canopy_baseline.py / potapov_baseline.py. Changing it here alone would score
# this baseline on a different pixel set than the models it is compared against.
GEDI_MAX_HEIGHT_CM = 12000
ECHOSAT_N_BANDS = 7
ECHOSAT_BASE_YEAR = 2018        # band 1; see the CAVEAT in the module docstring

# Written where ECHOSAT has no data. Chosen well outside the plausible 0..~200 m
# canopy range so it stays trivially separable from a real 0 m prediction.
SENTINEL = -9999.0

# MGRS latitude band letters (I and O are skipped), 8 deg each from 80S.
MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"

_RESAMPLING = {
    'average': Resampling.average,
    'nearest': Resampling.nearest,
    'bilinear': Resampling.bilinear,
}


# ---------------------------------------------------------------------------
# MGRS zone / band -> ECHOSAT tile id
# ---------------------------------------------------------------------------

def mgrs_latband(lat: float) -> str:
    """MGRS latitude band letter for a latitude (band X absorbs 72..84 N)."""
    if lat >= 84:
        return 'X'
    i = int((lat + 80) // 8)
    return MGRS_BANDS[min(max(i, 0), len(MGRS_BANDS) - 1)]


def utm_zone(lon: float) -> int:
    """UTM zone number for a longitude."""
    return int((lon + 180) // 6) + 1


def echosat_tiles_for_bounds(bounds) -> list:
    """Every ECHOSAT tile id whose zone/band cell touches `bounds` (EPSG:4326).

    Checking all four corners catches AOIs that straddle a zone seam (every 6 deg
    of longitude) or a latitude-band seam (every 8 deg) -- both of which occur
    inside the CA/OR/WA AOI.
    """
    minx, miny, maxx, maxy = bounds
    return sorted({f"T{utm_zone(lon):02d}{mgrs_latband(lat)}"
                   for lon in (minx, maxx) for lat in (miny, maxy)})


# ---------------------------------------------------------------------------
# Windowed read + reprojection onto the GEDI grid
# ---------------------------------------------------------------------------

def read_echosat_window(tile: str, bounds_4326, retries: int = 4):
    """Windowed-read one ECHOSAT tile over `bounds_4326`.

    Returns (array (7, h, w) float32 in METRES with SENTINEL for nodata,
             src_transform, src_crs) or None when the tile does not exist on the
    server or does not actually overlap the AOI.

    Only the internal 256x256 blocks intersecting the window are fetched over
    HTTP; the file itself is never downloaded.
    """
    url = f"/vsicurl/{ECHOSAT_URL.format(tile=tile)}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with rasterio.open(url) as src:
                b = transform_bounds("EPSG:4326", src.crs, *bounds_4326)
                win = from_bounds(*b, transform=src.transform)
                # Expand to whole pixels explicitly rather than via round_offsets/
                # round_lengths: their signatures changed across rasterio versions,
                # and flooring the offset while ceiling the length can still leave
                # the far edge a pixel short. Then clip to the raster -- a
                # corner-derived tile id can name a file whose extent misses the
                # AOI entirely.
                col_off = math.floor(win.col_off)
                row_off = math.floor(win.row_off)
                win = Window(
                    col_off, row_off,
                    math.ceil(win.col_off + win.width) - col_off,
                    math.ceil(win.row_off + win.height) - row_off,
                )
                try:
                    win = intersection(win, Window(0, 0, src.width, src.height))
                except WindowError:
                    return None
                if win.width < 1 or win.height < 1:
                    return None
                arr = src.read(window=win)                  # (7, h, w) int16, cm
                transform = src.window_transform(win)
                crs = src.crs
            out = np.where(arr == ECHOSAT_NODATA, SENTINEL,
                           arr.astype(np.float32) / 100.0).astype(np.float32)
            return out, transform, crs
        except RasterioIOError as exc:
            msg = str(exc)
            # A missing tile (ocean-only zone/band cell) is expected, not an error.
            if '404' in msg or 'does not exist' in msg.lower():
                return None
            last_err = exc
        except OSError as exc:
            # Transient network/socket trouble. Deliberately NOT a bare `except`:
            # a bare one would silently retry a programming error four times and
            # then report it as a network failure.
            last_err = exc
        wait = 2 ** attempt
        print(f"    {tile}: read attempt {attempt}/{retries} failed "
              f"({str(last_err)[:140]}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"{tile}: failed after {retries} attempts: {last_err}")


def build_tile(bounds_4326, dst_transform, dst_crs, width, height,
               resampling, retries=4):
    """ECHOSAT resampled onto the GEDI tile grid as (7, height, width) float32 m.

    Every ECHOSAT tile touching the AOI is reprojected into the output grid, take
    the first source with a valid value for a given pixel. UTM zone rasters
    overlap slightly at the seams, so some pixels are offered by more than one
    source; tile ids are processed in sorted order, making the choice
    deterministic. Returns (stack, tiles_used) or (None, []) if nothing overlapped.
    """
    dst = np.full((ECHOSAT_N_BANDS, height, width), SENTINEL, dtype=np.float32)
    filled = np.zeros((height, width), dtype=bool)
    used = []

    for tile in echosat_tiles_for_bounds(bounds_4326):
        got = read_echosat_window(tile, bounds_4326, retries=retries)
        if got is None:
            continue
        src_arr, src_transform, src_crs = got

        tmp = np.full((ECHOSAT_N_BANDS, height, width), SENTINEL, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=tmp,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=SENTINEL,
            dst_nodata=SENTINEL,
        )
        # Band 1 decides validity: the 7 years share one nodata footprint.
        new = np.isfinite(tmp[0]) & (tmp[0] != SENTINEL) & ~filled
        if not new.any():
            continue
        dst[:, new] = tmp[:, new]
        filled |= new
        used.append(tile)

    if not used:
        return None, []
    return dst, used


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shapefile',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/shapefiles/healpix_shapefile.shp')
    ap.add_argument('--gedi-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2020_healpix',
                    help='GEDI tiles to sample against.')
    ap.add_argument('--out-dir',
                    default='/Users/kdoerkse/Documents/ASU/JPL/data/echosat_baseline')
    ap.add_argument('--tiles-dir', default=None,
                    help='Directory for the cached per-tile ECHOSAT rasters '
                         '(default: <out-dir>/echosat_tiles). The cache holds all '
                         '7 years, so it can be re-sampled against any GEDI year '
                         'without re-downloading.')
    ap.add_argument('--split', default='test')
    ap.add_argument('--year', type=int, default=2020,
                    help='ECHOSAT year to evaluate. Should match the GEDI epoch in '
                         '--gedi-dir.')
    ap.add_argument('--base-year', type=int, default=ECHOSAT_BASE_YEAR,
                    help='Year of band 1 (default 2018). The published tifs carry no '
                         'band descriptions, so this mapping is inferred from the '
                         'stated 2018-2024 span -- see --all-years.')
    ap.add_argument('--all-years', action='store_true',
                    help='Also report metrics for every band against the same GEDI. '
                         'Use this to sanity-check --base-year: the band matching '
                         'your GEDI epoch should score best.')
    ap.add_argument('--resampling', default='average', choices=list(_RESAMPLING),
                    help='10 m -> 30 m resampling (default: average, best for '
                         'downsampling continuous canopy height).')
    ap.add_argument('--tile-id-column', default='healpix_id')
    ap.add_argument('--gedi-nodata', type=float, default=-9999.0)
    # Keep these identical to the RF runs or the stratified numbers are not
    # comparable (run_gedi_prediction_30m.py uses the same defaults).
    ap.add_argument('--metric-bin-size', type=float, default=10.0)
    ap.add_argument('--metric-max-height', type=float, default=60.0)
    ap.add_argument('--metric-min-bin-n', type=int, default=100)
    ap.add_argument('--metric-balance-seed', type=int, default=42)
    ap.add_argument('--tile-list', default=None, metavar='PATH',
                    help='File of healpix ids (one per line) restricting the run. Use '
                         'the master intersection list so the baseline is evaluated '
                         'on exactly the tiles the models were.')
    ap.add_argument('--max-tiles', type=int, default=None,
                    help='Optional cap on number of tiles (for quick tests).')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-download tiles even if a cached tif exists.')
    ap.add_argument('--cache-only', action='store_true',
                    help='Never contact the server: skip any tile without a cached '
                         'echosat tif instead of downloading it.')
    ap.add_argument('--no-eval', action='store_true',
                    help='Populate the tile cache and stop -- write no parquet and no '
                         'metrics. Use this for the parallel caching pass (a SLURM '
                         'array over tile slices), then run once more over the full '
                         'tile list with --cache-only to produce the outputs from the '
                         'shared cache. Without it, every array task would emit its '
                         'own partial parquet over just its slice.')
    ap.add_argument('--retries', type=int, default=4)
    args = ap.parse_args()

    band_idx = args.year - args.base_year
    if not 0 <= band_idx < ECHOSAT_N_BANDS:
        ap.error(f"--year {args.year} is outside the ECHOSAT range "
                 f"{args.base_year}..{args.base_year + ECHOSAT_N_BANDS - 1}")

    out_dir = Path(args.out_dir)
    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else out_dir / 'echosat_tiles'
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    resampling = _RESAMPLING[args.resampling]

    gdf = gpd.read_file(args.shapefile)
    tile_ids = gdf[gdf['split'] == args.split][args.tile_id_column].tolist()
    if args.tile_list:
        keep = {int(t) for t in Path(args.tile_list).read_text().split()
                if t.strip() and not t.strip().startswith('#')}
        before = len(tile_ids)
        tile_ids = [t for t in tile_ids if int(t) in keep]
        missing = sorted(keep - {int(t) for t in tile_ids})
        print(f"Tile list: {len(tile_ids)}/{before} '{args.split}' tiles kept "
              f"({args.tile_list})")
        if missing:
            print(f"  WARNING: {len(missing)} id(s) in the list are not in this "
                  f"split: {missing}")
        if not tile_ids:
            sys.exit("ERROR: tile list removed every tile.")
    if args.max_tiles:
        tile_ids = tile_ids[:args.max_tiles]
    print(f"{len(tile_ids)} '{args.split}' tiles to process.  "
          f"year={args.year} (band {band_idx + 1})  resampling={args.resampling}")

    rows, skipped = [], []
    for i, hid in enumerate(tile_ids):
        gedi_path = Path(args.gedi_dir) / f'healpix_{hid}_gedi.tif'
        if not gedi_path.exists():
            print(f"[{i+1}/{len(tile_ids)}] tile {hid}: missing GEDI tif -- skip")
            skipped.append(hid)
            continue

        with rasterio.open(gedi_path) as gsrc:
            gedi = gsrc.read(1)
            transform, crs = gsrc.transform, gsrc.crs
            width, height = gsrc.width, gsrc.height
            bounds = gsrc.bounds
            profile = gsrc.profile.copy()

        # 12000 cm matches run_gedi_prediction_30m.py:283, presto_processing.py,
        # eth_canopy_baseline.py and potapov_baseline.py. Keep it in step with them:
        # the baseline is only comparable to the models if it is scored on the same
        # pixel set. Raising it to 20000 admits ~17 GEDI artefacts in the 120-146 m
        # range across the 47 test tiles -- 0.0027% of pixels, but they land in the
        # top height bin, and the bin-balanced resample weights that bin equally with
        # bins 25x larger, so those few pixels swing balanced_r2 by ~12% relative.
        valid = (gedi != args.gedi_nodata) & (gedi > 0) & (gedi < GEDI_MAX_HEIGHT_CM)
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"[{i+1}/{len(tile_ids)}] tile {hid}: no valid GEDI pixels -- skip")
            skipped.append(hid)
            continue

        cache_tif = tiles_dir / f'healpix_{hid}_echosat.tif'
        if cache_tif.exists() and not args.overwrite:
            with rasterio.open(cache_tif) as psrc:
                stack = psrc.read().astype(np.float32)
            src_note = 'cached'
        elif args.cache_only:
            print(f"[{i+1}/{len(tile_ids)}] tile {hid}: no cached tif and "
                  f"--cache-only -- skip")
            skipped.append(hid)
            continue
        else:
            srcs = echosat_tiles_for_bounds(bounds)
            print(f"[{i+1}/{len(tile_ids)}] tile {hid}: reading ECHOSAT "
                  f"{srcs} ({width}x{height}, {n_valid} valid GEDI px)...")
            t0 = time.time()
            stack, used = build_tile(bounds, transform, crs, width, height,
                                     resampling, retries=args.retries)
            if stack is None:
                print(f"  tile {hid}: no ECHOSAT coverage from {srcs} -- skip")
                skipped.append(hid)
                continue
            # Inherit the GEDI grid but drop any block layout carried over from it:
            # a tiled TIFF needs block sizes that are multiples of 16, and these
            # tiles are arbitrary sizes (e.g. 3065 px wide).
            for k in ('tiled', 'blockxsize', 'blockysize', 'interleave'):
                profile.pop(k, None)
            profile.update(dtype='float32', count=ECHOSAT_N_BANDS, nodata=SENTINEL,
                           compress='lzw')
            with rasterio.open(cache_tif, 'w', **profile) as dst:
                dst.write(stack.astype(np.float32))
                for b in range(ECHOSAT_N_BANDS):
                    dst.set_band_description(b + 1, str(args.base_year + b))
            src_note = f"read {'+'.join(used)} in {time.time() - t0:.0f}s"

        if stack.shape[0] != ECHOSAT_N_BANDS or stack.shape[1:] != gedi.shape:
            print(f"  WARNING tile {hid}: shape mismatch echosat{stack.shape} vs "
                  f"gedi{gedi.shape} -- skip")
            skipped.append(hid)
            continue

        y_true_cm = gedi[valid].astype(np.float64)
        tile_rows = {}
        for b in range(ECHOSAT_N_BANDS):
            pred = stack[b]
            pred_v = pred[valid].astype(np.float64)
            masked = ~(np.isfinite(pred_v) & (pred_v != SENTINEL))
            keep = ~masked
            tile_rows[b] = (
                pd.DataFrame({'healpix_id': hid,
                              'y_true': (y_true_cm[keep] / 100.0).astype(np.float32),
                              'y_pred': pred_v[keep].astype(np.float32)}),
                pd.DataFrame({'healpix_id': hid,
                              'y_true': (y_true_cm / 100.0).astype(np.float32),
                              'y_pred': np.where(masked, 0.0, pred_v).astype(np.float32)}),
                int(masked.sum()),
            )
        rows.append(tile_rows)
        n_masked = tile_rows[band_idx][2]
        n_zero = int((tile_rows[band_idx][0]['y_pred'].values == 0).sum())
        print(f"  tile {hid} [{src_note}]: {n_valid} px, nodata(dropped)="
              f"{n_masked} ({100 * n_masked / n_valid:.1f}%), real 0 m kept={n_zero}")

    if args.no_eval:
        print(f"\n--no-eval: cached {len(rows)} tile(s), skipped {len(skipped)}. "
              f"Cache in {tiles_dir}.")
        print("Run again over the full tile list with --cache-only to write the "
              "parquet and metrics.")
        return

    if not rows:
        print("No data collected; nothing written.")
        sys.exit(1)

    full = pd.concat([r[band_idx][0] for r in rows], ignore_index=True)
    fill = pd.concat([r[band_idx][1] for r in rows], ignore_index=True)
    full_path = out_dir / 'predictions_test.parquet'
    fill_path = out_dir / 'predictions_test_maskfill0.parquet'
    full.to_parquet(full_path, index=False, compression='snappy')
    fill.to_parquet(fill_path, index=False, compression='snappy')

    print("\n" + "=" * 78)
    print(f"ECHOSAT {args.year} CANOPY HEIGHT BASELINE vs GEDI ({args.split} tiles)")
    print("=" * 78)
    print(f"tiles used: {len(rows)}  skipped: {len(skipped)}  "
          f"resampling: {args.resampling}  band: {band_idx + 1}\n")

    kw = dict(bin_size=args.metric_bin_size, max_height=args.metric_max_height,
              min_bin_n=args.metric_min_bin_n, balance_seed=args.metric_balance_seed)
    metrics = {
        'baseline': f'echosat_{args.year}',
        'split': args.split,
        'year': args.year,
        'base_year': args.base_year,
        'band': band_idx + 1,
        'resampling': args.resampling,
        'gedi_dir': args.gedi_dir,
        'n_tiles_used': len(rows),
        'n_tiles_skipped': len(skipped),
        'metric_bin_size': args.metric_bin_size,
        'metric_max_height': args.metric_max_height,
        'echosat_valid': evaluate_and_report(
            full['y_true'].values, full['y_pred'].values,
            f'ECHOSAT {args.year} valid-only (primary)', **kw),
    }
    print()
    metrics['masked_filled_zero'] = evaluate_and_report(
        fill['y_true'].values, fill['y_pred'].values,
        'nodata filled with 0 m', **kw)

    if args.all_years:
        print("\n" + "-" * 78)
        print("PER-YEAR CHECK -- the band matching this GEDI epoch should score best.")
        print("A flat profile means the years are too similar here to confirm the")
        print("band->year mapping; it does not confirm --base-year is right.")
        print("-" * 78)
        per_year = {}
        for b in range(ECHOSAT_N_BANDS):
            df = pd.concat([r[b][0] for r in rows], ignore_index=True)
            per_year[str(args.base_year + b)] = evaluate_and_report(
                df['y_true'].values, df['y_pred'].values,
                f'band {b + 1} (assumed {args.base_year + b})', **kw)
        metrics['per_year'] = per_year

    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nwrote {full_path}")
    print(f"wrote {fill_path}")
    print(f"wrote {metrics_path}")
    print(f"echosat tiles cached in {tiles_dir} (all 7 years, metres)")
    print("\nNOTE: R2 is the number comparable to the RF models; Pearson r is shown\n"
          "      alongside it because r ignores bias and scale, so a systematically\n"
          "      short baseline can post a high r and a negative R2.")
    print("NOTE: band->year is inferred, not documented -- see --all-years and the\n"
          "      module docstring before publishing these numbers.")


if __name__ == '__main__':
    main()
