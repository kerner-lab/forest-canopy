#!/usr/bin/env python3
"""
Build a Meta/WRI Canopy Height Map v2 (CHMv2) baseline over the GEDI *test* HEALPix
tiles, so it can be evaluated with the same scripts (evaluate.py / evaluate_bins.py)
as the embedding maps, the Potapov baseline and the ETH baseline.

CHMv2 (Brandt et al. 2026; DINOv3 Sat-L on Vantor/Maxar imagery, with WRI) is on the
AWS Open Data registry -- PUBLIC, no credentials, no aws CLI needed. Verified layout:

  bucket   dataforgood-fb-data           (us-east-1, anonymous read over https)
  prefix   forests/v2/global/dinov3_global_chm_v2_ml3/
             chm/<quadkey>.tif           the canopy height rasters
             metadata/<quadkey>.geojson  per-tile imagery ACQUISITION DATES
             tiles.geojson               global tile index (58 MB -- not needed here)

  Each chm/<quadkey>.tif is:
    CRS      EPSG:3857 (Web Mercator), one z=10 slippy/quadkey tile
    size     32768 x 32768, uint8, COG with overviews [2,4,8,16,32,64]
    res      1.1943 Mercator-m/px  ->  ~0.95 m on the ground at 37 N
    values   canopy height in WHOLE METRES. No nodata is set and 0 m is a REAL
             prediction (bare ground/rock), exactly like the ETH map and unlike
             Potapov where 0 == nodata. Absence of data = the .tif does not exist.

TEMPORAL CAVEAT -- READ THIS BEFORE WRITING THE METHODS SECTION
  CHMv2 is a SINGLE-EPOCH product, not an annual time series. There is no 2019 map and
  no 2020 map: the same raster is compared against both years of GEDI. What does vary is
  the underlying imagery acquisition date, which differs PER QUADKEY and is published in
  metadata/<quadkey>.geojson. This script harvests those dates and writes
  acquisition_dates.csv so you can state the actual temporal offset between the CHM
  imagery and each GEDI year rather than asserting a nominal one. Check that file before
  interpreting any 2019-vs-2020 difference: a difference in the CHM columns between the
  two years comes ONLY from the differing GEDI pixel masks, never from the CHM changing.

For each test tile we:
  1. Open the matching GEDI tile (healpix_<id>_gedi.tif) to get the exact output grid
     (EPSG:4326, transform, width, height) -- the same grid the embeddings land on, so
     pixel indices align directly.
  2. Find the z=10 quadkeys covering the tile, stream the overlapping window of each COG
     over https, and aggregate onto the GEDI 30 m grid. Aggregation is an exact
     area-weighted mean (Resampling.sum for both the data and a ones-mask, then divide),
     so quadkey seams inside a tile are handled correctly.
  3. Sample the result at the *same valid GEDI pixels* the models use
     (gedi != nodata, gedi > 0, gedi < 12000 cm), giving (y_true, y_pred) pairs in metres.

Outputs (in --out-dir):
  meta_chm_tiles/healpix_<id>_metachm.tif   reprojected CHM per tile (cache; nodata=NaN
                                            where no CHM tile exists). REUSED by the
                                            second year's run -- the raster is identical,
                                            so run 2019 first and 2020 costs almost nothing.
  predictions_test.parquet                  y_true, y_pred (m) at valid GEDI pixels where
                                            CHM exists. Real 0 m predictions are kept;
                                            only genuinely-absent pixels are dropped.
                                            This is the fair, primary file.
  predictions_test_maskfill0.parquet        same GEDI pixel set as the embedding maps,
                                            absent CHM filled with 0 m, for exact
                                            pixel-count parity.
  acquisition_dates.csv                     healpix_id, quadkey, acq_dates -- the imagery
                                            dates behind every tile you evaluated.

Usage (run once per GEDI year; --out-dir must differ, --tiles-dir can be SHARED):

  python data_processing/meta_chm_baseline.py \
      --shapefile /path/healpix_shapefile.shp \
      --gedi-dir  /path/gedi_tiles_2019_healpix \
      --out-dir   /path/baselines/meta_chm_2019 \
      --tiles-dir /path/baselines/meta_chm_tiles_shared

  python data_processing/meta_chm_baseline.py \
      --shapefile /path/healpix_shapefile.shp \
      --gedi-dir  /path/gedi_tiles_2020_healpix \
      --out-dir   /path/baselines/meta_chm_2020 \
      --tiles-dir /path/baselines/meta_chm_tiles_shared

Then:
  python evaluate.py --parquet <out-dir>/predictions_test.parquet
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

S3_BASE = ("https://dataforgood-fb-data.s3.amazonaws.com/"
           "forests/v2/global/dinov3_global_chm_v2_ml3")
ZOOM = 10                      # the quadkey level the CHM tiles are published at
NATIVE_RES_M = 1.1943285669559  # Mercator metres per pixel at z10 / 32768 px

# GDAL knobs that make streaming a COG over https sane: do not list the bucket on open,
# keep a big read cache, and merge nearby range requests.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="536870912",     # 512 MB
    GDAL_CACHEMAX=1024,             # MB -- rasterio requires a real int for this one
    GDAL_NUM_THREADS="ALL_CPUS",
)


# ---------------------------------------------------------------------------
# quadkey helpers (standard slippy-map / Bing scheme)
# ---------------------------------------------------------------------------

def lonlat_to_tile(lon, lat, z=ZOOM):
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    s = math.sin(math.radians(max(min(lat, 85.05112878), -85.05112878)))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_to_quadkey(x, y, z=ZOOM):
    qk = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        qk.append(str(digit))
    return "".join(qk)


def quadkeys_for_bounds(lon_min, lat_min, lon_max, lat_max, z=ZOOM):
    """Every z-level quadkey whose tile intersects the lon/lat box."""
    x0, y0 = lonlat_to_tile(lon_min, lat_max, z)   # NW corner
    x1, y1 = lonlat_to_tile(lon_max, lat_min, z)   # SE corner
    return [tile_to_quadkey(x, y, z)
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def fetch_acq_dates(quadkey, timeout=30):
    """
    Imagery acquisition dates for one quadkey, from metadata/<quadkey>.geojson.

    The file is a FeatureCollection whose features each carry an 'acq_date'; a tile
    mosaicked from several passes has several. Returns a sorted unique list, or []
    if the metadata is absent (which is not an error -- some tiles have none).
    """
    url = f"{S3_BASE}/metadata/{quadkey}.geojson"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            gj = json.loads(r.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return []
    dates = {f.get("properties", {}).get("acq_date")
             for f in gj.get("features", [])}
    return sorted(d for d in dates if d)


# ---------------------------------------------------------------------------
# aggregation onto the GEDI grid
# ---------------------------------------------------------------------------

def accumulate_quadkey(url, dst_transform, dst_crs, dst_h, dst_w, dst_bounds_4326,
                       agg, decimate, chunk_rows, acc_a, acc_b):
    """
    Stream one CHM quadkey and fold it into the accumulators.

    agg='mean' : acc_a += sum of source values per dst cell, acc_b += source pixel count.
                 Dividing at the end gives the exact area-weighted mean even where a dst
                 cell straddles two quadkeys -- which is why we sum rather than average
                 per quadkey and average the averages.
    agg='max'  : acc_a = elementwise max, acc_b = 1 where anything landed.

    Returns True if the tile existed and contributed, False if it is absent (404).

    `decimate` reads through the COG's overviews instead of the 1 m base level. At the
    default 8 that is ~9.6 m -- still 3x finer than the 30 m target, so the mean is
    essentially unchanged, at 1/64 the bytes over the wire. Overviews are average-built,
    so decimation is FORCED OFF for agg='max' (a max of local means is not a max).
    """
    try:
        src = rasterio.open(url)
    except rasterio.errors.RasterioIOError:
        return False        # no tile published here (ocean, or outside coverage)

    with src:
        # Window of the source covering the destination, in source (3857) coordinates.
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, *dst_bounds_4326, densify_pts=21)
        sb = src.bounds
        if left >= sb.right or right <= sb.left or bottom >= sb.top or top <= sb.bottom:
            return False
        win = from_bounds(max(left, sb.left), max(bottom, sb.bottom),
                          min(right, sb.right), min(top, sb.top),
                          src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return False

        step = 1 if agg == "max" else max(1, int(decimate))
        resamp = Resampling.max if agg == "max" else Resampling.sum
        contributed = False

        # Row-strip the read so a full 32768-wide window never lands in RAM at once.
        rows_per_chunk = max(step, (chunk_rows // step) * step)
        for r0 in range(0, int(win.height), rows_per_chunk):
            h = min(rows_per_chunk, int(win.height) - r0)
            if h < step:
                break
            sub = Window(win.col_off, win.row_off + r0, win.width, h)
            out_h, out_w = h // step, int(win.width) // step
            if out_h < 1 or out_w < 1:
                continue

            # out_shape triggers a decimated read; GDAL serves it from an overview.
            data = src.read(1, window=sub, out_shape=(out_h, out_w),
                            resampling=Resampling.average).astype(np.float32)
            sub_transform = src.window_transform(sub) * rasterio.Affine.scale(step, step)

            tmp = np.zeros((dst_h, dst_w), np.float32)
            reproject(data, tmp,
                      src_transform=sub_transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=resamp, src_nodata=None, dst_nodata=0.0,
                      init_dest_nodata=True)
            cnt = np.zeros((dst_h, dst_w), np.float32)
            reproject(np.ones_like(data), cnt,
                      src_transform=sub_transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.sum if agg != "max" else Resampling.max,
                      src_nodata=None, dst_nodata=0.0, init_dest_nodata=True)

            if agg == "max":
                hit = cnt > 0
                np.maximum(acc_a, np.where(hit, tmp, -np.inf), out=acc_a)
                acc_b[hit] = 1.0
            else:
                acc_a += tmp
                acc_b += cnt
            contributed = contributed or bool((cnt > 0).any())

    return contributed


def build_tile(hid, dst_transform, dst_crs, dst_h, dst_w, dst_bounds_4326,
               agg, decimate, chunk_rows, verbose=True):
    """Aggregate every intersecting quadkey onto the GEDI grid. NaN where no CHM."""
    qks = quadkeys_for_bounds(*dst_bounds_4326)
    acc_a = np.full((dst_h, dst_w), -np.inf, np.float32) if agg == "max" \
        else np.zeros((dst_h, dst_w), np.float32)
    acc_b = np.zeros((dst_h, dst_w), np.float32)

    used = []
    for qk in qks:
        url = f"/vsicurl/{S3_BASE}/chm/{qk}.tif"
        ok = accumulate_quadkey(url, dst_transform, dst_crs, dst_h, dst_w,
                                dst_bounds_4326, agg, decimate, chunk_rows, acc_a, acc_b)
        if ok:
            used.append(qk)
        if verbose:
            print(f"    quadkey {qk}: {'ok' if ok else 'absent'}")

    out = np.full((dst_h, dst_w), np.nan, np.float32)
    hit = acc_b > 0
    out[hit] = (acc_a[hit] if agg == "max" else acc_a[hit] / acc_b[hit])
    return out, used


def write_cache(path, arr, transform, crs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.write(arr, 1)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile", required=True,
                    help="healpix shapefile with healpix_id + split")
    ap.add_argument("--gedi-dir", required=True,
                    help="dir of healpix_<id>_gedi.tif for ONE year")
    ap.add_argument("--out-dir", required=True, help="per-year outputs go here")
    ap.add_argument("--tiles-dir", default=None,
                    help="cache dir for the reprojected CHM tifs. SHARE this between the "
                         "2019 and 2020 runs -- the CHM raster is identical, so the second "
                         "year is nearly free. Default: <out-dir>/meta_chm_tiles")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile-id-column", default="healpix_id")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"],
                    help="how 1 m CHM pixels collapse into a 30 m GEDI cell. 'mean' "
                         "(default) matches the ETH/Potapov baselines, so cross-baseline "
                         "numbers stay comparable. 'max' is arguably closer to what GEDI "
                         "RH98 measures (98th-percentile return height) and will read "
                         "several metres taller -- worth reporting as a sensitivity, not "
                         "as the headline. 'max' disables overview decimation and is slow.")
    ap.add_argument("--decimate", type=int, default=8,
                    help="read the COG at 1/N resolution via its overviews (N=8 -> ~9.6 m, "
                         "still 3x finer than the 30 m target). N=1 forces the full 1 m "
                         "base level: ~64x the bytes for a change well under the noise "
                         "floor. Ignored when --agg max.")
    ap.add_argument("--chunk-rows", type=int, default=8192,
                    help="source rows per read; lower it if memory is tight")
    ap.add_argument("--gedi-nodata", type=float, default=-9999.0)
    ap.add_argument("--max-tiles", type=int, default=None, help="stop after N tiles (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild cached per-tile tifs instead of reusing them")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else out_dir / "meta_chm_tiles"
    gedi_dir = Path(args.gedi_dir)

    if args.agg == "max" and args.decimate != 1:
        print("NOTE: --agg max reads the native 1 m level (overviews are average-built, "
              "so a max over them would be a max of means).")

    gdf = gpd.read_file(args.shapefile)
    ids = gdf[gdf["split"] == args.split][args.tile_id_column].astype(int).tolist()
    ids = sorted(ids)[:args.max_tiles] if args.max_tiles else sorted(ids)
    print(f"{args.split} tiles: {len(ids)}   GEDI: {gedi_dir}   agg={args.agg} "
          f"decimate={1 if args.agg == 'max' else args.decimate}")

    rows, skipped, date_rows = [], [], []

    with rasterio.Env(**GDAL_ENV):
        for i, hid in enumerate(ids):
            gedi_path = gedi_dir / f"healpix_{hid}_gedi.tif"
            if not gedi_path.exists():
                print(f"[{i+1}/{len(ids)}] tile {hid}: no GEDI raster -- skip")
                skipped.append(hid)
                continue

            with rasterio.open(gedi_path) as g:
                gedi = g.read(1)
                dst_transform, dst_crs = g.transform, g.crs
                dst_h, dst_w = g.height, g.width
                dst_bounds = tuple(g.bounds)
                nodata = g.nodata if g.nodata is not None else args.gedi_nodata

            valid = (gedi != nodata) & (gedi > 0) & (gedi < 12000)
            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"[{i+1}/{len(ids)}] tile {hid}: no valid GEDI pixels -- skip")
                skipped.append(hid)
                continue

            cache = tiles_dir / f"healpix_{hid}_metachm.tif"
            if cache.exists() and not args.overwrite:
                with rasterio.open(cache) as c:
                    pred = c.read(1)
                src_note = "cached"
                used = []
            else:
                print(f"[{i+1}/{len(ids)}] tile {hid}: streaming CHM "
                      f"({dst_w}x{dst_h}, {n_valid} valid GEDI px)...")
                pred, used = build_tile(hid, dst_transform, dst_crs, dst_h, dst_w,
                                        dst_bounds, args.agg, args.decimate,
                                        args.chunk_rows, verbose=not args.quiet)
                write_cache(cache, pred, dst_transform, dst_crs)
                src_note = "downloaded"

            # Acquisition dates are cheap and worth recording even on a cache hit.
            for qk in (used or quadkeys_for_bounds(*dst_bounds)):
                dates = fetch_acq_dates(qk)
                if dates:
                    date_rows.append({"healpix_id": hid, "quadkey": qk,
                                      "acq_dates": ";".join(dates),
                                      "first": dates[0], "last": dates[-1]})

            if pred.shape != gedi.shape:
                print(f"  WARNING tile {hid}: shape mismatch chm{pred.shape} vs "
                      f"gedi{gedi.shape} -- skip")
                skipped.append(hid)
                continue

            chm_valid = np.isfinite(pred)
            y_true_cm = gedi[valid].astype(np.float64)
            y_pred_full = pred[valid].astype(np.float64)
            missing = ~chm_valid[valid]
            n_missing = int(missing.sum())

            keep = ~missing
            tile_df = pd.DataFrame({
                "healpix_id": hid,
                "y_true": (y_true_cm[keep] / 100.0).astype(np.float32),
                "y_pred": y_pred_full[keep].astype(np.float32),
            })
            y_pred_fill = np.where(missing, 0.0, y_pred_full)
            tile_df_fill = pd.DataFrame({
                "healpix_id": hid,
                "y_true": (y_true_cm / 100.0).astype(np.float32),
                "y_pred": y_pred_fill.astype(np.float32),
            })
            rows.append((tile_df, tile_df_fill))
            n_zero = int((tile_df["y_pred"].values == 0).sum())
            print(f"  tile {hid} [{src_note}]: {n_valid} px, no-CHM(dropped)="
                  f"{n_missing} ({100 * n_missing / n_valid:.1f}%), 0 m kept={n_zero}")

    if not rows:
        print("No data collected; nothing written.")
        sys.exit(1)

    full = pd.concat([r[0] for r in rows], ignore_index=True)
    full.to_parquet(out_dir / "predictions_test.parquet", index=False, compression="snappy")
    fill = pd.concat([r[1] for r in rows], ignore_index=True)
    fill.to_parquet(out_dir / "predictions_test_maskfill0.parquet", index=False,
                    compression="snappy")
    if date_rows:
        dates_df = pd.DataFrame(date_rows).drop_duplicates()
        dates_df.to_csv(out_dir / "acquisition_dates.csv", index=False)

    def _stats(df):
        yt, yp = df["y_true"].values, df["y_pred"].values
        bias = float(np.mean(yp - yt))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
        return rmse, bias, r

    rmse_a, bias_a, r_a = _stats(full)
    rmse_f, bias_f, r_f = _stats(fill)
    print("\n" + "=" * 64)
    print(f"META CHMv2 BASELINE vs GEDI ({args.split} tiles)  agg={args.agg}")
    print("=" * 64)
    print(f"tiles used: {len(rows)}   skipped: {len(skipped)}")
    print(f"primary  n={len(full):>9,}  RMSE={rmse_a:6.2f} m  bias={bias_a:+6.2f} m  r={r_a:.3f}")
    print(f"fill0    n={len(fill):>9,}  RMSE={rmse_f:6.2f} m  bias={bias_f:+6.2f} m  r={r_f:.3f}")
    if date_rows:
        d = pd.DataFrame(date_rows)
        print(f"imagery acquisition dates: {d['first'].min()} .. {d['last'].max()} "
              f"({len(d)} quadkeys) -> {out_dir / 'acquisition_dates.csv'}")
        print("CHMv2 is single-epoch: the 2019 and 2020 runs share this raster and differ "
              "only in the GEDI mask.")
    print(f"wrote: {out_dir / 'predictions_test.parquet'}")


if __name__ == "__main__":
    main()
