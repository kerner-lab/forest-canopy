#!/usr/bin/env python3
"""
Validate the per-tile ECHOSAT cache written by data_processing/echosat_baseline.py.

  (default)        exists, opens, 7 bands, correct dtype/nodata, and -- the check
                   that actually matters for comparability -- the grid is IDENTICAL
                   to the tile's GEDI raster (CRS, transform, shape). A cache that
                   is off by one pixel still "looks fine" in QGIS but silently
                   misaligns every (y_true, y_pred) pair.
  --deep           reads every pixel. A job killed mid-write (full disk, timeout,
                   OOM) leaves a GeoTIFF whose header parses but whose LZW strips
                   are truncated; that only surfaces on read. Also reports nodata
                   fraction, value range, and flags all-nodata / constant bands.
  --verify-source  re-reads the LIVE ECHOSAT server at N random valid-GEDI pixels
                   per tile and correlates against the cache. This is the only
                   check that can catch a wrong CRS transform, a flipped axis, or
                   a cm/m unit error, because everything else is self-consistency.
                   Costs one small HTTP range read per tile.

Exit code is 0 only if nothing failed, so it can gate the phase-2 evaluation.

Usage:
  # fast structural pass over the split
  python check_echosat_cache.py --tiles-dir /data/.../echosat/echosat_tiles \\
      --gedi-dir /data/.../gedi/2019 --shapefile /scratch/.../healpix_shapefile.shp \\
      --split test

  # add truncation detection (reads every pixel; slower, local only)
  ... --deep

  # add an independent check against the source for 8 sampled tiles
  ... --deep --verify-source 8

  # write the tiles that need regenerating
  ... --deep --out-bad bad_tiles.txt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio

ECHOSAT_URL = "https://echosat.uni-muenster.de/ECHOSAT_{tile}.tif"
N_BANDS = 7
BASE_YEAR = 2018
SENTINEL = -9999.0
MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"

# Plausible canopy-height envelope in metres.
#
# MIN is NOT 0: ECHOSAT is a regression output, so it emits slightly negative
# heights over bare ground and water (observed down to -0.51 m across the 2020 test
# split). A 0.0 floor flagged two perfectly healthy tiles as "unit error?". The real
# unit-error signature is the MAXIMUM -- centimetres leaking through would put it
# near 12000, not the minimum a fraction below zero.
MIN_PLAUSIBLE_M = -5.0
MAX_PLAUSIBLE_M = 150.0
# Separately, a large share of negative pixels would mean something is actually
# wrong, even if none of them is individually extreme.
MAX_NEGATIVE_FRAC = 0.05


def mgrs_latband(lat):
    if lat >= 84:
        return 'X'
    return MGRS_BANDS[min(max(int((lat + 80) // 8), 0), len(MGRS_BANDS) - 1)]


def utm_zone(lon):
    return int((lon + 180) // 6) + 1


def echosat_tiles_for_bounds(bounds):
    minx, miny, maxx, maxy = bounds
    return sorted({f"T{utm_zone(lon):02d}{mgrs_latband(lat)}"
                   for lon in (minx, maxx) for lat in (miny, maxy)})


def check_tile(hid, tiles_dir, gedi_dir, deep, gedi_nodata=-9999.0,
               gedi_max_cm=12000):
    """Return (problems, info) for one tile. Empty problems list == healthy."""
    problems, info = [], {'healpix_id': int(hid)}

    cache = Path(tiles_dir) / f'healpix_{hid}_echosat.tif'
    if not cache.exists():
        return ['missing'], info
    info['size_mb'] = round(cache.stat().st_size / 1e6, 1)

    gedi_path = Path(gedi_dir) / f'healpix_{hid}_gedi.tif'
    if not gedi_path.exists():
        return ['gedi_missing'], info

    try:
        with rasterio.open(gedi_path) as g:
            g_shape, g_transform, g_crs = g.shape, g.transform, g.crs
            gedi = g.read(1) if deep else None
            g_bounds = g.bounds
    except Exception as exc:
        return [f'gedi_unreadable: {type(exc).__name__}'], info

    try:
        with rasterio.open(cache) as s:
            info['shape'] = list(s.shape)
            info['count'] = s.count
            info['dtype'] = s.dtypes[0]
            info['nodata'] = s.nodata
            info['descriptions'] = list(s.descriptions)

            if s.count != N_BANDS:
                problems.append(f'band_count={s.count} (expected {N_BANDS})')
            if s.dtypes[0] != 'float32':
                problems.append(f'dtype={s.dtypes[0]} (expected float32)')
            if s.nodata != SENTINEL:
                problems.append(f'nodata={s.nodata} (expected {SENTINEL})')

            # The alignment check. Anything but an exact match breaks the
            # pixel-for-pixel correspondence the whole comparison rests on.
            if s.shape != g_shape:
                problems.append(f'shape {s.shape} != gedi {g_shape}')
            if s.crs != g_crs:
                problems.append(f'crs {s.crs} != gedi {g_crs}')
            if s.transform != g_transform:
                problems.append('transform != gedi transform')

            expected_desc = tuple(str(BASE_YEAR + i) for i in range(N_BANDS))
            if s.count == N_BANDS and tuple(s.descriptions) != expected_desc:
                problems.append(f'band descriptions {s.descriptions} != {expected_desc}')

            if deep and not problems:
                # Reading every pixel is what exposes a truncated LZW stream.
                try:
                    stack = s.read()
                except Exception as exc:
                    problems.append(f'truncated/unreadable pixels: '
                                    f'{type(exc).__name__}: {exc}')
                    return problems, info

                valid = np.isfinite(stack) & (stack != SENTINEL)
                info['nodata_frac'] = round(1.0 - float(valid.mean()), 4)
                if not valid.any():
                    problems.append('all pixels nodata')
                    return problems, info

                v = stack[valid]
                info['min_m'] = round(float(v.min()), 2)
                info['max_m'] = round(float(v.max()), 2)
                info['mean_m'] = round(float(v.mean()), 2)

                if v.min() < MIN_PLAUSIBLE_M or v.max() > MAX_PLAUSIBLE_M:
                    problems.append(f'values out of range [{v.min():.1f}, '
                                    f'{v.max():.1f}] m -- unit error?')
                neg_frac = float((v < 0).mean())
                info['neg_frac'] = round(neg_frac, 5)
                if neg_frac > MAX_NEGATIVE_FRAC:
                    problems.append(f'{neg_frac:.1%} of pixels are negative '
                                    f'(min {v.min():.2f} m)')
                # A band that never varies means the source window was empty or
                # the write went wrong; heights are never uniform over a tile.
                for b in range(stack.shape[0]):
                    bv = stack[b][valid[b]] if valid[b].any() else np.array([])
                    if bv.size and float(bv.std()) == 0.0:
                        problems.append(f'band {b + 1} is constant ({bv[0]:.2f})')

                # Sanity vs GEDI: a gross misregistration shows up as a collapsed
                # correlation. Deliberately loose -- this is a smoke test, not a
                # metric. Real accuracy comes from the baseline's own metrics.json.
                gvalid = ((gedi != gedi_nodata) & (gedi > 0) & (gedi < gedi_max_cm))
                band = stack[2] if stack.shape[0] > 2 else stack[0]   # 2020 band as sanity check
                both = gvalid & np.isfinite(band) & (band != SENTINEL)
                info['n_gedi_px'] = int(gvalid.sum())
                info['n_compared'] = int(both.sum())
                if both.sum() >= 100:
                    yt = gedi[both].astype(np.float64) / 100.0
                    yp = band[both].astype(np.float64)
                    r = float(np.corrcoef(yt, yp)[0, 1])
                    info['r_vs_gedi'] = round(r, 3)
                    info['bias_m'] = round(float(np.mean(yp - yt)), 2)
                    if not np.isfinite(r) or r < 0.2:
                        problems.append(f'r vs GEDI = {r:.3f} -- possible '
                                        f'misregistration')
                elif gvalid.sum() >= 100:
                    problems.append(f'only {int(both.sum())} of {int(gvalid.sum())} '
                                    f'GEDI px have ECHOSAT data')
    except Exception as exc:
        problems.append(f'unreadable: {type(exc).__name__}: {exc}')

    info['sources'] = echosat_tiles_for_bounds(g_bounds)
    return problems, info


def verify_against_source(hid, tiles_dir, gedi_dir, n_points, rng):
    """Compare the cache against the LIVE server at n_points random locations.

    Samples the source with NEAREST at each point's centre while the cache holds a
    30 m AVERAGE of ~9 source pixels, so the two are not expected to be identical --
    but they must be strongly correlated with ~zero bias. A wrong CRS transform,
    flipped axis or cm/m error collapses the correlation, which no self-consistency
    check can detect.
    """
    cache = Path(tiles_dir) / f'healpix_{hid}_echosat.tif'
    gedi_path = Path(gedi_dir) / f'healpix_{hid}_gedi.tif'
    if not (cache.exists() and gedi_path.exists()):
        return None

    from rasterio.warp import transform as warp_transform

    with rasterio.open(cache) as c:
        band = c.read(3)                                  # 2020
        rows, cols = np.where(np.isfinite(band) & (band != SENTINEL))
        if len(rows) < n_points:
            return None
        pick = rng.choice(len(rows), size=n_points, replace=False)
        rr, cc = rows[pick], cols[pick]
        cached = band[rr, cc].astype(np.float64)
        xs, ys = rasterio.transform.xy(c.transform, rr, cc)
        crs = c.crs
        bounds = c.bounds

    lons, lats = warp_transform(crs, 'EPSG:4326', np.asarray(xs), np.asarray(ys))
    tiles = echosat_tiles_for_bounds((min(lons), min(lats), max(lons), max(lats)))

    src_vals = np.full(len(cached), np.nan)
    for tile in tiles:
        url = f"/vsicurl/{ECHOSAT_URL.format(tile=tile)}"
        try:
            with rasterio.open(url) as s:
                sx, sy = warp_transform('EPSG:4326', s.crs, lons, lats)
                todo = ~np.isfinite(src_vals)
                if not todo.any():
                    break
                pts = [(x, y) for x, y, t in zip(sx, sy, todo) if t]
                # indexes=[3] asks for band 3 (2020) ONLY, so each sample is a
                # 1-element array -- index [0], not [2].
                got = np.array([v[0] for v in s.sample(pts, indexes=[3])],
                               dtype=np.float64)
                got = np.where(got == -9999, np.nan, got / 100.0)
                src_vals[todo] = got
        except Exception as exc:                    # noqa: BLE001
            # Surface it: silently continuing here turns a code bug into the
            # indistinguishable "too few source samples", which is how the
            # v[2]/v[0] slip above went unnoticed.
            print(f"    {tile}: source read failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

    ok = np.isfinite(src_vals) & np.isfinite(cached)
    if ok.sum() < 10:
        return {'n': int(ok.sum()), 'error': 'too few source samples'}
    a, b = cached[ok], src_vals[ok]
    return {
        'n': int(ok.sum()),
        'r': round(float(np.corrcoef(a, b)[0, 1]), 4),
        'bias_m': round(float(np.mean(a - b)), 3),
        'rmse_m': round(float(np.sqrt(np.mean((a - b) ** 2))), 3),
        'tiles': tiles,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-dir', required=True)
    ap.add_argument('--gedi-dir', required=True)
    ap.add_argument('--shapefile', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--tile-id-column', default='healpix_id')
    ap.add_argument('--tile-list', default=None,
                    help='Restrict to these healpix ids (one per line).')
    ap.add_argument('--deep', action='store_true',
                    help='Read every pixel: catches truncated writes, and reports '
                         'value ranges plus a correlation smoke test against GEDI.')
    ap.add_argument('--verify-source', type=int, default=0, metavar='N_TILES',
                    help='Re-read the live ECHOSAT server for this many randomly '
                         'chosen tiles and correlate against the cache. The only '
                         'check that catches a wrong transform or unit error.')
    ap.add_argument('--verify-points', type=int, default=300,
                    help='Points sampled per tile for --verify-source.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-bad', default=None,
                    help='Write the ids that failed (one per line) for a re-run.')
    ap.add_argument('--out-json', default=None,
                    help='Write the full per-tile report as JSON.')
    args = ap.parse_args()

    import geopandas as gpd
    gdf = gpd.read_file(args.shapefile)
    ids = [int(t) for t in gdf[gdf['split'] == args.split][args.tile_id_column]]
    if args.tile_list:
        keep = {int(t) for t in Path(args.tile_list).read_text().split()
                if t.strip() and not t.strip().startswith('#')}
        ids = [t for t in ids if t in keep]
    ids.sort()

    print(f"Checking {len(ids)} '{args.split}' tiles")
    print(f"  cache : {args.tiles_dir}")
    print(f"  gedi  : {args.gedi_dir}")
    print(f"  mode  : {'deep (reads every pixel)' if args.deep else 'structural only'}")
    print()

    report, bad = {}, []
    for i, hid in enumerate(ids, 1):
        problems, info = check_tile(hid, args.tiles_dir, args.gedi_dir, args.deep)
        report[str(hid)] = {'problems': problems, **info}
        if problems:
            bad.append(hid)
            print(f"[{i}/{len(ids)}] tile {hid}: FAIL")
            for p in problems:
                print(f"    - {p}")
        elif args.deep:
            print(f"[{i}/{len(ids)}] tile {hid}: ok  "
                  f"nodata={info.get('nodata_frac', 0):.1%}  "
                  f"range=[{info.get('min_m')},{info.get('max_m')}]m  "
                  f"r_vs_gedi={info.get('r_vs_gedi')}  "
                  f"src={'+'.join(info.get('sources', []))}")

    healthy = [t for t in ids if t not in set(bad)]

    src_report = {}
    if args.verify_source and healthy:
        rng = np.random.default_rng(args.seed)
        sample = sorted(random.Random(args.seed).sample(
            healthy, min(args.verify_source, len(healthy))))
        print(f"\n{'=' * 70}")
        print(f"SOURCE VERIFICATION -- {len(sample)} tile(s) re-read from the server")
        print(f"{'=' * 70}")
        print("cache holds a 30 m average; source is sampled nearest at 10 m, so")
        print("expect r > 0.9 and |bias| < ~1 m. A collapsed r means the transform")
        print("or units are wrong.\n")
        for hid in sample:
            res = verify_against_source(hid, args.tiles_dir, args.gedi_dir,
                                        args.verify_points, rng)
            src_report[str(hid)] = res
            if res is None or 'error' in (res or {}):
                print(f"  tile {hid}: could not verify ({(res or {}).get('error')})")
                continue
            flag = '' if (res['r'] > 0.9 and abs(res['bias_m']) < 2.0) else '   <-- CHECK'
            print(f"  tile {hid}: n={res['n']:4d}  r={res['r']:.4f}  "
                  f"bias={res['bias_m']:+.3f} m  rmse={res['rmse_m']:.3f} m{flag}")
            if flag:
                bad.append(hid)

    print("\n" + "=" * 70)
    print(f"healthy : {len(ids) - len(set(bad))}/{len(ids)}")
    print(f"failed  : {len(set(bad))}")
    if bad:
        print(f"  {sorted(set(bad))}")
    print("=" * 70)

    if args.out_bad and bad:
        Path(args.out_bad).write_text("\n".join(str(t) for t in sorted(set(bad))) + "\n")
        print(f"wrote {args.out_bad} -- re-run those with --overwrite")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            {'split': args.split, 'tiles': report, 'source_verification': src_report},
            indent=2))
        print(f"wrote {args.out_json}")

    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
