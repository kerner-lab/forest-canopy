#!/usr/bin/env python3
"""
Generate WALL-TO-WALL TESSERA embeddings over an AOI.

The TESSERA analogue of alphaearth_wall_to_wall.py: same AOI inputs (--shapefile,
--aoi-tif, --aoi + AOI_YEAR), same target grid (--like, or --dst-crs/--resolution with
auto-UTM), same masking and same output layout — a DENSE, georeferenced, tiled float32
GeoTIFF, here with the 128 TESSERA channels as bands.

Pipeline:
  1. Read the AOI (shapefile polygon, or raster footprint); build the target grid.
     With --aoi/--aoi-tif the grid defaults to the AOI (ALS) raster's own grid
     (--own-grid to opt out), so embeddings and ALS pair pixel-for-pixel.
  2. Ask the GeoTessera registry which 0.1deg x 0.1deg blocks cover the AOI for --year.
     With --aoi-tif/--aoi (and masking on), drop blocks that touch no valid AOI pixel.
  3. Stream the blocks ONE AT A TIME (each ~485 MB as float32, in its own UTM CRS).
     Each block is reprojected (bilinear) onto just the sub-window of the target grid
     it covers, and blended with a FEATHERED WEIGHTED AVERAGE — identical to
     tessera_processing.reproject_tiles_to_gedi — into disk-backed accumulators:
         out = sum(value_i * weight_i) / sum(weight_i)
  4. Normalise acc / wsum window by window, mask outside the AOI to NaN, and write.

Why block-first rather than window-first (as the AEF script does): the aef-mosaic is a
lazy Zarr that can be subset cheaply per window, but TESSERA only serves whole blocks.
Fetching per output window would re-download each block for every window it touches.
Here every block is downloaded exactly once; the price is two temporary memmaps
(~ 128 * H * W * 4 bytes, the size of the uncompressed output) in --tmp-dir.

Reuses tessera_processing.py (GeoTessera client, block lookup/streaming, dequant) and
alphaearth_wall_to_wall.py (AOI_YEAR, AOI footprint, target grid) so the TESSERA values
and the grid match the per-tile extraction and the AEF wall-to-wall rasters exactly.

Usage — ALS cal/val AOI by name (year from AOI_YEAR), aligned to the AOI raster:
    python tessera_wall_to_wall.py \\
        --aoi CA_LosAngeles_1_B23_4326 --aoi-dir /path/ALS_CalVal \\
        --output-dir /path/tessera_als
    python tessera_wall_to_wall.py --list-aois

Usage — shapefile, auto UTM at 30 m:
    python tessera_wall_to_wall.py \\
        --shapefile /path/aoi.shp --year 2024 \\
        --output /path/tessera_2024_wall_to_wall.tif

Usage — check TESSERA coverage for the AOI/year without downloading embeddings:
    python tessera_wall_to_wall.py --aoi CA25_Allen_4326 --aoi-dir /path/ALS_CalVal \\
        --output-dir /path/tessera_als --dry-run

Notes:
  - TESSERA is 10 m, 128-D, one native UTM CRS per block; 30 m output is downsampled
    bilinearly (as in tessera_processing.py).
  - Year coverage is not uniform (2024 global; earlier years being back-filled). If the
    registry has no blocks for the AOI/year the script exits before allocating anything.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_processing as tp  # noqa: E402
import alphaearth_wall_to_wall as awtw  # noqa: E402  (AOI_YEAR + grid helpers)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

AOI_YEAR = awtw.AOI_YEAR
BLOCK_DEG = 0.1          # TESSERA blocks are 0.1deg squares centred on x.x5


def block_window(lon_c, lat_c, dst_crs, transform, width, height, pad=2):
    """Target-grid window covered by the TESSERA block centred at (lon_c, lat_c).

    The block's lon/lat square is transformed to dst_crs (all four corners), rounded
    outward, padded by `pad` px for bilinear edge support, and clipped to the grid.
    Returns None if the block falls entirely outside the grid.
    """
    h = BLOCK_DEG / 2
    bx0, by0, bx1, by1 = tp.bounds_to_crs(lon_c - h, lat_c - h, lon_c + h, lat_c + h,
                                          CRS.from_epsg(4326), dst_crs)
    win = from_bounds(bx0, by0, bx1, by1, transform=transform)
    c0 = max(0, int(np.floor(win.col_off)) - pad)
    r0 = max(0, int(np.floor(win.row_off)) - pad)
    c1 = min(width, int(np.ceil(win.col_off + win.width)) + pad)
    r1 = min(height, int(np.ceil(win.row_off + win.height)) + pad)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def aoi_valid_mask(aoi_vrt, win):
    """True where the (warped) AOI raster has valid, finite data in `win`."""
    return (aoi_vrt.read_masks(1, window=win) > 0) & np.isfinite(aoi_vrt.read(1, window=win))


def accumulate_block(emb, src_crs, src_transform, acc, wsum, win, grid_transform, dst_crs,
                     resampling=Resampling.bilinear):
    """Reproject one block onto `win` and add value*weight / weight into acc / wsum.

    Same feathered weighting as tessera_processing.reproject_tiles_to_gedi: the weight is
    a bilinearly reprojected unit mask (1 inside the block, ramping to 0 at its edges),
    zeroed wherever the block carried no embedding (0.0 = TESSERA nodata).
    """
    wt = window_transform(win, grid_transform)
    rs = slice(win.row_off, win.row_off + win.height)
    cs = slice(win.col_off, win.col_off + win.width)

    dst = np.zeros((emb.shape[0], win.height, win.width), dtype=np.float32)
    rasterio.warp.reproject(
        source=emb, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=wt, dst_crs=dst_crs,
        resampling=resampling, src_nodata=0.0, dst_nodata=0.0,
    )
    weight = np.zeros((win.height, win.width), dtype=np.float32)
    rasterio.warp.reproject(
        source=np.ones(emb.shape[1:], dtype=np.float32), destination=weight,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=wt, dst_crs=dst_crs,
        resampling=Resampling.bilinear, src_nodata=None, dst_nodata=0.0,
    )
    weight *= np.any(dst != 0.0, axis=0)

    acc[:, rs, cs] += dst * weight[None, :, :]
    wsum[rs, cs] += weight
    return bool(weight.any())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    aoi = ap.add_mutually_exclusive_group()
    aoi.add_argument("--shapefile", help="boundary shapefile (.shp)")
    aoi.add_argument("--aoi-tif", help="raster whose footprint (bounding box) is the AOI")
    aoi.add_argument("--aoi", help="AOI name from AOI_YEAR; reads <--aoi-dir>/<name>.tif "
                                   "and defaults --year to AOI_YEAR[name]")
    aoi.add_argument("--list-aois", action="store_true",
                     help="print 'name<TAB>year' for every AOI_YEAR entry and exit")
    ap.add_argument("--aoi-dir", help="folder holding the AOI tifs (with --aoi)")
    ap.add_argument("--year", type=int, default=None,
                    help="TESSERA year (default with --aoi/--aoi-tif: AOI_YEAR lookup by "
                         "tif stem)")
    ap.add_argument("--output", default=None, help="output GeoTIFF path (.tif)")
    ap.add_argument("--output-dir", default=None,
                    help="instead of --output: write <output-dir>/<aoi stem>_tessera_<year>.tif")
    ap.add_argument("--like", default=None,
                    help="reference raster to copy the grid from (crs/transform/size); "
                         "overrides --dst-crs/--resolution and aligns pixel-for-pixel. "
                         "Default with --aoi/--aoi-tif: the AOI (ALS) raster itself")
    ap.add_argument("--own-grid", action="store_true",
                    help="with --aoi/--aoi-tif, build an auto-UTM/--dst-crs grid at "
                         "--resolution instead of using the AOI raster's grid")
    ap.add_argument("--dst-crs", default=None,
                    help="target CRS (e.g. EPSG:32610). Default: auto-UTM from AOI centroid")
    ap.add_argument("--resolution", type=float, default=30.0, help="target pixel size (m)")
    ap.add_argument("--resampling", default="bilinear",
                    choices=("bilinear", "average", "nearest"),
                    help="10 m -> output-grid resampling. average aggregates every 10 m "
                         "pixel in the cell (olmoearth_wall_to_wall.py default); use the "
                         "same choice for every encoder you compare")
    ap.add_argument("--block-size", type=int, default=1024,
                    help="output window size in pixels for the final write pass")
    ap.add_argument("--buffer", type=float, default=0.01,
                    help="degrees to pad the AOI when selecting covering TESSERA blocks")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep the full bounding box (default: mask pixels outside the "
                         "polygon; with --aoi-tif/--aoi, outside the raster's valid pixels, "
                         "and skip TESSERA blocks that touch none without downloading them)")
    ap.add_argument("--cache-dir", default=None,
                    help="GeoTessera cache directory (default: library default)")
    ap.add_argument("--dataset-version", default=None,
                    help="GeoTessera dataset version (e.g. v1); default: library default (v1 in geotessera 0.8)")
    ap.add_argument("--tmp-dir", default=None,
                    help="where to put the accumulator memmaps (~ size of the uncompressed "
                         "output). Default: next to the output. Use node-local scratch on HPC.")
    ap.add_argument("--compress", default="lzw",
                    help="GeoTIFF compression (lzw, deflate, none). Default lzw — see --predictor")
    ap.add_argument("--predictor", type=int, default=0,
                    help="TIFF predictor; 0 = none (default). deflate+predictor=3 can crash "
                         "GDAL at close and leave an all-nodata tif (see "
                         "alphaearth_wall_to_wall.py --help).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid, covering block count and memmap size, then exit "
                         "(registry lookup only; no embedding downloads)")
    args = ap.parse_args()

    if args.list_aois:
        for name, year in AOI_YEAR.items():
            print(f"{name}\t{year}")
        return

    if args.aoi:
        if not args.aoi_dir:
            ap.error("--aoi needs --aoi-dir")
        if args.aoi not in AOI_YEAR:
            ap.error(f"--aoi {args.aoi!r} is not in AOI_YEAR: {sorted(AOI_YEAR)}")
        args.aoi_tif = str(Path(args.aoi_dir) / f"{args.aoi}.tif")
    if not (args.shapefile or args.aoi_tif):
        ap.error("give one of --shapefile, --aoi-tif, --aoi")

    aoi_path = args.shapefile or args.aoi_tif
    if not Path(aoi_path).exists():
        sys.exit(f"ERROR: AOI not found: {aoi_path}")
    stem = Path(aoi_path).stem
    if args.year is None:
        if args.aoi_tif and stem in AOI_YEAR:
            args.year = AOI_YEAR[stem]
        else:
            ap.error(f"--year is required ({stem!r} is not in AOI_YEAR)")
    elif args.aoi_tif and stem in AOI_YEAR and args.year != AOI_YEAR[stem]:
        print(f"Warning: --year {args.year} overrides AOI_YEAR[{stem!r}]={AOI_YEAR[stem]}",
              file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
    elif args.output_dir:
        out_path = Path(args.output_dir) / f"{stem}_tessera_{args.year}.tif"
    else:
        ap.error("give --output or --output-dir")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.aoi_tif:
        gdf = awtw.raster_footprint(args.aoi_tif)
        print(f"AOI raster footprint: {args.aoi_tif}  CRS={gdf.crs.to_string()}  year={args.year}")
        if args.like is None and not args.own_grid:
            args.like = args.aoi_tif        # align pixel-for-pixel with the ALS raster
    else:
        gdf = gpd.read_file(args.shapefile)
        if gdf.crs is None:
            print("Warning: shapefile has no CRS; assuming EPSG:4326", file=sys.stderr)
            gdf = gdf.set_crs(4326)
        print(f"Loaded shapefile: {len(gdf)} feature(s), CRS={gdf.crs}")

    dst_crs, transform, width, height = awtw.build_target_grid(
        gdf, args.like, args.dst_crs, args.resolution)
    dst_crs = CRS.from_user_input(dst_crs)
    res_x = abs(transform.a)
    n_ch = tp.TESSERA_N_CHANNELS
    est_gb = n_ch * width * height * 4 / 1e9
    res_str = f"{res_x:.6g} deg" if dst_crs.is_geographic else f"{res_x:.1f} m"
    print(f"Target grid : {width} x {height} px  @ {res_str}  CRS={dst_crs.to_string()}")
    print(f"Memmaps     : ~{est_gb:.1f} GB in --tmp-dir (also ~ uncompressed output size)")
    if args.no_mask:
        mask_desc = "OFF (full bbox)"
    elif args.aoi_tif:
        mask_desc = "ON (clip to AOI raster's valid pixels; skip blocks with none)"
    else:
        mask_desc = "ON (clip to polygon)"
    print(f"Mask        : {mask_desc}")

    # Mask source: an AOI raster's valid pixels (warped onto the output grid; exact when
    # --like is the AOI raster itself), else the shapefile polygon.
    aoi_vrt = gdf_dst = None
    if not args.no_mask:
        if args.aoi_tif:
            aoi_vrt = WarpedVRT(rasterio.open(args.aoi_tif), crs=dst_crs, transform=transform,
                                width=width, height=height, resampling=Resampling.nearest)
        else:
            gdf_dst = gdf.to_crs(dst_crs)

    # Registry lookup over the AOI's lon/lat bounds — cheap, no downloads.
    gt = tp.open_geotessera(args.cache_dir, args.dataset_version)
    try:
        years = gt.registry.get_available_years()
        if args.year not in years:
            print(f"Warning: year {args.year} not in TESSERA years {years}", file=sys.stderr)
    except AttributeError:
        pass
    lon0, lat0, lon1, lat1 = gdf.to_crs(4326).total_bounds
    blocks = tp.list_covering_blocks(gt, lon0, lat0, lon1, lat1, args.year, args.buffer)

    # Pair each block with the target-grid window it covers; drop off-grid blocks and,
    # with an AOI raster, blocks that touch no valid AOI pixel (never downloaded).
    jobs, n_no_aoi = [], 0
    for blk in blocks:
        _, lon_c, lat_c = blk
        win = block_window(lon_c, lat_c, dst_crs, transform, width, height)
        if win is None:
            continue
        if aoi_vrt is not None and not aoi_valid_mask(aoi_vrt, win).any():
            n_no_aoi += 1
            continue
        jobs.append((blk, win))
    print(f"TESSERA     : {len(blocks)} block(s) cover the AOI for {args.year}; "
          f"{len(jobs)} to fetch, {n_no_aoi} skipped (no AOI data)")
    if not jobs:
        sys.exit(f"ERROR: no TESSERA blocks to fetch for {stem} / {args.year} "
                 f"(no registry coverage, or none overlapping valid AOI pixels)")

    if args.dry_run:
        print("Dry run — re-run without --dry-run to generate.")
        return

    tmp_root = Path(args.tmp_dir) if args.tmp_dir else out_path.parent
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_path.stem}_", dir=tmp_root))
    t0 = time.time()
    try:
        # np.memmap with mode w+ zero-fills (sparse file), so untouched pixels stay 0.
        acc = np.memmap(tmp / "acc.f32", dtype=np.float32, mode="w+",
                        shape=(n_ch, height, width))
        wsum = np.memmap(tmp / "wsum.f32", dtype=np.float32, mode="w+",
                         shape=(height, width))

        # --- Pass 1: stream every block once, blend into the accumulators. ---
        win_of = {blk: win for blk, win in jobs}
        n_contrib = 0
        stream = tp.iter_covering_tiles(gt, [blk for blk, _ in jobs], _Log)
        it = tqdm(zip(jobs, stream), total=len(jobs), desc="TESSERA blocks") \
            if _HAS_TQDM else zip(jobs, stream)
        for (blk, _), (emb, src_crs, src_transform) in it:
            if emb.shape[0] != n_ch:
                sys.exit(f"ERROR: block {blk} has {emb.shape[0]} channels, expected {n_ch}")
            n_contrib += accumulate_block(emb, src_crs, src_transform, acc, wsum,
                                          win_of[blk], transform, dst_crs,
                                          Resampling[args.resampling])
            del emb
        acc.flush()
        wsum.flush()
        t_fetch = time.time() - t0
        print(f"Fetched + blended {len(jobs)} block(s) ({n_contrib} with data) "
              f"in {t_fetch:.0f}s")

        # --- Pass 2: normalise, mask, write. ---
        nodata = np.float32("nan")
        profile = dict(
            driver="GTiff", dtype="float32", count=n_ch, crs=dst_crs,
            transform=transform, width=width, height=height, nodata=nodata,
            tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
        )
        if args.compress and args.compress.lower() != "none":
            profile["compress"] = args.compress
            if args.predictor:
                profile["predictor"] = args.predictor

        n_written = n_empty = 0
        probe = None    # (window, row, col) of one pixel known to be finite when written
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(n_ch):
                dst.set_band_description(b + 1, f"tessera_ch_{b + 1:03d}")
            for win in awtw.iter_windows(width, height, args.block_size):
                rs = slice(win.row_off, win.row_off + win.height)
                cs = slice(win.col_off, win.col_off + win.width)
                w = np.asarray(wsum[rs, cs])
                valid = w > 0
                if aoi_vrt is not None:
                    valid &= aoi_valid_mask(aoi_vrt, win)
                if gdf_dst is not None:
                    valid &= ~geometry_mask(gdf_dst.geometry, out_shape=w.shape,
                                            transform=window_transform(win, transform),
                                            invert=False)   # True where OUTSIDE polygon
                if not valid.any():
                    n_empty += 1        # left unwritten → reads as nodata
                    continue
                block = np.full((n_ch, win.height, win.width), np.nan, dtype=np.float32)
                block[:, valid] = np.asarray(acc[:, rs, cs])[:, valid] / w[valid]
                dst.write(block, window=win)
                n_written += 1
                if probe is None:
                    r, c = np.argwhere(valid)[0]
                    probe = (win, int(r), int(c))

        # Same finalisation check as alphaearth_wall_to_wall.py: a GDAL crash at close
        # leaves a full-size tif with no tile index that reads as nodata everywhere.
        if probe is not None:
            win, r, c = probe
            with rasterio.open(out_path) as chk:
                got = chk.read(1, window=Window(win.col_off + c, win.row_off + r, 1, 1))
            if not np.isfinite(got).any():
                sys.exit(f"ERROR: {out_path} did not finalize — a pixel that was finite "
                         f"when written now reads as nodata. Re-run with --compress lzw "
                         f"--predictor 0.")
        else:
            print("Warning: no valid pixels were written (TESSERA has no data inside "
                  "the AOI mask).", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.0f}s. windows written={n_written} empty={n_empty}")
    print(f"Output: {out_path}  ({size_mb:.1f} MB on disk, {n_ch} bands)")

    summary = {
        "shapefile": args.shapefile, "aoi_tif": args.aoi_tif, "year": args.year,
        "output": str(out_path), "source": "tessera",
        "dataset_version": args.dataset_version,
        "dst_crs": dst_crs.to_string(), "resolution_m": res_x,
        "width": width, "height": height, "n_channels": n_ch,
        "masked": not args.no_mask, "resampling": args.resampling,
        "tessera_blocks_covering": len(blocks), "tessera_blocks_fetched": len(jobs),
        "tessera_blocks_with_data": n_contrib, "tessera_blocks_no_aoi_data": n_no_aoi,
        "windows_written": n_written, "windows_empty": n_empty,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sidecar: {out_path.with_suffix('.json')}")


class _Log:
    """Minimal logger for tp._to_float32_chw's integer-dtype warning."""
    @staticmethod
    def warning(msg, *a):
        print("Warning: " + (msg % a if a else msg), file=sys.stderr)


if __name__ == "__main__":
    main()
