#!/usr/bin/env python3
"""
Generate WALL-TO-WALL Alpha Earth Foundation (aef-mosaic) embeddings over an AOI.

Produces a DENSE, georeferenced multi-band GeoTIFF of the 64 AEF channels at a
chosen resolution (default 30 m) over the full extent of an AOI, given as either
a boundary shapefile (--shapefile) or a raster whose footprint is the AOI
(--aoi-tif / --aoi, e.g. the ALS cal/val tifs).

Pipeline:
  1. Read the AOI (shapefile polygon, or raster footprint); compute its extent.
  2. Build a target grid: either copied from a reference raster (--like) so it aligns
     pixel-for-pixel with your GEDI/model grid, or constructed from --dst-crs + --resolution.
  3. Process the output in windows (bounded memory / S3 fetch):
       - transform each window's bounds to the aef CRS (EPSG:4326),
       - subset the aef-mosaic to that window + year,
       - dequantize int8 -> float32 (signed-square formula),
       - reproject onto the window (bilinear),
       - optionally mask pixels outside the shapefile polygon to nodata.
  4. Write a tiled, compressed float32 GeoTIFF (64 bands), one band per AEF channel.

Reuses the aef helpers from alphaearth_processing.py so the mosaic access, dequantization,
and reprojection match your GEDI extraction exactly.

Usage — auto UTM grid at 30 m, masked to the polygon:
    python alphaearth_wall_to_wall.py \\
        --shapefile /path/aoi.shp --year 2020 \\
        --output /path/aef_2020_wall_to_wall.tif

Usage — ALS cal/val AOI by name (year looked up in AOI_YEAR, tif found in --aoi-dir):
    python alphaearth_wall_to_wall.py \\
        --aoi CA_LosAngeles_1_B23 --aoi-dir /path/ALS_CalVal \\
        --output-dir /path/aef_als
    python alphaearth_wall_to_wall.py --list-aois    # "name<TAB>year" per line, for bash

Usage — align to an existing GEDI/prediction raster grid (recommended for validation):
    python alphaearth_wall_to_wall.py \\
        --shapefile /path/aoi.shp --year 2020 \\
        --like /path/reference_30m.tif \\
        --output /path/aef_2020_wall_to_wall.tif

Notes:
  - aef-mosaic CRS is EPSG:4326 (~10 m GSD). 30 m output is upsampled from ~10 m source.
  - "30 m/pixel" requires a metric CRS; the default auto-UTM (or --dst-crs) ensures that.
  - Output can be large: count*height*width*4 bytes. Tiled+deflate keeps it reasonable;
    BIGTIFF is enabled when needed.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.transform import from_origin, array_bounds
from rasterio.windows import Window, transform as window_transform
from rasterio.features import geometry_mask
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from affine import Affine
from pyproj import CRS
from shapely.geometry import box

# Reuse the proven aef-mosaic helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import alphaearth_processing as aep  # noqa: E402

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ALS cal/val AOIs -> AEF year to export. Keys are the AOI tif stems (<key>.tif in
# --aoi-dir); each year matches the ALS acquisition so embeddings and lidar agree.
AOI_YEAR = {
    'CA_LosAngeles_1_B23_4326': 2024,
    'CA_CaliforniaGaps_4_B23_4326': 2023,
    'OR_SWCentralSycan_4_B21_4326': 2022,
    'CA_NorthernCA_1_B22_4326': 2022,
    'CA25_Allen_4326': 2018,
    'CA20_Brande_4326': 2018,
    '2017_USFS_Lidar_Umpqua_National_Forest_Tiller_OR_4326': 2017,
    '2017_USFS_Lidar_Diamond_Lake_OR_4326': 2017,
}


def raster_footprint(path: str) -> gpd.GeoDataFrame:
    """The raster's bounding box as a one-polygon GeoDataFrame in its own CRS.

    Edges are densified so the box stays faithful when reprojected (e.g. a
    EPSG:4326 tif onto UTM, where straight lon/lat edges become curves).
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            sys.exit(f"ERROR: {path} has no CRS")
        b = src.bounds
        geom = box(*b).segmentize(max(b.right - b.left, b.top - b.bottom) / 100)
        return gpd.GeoDataFrame(geometry=[geom], crs=src.crs)


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def infer_n_channels(ds) -> int:
    """Number of AEF embedding channels (the dim that isn't time/x/y)."""
    emb = ds["embeddings"]
    band_dims = [d for d in emb.dims if d not in ("time", "x", "y")]
    if not band_dims:
        raise ValueError(f"Could not find a band dim in {emb.dims}")
    return int(emb.sizes[band_dims[0]])


def build_target_grid(gdf: gpd.GeoDataFrame, like: str, dst_crs_arg: str, resolution: float):
    """Return (dst_crs, transform, width, height)."""
    if like:
        with rasterio.open(like) as ref:
            return ref.crs, ref.transform, ref.width, ref.height

    # Pick CRS: explicit, else auto-UTM from the AOI centroid (in lon/lat).
    if dst_crs_arg:
        dst_crs = CRS.from_user_input(dst_crs_arg)
    else:
        c = gdf.to_crs(4326).union_all().centroid
        dst_crs = CRS.from_epsg(utm_epsg_from_lonlat(c.x, c.y))

    bounds = gdf.to_crs(dst_crs).total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    # Snap extent outward to a whole number of pixels.
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    return dst_crs, transform, width, height


def iter_windows(width: int, height: int, block: int):
    for row in range(0, height, block):
        for col in range(0, width, block):
            yield Window(col, row, min(block, width - col), min(block, height - row))


def fetch_aef_block(ds, aef_crs, dst_crs, win_transform, win_w, win_h, year, buffer):
    """
    Subset + dequantize the aef-mosaic over a destination window's footprint.
    Returns (data (C,Hs,Ws) float32, src_transform) or (None, None) if empty.
    """
    minx, miny, maxx, maxy = array_bounds(win_h, win_w, win_transform)
    ax0, ay0, ax1, ay1 = aep.bounds_to_crs(minx, miny, maxx, maxy, dst_crs, aef_crs)
    subset = aep.subset_aef(ds, ax0, ay0, ax1, ay1, year, buffer=buffer)
    if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
        return None, None
    raw = subset.compute().values
    if raw.ndim == 4:
        raw = raw.squeeze(axis=0)
    if raw.ndim != 3:
        raise ValueError(f"Unexpected aef ndim={raw.ndim}, shape={raw.shape}")
    if raw.shape[0] > raw.shape[-1]:        # (H, W, C) -> (C, H, W)
        raw = np.transpose(raw, (2, 0, 1))
    data = aep.dequantize_aef(raw)
    src_transform = aep.src_transform_from_coords(subset.x.values, subset.y.values)
    return data, src_transform


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
                    help="aef year, e.g. 2020 (default with --aoi/--aoi-tif: AOI_YEAR "
                         "lookup by tif stem)")
    ap.add_argument("--output", default=None, help="output GeoTIFF path (.tif)")
    ap.add_argument("--output-dir", default=None,
                    help="instead of --output: write <output-dir>/<aoi stem>_aef_<year>.tif")
    ap.add_argument("--like", default=None,
                    help="reference raster to copy the grid from (crs/transform/size); "
                         "overrides --dst-crs/--resolution and aligns pixel-for-pixel")
    ap.add_argument("--dst-crs", default=None,
                    help="target CRS (e.g. EPSG:32610). Default: auto-UTM from AOI centroid")
    ap.add_argument("--resolution", type=float, default=30.0, help="target pixel size (m)")
    ap.add_argument("--block-size", type=int, default=1024,
                    help="output window size in pixels for memory-bounded processing")
    ap.add_argument("--buffer", type=float, default=0.01,
                    help="degrees to pad each window before aef subsetting (edge coverage)")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep the full bounding box (default: mask pixels outside the "
                         "polygon; with --aoi-tif/--aoi, outside the raster's valid pixels, "
                         "and skip blocks with none without fetching aef)")
    ap.add_argument("--compress", default="lzw",
                    help="GeoTIFF compression (lzw, deflate, none). Default lzw — see --predictor")
    ap.add_argument("--predictor", type=int, default=0,
                    help="TIFF predictor; 0 = none (default). deflate+predictor=3 (float "
                         "predictor) can abort with 'double free / corruption' at GDAL close "
                         "on some libtiff builds. That crash happens AFTER every block is "
                         "written but BEFORE GDAL patches TileOffsets/TileByteCounts into the "
                         "header, leaving a full-size .tif whose tiles all read as nodata and "
                         "no sidecar .json. Only set this if you know your build is safe.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the target grid + block count and exit (no S3 fetch)")
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
        out_path = Path(args.output_dir) / f"{stem}_aef_{args.year}.tif"
    else:
        ap.error("give --output or --output-dir")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.aoi_tif:
        gdf = raster_footprint(args.aoi_tif)
        print(f"AOI raster footprint: {args.aoi_tif}  CRS={gdf.crs.to_string()}  year={args.year}")
    else:
        gdf = gpd.read_file(args.shapefile)
        if gdf.crs is None:
            print("Warning: shapefile has no CRS; assuming EPSG:4326", file=sys.stderr)
            gdf = gdf.set_crs(4326)
        print(f"Loaded shapefile: {len(gdf)} feature(s), CRS={gdf.crs}")

    dst_crs, transform, width, height = build_target_grid(
        gdf, args.like, args.dst_crs, args.resolution)
    res_x = abs(transform.a)
    n_blocks = len(list(iter_windows(width, height, args.block_size)))
    est_gb = 64 * width * height * 4 / 1e9
    res_str = f"{res_x:.6g} deg" if CRS.from_user_input(dst_crs).is_geographic else f"{res_x:.1f} m"
    print(f"Target grid : {width} x {height} px  @ {res_str}  CRS={dst_crs.to_string()}")
    print(f"Blocks      : {n_blocks} ({args.block_size}px)   uncompressed ~{est_gb:.1f} GB (64 band f32)")
    if args.no_mask:
        mask_desc = "OFF (full bbox)"
    elif args.aoi_tif:
        mask_desc = "ON (clip to AOI raster's valid pixels; skip blocks with none)"
    else:
        mask_desc = "ON (clip to polygon)"
    print(f"Mask        : {mask_desc}")

    if args.dry_run:
        print("Dry run — re-run without --dry-run to generate.")
        return

    print("\nOpening aef-mosaic (lazy)...")
    ds = aep.open_aef_mosaic()
    aef_crs = aep.get_aef_crs(ds)
    n_channels = infer_n_channels(ds)
    years = ds["embeddings"].coords["time"].values.tolist()
    print(f"aef CRS={aef_crs.to_epsg() or aef_crs.name}  channels={n_channels}  years={years}")
    if args.year not in years:
        print(f"Warning: year {args.year} not in {years}", file=sys.stderr)

    # Mask source: an AOI raster's valid pixels (warped onto the output grid; exact when
    # --like is the AOI raster itself), else the shapefile polygon.
    aoi_vrt = gdf_dst = None
    if not args.no_mask:
        if args.aoi_tif:
            aoi_vrt = WarpedVRT(rasterio.open(args.aoi_tif), crs=dst_crs, transform=transform,
                                width=width, height=height, resampling=Resampling.nearest)
        else:
            gdf_dst = gdf.to_crs(dst_crs)
    nodata = np.float32("nan")

    profile = dict(
        driver="GTiff", dtype="float32", count=n_channels, crs=dst_crs,
        transform=transform, width=width, height=height, nodata=nodata,
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    )
    if args.compress and args.compress.lower() != "none":
        profile["compress"] = args.compress
        if args.predictor:
            profile["predictor"] = args.predictor

    t0 = time.time()
    n_written = n_empty = n_no_aoi = 0
    probe = None        # (window, row, col) of one pixel known to be finite when written
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(n_channels):
            dst.set_band_description(b + 1, f"aef_ch_{b + 1:03d}")

        windows = list(iter_windows(width, height, args.block_size))
        it = tqdm(windows, desc="blocks") if _HAS_TQDM else windows
        for win in it:
            wt = window_transform(win, transform)
            aoi_valid = None
            if aoi_vrt is not None:
                aoi_valid = (aoi_vrt.read_masks(1, window=win) > 0) & \
                    np.isfinite(aoi_vrt.read(1, window=win))
                if not aoi_valid.any():     # no AOI data here: don't fetch aef at all
                    n_no_aoi += 1
                    continue
            data, src_transform = fetch_aef_block(
                ds, aef_crs, dst_crs, wt, win.width, win.height, args.year, args.buffer)
            if data is None:
                n_empty += 1
                continue

            block = np.full((n_channels, win.height, win.width), np.nan, dtype=np.float32)
            rasterio.warp.reproject(
                source=data, destination=block,
                src_transform=src_transform, src_crs=aef_crs,
                dst_transform=wt, dst_crs=dst_crs,
                src_nodata=0.0, dst_nodata=np.nan, resampling=Resampling.bilinear,
            )

            if gdf_dst is not None:
                outside = geometry_mask(gdf_dst.geometry, out_shape=(win.height, win.width),
                                        transform=wt, invert=False)  # True where OUTSIDE polygon
                block[:, outside] = np.nan
            if aoi_valid is not None:
                block[:, ~aoi_valid] = np.nan

            dst.write(block, window=win)
            n_written += 1
            if probe is None:
                fin = np.argwhere(np.isfinite(block[0]))
                if len(fin):
                    probe = (win, int(fin[0][0]), int(fin[0][1]))

    # Verify the file actually finalized. If GDAL crashes at close (see --predictor), the
    # tile data is on disk but TileOffsets/TileByteCounts stay all-zero: the .tif opens,
    # reports the right size and CRS, and reads as ALL NODATA everywhere. Re-read one pixel
    # that was finite when written — the cheapest possible proof that the index survived.
    if probe is not None:
        win, r, c = probe
        with rasterio.open(out_path) as chk:
            got = chk.read(1, window=Window(win.col_off + c, win.row_off + r, 1, 1))
        if not np.isfinite(got).any():
            sys.exit(f"ERROR: {out_path} did not finalize — a pixel that was finite when "
                     f"written now reads as nodata. The tile index was not written (GDAL "
                     f"likely crashed at close). Re-run with --compress lzw --predictor 0.")
    elif n_written:
        print("Warning: every written block was all-nodata; could not verify the file "
              "finalized.", file=sys.stderr)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.0f}s. blocks written={n_written} empty={n_empty} "
          f"skipped (no AOI data)={n_no_aoi}")
    print(f"Output: {out_path}  ({size_mb:.1f} MB on disk, {n_channels} bands)")

    summary = {
        "shapefile": args.shapefile, "aoi_tif": args.aoi_tif, "year": args.year, "output": str(out_path),
        "dst_crs": dst_crs.to_string(), "resolution_m": res_x,
        "width": width, "height": height, "n_channels": n_channels,
        "masked": not args.no_mask, "blocks_written": n_written, "blocks_empty": n_empty,
        "blocks_no_aoi_data": n_no_aoi,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sidecar: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
