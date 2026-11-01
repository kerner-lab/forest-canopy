#!/usr/bin/env python3
"""
Generate WALL-TO-WALL OlmoEarth v1.3 embeddings over an AOI from the precomputed
Source Cooperative store (https://source.coop/olmoearth/olmoearth-v1-3-embeddings).

The OlmoEarth analogue of alphaearth_wall_to_wall.py / tessera_wall_to_wall.py: same AOI
inputs (--aoi/--aoi-dir, --aoi-tif, --shapefile, AOI_YEAR), same masking and the same
output layout — a tiled float32 GeoTIFF, here with the 128 OlmoEarth channels as bands.

Store layout (s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr, anonymous):
  one group per UTM zone, utm01..utm60, each with
    embeddings  int8 (time, 128, y, x)   10 m, nodata -128
    time        years (2017..2025)
    attrs       proj:code (EPSG:326NN), spatial:transform (Affine, 6 coeffs)
  dequantize: (abs(q) / 127.5) ** 2 * sign(q)   (geoemb:quantization in the attrs)

Default grid with --aoi/--aoi-tif is the AOI (ALS) raster's OWN grid, so the output is
pixel-for-pixel aligned with the ALS ground truth and predict_als_aoi.py /
read_aoi_pairs pair them without regridding (its QA rejects mismatched grids). Pass
--own-grid to instead build an auto-UTM grid at --resolution, or --like for any other
reference raster.

Pipeline, per output window (bounded memory; only the needed zarr chunks are read):
  1. Work out which UTM zone(s) the window touches (normally one).
  2. Transform the window bounds into the zone CRS, convert to row/col via the zone's
     spatial:transform, and read embeddings[year, :, rows, cols] (int8).
  3. Dequantize (-128 -> NaN) and reproject onto the window with AREA AVERAGING by
     default (10 m -> ~30 m ALS grid: each output cell is the mean of the ~9 source
     pixels inside it, matching how ALS metrics summarise the cell; bilinear would only
     sample the few pixels nearest the cell centre). With two zones, the second only fills pixels still NaN.
  4. Mask outside the AOI raster's valid pixels (or the shapefile polygon) and write.

Usage — ALS cal/val AOI by name (year from AOI_YEAR), on the ALS grid:
    python olmoearth_wall_to_wall.py \\
        --aoi CA_LosAngeles_1_B23_4326 --aoi-dir /path/ALS_CalVal \\
        --output-dir /path/olmo_als
    python olmoearth_wall_to_wall.py --list-aois

Usage — any ALS tif, explicit year:
    python olmoearth_wall_to_wall.py \\
        --aoi-tif /path/2020_Lidar_Santa_Clara_County_CA_4326.tif --year 2020 \\
        --output-dir /path/olmo_als

Notes:
  - These are OlmoEarth-v1_3-BASE embeddings at 10 m. They are NOT the same features as
    the OLMo NANO (patch 3, 30 m) embeddings from unified_embedding_pipeline.py, so an RF
    trained on those cannot be applied to this output — train on v1.3 features at the
    GEDI tiles too (same store, same reader).
  - geoemb:matryoshka_dims = [128, 64]: --n-channels 64 keeps the first 64 channels as a
    valid smaller embedding (and halves the S3 read).
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
import zarr
from affine import Affine
from obstore.store import S3Store
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window, transform as window_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alphaearth_wall_to_wall as awtw  # noqa: E402  (AOI_YEAR + grid helpers)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
STORE_ENDPOINT = "https://data.source.coop"
NODATA_Q = -128
AOI_YEAR = awtw.AOI_YEAR


# ---------------------------------------------------------------------------
# Store access (from the Source Cooperative tutorial)
# ---------------------------------------------------------------------------

def open_olmo_store():
    """Anonymous, read-only zarr root of the v1.3 embeddings."""
    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(STORE_URL, endpoint=STORE_ENDPOINT, skip_signature=True)
    store = zarr.storage.ObjectStore(s3, read_only=True)
    return zarr.open_group(store, mode="r")


def dequantize(q):
    """int8 -> float32 via the store's signed-power formula; -128 (nodata) -> NaN."""
    q = np.asarray(q)
    v = np.where(q == NODATA_Q, np.nan, q).astype("float32")
    return (np.abs(v) / 127.5) ** 2.0 * np.sign(v)


class Zone:
    """One utmNN group: its CRS, affine transform, array and year -> time index."""

    def __init__(self, root, number):
        self.name = f"utm{number:02d}"
        g = root[self.name]
        self.crs = CRS.from_user_input(g.attrs["proj:code"])
        a, b, c, d, e, f = g.attrs["spatial:transform"]
        self.transform = Affine(a, b, c, d, e, f)
        self.emb = g["embeddings"]                  # (time, C, y, x) int8
        self.years = [int(t) for t in g["time"][:]]

    def t_index(self, year):
        if year not in self.years:
            sys.exit(f"ERROR: year {year} not in {self.name} years {self.years}")
        return self.years.index(year)

    def read(self, t, n_channels, minx, miny, maxx, maxy, pad=2):
        """Dequantized (C, H, W) block covering bounds given in THIS zone's CRS, and its
        transform; (None, None) if it falls outside the zone array."""
        inv = ~self.transform
        cols, rows = zip(*(inv * (x, y) for x, y in
                           ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))))
        _, _, ny, nx = self.emb.shape
        r0 = max(0, int(np.floor(min(rows))) - pad)
        r1 = min(ny, int(np.ceil(max(rows))) + pad)
        c0 = max(0, int(np.floor(min(cols))) - pad)
        c1 = min(nx, int(np.ceil(max(cols))) + pad)
        if r1 <= r0 or c1 <= c0:
            return None, None
        q = self.emb[t, :n_channels, r0:r1, c0:c1]
        return dequantize(q), self.transform * Affine.translation(c0, r0)


def utm_zone_numbers(lon_min, lon_max):
    """UTM zone numbers spanned by a longitude range."""
    z0 = int((lon_min + 180) // 6) + 1
    z1 = int((lon_max + 180) // 6) + 1
    return [((z - 1) % 60) + 1 for z in range(z0, z1 + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    aoi = ap.add_mutually_exclusive_group()
    aoi.add_argument("--shapefile", help="boundary shapefile (.shp)")
    aoi.add_argument("--aoi-tif", help="AOI raster (e.g. ALS tif); its grid is the default "
                                       "output grid and its valid pixels the mask")
    aoi.add_argument("--aoi", help="AOI name from AOI_YEAR; reads <--aoi-dir>/<name>.tif "
                                   "and defaults --year to AOI_YEAR[name]")
    aoi.add_argument("--list-aois", action="store_true",
                     help="print 'name<TAB>year' for every AOI_YEAR entry and exit")
    ap.add_argument("--aoi-dir", help="folder holding the AOI tifs (with --aoi)")
    ap.add_argument("--year", type=int, default=None,
                    help="embedding year, 2017-2025 (default with --aoi/--aoi-tif: "
                         "AOI_YEAR lookup by tif stem)")
    ap.add_argument("--output", default=None, help="output GeoTIFF path (.tif)")
    ap.add_argument("--output-dir", default=None,
                    help="instead of --output: write <output-dir>/<aoi stem>_olmo_<year>.tif")
    ap.add_argument("--like", default=None,
                    help="reference raster to copy the grid from (crs/transform/size). "
                         "Default with --aoi/--aoi-tif: the AOI raster itself")
    ap.add_argument("--own-grid", action="store_true",
                    help="with --aoi/--aoi-tif, build an auto-UTM/--dst-crs grid at "
                         "--resolution instead of using the AOI raster's grid")
    ap.add_argument("--dst-crs", default=None,
                    help="target CRS when not using a reference grid. Default: auto-UTM")
    ap.add_argument("--resolution", type=float, default=30.0,
                    help="target pixel size (m) when not using a reference grid")
    ap.add_argument("--n-channels", type=int, default=128, choices=(64, 128),
                    help="keep the first N channels (matryoshka dims 128/64)")
    ap.add_argument("--resampling", default="average",
                    choices=("bilinear", "average", "nearest"),
                    help="10 m -> output-grid resampling. average (default) aggregates every 10 m "
                         "pixel in the cell; bilinear matches alphaearth_wall_to_wall.py")
    ap.add_argument("--block-size", type=int, default=512,
                    help="output window size in pixels. Each 30 m window reads ~(3*N)^2 "
                         "x C int8 from S3 (512 -> ~300 MB), held as float32 (x4)")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep the full grid (default: mask outside the AOI raster's valid "
                         "pixels / the polygon, and skip windows with none without reading)")
    ap.add_argument("--compress", default="lzw",
                    help="GeoTIFF compression (lzw, deflate, none). Default lzw — see --predictor")
    ap.add_argument("--predictor", type=int, default=0,
                    help="TIFF predictor; 0 = none (default). deflate+predictor=3 can crash "
                         "GDAL at close and leave an all-nodata tif (see "
                         "alphaearth_wall_to_wall.py --help).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid, zones and window count, then exit (metadata only)")
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
        out_path = Path(args.output_dir) / f"{stem}_olmo_{args.year}.tif"
    else:
        ap.error("give --output or --output-dir")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.aoi_tif:
        gdf = awtw.raster_footprint(args.aoi_tif)
        print(f"AOI raster: {args.aoi_tif}  CRS={gdf.crs.to_string()}  year={args.year}")
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
    n_ch = args.n_channels
    res_x = abs(transform.a)
    res_str = f"{res_x:.6g} deg" if dst_crs.is_geographic else f"{res_x:.1f} m"
    windows = list(awtw.iter_windows(width, height, args.block_size))
    lon0, _, lon1, _ = gdf.to_crs(4326).total_bounds
    zone_nums = utm_zone_numbers(lon0, lon1)
    print(f"Target grid : {width} x {height} px  @ {res_str}  CRS={dst_crs.to_string()}"
          f"{'  (= ' + Path(args.like).name + ')' if args.like else ''}")
    print(f"Windows     : {len(windows)} ({args.block_size}px)   uncompressed "
          f"~{n_ch * width * height * 4 / 1e9:.2f} GB ({n_ch} band f32)")
    print(f"UTM zones   : {', '.join(f'utm{z:02d}' for z in zone_nums)}")
    if args.no_mask:
        mask_desc = "OFF (full grid)"
    elif args.aoi_tif:
        mask_desc = "ON (AOI raster's valid pixels; skip windows with none)"
    else:
        mask_desc = "ON (clip to polygon)"
    print(f"Mask        : {mask_desc}")

    root = open_olmo_store()
    zones = [Zone(root, z) for z in zone_nums]
    t_idx = [z.t_index(args.year) for z in zones]
    print(f"Store       : {STORE_URL}  years={zones[0].years}")

    if args.dry_run:
        print("Dry run — re-run without --dry-run to generate.")
        return

    aoi_vrt = gdf_dst = None
    if not args.no_mask:
        if args.aoi_tif:
            aoi_vrt = WarpedVRT(rasterio.open(args.aoi_tif), crs=dst_crs, transform=transform,
                                width=width, height=height, resampling=Resampling.nearest)
        else:
            gdf_dst = gdf.to_crs(dst_crs)

    profile = dict(
        driver="GTiff", dtype="float32", count=n_ch, crs=dst_crs,
        transform=transform, width=width, height=height, nodata=np.float32("nan"),
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    )
    if args.compress and args.compress.lower() != "none":
        profile["compress"] = args.compress
        if args.predictor:
            profile["predictor"] = args.predictor
    resampling = Resampling[args.resampling]

    t0 = time.time()
    n_written = n_empty = n_no_aoi = 0
    probe = None    # (window, row, col) of one pixel known to be finite when written
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(n_ch):
            dst.set_band_description(b + 1, f"olmo_ch_{b + 1:03d}")

        for win in (tqdm(windows, desc="windows") if _HAS_TQDM else windows):
            wt = window_transform(win, transform)
            aoi_valid = None
            if aoi_vrt is not None:
                aoi_valid = (aoi_vrt.read_masks(1, window=win) > 0) & \
                    np.isfinite(aoi_vrt.read(1, window=win))
                if not aoi_valid.any():     # no ALS data here: don't touch S3
                    n_no_aoi += 1
                    continue

            block = np.full((n_ch, win.height, win.width), np.nan, dtype=np.float32)
            bounds = array_bounds(win.height, win.width, wt)   # (minx, miny, maxx, maxy)
            for zone, t in zip(zones, t_idx):
                zb = awtw.aep.bounds_to_crs(*bounds, dst_crs, zone.crs)
                data, src_transform = zone.read(t, n_ch, *zb)
                if data is None or not np.isfinite(data).any():
                    continue
                part = np.full_like(block, np.nan)
                rasterio.warp.reproject(
                    source=data, destination=part,
                    src_transform=src_transform, src_crs=zone.crs,
                    dst_transform=wt, dst_crs=dst_crs,
                    src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling,
                )
                fill = np.isnan(block[0]) & np.isfinite(part).all(axis=0)
                block[:, fill] = part[:, fill]
                del data, part

            if gdf_dst is not None:
                outside = geometry_mask(gdf_dst.geometry, out_shape=(win.height, win.width),
                                        transform=wt, invert=False)
                block[:, outside] = np.nan
            if aoi_valid is not None:
                block[:, ~aoi_valid] = np.nan

            if not np.isfinite(block[0]).any():
                n_empty += 1
                continue
            dst.write(block, window=win)
            n_written += 1
            if probe is None:
                r, c = np.argwhere(np.isfinite(block[0]))[0]
                probe = (win, int(r), int(c))

    # Same finalisation check as alphaearth_wall_to_wall.py.
    if probe is not None:
        win, r, c = probe
        with rasterio.open(out_path) as chk:
            got = chk.read(1, window=Window(win.col_off + c, win.row_off + r, 1, 1))
        if not np.isfinite(got).any():
            sys.exit(f"ERROR: {out_path} did not finalize — a pixel that was finite when "
                     f"written now reads as nodata. Re-run with --compress lzw --predictor 0.")
    else:
        print("Warning: no valid pixels were written (no OlmoEarth data inside the AOI "
              "mask for this year).", file=sys.stderr)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.0f}s. windows written={n_written} empty={n_empty} "
          f"skipped (no AOI data)={n_no_aoi}")
    print(f"Output: {out_path}  ({size_mb:.1f} MB on disk, {n_ch} bands)")

    summary = {
        "shapefile": args.shapefile, "aoi_tif": args.aoi_tif, "like": args.like,
        "year": args.year, "output": str(out_path), "source": STORE_URL,
        "model": "OlmoEarth-v1_3-Base", "n_channels": n_ch, "resampling": args.resampling,
        "utm_zones": [z.name for z in zones],
        "dst_crs": dst_crs.to_string(), "resolution": res_x,
        "width": width, "height": height, "masked": not args.no_mask,
        "windows_written": n_written, "windows_empty": n_empty,
        "windows_no_aoi_data": n_no_aoi,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sidecar: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
