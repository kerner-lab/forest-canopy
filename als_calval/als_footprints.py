#!/usr/bin/env python3
"""
Valid-data footprint polygons for ALS canopy-height tifs, written to one shapefile.

For each tif the valid mask of one band (default rh98, band 2; nodata = NaN) is
polygonized and dissolved into a single (multi)polygon per file. Optional
clean-up for an AOI request: morphological closing (--close-px) bridges thin
gaps between flight lines, holes smaller than --min-hole-km2 are filled, and the
outline is simplified (--simplify-m) then buffered outward by the same distance,
so the polygon always contains every valid pixel (checked; the script fails
otherwise). --min-part-km2 > 0 drops small slivers and so does lose data.
Areas are computed in an equal-area CRS (EPSG:6933).

Columns: file, mask_km2 (raw valid pixels), area_km2 (cleaned polygon).

Example:
  python als_calval/als_footprints.py /Volumes/PRO-G40/ASU/JPL/data/ALS_CalVal/*.tif \
      --out /Volumes/PRO-G40/ASU/JPL/data/ALS_CalVal/als_footprints.shp
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

EQUAL_AREA = "EPSG:6933"


def clean(geom, min_hole_m2, min_part_m2):
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for p in parts:
        if p.area < min_part_m2:
            continue
        holes = [h for h in p.interiors if Polygon(h).area >= min_hole_m2]
        out.append(Polygon(p.exterior, holes))
    return unary_union(out)


def polygonize(valid, transform):
    return unary_union([shape(g) for g, v in shapes(valid.astype(np.uint8), mask=valid,
                                                     transform=transform) if v == 1])


def footprint(path, band, close_px):
    """Return (cleaned-mask polygon, raw valid-pixel polygon, crs, n valid px, res)."""
    with rasterio.open(path) as src:
        a = src.read(band)
        valid = np.isfinite(a)
        if src.nodata is not None and not np.isnan(src.nodata):
            valid &= a != src.nodata
        raw = polygonize(valid, src.transform)
        geom = raw
        if close_px:
            # pad first: erosion treats outside-the-array as 0 and would eat
            # valid pixels along the raster edge
            closed = ndimage.binary_closing(np.pad(valid, close_px), iterations=close_px)
            closed = closed[close_px:-close_px, close_px:-close_px] | valid
            geom = polygonize(closed, src.transform)
        return geom, raw, src.crs, int(valid.sum()), src.res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tifs", nargs="+")
    ap.add_argument("--out", required=True, help="output .shp (or .gpkg/.geojson)")
    ap.add_argument("--band", type=int, default=2, help="band whose valid mask is used (2 = rh98)")
    ap.add_argument("--close-px", type=int, default=3,
                    help="binary-closing iterations (pixels) to bridge small gaps; 0 = off")
    ap.add_argument("--min-hole-km2", type=float, default=1.0)
    ap.add_argument("--min-part-km2", type=float, default=0.0,
                    help="drop parts smaller than this (loses data; default keep all)")
    ap.add_argument("--simplify-m", type=float, default=60.0,
                    help="simplify tolerance; the result is buffered out by the same "
                         "distance so no data is cut off. 0 = exact pixel outline")
    args = ap.parse_args()

    rows = []
    for t in args.tifs:
        geom, raw, crs, mask_px, res = footprint(t, args.band, args.close_px)
        g = gpd.GeoSeries([geom, raw], crs=crs).to_crs(EQUAL_AREA)
        # raw valid-pixel area: pixel count x pixel area at the footprint centroid
        cy = raw.centroid.y
        px_m2 = (res[0] * 111_320 * np.cos(np.radians(cy))) * (res[1] * 110_574)
        e = clean(g.iloc[0], args.min_hole_km2 * 1e6, args.min_part_km2 * 1e6)
        if args.simplify_m:
            # quad_segs=2 corners are chords reaching only cos(pi/8)=0.92 of the
            # radius, hence the 1.1x so the buffer fully covers the tolerance
            e = e.simplify(args.simplify_m, preserve_topology=True).buffer(
                1.1 * args.simplify_m, quad_segs=2)
        outside_km2 = g.iloc[1].difference(e).area / 1e6
        if args.min_part_km2 == 0 and outside_km2 > 1e-3:
            raise SystemExit(f"{Path(t).stem}: {outside_km2:.4f} km2 of valid data "
                             f"outside the polygon")
        rows.append({"file": Path(t).stem, "mask_km2": mask_px * px_m2 / 1e6,
                     "area_km2": e.area / 1e6, "geometry": e})
        print(f"{Path(t).stem:50s} valid {mask_px * px_m2 / 1e6:10,.1f} km2 "
              f"-> polygon {e.area / 1e6:10,.1f} km2 (data outside: {outside_km2:.4f} km2)")

    gdf = gpd.GeoDataFrame(rows, crs=EQUAL_AREA).to_crs("EPSG:4326")
    gdf.to_file(args.out)
    tot = unary_union(gpd.GeoSeries(gdf.geometry, crs=4326).to_crs(EQUAL_AREA)).area / 1e6
    print(f"\n{len(gdf)} footprints, union area {tot:,.1f} km2 -> {args.out}")


if __name__ == "__main__":
    main()
