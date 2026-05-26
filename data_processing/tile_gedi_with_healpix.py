#!/usr/bin/env python3
"""
Split a large GEDI GeoTIFF into tiles using HEALPix shapefile.
This creates individual GEDI tiles that match the OLMo embedding tile structure.
Output naming: healpix_{tile_id}_gedi.tif
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from pathlib import Path
from typing import List
import warnings

warnings.filterwarnings('ignore')


def tile_gedi_with_healpix(
        gedi_tif_path: str,
        healpix_shapefile: str,
        output_dir: str,
        tile_ids: List[int] = None,
        tile_id_column: str = 'healpix_id',
        nodata_value: float = -9999.0
):
    """
    Split GEDI GeoTIFF into tiles using HEALPix shapefile.

    Args:
        gedi_tif_path: Path to the large GEDI canopy height GeoTIFF
        healpix_shapefile: Path to HEALPix tile boundaries shapefile
        output_dir: Directory to save individual tile GeoTIFFs
        tile_ids: Optional list of specific tile IDs to process (e.g., [1,2,3,4,5])
                 If None, processes all tiles in shapefile
        tile_id_column: Column name in shapefile containing tile IDs
        nodata_value: NoData value for output rasters
    """

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("TILING GEDI DATA WITH HEALPIX")
    print("=" * 60)

    # Load shapefile
    print(f"\nLoading HEALPix shapefile: {healpix_shapefile}")
    tiles_gdf = gpd.read_file(healpix_shapefile)
    print(f"  Total tiles in shapefile: {len(tiles_gdf)}")
    print(f"  CRS: {tiles_gdf.crs}")
    print(f"  Columns: {list(tiles_gdf.columns)}")

    # Verify tile ID column exists
    if tile_id_column not in tiles_gdf.columns:
        print(f"\n❌ Error: Column '{tile_id_column}' not found in shapefile")
        print(f"   Available columns: {list(tiles_gdf.columns)}")
        print(f"   Please specify the correct tile_id_column parameter")
        return

    # Filter to specific tiles if requested
    if tile_ids is not None:
        tiles_gdf = tiles_gdf[tiles_gdf[tile_id_column].isin(tile_ids)]
        print(f"  Filtered to {len(tiles_gdf)} tiles: {tile_ids}")

    # Load GEDI raster
    print(f"\nLoading GEDI raster: {gedi_tif_path}")
    with rasterio.open(gedi_tif_path) as src:
        gedi_crs = src.crs
        gedi_bounds = src.bounds
        gedi_shape = src.shape
        gedi_dtype = src.dtypes[0]

        print(f"  Shape: {gedi_shape}")
        print(f"  CRS: {gedi_crs}")
        print(f"  Bounds: {gedi_bounds}")
        print(f"  Data type: {gedi_dtype}")

        # Check if CRS match
        if tiles_gdf.crs != gedi_crs:
            print(f"\n⚠️  CRS mismatch - reprojecting shapefile to match raster")
            tiles_gdf = tiles_gdf.to_crs(gedi_crs)

    # Process each tile
    print(f"\n{'=' * 60}")
    print("PROCESSING TILES")
    print('=' * 60)

    processed_count = 0
    skipped_count = 0

    for idx, tile_row in tiles_gdf.iterrows():
        tile_id = tile_row[tile_id_column]
        tile_geom = tile_row.geometry

        print(f"\nTile {tile_id}:")

        # Check if tile intersects with GEDI bounds
        tile_bounds = tile_geom.bounds
        if not _bounds_intersect(gedi_bounds, tile_bounds):
            print(f"  ⚠️  No intersection with GEDI data - skipping")
            skipped_count += 1
            continue

        try:
            with rasterio.open(gedi_tif_path) as src:
                # Mask the raster to the tile geometry
                out_image, out_transform = mask(
                    src,
                    [tile_geom],
                    crop=True,
                    nodata=nodata_value,
                    all_touched=True  # Include pixels that touch the boundary
                )

                out_meta = src.meta.copy()

                # Update metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": nodata_value,
                    "compress": "lzw"  # Add compression to save space
                })

                # Check if tile has any valid data
                valid_pixels = np.sum(out_image[0] != nodata_value)
                total_pixels = out_image[0].size
                valid_pct = (valid_pixels / total_pixels) * 100

                print(f"  Shape: {out_image.shape[1:]} (H x W)")
                print(f"  Valid pixels: {valid_pixels:,} ({valid_pct:.1f}%)")

                if valid_pixels == 0:
                    print(f"  ⚠️  No valid data - skipping")
                    skipped_count += 1
                    continue

                # Save the tile
                output_file = output_path / f"healpix_{tile_id}_gedi.tif"
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"  ✓ Saved to {output_file.name}")
                processed_count += 1

        except Exception as e:
            print(f"  ❌ Error processing tile: {e}")
            skipped_count += 1
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(f"Total tiles processed: {processed_count}")
    print(f"Tiles skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")

    if processed_count > 0:
        print(f"\n✓ Successfully created {processed_count} GEDI tile(s)")
    else:
        print(f"\n⚠️  No tiles were created - check inputs")


def _bounds_intersect(bounds1, bounds2):
    """Check if two bounding boxes intersect."""
    left1, bottom1, right1, top1 = bounds1
    left2, bottom2, right2, top2 = bounds2

    return not (right1 < left2 or right2 < left1 or
                top1 < bottom2 or top2 < bottom1)


def inspect_healpix_shapefile(shapefile_path: str):
    """
    Inspect HEALPix shapefile to understand structure.
    Use this to determine the correct tile_id_column name.
    """
    print("=" * 60)
    print("HEALPIX SHAPEFILE INSPECTION")
    print("=" * 60)

    gdf = gpd.read_file(shapefile_path)

    print(f"\nBasic Info:")
    print(f"  Total features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Geometry type: {gdf.geometry.type.iloc[0]}")

    print(f"\nColumns:")
    for col in gdf.columns:
        if col != 'geometry':
            print(f"  {col}: {gdf[col].dtype}")
            # Show unique values if not too many
            unique_vals = gdf[col].unique()
            if len(unique_vals) <= 20:
                print(f"    Values: {sorted(unique_vals)}")
            else:
                print(f"    Values: {len(unique_vals)} unique values")
                print(f"    Range: {gdf[col].min()} to {gdf[col].max()}")

    print(f"\nFirst few rows:")
    print(gdf.head())

    print(f"\nBounds:")
    print(f"  {gdf.total_bounds}")

    return gdf


def verify_tile_output(output_dir: str, tile_ids: List[int] = None):
    """
    Verify that tiles were created correctly.

    Args:
        output_dir: Directory containing tiled GeoTIFFs
        tile_ids: Optional list of tile IDs to check
    """
    print("=" * 60)
    print("VERIFYING TILE OUTPUT")
    print("=" * 60)

    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return

    # Find all GEDI tiles
    gedi_tiles = sorted(output_path.glob("healpix_*_gedi.tif"))

    print(f"\nFound {len(gedi_tiles)} GEDI tiles")

    if tile_ids:
        print(f"\nChecking specific tiles: {tile_ids}")
        for tile_id in tile_ids:
            tile_file = output_path / f"healpix_{tile_id}_gedi.tif"
            if tile_file.exists():
                with rasterio.open(tile_file) as src:
                    data = src.read(1)
                    valid = np.sum(data != src.nodata) if src.nodata else np.sum(np.isfinite(data))
                    print(f"  ✓ Tile {tile_id}: {src.shape}, {valid:,} valid pixels")
            else:
                print(f"  ❌ Tile {tile_id}: NOT FOUND")
    else:
        # Show all tiles
        for tile_file in gedi_tiles:
            # Extract tile ID from filename
            tile_name = tile_file.stem  # e.g., "healpix_9_gedi"
            tile_id = tile_name.split('_')[1]

            with rasterio.open(tile_file) as src:
                data = src.read(1)
                valid = np.sum(data != src.nodata) if src.nodata else np.sum(np.isfinite(data))
                print(f"  Tile {tile_id}: {src.shape}, {valid:,} valid pixels")


if __name__ == "__main__":
    data_dir = '/Users/kdoerkse/Documents/ASU/JPL/data'

    GEDI_TIF = "{}/gedi_rh98_30.0m_2020.tif".format(data_dir)
    HEALPIX_SHAPEFILE = "{}/shapefiles/healpix_shapefile.shp".format(data_dir)
    OUTPUT_DIR = "{}/gedi_tiles_2020_healpix".format(data_dir)

    TILE_ID_COLUMN = 'healpix_id'
    gdf = gpd.read_file(HEALPIX_SHAPEFILE)
    TILE_IDS = gdf.healpix_id.values


    # Tile the GEDI data
    print("\nTiling GEDI data...")
    tile_gedi_with_healpix(
        gedi_tif_path=GEDI_TIF,
        healpix_shapefile=HEALPIX_SHAPEFILE,
        output_dir=OUTPUT_DIR,
        tile_ids=TILE_IDS,
        tile_id_column=TILE_ID_COLUMN,
        nodata_value=-9999.0
    )

    # Verify output
    print("\nVerifying output...")
    verify_tile_output(OUTPUT_DIR, tile_ids=TILE_IDS)

    print("\n" + "=" * 60)
    print("TILING COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run train_gedi_rf.py with:")
    print(f"  GEDI_TILES_DIR = '{OUTPUT_DIR}'")