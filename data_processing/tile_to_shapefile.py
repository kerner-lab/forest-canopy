#!/usr/bin/env python3
"""
Create a spatial index of GDAL tiles with train/val/test split assignments from HEALPix pickle file.
"""

import os
import re
import pickle
from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
import pandas as pd


def parse_tile_filename(filename):
    """
    Parse tile filename to extract index.

    Handles multiple formats:
        gedi_rh98_30.0m_2020_001_002.tif -> 2020_001_002
        gedi_rh98_30.0m_2021_v6_309_175.tif -> 2021_v6_309_175
    """
    # Remove .tif extension
    name = filename.replace('.tif', '')

    # Try pattern with version: year_vX_row_col
    pattern_with_version = r'(\d{4})_(v\d+)_(\d+)_(\d+)$'
    match = re.search(pattern_with_version, name)
    if match:
        year, version, row, col = match.groups()
        return f"{year}_{version}_{row}_{col}"

    # Try pattern without version: year_row_col
    pattern_no_version = r'(\d{4})_(\d+)_(\d+)$'
    match = re.search(pattern_no_version, name)
    if match:
        year, row, col = match.groups()
        return f"{year}_{row}_{col}"

    return None


def create_tile_to_split_mapping(healpix_dict):
    """
    Create a mapping from tile_index to trainvaltest split.

    Args:
        healpix_dict: Dictionary with healpix pixel data

    Returns:
        dict: {tile_index: split}
    """
    tile_to_split = {}

    for healpix_id, data in healpix_dict.items():
        split = data['trainvaltest']
        tile_indices = data['tile_indices']

        for tile_idx in tile_indices:
            if tile_idx in tile_to_split and tile_to_split[tile_idx] != split:
                print(f"Warning: Tile {tile_idx} appears in multiple healpix pixels with different splits!")
            tile_to_split[tile_idx] = split

    return tile_to_split


def normalize_tile_index(tile_idx):
    """
    Create variations of tile index for flexible matching.

    Returns list of possible formats:
        2021_v6_309_175 -> ['2021_v6_309_175', '2021_309_175']
        2020_107_064 -> ['2020_107_064', '2020_v6_107_064']
    """
    variations = [tile_idx]

    # If has version, also try without version
    if '_v' in tile_idx:
        # 2021_v6_309_175 -> 2021_309_175
        no_version = re.sub(r'_v\d+_', '_', tile_idx)
        variations.append(no_version)
    else:
        # 2020_107_064 -> try adding common version patterns
        parts = tile_idx.split('_')
        if len(parts) == 3:
            year, row, col = parts
            variations.append(f"{year}_v6_{row}_{col}")

    return variations


def get_tile_bounds(tile_path):
    """Get bounds from a raster tile."""
    with rasterio.open(tile_path) as src:
        bounds = src.bounds
        crs = src.crs
        return bounds, crs


def main():
    # Configuration
    tile_dir = Path("/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2019")
    pickle_path = Path("/Users/kdoerkse/Documents/ASU/JPL/data/healpix_dict_assigned.pkl")

    # Load pickle file
    print(f"Loading HEALPix pickle file: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        healpix_dict = pickle.load(f)

    print(f"Loaded {len(healpix_dict)} HEALPix pixels")

    # Create tile-to-split mapping
    print("Creating tile-to-split mapping...")
    tile_to_split = create_tile_to_split_mapping(healpix_dict)
    print(f"Found {len(tile_to_split)} unique tile indices in pickle file")

    # Show sample from pickle
    sample_tiles = list(tile_to_split.keys())[:3]
    print(f"\nSample tile indices from pickle: {sample_tiles}")

    # Find all tile files
    print("\nFinding tile files...")
    tile_files = list(tile_dir.glob("gedi_rh98_30.0m_*.tif"))
    print(f"Found {len(tile_files)} tile files")

    # Process each tile
    data = []
    crs = None
    matched_count = 0
    unmatched_count = 0

    for tile_path in tile_files:
        try:
            # Parse index from filename
            tile_index = parse_tile_filename(tile_path.name)

            if tile_index is None:
                print(f"Warning: Could not parse index from {tile_path.name}")
                continue

            # Get bounds
            bounds, tile_crs = get_tile_bounds(tile_path)

            if crs is None:
                crs = tile_crs

            # Create polygon from bounds
            polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

            # Try to match tile index to split
            split = None
            matched_variation = None

            # Try all variations of the tile index
            for variation in normalize_tile_index(tile_index):
                if variation in tile_to_split:
                    split = tile_to_split[variation]
                    matched_variation = variation
                    break

            if split is None:
                split = 'unknown'
                unmatched_count += 1
            else:
                matched_count += 1

            # Add to data
            data.append({
                'tile_index': tile_index,
                'filename': tile_path.name,
                'split': split,
                'geometry': polygon,
                'matched_as': matched_variation if matched_variation else None
            })

        except Exception as e:
            print(f"Error processing {tile_path.name}: {e}")
            continue

    print(f"\nProcessed {len(data)} tiles successfully")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs=crs)

    # Diagnostic information
    print("\n" + "=" * 60)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 60)

    # Check format patterns
    tile_formats = set()
    for idx in gdf['tile_index'].head(10):
        if '_v' in idx:
            tile_formats.add("with_version (e.g., 2021_v6_309_175)")
        else:
            tile_formats.add("no_version (e.g., 2020_107_064)")

    pickle_formats = set()
    for idx in list(tile_to_split.keys())[:10]:
        if '_v' in idx:
            pickle_formats.add("with_version (e.g., 2021_v6_309_175)")
        else:
            pickle_formats.add("no_version (e.g., 2020_107_064)")

    print(f"\nTile filename formats: {tile_formats}")
    print(f"Pickle index formats: {pickle_formats}")

    # Show sample indices
    print(f"\nSample tile indices (first 5): {list(gdf['tile_index'].head())}")
    print(f"Sample pickle indices (first 5): {sample_tiles}")

    if unmatched_count > 0:
        print(f"\n⚠️  WARNING: {unmatched_count} tiles could not be matched to pickle data")
        print("Sample unmatched tiles:")
        unmatched = gdf[gdf['split'] == 'unknown']['tile_index'].head()
        for idx in unmatched:
            print(f"  - {idx}")
            print(f"    Tried variations: {normalize_tile_index(idx)}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SPLIT DISTRIBUTION")
    print("=" * 60)
    print(gdf['split'].value_counts())

    # Save in multiple formats
    output_dir = tile_dir / "tile_index"
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving outputs to {output_dir}")

    # Drop the 'matched_as' column before saving (it's just for debugging)
    gdf_output = gdf.drop(columns=['matched_as'])

    # Save as shapefile
    shapefile_path = output_dir / "tile_index.shp"
    gdf_output.to_file(shapefile_path)
    print(f"Saved shapefile: {shapefile_path}")

    # Save as GeoJSON
    geojson_path = output_dir / "tile_index.geojson"
    gdf_output.to_file(geojson_path, driver='GeoJSON')
    print(f"Saved GeoJSON: {geojson_path}")

    # Save as GeoParquet
    geoparquet_path = output_dir / "tile_index.parquet"
    gdf_output.to_parquet(geoparquet_path)
    print(f"Saved GeoParquet: {geoparquet_path}")

    # Save as CSV with WKT geometry for reference
    csv_path = output_dir / "tile_index.csv"
    gdf_csv = gdf_output.copy()
    gdf_csv['geometry_wkt'] = gdf_csv['geometry'].to_wkt()
    gdf_csv.drop(columns=['geometry']).to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\nDone!")
    print(f"CRS: {crs}")
    print(f"Total tiles: {len(gdf)}")
    print(f"Bounds: {gdf.total_bounds}")


if __name__ == "__main__":
    main()