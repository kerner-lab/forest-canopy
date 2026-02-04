"""
Script to run to query embeddings over gedi data
Steps:
1. load gedi data
2. split into tiles, keeping geo information
3. queru gee embeddings collection
4. subset over tile
5. save
"""

import ee
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import os
from pathlib import Path
import json
import requests

# Authenticate and initialize Earth Engine
ee.Initialize(project='ee-kelseydoerksen')

# Configuration
input_tiff = '/Users/kdoerkse/Documents/JPL/data/gedi_rh98_30.0m_2019.tif'
output_dir = '/Users/kdoerkse/Documents/JPL/data/tiles'
tile_size = 224
embedding_year = 2019

# Create output directories
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(f"{output_dir}/gedi_tiles").mkdir(exist_ok=True)
Path(f"{output_dir}/embeddings").mkdir(exist_ok=True)


def split_tiff_into_tiles(input_path, tile_size=224):
    """
    Split large TIFF into tiles and return tile information
    """
    tiles_info = []

    with rasterio.open(input_path) as src:
        print(f"Input image shape: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")
        print(f"Bands: {src.count}")

        # Store profile before closing
        src_profile = src.profile.copy()
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height

        # Calculate number of tiles
        n_tiles_x = (src_width + tile_size - 1) // tile_size
        n_tiles_y = (src_height + tile_size - 1) // tile_size

        print(f"Will create {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")

        # Iterate through tiles
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Calculate window
                col_off = j * tile_size
                row_off = i * tile_size
                width = min(tile_size, src_width - col_off)
                height = min(tile_size, src_height - row_off)

                window = Window(col_off, row_off, width, height)

                # Get transform for this tile
                tile_transform = src.window_transform(window)

                # Calculate geographic bounds
                left = tile_transform.c
                top = tile_transform.f
                right = left + width * tile_transform.a
                bottom = top + height * tile_transform.e

                # Create bounds in lat/lon (EPSG:4326) for Earth Engine
                if str(src_crs) != 'EPSG:4326':
                    bounds_4326 = transform_bounds(src_crs, 'EPSG:4326',
                                                   left, bottom, right, top)
                else:
                    bounds_4326 = (left, bottom, right, top)

                tile_info = {
                    'tile_id': f"tile_{i:04d}_{j:04d}",
                    'row': i,
                    'col': j,
                    'window': (col_off, row_off, width, height),  # Store as tuple
                    'transform': tile_transform,
                    'bounds': (left, bottom, right, top),
                    'bounds_4326': bounds_4326,  # For Earth Engine
                    'crs': src_crs,
                    'width': width,
                    'height': height
                }

                tiles_info.append(tile_info)

    return tiles_info, src_profile


def save_gedi_tile(input_path, tile_info, output_dir):
    """
    Save the GEDI TIFF tile with proper georeferencing
    """
    try:
        with rasterio.open(input_path) as src:
            # Read the window
            col_off, row_off, width, height = tile_info['window']
            window = Window(col_off, row_off, width, height)

            # Read data
            data = src.read(window=window)

            # Update profile for this tile
            tile_profile = src.profile.copy()
            tile_profile.update({
                'height': height,
                'width': width,
                'transform': tile_info['transform']
            })

            # Save tile
            output_path = os.path.join(output_dir, 'gedi_tiles',
                                       f"{tile_info['tile_id']}_gedi.tif")

            with rasterio.open(output_path, 'w', **tile_profile) as dst:
                dst.write(data)

            print(f"  Saved GEDI tile: {output_path}")
            return output_path

    except Exception as e:
        print(f"  Error saving GEDI tile {tile_info['tile_id']}: {e}")
        return None


def get_embeddings_for_tile(tile_info, year):
    """
    Extract embeddings for a specific tile using Earth Engine
    Resample to match the GEDI resolution
    """
    try:
        # Create Earth Engine geometry from bounds (in EPSG:4326)
        west, south, east, north = tile_info['bounds_4326']
        region = ee.Geometry.Rectangle([west, south, east, north])

        # Load embedding collection
        ae = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

        # Filter to tile and year
        embedding = ae.filterDate(f'{year}-01-01', f'{year + 1}-01-01') \
            .filterBounds(region) \
            .median() \
            .clip(region)

        # get projection
        native_projection = ae.first().projection()

        # Set the default projection before reduceResolution
        embedding = embedding.setDefaultProjection(native_projection)

        # Aggregate from native resolution to 30m using MEAN
        # This properly averages the 10m pixels into 30m pixels
        embedding_30m = embedding.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=1024
        ).reproject(
            crs='EPSG:4326',
            scale=30
        )

        # Get URL for download
        url = embedding_30m.getDownloadURL({
            'scale': 30,
            'region': region,
            'format': 'GEO_TIFF'
        })

        return {'url': url, 'region': region}

    except Exception as e:
        print(f"  Error getting embeddings for tile {tile_info['tile_id']}: {e}")
        return None


def download_and_save_embedding(embedding_info, tile_info, output_dir):
    """
    Download embedding from Earth Engine and save as GeoTIFF
    """
    if embedding_info is None:
        return None

    try:
        # Download the file
        response = requests.get(embedding_info['url'])

        if response.status_code == 200:
            # Save directly
            output_path = os.path.join(output_dir, 'embeddings',
                                       f"{tile_info['tile_id']}_embeddings.tif")

            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"  Saved embedding tile: {output_path}")
            return output_path
        else:
            print(f"  Failed to download tile {tile_info['tile_id']}: {response.status_code}")
            return None

    except Exception as e:
        print(f"  Error downloading tile {tile_info['tile_id']}: {e}")
        return None


def process_tiff_with_embeddings(input_tiff, output_dir, tile_size=224, embedding_year=2019,
                                 max_tiles=None, start_tile=0):
    """
    Main function to process large TIFF - saves both GEDI tiles and embeddings

    Args:
        max_tiles: Optional limit on number of tiles to process (for testing)
        start_tile: Start from this tile index (for resuming)
    """
    print("Step 1: Splitting TIFF into tiles...")
    tiles_info, src_profile = split_tiff_into_tiles(input_tiff, tile_size)

    # Optionally limit tiles for testing
    if max_tiles:
        tiles_info = tiles_info[start_tile:start_tile + max_tiles]
        print(f"Processing tiles {start_tile} to {start_tile + len(tiles_info)}")

    print(f"\nStep 2: Processing {len(tiles_info)} tiles...")

    successful_gedi = 0
    successful_embeddings = 0
    failed = 0

    for idx, tile_info in enumerate(tiles_info):
        print(f"\nProcessing tile {idx + 1}/{len(tiles_info)}: {tile_info['tile_id']}")

        # Save GEDI tile
        gedi_path = save_gedi_tile(input_tiff, tile_info, output_dir)
        if gedi_path:
            successful_gedi += 1

        # Get embeddings from Earth Engine
        embedding_info = get_embeddings_for_tile(tile_info, embedding_year)

        # Download and save the embedding tile
        if embedding_info:
            result = download_and_save_embedding(embedding_info, tile_info, output_dir)
            if result:
                successful_embeddings += 1
            else:
                failed += 1
        else:
            failed += 1

        # Save tile metadata
        metadata_path = os.path.join(output_dir,
                                     f"{tile_info['tile_id']}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'tile_id': tile_info['tile_id'],
                'bounds': tile_info['bounds'],
                'bounds_4326': tile_info['bounds_4326'],
                'crs': str(tile_info['crs']),
                'width': tile_info['width'],
                'height': tile_info['height'],
                'row': tile_info['row'],
                'col': tile_info['col'],
                'gedi_tile': f"gedi_tiles/{tile_info['tile_id']}_gedi.tif" if gedi_path else None,
                'embedding_tile': f"embeddings/{tile_info['tile_id']}_embeddings.tif"
            }, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"GEDI tiles saved: {successful_gedi}")
    print(f"Embedding tiles saved: {successful_embeddings}")
    print(f"Failed embeddings: {failed}")


if __name__ == "__main__":
    process_tiff_with_embeddings(input_tiff=input_tiff,
                                 output_dir=output_dir,
                                 tile_size=tile_size,
                                 embedding_year=embedding_year)