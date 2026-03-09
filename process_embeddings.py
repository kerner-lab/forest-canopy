#!/usr/bin/env python3
"""
Script to tile Alpha Earth TIF files and match them with GEDI data.
Creates 224x224 tiles with spatial alignment between input and target data.
"""

import os
import numpy as np
from osgeo import gdal, osr
from pathlib import Path
import argparse


class GeoTiffTiler:
    def __init__(self, tile_size=224):
        self.tile_size = tile_size

    def get_geotransform_info(self, dataset):
        """Extract geotransform and projection info from dataset."""
        gt = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        return gt, proj

    def pixel_to_geo(self, geotransform, px, py):
        """Convert pixel coordinates to geographic coordinates."""
        geo_x = geotransform[0] + px * geotransform[1] + py * geotransform[2]
        geo_y = geotransform[3] + px * geotransform[4] + py * geotransform[5]
        return geo_x, geo_y

    def geo_to_pixel(self, geotransform, geo_x, geo_y):
        """Convert geographic coordinates to pixel coordinates."""
        det = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        px = (geotransform[5] * (geo_x - geotransform[0]) -
              geotransform[2] * (geo_y - geotransform[3])) / det
        py = (-geotransform[4] * (geo_x - geotransform[0]) +
              geotransform[1] * (geo_y - geotransform[3])) / det
        return int(round(px)), int(round(py))

    def get_overlap_region(self, src_ds, target_ds):
        """Calculate the overlapping region between source and target datasets."""
        src_gt, src_proj = self.get_geotransform_info(src_ds)
        target_gt, target_proj = self.get_geotransform_info(target_ds)

        # Get bounds of source image
        src_width = src_ds.RasterXSize
        src_height = src_ds.RasterYSize

        src_ul_x, src_ul_y = self.pixel_to_geo(src_gt, 0, 0)
        src_lr_x, src_lr_y = self.pixel_to_geo(src_gt, src_width, src_height)

        # Get bounds of target image
        target_width = target_ds.RasterXSize
        target_height = target_ds.RasterYSize

        target_ul_x, target_ul_y = self.pixel_to_geo(target_gt, 0, 0)
        target_lr_x, target_lr_y = self.pixel_to_geo(target_gt, target_width, target_height)

        # Calculate overlap in geographic coordinates
        overlap_ul_x = max(src_ul_x, target_ul_x)
        overlap_ul_y = min(src_ul_y, target_ul_y)
        overlap_lr_x = min(src_lr_x, target_lr_x)
        overlap_lr_y = max(src_lr_y, target_lr_y)

        # Check if there's any overlap
        if overlap_ul_x >= overlap_lr_x or overlap_ul_y <= overlap_lr_y:
            return None

        # Convert overlap to pixel coordinates for both images
        src_overlap_px_ul = self.geo_to_pixel(src_gt, overlap_ul_x, overlap_ul_y)
        src_overlap_px_lr = self.geo_to_pixel(src_gt, overlap_lr_x, overlap_lr_y)

        target_overlap_px_ul = self.geo_to_pixel(target_gt, overlap_ul_x, overlap_ul_y)
        target_overlap_px_lr = self.geo_to_pixel(target_gt, overlap_lr_x, overlap_lr_y)

        return {
            'src_offset': src_overlap_px_ul,
            'src_size': (src_overlap_px_lr[0] - src_overlap_px_ul[0],
                         src_overlap_px_lr[1] - src_overlap_px_ul[1]),
            'target_offset': target_overlap_px_ul,
            'target_size': (target_overlap_px_lr[0] - target_overlap_px_ul[0],
                            target_overlap_px_lr[1] - target_overlap_px_ul[1]),
            'geo_bounds': (overlap_ul_x, overlap_ul_y, overlap_lr_x, overlap_lr_y)
        }

    def dequantize_embeddings(self, values):
        """
        Dequantize embedding values.

        Args:
            values: Quantized embedding values

        Returns:
            Dequantized values
        """
        return ((values / 127.5) ** 2) * np.sign(values)

    def create_tiles(self, src_path, target_path, output_dir, save_embeddings_as_npy=True):
        """
        Create matched tiles from source and target datasets.

        Args:
            src_path: Path to Alpha Earth TIF file
            target_path: Path to GEDI data TIF file
            output_dir: Directory to save tiles
            save_embeddings_as_npy: If True, dequantize and save embeddings as .npy files
        """
        # Open datasets
        src_ds = gdal.Open(src_path)
        target_ds = gdal.Open(target_path)

        if src_ds is None:
            raise ValueError(f"Could not open source file: {src_path}")
        if target_ds is None:
            raise ValueError(f"Could not open target file: {target_path}")

        # Get overlap region
        overlap = self.get_overlap_region(src_ds, target_ds)

        if overlap is None:
            print(f"No overlap found between {src_path} and {target_path}")
            return 0

        # Create output directories
        src_basename = Path(src_path).stem
        src_tile_dir = Path(output_dir) / 'input_tiles' / src_basename
        target_tile_dir = Path(output_dir) / 'target_tiles' / src_basename
        src_tile_dir.mkdir(parents=True, exist_ok=True)
        target_tile_dir.mkdir(parents=True, exist_ok=True)

        # Read overlap regions
        src_overlap_data = src_ds.ReadAsArray(
            overlap['src_offset'][0],
            overlap['src_offset'][1],
            overlap['src_size'][0],
            overlap['src_size'][1]
        )

        target_overlap_data = target_ds.ReadAsArray(
            overlap['target_offset'][0],
            overlap['target_offset'][1],
            overlap['target_size'][0],
            overlap['target_size'][1]
        )

        # Ensure both have the same shape (handle potential rounding differences)
        min_width = min(src_overlap_data.shape[-1], target_overlap_data.shape[-1])
        min_height = min(src_overlap_data.shape[-2], target_overlap_data.shape[-2])

        if len(src_overlap_data.shape) == 3:
            src_overlap_data = src_overlap_data[:, :min_height, :min_width]
        else:
            src_overlap_data = src_overlap_data[:min_height, :min_width]

        if len(target_overlap_data.shape) == 3:
            target_overlap_data = target_overlap_data[:, :min_height, :min_width]
        else:
            target_overlap_data = target_overlap_data[:min_height, :min_width]

        # Calculate number of tiles
        n_tiles_x = min_width // self.tile_size
        n_tiles_y = min_height // self.tile_size

        tile_count = 0
        src_gt, _ = self.get_geotransform_info(src_ds)

        # Create tiles
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Extract tile
                y_start = i * self.tile_size
                x_start = j * self.tile_size

                if len(src_overlap_data.shape) == 3:
                    src_tile = src_overlap_data[:, y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size]
                else:
                    src_tile = src_overlap_data[y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size]

                if len(target_overlap_data.shape) == 3:
                    target_tile = target_overlap_data[:, y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size]
                else:
                    target_tile = target_overlap_data[y_start:y_start + self.tile_size,
                    x_start:x_start + self.tile_size]

                # Save tiles
                tile_basename = f"tile_{i:04d}_{j:04d}"

                # Save input (embedding) tiles
                if save_embeddings_as_npy:
                    # Dequantize embeddings
                    dequantized_tile = self.dequantize_embeddings(src_tile.astype(np.float32))

                    # Save as .npy
                    npy_path = src_tile_dir / f"{tile_basename}.npy"
                    np.save(npy_path, dequantized_tile)

                    # Save first 3 tiles as TIF for QGIS verification
                    if tile_count < 3:
                        tif_path = src_tile_dir / f"{tile_basename}.tif"
                        self.save_tile(dequantized_tile, tif_path, src_ds,
                                       overlap['src_offset'][0] + x_start,
                                       overlap['src_offset'][1] + y_start)
                else:
                    # Save as TIF
                    tif_path = src_tile_dir / f"{tile_basename}.tif"
                    self.save_tile(src_tile, tif_path, src_ds,
                                   overlap['src_offset'][0] + x_start,
                                   overlap['src_offset'][1] + y_start)

                # Save target (GEDI) tiles as TIF
                target_tile_name = f"{tile_basename}.tif"
                self.save_tile(target_tile, target_tile_dir / target_tile_name, target_ds,
                               overlap['target_offset'][0] + x_start,
                               overlap['target_offset'][1] + y_start)

                tile_count += 1

        print(f"Created {tile_count} tile pairs from {src_basename}")
        return tile_count

    def save_tile(self, tile_data, output_path, reference_ds, x_offset, y_offset):
        """Save a tile as a GeoTIFF with proper georeferencing."""
        driver = gdal.GetDriverByName('GTiff')

        # Handle multi-band vs single-band
        if len(tile_data.shape) == 3:
            n_bands = tile_data.shape[0]
            height = tile_data.shape[1]
            width = tile_data.shape[2]
        else:
            n_bands = 1
            height = tile_data.shape[0]
            width = tile_data.shape[1]
            tile_data = tile_data.reshape(1, height, width)

        # Create output dataset
        out_ds = driver.Create(str(output_path), width, height, n_bands,
                               gdal.GDT_Float32)

        # Set geotransform
        ref_gt = reference_ds.GetGeoTransform()
        new_gt = list(ref_gt)
        new_gt[0] = ref_gt[0] + x_offset * ref_gt[1]
        new_gt[3] = ref_gt[3] + y_offset * ref_gt[5]
        out_ds.SetGeoTransform(new_gt)

        # Set projection
        out_ds.SetProjection(reference_ds.GetProjection())

        # Write data
        for i in range(n_bands):
            out_band = out_ds.GetRasterBand(i + 1)
            out_band.WriteArray(tile_data[i])
            out_band.FlushCache()

        out_ds = None


def main():
    parser = argparse.ArgumentParser(
        description='Tile Alpha Earth files and match with GEDI data'
    )
    parser.add_argument('--alpha-earth-dir', required=True,
                        help='Directory containing Alpha Earth TIF files')
    parser.add_argument('--gedi-file', required=True,
                        help='Path to GEDI data TIF file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for tiles')
    parser.add_argument('--tile-size', type=int, default=224,
                        help='Tile size (default: 224)')
    parser.add_argument('--save-embeddings-as-tif', action='store_true',
                        help='Save embeddings as TIF instead of dequantized NPY (default: save as NPY)')

    args = parser.parse_args()

    # Create tiler
    tiler = GeoTiffTiler(tile_size=args.tile_size)

    # Get all Alpha Earth files
    alpha_earth_files = list(Path(args.alpha_earth_dir).glob('*.tif')) + \
                        list(Path(args.alpha_earth_dir).glob('*.tiff'))

    if not alpha_earth_files:
        print(f"No TIF files found in {args.alpha_earth_dir}")
        return

    print(f"Found {len(alpha_earth_files)} Alpha Earth files")
    print(f"Processing with tile size: {args.tile_size}x{args.tile_size}")

    total_tiles = 0
    for alpha_file in alpha_earth_files:
        print(f"\nProcessing: {alpha_file.name}")
        try:
            tile_count = tiler.create_tiles(
                str(alpha_file),
                args.gedi_file,
                args.output_dir,
                save_embeddings_as_npy=not args.save_embeddings_as_tif
            )
            total_tiles += tile_count
        except Exception as e:
            print(f"Error processing {alpha_file.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Total tiles created: {total_tiles}")
    print(f"Output directory: {args.output_dir}")
    if args.save_embeddings_as_tif:
        print(f"  - Input tiles (TIF): {args.output_dir}/input_tiles/")
    else:
        print(f"  - Input tiles (NPY, dequantized): {args.output_dir}/input_tiles/")
        print(f"  - First 3 tiles also saved as TIF for QGIS verification")
    print(f"  - Target tiles (TIF): {args.output_dir}/target_tiles/")


if __name__ == '__main__':
    main()