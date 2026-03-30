#!/usr/bin/env python3
"""
Convert rslearn GeoTIFF embeddings to compressed Zarr format.
Creates both float32 (lossless) and int8 (quantized) versions.
"""

import zarr
import rasterio
import numpy as np
import json
import argparse
from pathlib import Path
from numcodecs import Blosc


def geotiff_to_zarr_float32(geotiff_path, zarr_output_path, healpix_id):
    """
    Convert GeoTIFF to compressed Zarr (float32, lossless).
    """

    # Read GeoTIFF
    with rasterio.open(geotiff_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

        metadata = {
            'healpix_id': healpix_id,
            'transform': list(transform),
            'crs': str(crs),
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'bounds': list(src.bounds)
        }

    zarr_output_path = Path(zarr_output_path)
    zarr_output_path.mkdir(parents=True, exist_ok=True)

    zarr_file = zarr_output_path / f"healpix_{healpix_id}_embeddings_float32.zarr"

    # Save to Zarr - CORRECTED LINE
    z = zarr.open(
        str(zarr_file),
        mode='w',
        shape=data.shape,
        chunks=(1, 256, 256),
        dtype=data.dtype,
        zarr_format=2,
        compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)  # FIXED
    )
    z[:] = data

    # Calculate sizes
    original_size = data.nbytes
    zarr_size = sum(f.stat().st_size for f in zarr_file.rglob('*') if f.is_file())
    compression_ratio = original_size / zarr_size

    original_size_mb = original_size / (1024 ** 2)
    zarr_size_mb = zarr_size / (1024 ** 2)

    return str(zarr_file), compression_ratio, original_size_mb, zarr_size_mb, metadata


def geotiff_to_zarr_int8(geotiff_path, zarr_output_path, healpix_id):
    """
    Convert GeoTIFF to compressed Zarr (int8 quantized).

    Returns:
        tuple: (zarr_path, compression_ratio, original_size_mb, zarr_size_mb, quantization_params)
    """

    # Read GeoTIFF
    with rasterio.open(geotiff_path) as src:
        data = src.read()  # Shape: (bands, height, width)
        transform = src.transform
        crs = src.crs

    # Quantize per band
    num_bands = data.shape[0]
    scales = []
    offsets = []
    data_int8 = np.zeros_like(data, dtype=np.int8)

    for i in range(num_bands):
        band = data[i]

        # Find min/max for this dimension
        vmin = np.nanmin(band)
        vmax = np.nanmax(band)

        # Linear mapping: [vmin, vmax] -> [-128, 127]
        scale = (vmax - vmin) / 255.0
        offset = vmin

        # Quantize to int8
        band_quantized = np.round((band - offset) / scale - 128).astype(np.int8)
        data_int8[i] = band_quantized

        scales.append(float(scale))
        offsets.append(float(offset))

    # Quantization parameters
    quantization_params = {
        'healpix_id': healpix_id,
        'scales': scales,
        'offsets': offsets,
        'num_bands': num_bands,
        'dtype_original': str(data.dtype),
        'transform': list(transform),
        'crs': str(crs),
        'shape': list(data_int8.shape)
    }

    # Create output directory
    zarr_output_path = Path(zarr_output_path)
    zarr_output_path.mkdir(parents=True, exist_ok=True)

    # Zarr filename with int8 suffix
    zarr_file = zarr_output_path / f"healpix_{healpix_id}_embeddings_int8.zarr"

    # Save to Zarr with compression (Zarr v3 compatible)
    z = zarr.open(
        str(zarr_file),
        mode='w',
        shape=data_int8.shape,
        chunks=(1, 256, 256),
        dtype=np.int8,
        zarr_format=2,  # Use Zarr v2 format
        compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
    )
    z[:] = data_int8

    # Calculate sizes
    original_size = data.nbytes
    zarr_size = sum(f.stat().st_size for f in zarr_file.rglob('*') if f.is_file())
    compression_ratio = original_size / zarr_size

    original_size_mb = original_size / (1024 ** 2)
    zarr_size_mb = zarr_size / (1024 ** 2)

    return str(zarr_file), compression_ratio, original_size_mb, zarr_size_mb, quantization_params


def process_tile_embeddings(dataset_path, tile_name, zarr_output_path, healpix_id,
                            create_float32=True, create_int8=True, delete_geotiff=True):
    """
    Process all embedding GeoTIFFs for a tile.
    Creates both float32 and int8 Zarr versions.
    """

    embeddings_path = Path(dataset_path) / "windows" / tile_name / tile_name / "layers" / "embeddings"

    if not embeddings_path.exists():
        print(f"  WARNING: Embeddings path does not exist: {embeddings_path}")
        return None

    # Find all geotiff files
    geotiff_files = list(embeddings_path.glob("*/geotiff.tif"))

    if not geotiff_files:
        print(f"  WARNING: No GeoTIFF files found in {embeddings_path}")
        return None

    results = {
        'tile_name': tile_name,
        'healpix_id': healpix_id,
        'num_files': len(geotiff_files),
        'float32': {
            'total_original_mb': 0,
            'total_zarr_mb': 0,
            'total_saved_mb': 0,
            'avg_compression_ratio': 0,
            'zarr_files': []
        },
        'int8': {
            'total_original_mb': 0,
            'total_zarr_mb': 0,
            'total_saved_mb': 0,
            'avg_compression_ratio': 0,
            'zarr_files': [],
            'quantization_files': []
        }
    }

    for geotiff_file in geotiff_files:
        print(f"  Converting: {geotiff_file.name}")

        # Determine timestep suffix if multiple files
        if len(geotiff_files) > 1:
            timestep = geotiff_file.parent.name
            if timestep == 'embeddings':
                healpix_id_with_timestep = healpix_id
            else:
                timestep_suffix = timestep.split('.')[-1]
                healpix_id_with_timestep = f"{healpix_id}_t{timestep_suffix}"
        else:
            healpix_id_with_timestep = healpix_id

        # Convert to float32 Zarr (lossless)
        if create_float32:
            print(f"    → Creating float32 Zarr (lossless)...")
            zarr_path, ratio, orig_mb, zarr_mb, metadata = geotiff_to_zarr_float32(
                str(geotiff_file),
                zarr_output_path,
                healpix_id_with_timestep
            )

            saved_mb = orig_mb - zarr_mb

            print(f"      ✓ {ratio:.2f}× compression")
            print(f"      ✓ Original: {orig_mb:.1f} MB → Zarr: {zarr_mb:.1f} MB (saved {saved_mb:.1f} MB)")
            print(f"      ✓ File: {Path(zarr_path).name}")

            results['float32']['total_original_mb'] += orig_mb
            results['float32']['total_zarr_mb'] += zarr_mb
            results['float32']['total_saved_mb'] += saved_mb
            results['float32']['avg_compression_ratio'] += ratio
            results['float32']['zarr_files'].append(str(zarr_path))

            # Save metadata
            metadata_file = Path(zarr_output_path) / f"healpix_{healpix_id_with_timestep}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Convert to int8 Zarr (quantized)
        if create_int8:
            print(f"    → Creating int8 Zarr (quantized)...")
            zarr_path, ratio, orig_mb, zarr_mb, quant_params = geotiff_to_zarr_int8(
                str(geotiff_file),
                zarr_output_path,
                healpix_id_with_timestep
            )

            saved_mb = orig_mb - zarr_mb

            print(f"      ✓ {ratio:.2f}× compression")
            print(f"      ✓ Original: {orig_mb:.1f} MB → Zarr: {zarr_mb:.1f} MB (saved {saved_mb:.1f} MB)")
            print(f"      ✓ File: {Path(zarr_path).name}")

            results['int8']['total_original_mb'] += orig_mb
            results['int8']['total_zarr_mb'] += zarr_mb
            results['int8']['total_saved_mb'] += saved_mb
            results['int8']['avg_compression_ratio'] += ratio
            results['int8']['zarr_files'].append(str(zarr_path))

            # Save quantization parameters
            quant_file = Path(zarr_output_path) / f"healpix_{healpix_id_with_timestep}_quantization_params.json"
            with open(quant_file, 'w') as f:
                json.dump(quant_params, f, indent=2)
            results['int8']['quantization_files'].append(str(quant_file))
            print(f"      ✓ Quantization params: {quant_file.name}")

        # Delete original GeoTIFF
        if delete_geotiff:
            geotiff_file.unlink()
            print(f"    ✓ Deleted original GeoTIFF")

        print()

    # Calculate averages
    if create_float32 and results['float32']['avg_compression_ratio'] > 0:
        results['float32']['avg_compression_ratio'] /= len(geotiff_files)

    if create_int8 and results['int8']['avg_compression_ratio'] > 0:
        results['int8']['avg_compression_ratio'] /= len(geotiff_files)

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert rslearn embeddings to compressed Zarr')
    parser.add_argument('--dataset-path', required=True, help='rslearn dataset root path')
    parser.add_argument('--tile-name', required=True, help='Tile name (e.g., tile_0)')
    parser.add_argument('--zarr-output', required=True, help='Output directory for Zarr files')
    parser.add_argument('--healpix-id', required=True, help='HEALPix tile ID')
    parser.add_argument('--keep-geotiff', action='store_true', help='Keep original GeoTIFF')
    parser.add_argument('--skip-float32', action='store_true', help='Skip float32 version')
    parser.add_argument('--skip-int8', action='store_true', help='Skip int8 version')

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"Converting tile {args.tile_name} (HEALPix ID: {args.healpix_id})")
    print(f"{'=' * 70}")

    results = process_tile_embeddings(
        args.dataset_path,
        args.tile_name,
        args.zarr_output,
        args.healpix_id,
        create_float32=not args.skip_float32,
        create_int8=not args.skip_int8,
        delete_geotiff=not args.keep_geotiff
    )

    if results:
        print(f"{'=' * 70}")
        print(f"Summary for {args.tile_name}")
        print(f"{'=' * 70}")
        print(f"Files processed: {results['num_files']}")

        if results['float32']['zarr_files']:
            print(f"\nFloat32 (lossless):")
            print(f"  Original size: {results['float32']['total_original_mb']:.1f} MB")
            print(f"  Zarr size: {results['float32']['total_zarr_mb']:.1f} MB")
            print(f"  Space saved: {results['float32']['total_saved_mb']:.1f} MB")
            print(f"  Avg compression: {results['float32']['avg_compression_ratio']:.2f}×")
            print(f"  Files created:")
            for zf in results['float32']['zarr_files']:
                print(f"    - {Path(zf).name}")

        if results['int8']['zarr_files']:
            print(f"\nInt8 (quantized):")
            print(f"  Original size: {results['int8']['total_original_mb']:.1f} MB")
            print(f"  Zarr size: {results['int8']['total_zarr_mb']:.1f} MB")
            print(f"  Space saved: {results['int8']['total_saved_mb']:.1f} MB")
            print(f"  Avg compression: {results['int8']['avg_compression_ratio']:.2f}×")
            print(f"  Files created:")
            for zf in results['int8']['zarr_files']:
                print(f"    - {Path(zf).name}")
            print(f"  Quantization params:")
            for qf in results['int8']['quantization_files']:
                print(f"    - {Path(qf).name}")

        print(f"{'=' * 70}\n")

        return 0
    else:
        print(f"\nERROR: Failed to process tile {args.tile_name}")
        return 1


if __name__ == '__main__':
    exit(main())