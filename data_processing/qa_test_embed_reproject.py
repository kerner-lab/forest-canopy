#!/usr/bin/env python3
"""
Test script to verify embeddings reproject correctly to GEDI grid
"""

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
from pathlib import Path
import json

# =============================================================================
# Configuration
# =============================================================================

HEALPIX_ID = "25611"
EMBEDDINGS_GEOTIFF = f"/scratch/kdoerkse/olmo/dataset/windows/healpix_{HEALPIX_ID}/tile_27/layers/embeddings/*/geotiff.tif"
GEDI_TILE = f"/scratch/kdoerkse/olmo/gedi/healpix_{HEALPIX_ID}_gedi.tif"
OUTPUT_DIR = "/scratch/kdoerkse/olmo/test_reprojection"

# Find the actual geotiff path
import glob

geotiff_files = glob.glob(EMBEDDINGS_GEOTIFF)
if not geotiff_files:
    raise FileNotFoundError(f"No embeddings found: {EMBEDDINGS_GEOTIFF}")
EMBEDDINGS_GEOTIFF = geotiff_files[0]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("REPROJECTION TEST")
print("=" * 70)
print(f"Embeddings: {EMBEDDINGS_GEOTIFF}")
print(f"GEDI tile: {GEDI_TILE}")
print(f"Output: {OUTPUT_DIR}")
print()

# =============================================================================
# Step 1: Load GEDI tile to get target grid
# =============================================================================

print("Step 1: Reading GEDI tile (target grid)...")
with rasterio.open(GEDI_TILE) as gedi:
    gedi_profile = gedi.profile.copy()
    gedi_bounds = gedi.bounds
    gedi_transform = gedi.transform
    gedi_shape = gedi.shape
    gedi_crs = gedi.crs
    gedi_res = gedi.res

    print(f"  CRS: {gedi_crs}")
    print(f"  Shape: {gedi_shape}")
    print(f"  Bounds: {gedi_bounds}")
    print(f"  Resolution: {gedi_res}")
    print(f"  Transform: {gedi_transform}")
    print()

# =============================================================================
# Step 2: Load embeddings source
# =============================================================================

print("Step 2: Reading embeddings (source)...")
with rasterio.open(EMBEDDINGS_GEOTIFF) as src:
    src_profile = src.profile.copy()
    src_bounds = src.bounds
    src_crs = src.crs
    src_shape = src.shape
    src_count = src.count

    print(f"  CRS: {src_crs}")
    print(f"  Shape: {src_shape}")
    print(f"  Bands: {src_count}")
    print(f"  Bounds: {src_bounds}")
    print(f"  Resolution: {src.res}")
    print()

# =============================================================================
# Step 3: Reproject embeddings to GEDI grid
# =============================================================================

print("Step 3: Reprojecting embeddings to GEDI grid...")

# Create output profile matching GEDI
dst_profile = src_profile.copy()
dst_profile.update({
    'crs': gedi_crs,
    'transform': gedi_transform,
    'width': gedi_shape[1],
    'height': gedi_shape[0],
    'dtype': 'float32',
    'compress': 'lzw'
})

# Output path
output_path = Path(OUTPUT_DIR) / f"healpix_{HEALPIX_ID}_embeddings_reprojected.tif"

print(f"  Output shape: ({dst_profile['height']}, {dst_profile['width']})")
print(f"  Output bands: {dst_profile['count']}")
print(f"  Writing to: {output_path}")
print()

# Reproject band by band (to handle 128 bands efficiently)
with rasterio.open(EMBEDDINGS_GEOTIFF) as src:
    with rasterio.open(output_path, 'w', **dst_profile) as dst:

        # Process in chunks to save memory
        chunk_size = 16  # Process 16 bands at a time

        for start_band in range(0, src_count, chunk_size):
            end_band = min(start_band + chunk_size, src_count)
            print(f"  Processing bands {start_band + 1}-{end_band}/{src_count}...")

            for band_idx in range(start_band, end_band):
                # Read source band
                src_data = src.read(band_idx + 1)

                # Create destination array
                dst_data = np.zeros((dst_profile['height'], dst_profile['width']), dtype=np.float32)

                # Reproject
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=gedi_transform,
                    dst_crs=gedi_crs,
                    resampling=Resampling.bilinear
                )

                # Write to output
                dst.write(dst_data, band_idx + 1)

print(f"✓ Reprojection complete!")
print()

# =============================================================================
# Step 4: Verify reprojected embeddings
# =============================================================================

print("Step 4: Verifying reprojected embeddings...")

with rasterio.open(output_path) as reproj:
    with rasterio.open(GEDI_TILE) as gedi:

        # Check CRS
        if reproj.crs == gedi.crs:
            print(f"  ✓ CRS matches: {reproj.crs}")
        else:
            print(f"  ✗ CRS mismatch: {reproj.crs} vs {gedi.crs}")

        # Check bounds (allow small floating point differences)
        bounds_diff = [abs(a - b) for a, b in zip(reproj.bounds, gedi.bounds)]
        max_bounds_diff = max(bounds_diff)

        if max_bounds_diff < 1e-6:
            print(f"  ✓ Bounds match exactly")
        elif max_bounds_diff < 1e-3:
            print(f"  ✓ Bounds match within tolerance ({max_bounds_diff:.2e})")
        else:
            print(f"  ⚠ Bounds differ by {max_bounds_diff:.6f}")
            print(f"    Embeddings: {reproj.bounds}")
            print(f"    GEDI:       {gedi.bounds}")

        # Check shape
        if reproj.shape == gedi.shape:
            print(f"  ✓ Shape matches: {reproj.shape}")
        else:
            print(f"  ✗ Shape mismatch: {reproj.shape} vs {gedi.shape}")

        # Check resolution
        res_diff = [abs(a - b) for a, b in zip(reproj.res, gedi.res)]
        max_res_diff = max(res_diff)

        if max_res_diff < 1e-9:
            print(f"  ✓ Resolution matches: {reproj.res}")
        elif max_res_diff < 1e-6:
            print(f"  ✓ Resolution matches within tolerance")
            print(f"    Embeddings: {reproj.res}")
            print(f"    GEDI:       {gedi.res}")
        else:
            print(f"  ⚠ Resolution differs")
            print(f"    Embeddings: {reproj.res}")
            print(f"    GEDI:       {gedi.res}")

        # Check transform
        if reproj.transform == gedi.transform:
            print(f"  ✓ Transform matches exactly")
        else:
            # Check if transforms are close
            emb_params = reproj.transform.to_gdal()
            gedi_params = gedi.transform.to_gdal()
            param_diffs = [abs(a - b) for a, b in zip(emb_params, gedi_params)]
            max_param_diff = max(param_diffs)

            if max_param_diff < 1e-9:
                print(f"  ✓ Transform matches within tolerance")
            else:
                print(f"  ⚠ Transform differs (max diff: {max_param_diff:.2e})")
                print(f"    Embeddings: {emb_params}")
                print(f"    GEDI:       {gedi_params}")

print()

# =============================================================================
# Step 5: Pixel alignment check
# =============================================================================

print("Step 5: Checking pixel center alignment...")

with rasterio.open(output_path) as reproj:
    with rasterio.open(GEDI_TILE) as gedi:

        # Sample pixel centers
        test_rows = [0, gedi.height // 2, gedi.height - 1]
        test_cols = [0, gedi.width // 2, gedi.width - 1]

        max_offset = 0

        for row in test_rows:
            for col in test_cols:
                # Get pixel center coordinates from both
                emb_xy = reproj.xy(row, col)
                gedi_xy = gedi.xy(row, col)

                # Calculate offset
                offset_x = abs(emb_xy[0] - gedi_xy[0])
                offset_y = abs(emb_xy[1] - gedi_xy[1])
                max_offset = max(max_offset, offset_x, offset_y)

        if max_offset < 1e-9:
            print(f"  ✓ Pixel centers align perfectly")
        elif max_offset < gedi.res[0] * 0.01:  # Within 1% of pixel size
            print(f"  ✓ Pixel centers align well (max offset: {max_offset:.2e}°)")
        else:
            print(f"  ⚠ Pixel centers may not align (max offset: {max_offset:.6f}°)")

print()

# =============================================================================
# Step 6: Data quality check
# =============================================================================

print("Step 6: Checking data quality...")

with rasterio.open(output_path) as reproj:
    # Sample first band
    reproj_data = reproj.read(1)

    # Check for valid data
    valid_pixels = np.isfinite(reproj_data).sum()
    total_pixels = reproj_data.size
    valid_pct = (valid_pixels / total_pixels) * 100

    print(f"  Valid pixels: {valid_pixels:,} / {total_pixels:,} ({valid_pct:.2f}%)")
    print(
        f"  Value range: [{reproj_data[np.isfinite(reproj_data)].min():.6f}, {reproj_data[np.isfinite(reproj_data)].max():.6f}]")

    if valid_pct > 90:
        print(f"  ✓ Good data coverage")
    else:
        print(f"  ⚠ Low data coverage - may indicate reprojection issues")

print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Reprojected embeddings saved to:")
print(f"  {output_path}")
print(f"File size: {output_path.stat().st_size / (1024 ** 3):.2f} GB")
print()
print("Next steps:")
print("  1. Visually inspect in QGIS alongside GEDI tile")
print("  2. Check if pixel values make sense")
print("  3. If good, integrate into ML pipeline")
print("=" * 70)