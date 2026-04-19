#!/usr/bin/env python3
"""
Parallelized Zarr reprojection - processes a chunk of tiles
"""
import zarr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import json
from pathlib import Path
import time
import re
import sys

# Get chunk parameters from command line
if len(sys.argv) != 3:
    print("Usage: qa_zarr_reprojection_parallel.py <start_idx> <end_idx>")
    sys.exit(1)

START_IDX = int(sys.argv[1])
END_IDX = int(sys.argv[2])

ZARR_DIR = "/scratch/kdoerkse/olmo/california_embeddings_zarr_30m"
GEDI_DIR = "/scratch/kdoerkse/olmo/gedi"
OUTPUT_DIR = "/scratch/kdoerkse/olmo/california_embeddings_30m_reproj"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Discover all healpix IDs from GEDI tile filenames
gedi_files = sorted(Path(GEDI_DIR).glob("healpix_*_gedi.tif"))
healpix_ids = []
for f in gedi_files:
    m = re.match(r"healpix_(\d+)_gedi\.tif", f.name)
    if m:
        healpix_ids.append(m.group(1))

if not healpix_ids:
    raise RuntimeError(f"No GEDI tiles found in {GEDI_DIR}")

# Process only this chunk
chunk_ids = healpix_ids[START_IDX:END_IDX]

print("=" * 70)
print(f"ZARR REPROJECTION - CHUNK {START_IDX}-{END_IDX}")
print("=" * 70)
print(f"Total tiles available: {len(healpix_ids)}")
print(f"Processing chunk: {len(chunk_ids)} tiles (indices {START_IDX}-{END_IDX - 1})")
print(f"Output: {OUTPUT_DIR}")
print()

failed = []
completed = 0
skipped = 0

chunk_start = time.time()

for tile_idx, HEALPIX_ID in enumerate(chunk_ids):
    global_idx = START_IDX + tile_idx

    print(f"\n{'=' * 70}")
    print(
        f"Tile {global_idx + 1}/{len(healpix_ids)}: healpix_{HEALPIX_ID} (chunk progress: {tile_idx + 1}/{len(chunk_ids)})")
    print(f"{'=' * 70}")

    GEDI_TILE = f"{GEDI_DIR}/healpix_{HEALPIX_ID}_gedi.tif"
    FLOAT32_ZARR = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_embeddings_float32.zarr"
    INT8_ZARR = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_embeddings_int8.zarr"
    METADATA_FILE = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_metadata.json"
    QUANT_PARAMS = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_quantization_params.json"

    OUT_FLOAT32 = f"{OUTPUT_DIR}/healpix_{HEALPIX_ID}_embeddings_float32_reproj.zarr"
    OUT_INT8 = f"{OUTPUT_DIR}/healpix_{HEALPIX_ID}_embeddings_int8_reproj.zarr"

    # Skip if both outputs already exist
    if Path(OUT_FLOAT32).exists() and Path(OUT_INT8).exists():
        print(f"  ✓ Skipping — outputs already exist")
        skipped += 1
        continue

    # Check inputs exist
    missing = [p for p in [FLOAT32_ZARR, INT8_ZARR, METADATA_FILE, QUANT_PARAMS]
               if not Path(p).exists()]
    if missing:
        print(f"  ✗ Skipping — missing inputs: {[Path(p).name for p in missing]}")
        failed.append((HEALPIX_ID, "missing inputs"))
        continue

    tile_start = time.time()

    try:
        # Load GEDI target grid
        with rasterio.open(GEDI_TILE) as gedi:
            gedi_crs = gedi.crs
            gedi_transform = gedi.transform
            gedi_shape = gedi.shape

        # Load metadata
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

        src_crs = metadata['crs']
        src_bounds = metadata['bounds']
        src_shape = metadata['shape']
        num_bands, src_height, src_width = src_shape

        src_transform = from_bounds(
            src_bounds[0], src_bounds[1], src_bounds[2], src_bounds[3],
            src_width, src_height
        )

        # Float32 reprojection
        if not Path(OUT_FLOAT32).exists():
            print(f"  Reprojecting float32...")
            z_float32 = zarr.open(FLOAT32_ZARR, mode='r')
            float32_data = z_float32[:]

            float32_reproj = np.zeros((num_bands, gedi_shape[0], gedi_shape[1]), dtype=np.float32)

            for band_idx in range(num_bands):
                reproject(
                    source=float32_data[band_idx],
                    destination=float32_reproj[band_idx],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=gedi_transform,
                    dst_crs=gedi_crs,
                    resampling=Resampling.bilinear
                )

            z_out = zarr.open(OUT_FLOAT32, mode='w', shape=float32_reproj.shape,
                              dtype=np.float32, chunks=(1, 256, 256))
            z_out[:] = float32_reproj
            print(f"    ✓ Float32 saved")
            del float32_data, float32_reproj
        else:
            print(f"  ✓ Float32 exists")
            float32_reproj = zarr.open(OUT_FLOAT32, mode='r')[:]

        # Int8 reprojection
        if not Path(OUT_INT8).exists():
            print(f"  Reprojecting int8...")
            z_int8 = zarr.open(INT8_ZARR, mode='r')
            int8_data = z_int8[:]

            with open(QUANT_PARAMS) as f:
                quant_params = json.load(f)

            # Dequantize
            int8_dequant = np.zeros_like(int8_data, dtype=np.float32)
            for i in range(len(quant_params['scales'])):
                int8_dequant[i] = (int8_data[i].astype(np.float32) + 128) * \
                                  quant_params['scales'][i] + quant_params['offsets'][i]

            int8_reproj = np.zeros((num_bands, gedi_shape[0], gedi_shape[1]), dtype=np.float32)

            for band_idx in range(num_bands):
                reproject(
                    source=int8_dequant[band_idx],
                    destination=int8_reproj[band_idx],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=gedi_transform,
                    dst_crs=gedi_crs,
                    resampling=Resampling.bilinear
                )

            z_out = zarr.open(OUT_INT8, mode='w', shape=int8_reproj.shape,
                              dtype=np.float32, chunks=(1, 256, 256))
            z_out[:] = int8_reproj
            print(f"    ✓ Int8 saved")
            del int8_data, int8_dequant, int8_reproj
        else:
            print(f"  ✓ Int8 exists")

        tile_duration = time.time() - tile_start
        print(f"  ✓ Complete in {tile_duration:.1f}s")
        completed += 1

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        failed.append((HEALPIX_ID, str(e)))
        continue

# Summary
chunk_duration = time.time() - chunk_start
print("\n" + "=" * 70)
print(f"CHUNK {START_IDX}-{END_IDX} COMPLETE")
print("=" * 70)
print(f"Tiles in chunk: {len(chunk_ids)}")
print(f"Completed: {completed}")
print(f"Skipped (already done): {skipped}")
print(f"Failed: {len(failed)}")
print(f"Total time: {chunk_duration / 60:.1f} minutes")
if completed > 0:
    print(f"Average time per tile: {chunk_duration / completed:.1f}s")
if failed:
    print("\nFailed tiles:")
    for hid, reason in failed:
        print(f"  healpix_{hid}: {reason}")
print("=" * 70)

sys.exit(0 if len(failed) == 0 else 1)