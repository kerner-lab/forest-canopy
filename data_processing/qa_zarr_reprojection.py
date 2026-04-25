#!/usr/bin/env python3
"""
QA Script: Test Zarr embeddings reprojection to GEDI grid

Usage:
    # Process all tiles discovered from GEDI dir
    python qa_zarr_reprojection.py

    # Process specific tiles only
    python qa_zarr_reprojection.py --tiles 12345 67890 11111

    # Override directories
    python qa_zarr_reprojection.py --zarr-dir /path/to/zarr --gedi-dir /path/to/gedi --output-dir /path/to/out
"""

import argparse
import zarr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import json
from pathlib import Path
import time
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reproject zarr embeddings to GEDI grid and run QA comparison'
    )
    parser.add_argument('--zarr-dir', default='/scratch/kdoerkse/olmo/california_embeddings_zarr_30m',
                        help='Directory containing source zarr embedding files')
    parser.add_argument('--gedi-dir', default='/scratch/kdoerkse/olmo/gedi',
                        help='Directory containing GEDI tif files')
    parser.add_argument('--output-dir', default='/scratch/kdoerkse/olmo/california_embeddings_30m_reproj',
                        help='Directory to write reprojected zarr files')
    parser.add_argument('--tiles', nargs='+', metavar='HEALPIX_ID',
                        help='Specific healpix IDs to process (default: all tiles found in --gedi-dir)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-process tiles whose output zarrs already exist')
    return parser.parse_args()


args = parse_args()

ZARR_DIR = args.zarr_dir
GEDI_DIR = args.gedi_dir
OUTPUT_DIR = args.output_dir

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

if args.tiles:
    healpix_ids = [str(t) for t in args.tiles]
    print(f"Processing {len(healpix_ids)} specified tiles: {healpix_ids}")
else:
    gedi_files = sorted(Path(GEDI_DIR).glob("healpix_*_gedi.tif"))
    healpix_ids = []
    for f in gedi_files:
        m = re.match(r"healpix_(\d+)_gedi\.tif", f.name)
        if m:
            healpix_ids.append(m.group(1))

    if not healpix_ids:
        raise RuntimeError(f"No GEDI tiles found in {GEDI_DIR}")

print("=" * 70)
print("ZARR REPROJECTION - BATCH PROCESSING")
print("=" * 70)
print(f"Found {len(healpix_ids)} healpix tiles")
print(f"Output: {OUTPUT_DIR}")
print()

failed = []

for tile_idx, HEALPIX_ID in enumerate(healpix_ids):
    print(f"\n{'=' * 70}")
    print(f"Tile {tile_idx + 1}/{len(healpix_ids)}: healpix_{HEALPIX_ID}")
    print(f"{'=' * 70}")

    GEDI_TILE = f"{GEDI_DIR}/healpix_{HEALPIX_ID}_gedi.tif"
    FLOAT32_ZARR = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_embeddings_float32.zarr"
    INT8_ZARR = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_embeddings_int8.zarr"
    METADATA_FILE = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_metadata.json"
    QUANT_PARAMS = f"{ZARR_DIR}/healpix_{HEALPIX_ID}_quantization_params.json"

    OUT_FLOAT32 = f"{OUTPUT_DIR}/healpix_{HEALPIX_ID}_embeddings_float32_reproj.zarr"
    OUT_INT8 = f"{OUTPUT_DIR}/healpix_{HEALPIX_ID}_embeddings_int8_reproj.zarr"

    # Skip if both outputs already exist (unless --overwrite)
    if not args.overwrite and Path(OUT_FLOAT32).exists() and Path(OUT_INT8).exists():
        print(f"  Skipping — outputs already exist")
        continue

    # Check inputs exist
    missing = [p for p in [FLOAT32_ZARR, INT8_ZARR, METADATA_FILE, QUANT_PARAMS] if not Path(p).exists()]
    if missing:
        print(f"  Skipping — missing inputs: {missing}")
        failed.append((HEALPIX_ID, "missing inputs"))
        continue

    try:
        # =========================================================
        # Step 1: Load GEDI target grid
        # =========================================================
        print("Step 1: Loading GEDI target grid...")
        with rasterio.open(GEDI_TILE) as gedi:
            gedi_crs = gedi.crs
            gedi_transform = gedi.transform
            gedi_shape = gedi.shape
            gedi_bounds = gedi.bounds
            gedi_res = gedi.res

        print(f"  CRS: {gedi_crs}  Shape: {gedi_shape}  Res: {gedi_res}")

        # =========================================================
        # Step 2: Load metadata
        # =========================================================
        print("Step 2: Loading embeddings metadata...")
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

        src_crs = metadata['crs']
        src_bounds = metadata['bounds']
        src_shape = metadata['shape']

        num_bands = src_shape[0]
        src_height = src_shape[1]
        src_width = src_shape[2]

        src_transform = from_bounds(
            src_bounds[0], src_bounds[1], src_bounds[2], src_bounds[3],
            src_width, src_height
        )
        print(f"  Source CRS: {src_crs}  Shape: ({num_bands} bands, {src_height}, {src_width})")

        # =========================================================
        # Step 3: Float32 reprojection
        # =========================================================
        if not Path(OUT_FLOAT32).exists():
            print("Step 3: Float32 reprojection...")
            start = time.time()

            z_float32 = zarr.open(FLOAT32_ZARR, mode='r')
            float32_data = z_float32[:]

            float32_reproj = np.zeros((float32_data.shape[0], gedi_shape[0], gedi_shape[1]), dtype=np.float32)
            for band_idx in range(float32_data.shape[0]):
                if band_idx % 16 == 0:
                    print(f"    Band {band_idx + 1}/{float32_data.shape[0]}...")
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
                              dtype=np.float32, chunks=(1, gedi_shape[0], gedi_shape[1]))
            z_out[:] = float32_reproj
            print(f"  Saved float32 reproj: {float32_reproj.shape}  ({time.time() - start:.1f}s)")
        else:
            print("Step 3: Float32 output exists, loading from disk...")
            float32_reproj = zarr.open(OUT_FLOAT32, mode='r')[:]

        # =========================================================
        # Step 4: Int8 reprojection
        # =========================================================
        if not Path(OUT_INT8).exists():
            print("Step 4: Int8 reprojection...")
            start = time.time()

            z_int8 = zarr.open(INT8_ZARR, mode='r')
            int8_data = z_int8[:]

            with open(QUANT_PARAMS) as f:
                quant_params = json.load(f)

            int8_dequant = np.zeros_like(int8_data, dtype=np.float32)
            for i in range(len(quant_params['scales'])):
                int8_dequant[i] = (int8_data[i].astype(np.float32) + 128) * quant_params['scales'][i] + quant_params['offsets'][i]

            int8_reproj = np.zeros((int8_dequant.shape[0], gedi_shape[0], gedi_shape[1]), dtype=np.float32)
            for band_idx in range(int8_dequant.shape[0]):
                if band_idx % 16 == 0:
                    print(f"    Band {band_idx + 1}/{int8_dequant.shape[0]}...")
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
                              dtype=np.float32, chunks=(1, gedi_shape[0], gedi_shape[1]))
            z_out[:] = int8_reproj
            print(f"  Saved int8 reproj: {int8_reproj.shape}  ({time.time() - start:.1f}s)")
        else:
            print("Step 4: Int8 output exists, loading from disk...")
            int8_reproj = zarr.open(OUT_INT8, mode='r')[:]

        # =========================================================
        # Step 5: QA comparison
        # =========================================================
        print("Step 5: QA comparison...")
        diff = np.abs(float32_reproj - int8_reproj)
        mae = np.mean(diff)
        relative_error = mae / (np.mean(np.abs(float32_reproj)) + 1e-8)

        cos_sims = []
        for i in range(min(100, float32_reproj.shape[0])):
            flat_f32 = float32_reproj[i].flatten()
            flat_i8 = int8_reproj[i].flatten()
            cos_sim = np.dot(flat_f32, flat_i8) / (np.linalg.norm(flat_f32) * np.linalg.norm(flat_i8) + 1e-8)
            cos_sims.append(cos_sim)
        avg_cos_sim = np.mean(cos_sims)

        status = "EXCELLENT" if relative_error < 0.02 and avg_cos_sim > 0.99 else \
                 "GOOD" if relative_error < 0.05 and avg_cos_sim > 0.95 else "WARNING"
        print(f"  MAE: {mae:.6f}  Relative error: {relative_error:.2%}  Cosine sim: {avg_cos_sim:.6f}  [{status}]")

    except Exception as e:
        print(f"  ERROR processing healpix_{HEALPIX_ID}: {e}")
        failed.append((HEALPIX_ID, str(e)))
        continue

# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "=" * 70)
print("BATCH PROCESSING COMPLETE")
print("=" * 70)
print(f"Total tiles: {len(healpix_ids)}")
print(f"Failed: {len(failed)}")
if failed:
    print("Failed tiles:")
    for hid, reason in failed:
        print(f"  healpix_{hid}: {reason}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)
