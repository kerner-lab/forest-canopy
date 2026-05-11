#!/usr/bin/env python3
"""
One-time preprocessing: extract (X, y) from zarr embeddings + GEDI tifs and
save per-tile compressed .npz files so ML training can skip the I/O entirely.

Usage:
    python preprocess_olmo_to_gedi_tiles.py \
        --shapefile /path/to/healpix.shp \
        --gedi-dir /path/to/gedi_tifs \
        --embeddings-dir /path/to/zarr_embeddings \
        --output-dir /path/to/cache \
        --embedding-format float32

Subsequent runs skip tiles that already have a cache file. Use --overwrite to
force re-processing. The output directory can be passed directly to
run_gedi_prediction_30m.py via --preprocessed-dir.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import zarr
from pathlib import Path
from tqdm import tqdm
import json
import logging
import argparse
import time
import os
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def setup_logging(output_dir):
    logger = logging.getLogger('GEDIPreprocess')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_file = output_dir / f'preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def tile_cache_path(output_dir, healpix_id, embedding_format):
    return output_dir / f"healpix_{healpix_id}_{embedding_format}.npz"


def load_embeddings(healpix_id, embeddings_dir, embedding_format, logger):
    emb_path = embeddings_dir / f"healpix_{healpix_id}_embeddings_{embedding_format}_reproj.zarr"
    if not emb_path.exists():
        logger.warning(f"Embedding not found: {emb_path}")
        return None
    try:
        z = zarr.open(str(emb_path), mode='r')
        embeddings = np.array(z)  # (C, H, W)
        if embeddings.ndim != 3:
            logger.error(f"Unexpected embedding shape for tile {healpix_id}: {embeddings.shape}")
            return None
        embeddings = np.transpose(embeddings, (1, 2, 0))  # -> (H, W, C)
        if embedding_format == 'int8':
            embeddings = embeddings.astype(np.int8)   # keep raw int8; training normalises
        else:
            embeddings = embeddings.astype(np.float32)
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings for tile {healpix_id}: {e}")
        return None


def load_gedi(healpix_id, gedi_dir, logger):
    for ext in ('.tif', '.tiff', ''):
        gedi_path = gedi_dir / f"healpix_{healpix_id}_gedi{ext}"
        if gedi_path.exists():
            try:
                with rasterio.open(gedi_path) as src:
                    data = src.read(1)
                    nodata = src.nodata if src.nodata is not None else -9999
                return data, nodata
            except Exception as e:
                logger.error(f"Error reading GEDI file {gedi_path}: {e}")
                return None, None
    logger.warning(f"GEDI file not found for tile {healpix_id}")
    return None, None


def process_tile(healpix_id, embeddings_dir, gedi_dir, embedding_format, logger):
    """Returns (X, y) float32 arrays or (None, None) on failure."""
    embeddings = load_embeddings(healpix_id, embeddings_dir, embedding_format, logger)
    if embeddings is None:
        return None, None

    gedi_data, nodata = load_gedi(healpix_id, gedi_dir, logger)
    if gedi_data is None:
        return None, None

    emb_h, emb_w, _ = embeddings.shape
    gedi_h, gedi_w = gedi_data.shape

    valid_mask = (
        (gedi_data != nodata) &
        (gedi_data > 0) &       # non-zero forest height only
        (gedi_data < 12000)     # exclude outliers (stored in cm)
    )
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        logger.info(f"Tile {healpix_id}: no valid non-zero GEDI pixels")
        return None, None

    if gedi_h != emb_h or gedi_w != emb_w:
        emb_rows = np.clip(np.round(rows * emb_h / gedi_h).astype(int), 0, emb_h - 1)
        emb_cols = np.clip(np.round(cols * emb_w / gedi_w).astype(int), 0, emb_w - 1)
        logger.info(
            f"Tile {healpix_id}: GEDI {gedi_data.shape} vs embedding {embeddings.shape[:2]} "
            f"— scaling indices"
        )
    else:
        emb_rows, emb_cols = rows, cols

    y = (gedi_data[rows, cols] / 100.0).astype(np.float32)  # cm -> m
    X = embeddings[emb_rows, emb_cols, :]

    logger.info(
        f"Tile {healpix_id}: {len(y)} samples, height [{y.min():.2f}, {y.max():.2f}] m"
    )
    return X, y


def check_inputs(tiles_df, embeddings_dir, gedi_dir, embedding_format, output_dir, overwrite):
    """Print a pre-flight existence report for all tiles and return counts."""
    n_ready = n_missing_emb = n_missing_gedi = n_cached = 0
    missing_emb_ids, missing_gedi_ids = [], []

    for _, row in tiles_df.iterrows():
        hid = row['healpix_id']
        cache_path = tile_cache_path(output_dir, hid, embedding_format)
        if not overwrite and cache_path.exists():
            n_cached += 1
            continue
        emb_path = embeddings_dir / f"healpix_{hid}_embeddings_{embedding_format}_reproj.zarr"
        gedi_found = any(
            (gedi_dir / f"healpix_{hid}_gedi{ext}").exists()
            for ext in ('.tif', '.tiff', '')
        )
        if not emb_path.exists():
            n_missing_emb += 1
            missing_emb_ids.append(int(hid))
        elif not gedi_found:
            n_missing_gedi += 1
            missing_gedi_ids.append(int(hid))
        else:
            n_ready += 1

    print(f"\n{'='*60}")
    print(f"Input existence check ({len(tiles_df)} tiles)")
    print(f"  Ready to process : {n_ready}")
    print(f"  Already cached   : {n_cached}")
    print(f"  Missing embedding: {n_missing_emb}")
    if missing_emb_ids:
        print(f"    {missing_emb_ids}")
    print(f"  Missing GEDI     : {n_missing_gedi}")
    if missing_gedi_ids:
        print(f"    {missing_gedi_ids}")
    print(f"{'='*60}")
    return n_ready, n_missing_emb, n_missing_gedi, n_cached


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess GEDI tiles to .npz cache for fast ML training'
    )
    parser.add_argument('--shapefile', required=True,
                        help='Healpix shapefile (.shp) with split column')
    parser.add_argument('--gedi-dir', required=True,
                        help='Directory containing GEDI tif files')
    parser.add_argument('--embeddings-dir', required=True,
                        help='Directory containing zarr embedding files')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write per-tile .npz cache files')
    parser.add_argument('--embedding-format', choices=['float32', 'int8'], default='float32')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Which splits to process (default: all three)')
    parser.add_argument('--tiles', nargs='+', metavar='HEALPIX_ID', type=int,
                        help='Process only these healpix IDs (must belong to the selected splits)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-process tiles that already have a cached .npz')
    parser.add_argument('--check-only', action='store_true',
                        help='Check input file existence and print a report, then exit without processing')
    parser.add_argument('--subset-fraction', type=float, default=None,
                        help='Randomly sample this fraction (0, 1] of tiles per split before processing. '
                             'Selected tile IDs are saved to subset_tiles.json.')
    parser.add_argument('--subset-seed', type=int, default=42,
                        help='Random seed for subset tile sampling (default: 42)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    embeddings_dir = Path(args.embeddings_dir)
    gedi_dir = Path(args.gedi_dir)

    gdf = gpd.read_file(args.shapefile)
    print(f"Loaded {len(gdf)} tiles")
    print(f"Split distribution:\n{gdf['split'].value_counts()}")

    tiles_df = gdf[gdf['split'].isin(args.splits)][['healpix_id', 'split']].copy()

    if args.subset_fraction is not None and args.subset_fraction < 1.0:
        rng = np.random.default_rng(args.subset_seed)
        sampled_parts = []
        subset_record = {}
        for split in args.splits:
            split_df = tiles_df[tiles_df['split'] == split]
            n_subset = max(1, int(round(len(split_df) * args.subset_fraction)))
            chosen_ids = rng.choice(split_df['healpix_id'].values, size=n_subset, replace=False)
            chosen_ids = np.sort(chosen_ids)
            sampled_parts.append(split_df[split_df['healpix_id'].isin(chosen_ids)])
            subset_record[split] = [int(i) for i in chosen_ids]
            print(f"  Subset '{split}': {n_subset}/{len(split_df)} tiles sampled")
        tiles_df = pd.concat(sampled_parts).copy()
        subset_path = output_dir / 'subset_tiles.json'
        with open(subset_path, 'w') as f:
            json.dump({'subset_fraction': args.subset_fraction,
                       'subset_seed': args.subset_seed,
                       'splits': subset_record}, f, indent=2)
        print(f"Subset tile IDs saved: {subset_path}")

    if args.tiles:
        requested = set(args.tiles)
        tiles_df = tiles_df[tiles_df['healpix_id'].isin(requested)].copy()
        found = set(tiles_df['healpix_id'].tolist())
        missing_from_shp = requested - found
        if missing_from_shp:
            print(f"Warning: {len(missing_from_shp)} requested tiles not found in splits {args.splits}: "
                  f"{sorted(missing_from_shp)}")
        print(f"\nProcessing {len(tiles_df)} specified tiles across splits: {args.splits}")
    else:
        print(f"\nProcessing {len(tiles_df)} tiles across splits: {args.splits}")

    if args.check_only:
        check_inputs(tiles_df, embeddings_dir, gedi_dir, args.embedding_format, output_dir, args.overwrite)
        return

    check_inputs(tiles_df, embeddings_dir, gedi_dir, args.embedding_format, output_dir, args.overwrite)
    logger.info(f"Processing {len(tiles_df)} tiles, format={args.embedding_format}")

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    missing_embeddings = []
    missing_gedi = []
    total_samples = 0
    start = time.time()

    for _, row in tqdm(tiles_df.iterrows(), total=len(tiles_df), desc="Preprocessing tiles"):
        hid = row['healpix_id']
        split = row['split']
        cache_path = tile_cache_path(output_dir, hid, args.embedding_format)

        if not args.overwrite and cache_path.exists():
            n_skipped += 1
            logger.info(f"Tile {hid}: cached, skipping")
            continue

        t0 = time.time()
        X, y = process_tile(hid, embeddings_dir, gedi_dir, args.embedding_format, logger)
        elapsed = time.time() - t0

        if X is None:
            n_failed += 1
            emb_path = embeddings_dir / f"healpix_{hid}_embeddings_{args.embedding_format}_reproj.zarr"
            if not emb_path.exists():
                skip_reason = "missing_embedding"
                missing_embeddings.append(int(hid))
            else:
                gedi_found = any(
                    (gedi_dir / f"healpix_{hid}_gedi{ext}").exists()
                    for ext in ('.tif', '.tiff', '')
                )
                if not gedi_found:
                    skip_reason = "missing_gedi"
                    missing_gedi.append(int(hid))
                else:
                    skip_reason = "no_valid_pixels"
            logger.info(f"Tile {hid}: skipped [{skip_reason}]")
            continue

        np.savez_compressed(cache_path, X=X, y=y)
        n_processed += 1
        total_samples += len(y)

        mem_str = ''
        if _HAS_PSUTIL:
            mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
            mem_str = f'  mem={mem_gb:.2f}GB'
        logger.info(
            f"Tile {hid} ({split}): {len(y)} samples saved in {elapsed:.1f}s{mem_str}"
        )

    elapsed_total = time.time() - start

    print(f"\n{'='*60}")
    print(f"Preprocessing complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Processed : {n_processed} tiles  ({total_samples:,} total samples)")
    print(f"  Skipped   : {n_skipped} tiles (already cached)")
    print(f"  Failed    : {n_failed} tiles")
    if missing_embeddings:
        print(f"  Missing embeddings: {missing_embeddings}")
    if missing_gedi:
        print(f"  Missing GEDI: {missing_gedi}")
    print(f"\nCache directory: {output_dir}")
    print(f"Pass --preprocessed-dir {output_dir} to run_gedi_prediction_30m.py to use this cache.")

    summary = {
        'n_processed': n_processed,
        'n_skipped': n_skipped,
        'n_failed': n_failed,
        'total_samples': total_samples,
        'elapsed_s': round(elapsed_total, 1),
        'embedding_format': args.embedding_format,
        'splits': args.splits,
        'missing_embeddings': missing_embeddings,
        'missing_gedi': missing_gedi,
    }
    splits_tag = "_".join(args.splits)
    summary_path = output_dir / f'preprocess_summary_{splits_tag}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {summary_path}")


if __name__ == '__main__':
    main()
