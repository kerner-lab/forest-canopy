#!/usr/bin/env python3
"""
Diagnose OLMo Embeddings Data Issues

Checks:
1. Which tiles have metadata
2. Which zarr files are valid
3. Which zarr files are corrupted/empty
"""

import zarr
import json
from pathlib import Path
import argparse
from collections import defaultdict


def check_zarr_file(zarr_path):
    """Check if zarr file is valid"""
    try:
        # Try as group
        try:
            z = zarr.open(str(zarr_path), mode='r')
            return 'valid_group', z.shape if hasattr(z, 'shape') else 'group'
        except:
            # Try as array
            z = zarr.open_array(str(zarr_path), mode='r')
            return 'valid_array', z.shape
    except Exception as e:
        return 'corrupt', str(e)


def check_metadata(metadata_path, zarr_path):
    """Check if metadata exists"""
    # Check JSON file
    if metadata_path.exists():
        return 'json_file', metadata_path

    # Check zarr attrs
    try:
        try:
            z = zarr.open(str(zarr_path), mode='r')
        except:
            z = zarr.open_array(str(zarr_path), mode='r')

        attrs = dict(z.attrs)
        if 'transform' in attrs and 'crs' in attrs:
            return 'zarr_attrs', None
    except:
        pass

    return 'missing', None


def main():
    parser = argparse.ArgumentParser(description='Diagnose OLMo embeddings data')
    parser.add_argument('--embeddings-dir', required=True, help='Directory containing zarr files')

    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    # Find all zarr files
    zarr_files = sorted(emb_dir.glob("healpix_*_embeddings_int8.zarr"))

    print(f"Found {len(zarr_files)} zarr files in {emb_dir}\n")

    # Track status
    status = defaultdict(list)

    for zarr_path in zarr_files:
        healpix_id = zarr_path.stem.split("_")[1]
        metadata_path = emb_dir / f"healpix_{healpix_id}_metadata.json"

        # Check zarr
        zarr_status, zarr_info = check_zarr_file(zarr_path)

        # Check metadata
        meta_status, meta_path = check_metadata(metadata_path, zarr_path)

        # Categorize
        if zarr_status == 'corrupt':
            status['corrupt'].append((healpix_id, zarr_info))
        elif meta_status == 'missing':
            status['missing_metadata'].append(healpix_id)
        else:
            status['ok'].append(healpix_id)

    # Report
    print("=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    print(f"\n✓ OK: {len(status['ok'])} tiles")
    print(f"  These tiles are ready to use")

    print(f"\n⚠️  MISSING METADATA: {len(status['missing_metadata'])} tiles")
    if status['missing_metadata']:
        print(f"  Tile IDs: {status['missing_metadata'][:10]}")
        if len(status['missing_metadata']) > 10:
            print(f"  ... and {len(status['missing_metadata']) - 10} more")
        print(f"\n  FIX: Run create_metadata_files.py")

    print(f"\n✗ CORRUPTED ZARR: {len(status['corrupt'])} tiles")
    if status['corrupt']:
        print(f"  These zarr directories exist but contain no valid data:")
        for tile_id, error in status['corrupt'][:10]:
            print(f"    Tile {tile_id}: {error[:50]}")
        if len(status['corrupt']) > 10:
            print(f"  ... and {len(status['corrupt']) - 10} more")
        print(f"\n  FIX: These tiles need to be regenerated")

    # Summary
    total = len(zarr_files)
    usable = len(status['ok']) + len(status['missing_metadata'])

    print(f"\n" + "=" * 60)
    print(f"TOTAL: {total} tiles")
    print(f"USABLE (after metadata fix): {usable} ({100 * usable / total:.1f}%)")
    print(f"UNUSABLE (corrupted): {len(status['corrupt'])} ({100 * len(status['corrupt']) / total:.1f}%)")
    print("=" * 60)

    # Save report
    report = {
        'total': total,
        'ok': status['ok'],
        'missing_metadata': status['missing_metadata'],
        'corrupted': [t[0] for t in status['corrupt']]
    }

    report_file = emb_dir / 'data_diagnosis.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    return 0 if len(status['corrupt']) == 0 else 1


if __name__ == '__main__':
    import sys

    sys.exit(main())