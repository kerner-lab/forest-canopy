#!/usr/bin/env python3
"""
Create Metadata JSON Files from Zarr Attributes

This script extracts metadata from zarr files and saves them as JSON files
for use with the CRS-aware GEDI prediction script.

Usage:
    python create_metadata_files.py --embeddings-dir /path/to/olmo/2021
"""

import zarr
import json
from pathlib import Path
import argparse


def create_metadata_file(zarr_path, output_path):
    """Extract metadata from zarr and save as JSON"""
    try:
        # Try opening as a group first, then as an array
        try:
            z = zarr.open(str(zarr_path), mode='r')
        except Exception:
            # If group fails, try as array
            z = zarr.open_array(str(zarr_path), mode='r')

        attrs = dict(z.attrs)

        # Ensure required fields are present
        required = ['transform', 'crs', 'shape']
        missing = [f for f in required if f not in attrs]

        if missing:
            print(f"  ⚠️  Missing fields: {missing}")
            return False

        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(attrs, f, indent=2)

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Create metadata JSON files from zarr attributes')
    parser.add_argument('--embeddings-dir', required=True, help='Directory containing zarr files')
    parser.add_argument('--force', action='store_true', help='Overwrite existing JSON files')

    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    if not emb_dir.exists():
        print(f"✗ Directory not found: {emb_dir}")
        return 1

    # Find all zarr files
    zarr_files = sorted(emb_dir.glob("healpix_*_embeddings_int8.zarr"))

    print(f"Found {len(zarr_files)} zarr files in {emb_dir}")
    print()

    created = 0
    skipped = 0
    failed = 0

    for zarr_path in zarr_files:
        # Extract healpix_id from filename
        healpix_id = zarr_path.stem.split("_")[1]

        # Output path
        metadata_file = emb_dir / f"healpix_{healpix_id}_metadata.json"

        # Check if already exists
        if metadata_file.exists() and not args.force:
            print(f"Tile {healpix_id}: Skipped (already exists)")
            skipped += 1
            continue

        print(f"Tile {healpix_id}: Creating metadata...")

        if create_metadata_file(zarr_path, metadata_file):
            print(f"  ✓ Created {metadata_file.name}")
            created += 1
        else:
            print(f"  ✗ Failed to create metadata")
            failed += 1

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print()

    if created > 0:
        print(f"✓ Metadata files saved to: {emb_dir}")
        print()
        print("You can now run the GEDI prediction script!")
        return 0
    elif failed > 0:
        print("✗ Some metadata files failed to create")
        print("Check that zarr files have the required attributes")
        return 1
    else:
        print("All metadata files already exist")
        print("Use --force to overwrite")
        return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())