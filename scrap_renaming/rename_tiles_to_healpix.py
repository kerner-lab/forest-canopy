#!/usr/bin/env python3
"""
Rename tile files from index-based naming to healpix_id-based naming.

Converts:
  tile_171 -> healpix_36436 (or whatever the actual healpix_id is)

Handles:
  - Directory names: /dataset/windows/tile_171/ -> /dataset/windows/healpix_36436/
  - Zarr arrays: healpix_171_embeddings_int8.zarr -> healpix_36436_embeddings_int8.zarr
  - JSON files: healpix_171_metadata.json -> healpix_36436_metadata.json
"""

import struct
import os
import shutil
from pathlib import Path


def read_healpix_mapping(shapefile_dbf: str) -> dict:
    """
    Read shapefile DBF and create index -> healpix_id mapping.

    Returns:
        dict: {index: healpix_id}
    """
    with open(shapefile_dbf, 'rb') as f:
        # Read header
        data = f.read(32)
        num_records = struct.unpack('<I', data[4:8])[0]
        header_len = struct.unpack('<H', data[8:10])[0]
        record_len = struct.unpack('<H', data[10:12])[0]

        # Calculate number of fields
        num_fields = (header_len - 33) // 32

        # Read field descriptors
        fields = []
        for i in range(num_fields):
            field_data = f.read(32)
            name = field_data[0:11].rstrip(b'\x00 ').decode('ascii')
            field_type = chr(field_data[11])
            length = field_data[16]
            decimal = field_data[17]
            fields.append({'name': name, 'type': field_type, 'length': length, 'decimal': decimal})

        # Read terminator
        f.read(1)

        # Read records
        mapping = {}
        for idx in range(num_records):
            rec_data = f.read(record_len)
            if rec_data[0:1] == b' ':  # Not deleted
                pos = 1
                record = {}
                for field in fields:
                    field_value = rec_data[pos:pos + field['length']]
                    pos += field['length']

                    if field['type'] == 'N':  # Numeric
                        val_str = field_value.strip()
                        if val_str:
                            try:
                                if field['decimal'] > 0:
                                    record[field['name']] = float(val_str)
                                else:
                                    record[field['name']] = int(val_str)
                            except:
                                record[field['name']] = None
                        else:
                            record[field['name']] = None
                    else:
                        record[field['name']] = field_value.strip().decode('ascii', errors='ignore')

                if 'healpix_id' in record:
                    mapping[idx] = record['healpix_id']

    return mapping


def rename_windows_directories(base_dir: str, mapping: dict, dry_run: bool = True):
    """
    Rename directories in /dataset/windows/tile_X/ to healpix_Y/.

    Args:
        base_dir: Path to /scratch/kdoerkse/olmo/dataset/windows/
        mapping: Index -> healpix_id mapping
        dry_run: If True, only print what would be renamed
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"⚠️  Directory not found: {base_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"RENAMING WINDOWS DIRECTORIES")
    print('=' * 60)

    renamed_count = 0

    for idx, healpix_id in sorted(mapping.items()):
        old_dir = base_path / f"tile_{idx}"
        new_dir = base_path / f"healpix_{healpix_id}"

        if old_dir.exists():
            if dry_run:
                print(f"[DRY RUN] Would rename:")
                print(f"  {old_dir}")
                print(f"  -> {new_dir}")
            else:
                print(f"Renaming: tile_{idx} -> healpix_{healpix_id}")
                old_dir.rename(new_dir)
            renamed_count += 1

    print(f"\n{'Dry run' if dry_run else 'Actual'} renamed: {renamed_count} directories")


def rename_zarr_and_json_files(base_dir: str, mapping: dict, dry_run: bool = True):
    """
    Rename zarr arrays and JSON files in california_embeddings_zarr/.

    Handles:
      - healpix_171_embeddings_int8.zarr -> healpix_36436_embeddings_int8.zarr
      - healpix_171_embeddings_float32.zarr -> healpix_36436_embeddings_float32.zarr
      - healpix_171_quantization_params.json -> healpix_36436_quantization_params.json
      - healpix_171_metadata.json -> healpix_36436_metadata.json

    Args:
        base_dir: Path to /scratch/kdoerkse/olmo/california_embeddings_zarr/
        mapping: Index -> healpix_id mapping
        dry_run: If True, only print what would be renamed
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"⚠️  Directory not found: {base_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"RENAMING ZARR AND JSON FILES")
    print('=' * 60)

    renamed_count = 0

    # Patterns to match
    patterns = [
        "_embeddings_int8.zarr",
        "_embeddings_float32.zarr",
        "_quantization_params.json",
        "_metadata.json"
    ]

    for idx, healpix_id in sorted(mapping.items()):
        for pattern in patterns:
            old_name = f"healpix_{idx}{pattern}"
            new_name = f"healpix_{healpix_id}{pattern}"

            old_path = base_path / old_name
            new_path = base_path / new_name

            if old_path.exists():
                if dry_run:
                    print(f"[DRY RUN] Would rename:")
                    print(f"  {old_name}")
                    print(f"  -> {new_name}")
                else:
                    print(f"Renaming: {old_name} -> {new_name}")
                    old_path.rename(new_path)
                renamed_count += 1

    print(f"\n{'Dry run' if dry_run else 'Actual'} renamed: {renamed_count} files/directories")


def verify_mapping(shapefile_dbf: str, check_indices: list = None):
    """
    Print the index -> healpix_id mapping for verification.

    Args:
        shapefile_dbf: Path to shapefile DBF
        check_indices: Optional list of specific indices to check
    """
    mapping = read_healpix_mapping(shapefile_dbf)

    print(f"\n{'=' * 60}")
    print(f"HEALPIX MAPPING VERIFICATION")
    print('=' * 60)
    print(f"Total tiles: {len(mapping)}")

    if check_indices:
        print(f"\nChecking specific indices: {check_indices}")
        for idx in check_indices:
            if idx in mapping:
                print(f"  Index {idx} -> healpix_id {mapping[idx]}")
            else:
                print(f"  Index {idx} -> NOT FOUND")
    else:
        print(f"\nFirst 20 mappings:")
        for idx in sorted(mapping.keys())[:20]:
            print(f"  Index {idx} -> healpix_id {mapping[idx]}")

        print(f"\nLast 20 mappings:")
        for idx in sorted(mapping.keys())[-20:]:
            print(f"  Index {idx} -> healpix_id {mapping[idx]}")

    return mapping


def main():
    # ================================================================
    # CONFIGURATION
    # ================================================================

    SHAPEFILE_DBF = "/scratch/kdoerkse/olmo/healpix_shapefile.dbf"  # UPDATE THIS
    WINDOWS_DIR = "/scratch/kdoerkse/olmo/dataset/windows"
    EMBEDDINGS_DIR = "/scratch/kdoerkse/olmo/california_embeddings_zarr"

    # Set to False to actually perform the renaming
    DRY_RUN = False

    # Optional: Check specific indices for verification
    CHECK_INDICES = [0, 1, 2, 171, 310]  # Add indices you want to verify

    # ================================================================
    # EXECUTION
    # ================================================================

    print("=" * 60)
    print("TILE RENAMING SCRIPT")
    print("=" * 60)

    if DRY_RUN:
        print("\n⚠️  DRY RUN MODE - No files will be modified")
        print("Set DRY_RUN = False to actually rename files\n")
    else:
        print("\n🚨 LIVE MODE - Files will be renamed!")
        print("Make sure you have backups!\n")

    # Step 1: Read and verify mapping
    print("\nStep 1: Reading shapefile mapping...")
    mapping = verify_mapping(SHAPEFILE_DBF, check_indices=CHECK_INDICES)

    # Step 2: Rename windows directories
    print("\nStep 2: Renaming windows directories...")
    rename_windows_directories(WINDOWS_DIR, mapping, dry_run=DRY_RUN)

    # Step 3: Rename zarr and JSON files
    print("\nStep 3: Renaming zarr arrays and JSON files...")
    rename_zarr_and_json_files(EMBEDDINGS_DIR, mapping, dry_run=DRY_RUN)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    if DRY_RUN:
        print("✓ Dry run complete - review the output above")
        print("\nTo actually rename files:")
        print("  1. Verify the mappings look correct")
        print("  2. Set DRY_RUN = False")
        print("  3. Run the script again")
        print("\n⚠️  IMPORTANT: Make backups before running with DRY_RUN=False!")
    else:
        print("✓ Renaming complete!")
        print("\nVerify the results:")
        print(f"  ls {WINDOWS_DIR}/healpix_*/")
        print(f"  ls {EMBEDDINGS_DIR}/healpix_*")


if __name__ == "__main__":
    main()