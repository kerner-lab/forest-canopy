# !/usr/bin/env python3
"""
Quick script to view the index -> healpix_id mapping from shapefile.
Use this to verify mappings before running the renaming script.
"""


import struct
import sys


def read_healpix_mapping(shapefile_dbf: str) -> dict:
    """Read shapefile and return index -> healpix_id mapping."""
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
            if rec_data[0:1] == b' ':
                pos = 1
                record = {}
                for field in fields:
                    field_value = rec_data[pos:pos + field['length']]
                    pos += field['length']

                    if field['type'] == 'N':
                        val_str = field_value.strip()
                        if val_str:
                            try:
                                record[field['name']] = int(val_str) if field['decimal'] == 0 else float(val_str)
                            except:
                                record[field['name']] = None
                        else:
                            record[field['name']] = None
                    else:
                        record[field['name']] = field_value.strip().decode('ascii', errors='ignore')

                if 'healpix_id' in record:
                    mapping[idx] = record['healpix_id']

    return mapping


def main():
    if len(sys.argv) > 1:
        shapefile_dbf = sys.argv[1]
    else:
        shapefile_dbf = "/path/to/healpix_shapefile.dbf"  # UPDATE THIS

    print("=" * 70)
    print("HEALPIX INDEX -> ID MAPPING")
    print("=" * 70)

    mapping = read_healpix_mapping(shapefile_dbf)

    print(f"\nTotal tiles: {len(mapping)}")
    print(f"\nIndex -> HEALPix ID Mapping:")
    print("-" * 70)

    for idx in sorted(mapping.keys()):
        healpix_id = mapping[idx]
        print(f"  tile_{idx:<4} -> healpix_{healpix_id}")

    print("=" * 70)

    # Export to CSV for reference
    csv_file = "healpix_mapping.csv"
    with open(csv_file, 'w') as f:
        f.write("index,healpix_id,old_name,new_name\n")
        for idx in sorted(mapping.keys()):
            healpix_id = mapping[idx]
            f.write(f"{idx},{healpix_id},tile_{idx},healpix_{healpix_id}\n")

    print(f"\n✓ Mapping exported to {csv_file}")


if __name__ == "__main__":
    main()