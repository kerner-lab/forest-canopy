#!/usr/bin/env python3
"""Read tiles from shapefile and output JSON."""
import geopandas as gpd
import sys
import json

if len(sys.argv) != 4:
    print("Usage: read_shapefile_tiles.py <shapefile_path> <start_idx> <end_idx>", file=sys.stderr)
    sys.exit(1)

shapefile_path = sys.argv[1]
start_idx = int(sys.argv[2])
end_idx = int(sys.argv[3])

try:
    gdf = gpd.read_file(shapefile_path)
except Exception as e:
    print(f"ERROR reading shapefile: {e}", file=sys.stderr)
    sys.exit(1)

tiles_info = []
for idx in range(start_idx, min(end_idx, len(gdf))):
    tile = gdf.iloc[idx]
    bounds = tile.geometry.bounds

    # Get HEALPix ID
    healpix_id = None
    for col in ['HEALPIX', 'healpix', 'ID', 'id', 'PIXEL', 'pixel', 'hp_id', 'NEST', 'nest']:
        if col in tile.index:
            healpix_id = str(tile[col])
            break

    if healpix_id is None:
        healpix_id = str(idx)

    tiles_info.append({
        'idx': idx,
        'healpix_id': healpix_id,
        'minx': bounds[0],
        'miny': bounds[1],
        'maxx': bounds[2],
        'maxy': bounds[3]
    })

print(json.dumps(tiles_info))