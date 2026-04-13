import pickle
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import sys


def pkl_to_shapefile(pkl_path, output_shapefile):
    """
    Convert a pickle file with HEALPix data to a shapefile.

    Parameters:
    -----------
    pkl_path : str
        Path to the input pickle file
    output_shapefile : str
        Path to the output shapefile (without .shp extension)
    """

    # Load the pickle file
    print("Loading pickle file...")
    with open(pkl_path, 'rb') as f:
        healpix_dict = pickle.load(f)

    print(f"Loaded {len(healpix_dict)} HEALPix pixels")

    # Prepare lists to store data
    healpix_ids = []
    geometries = []
    tile_indices_list = []
    trainvaltest_list = []

    # Extract data from dictionary
    print("Extracting data...")
    for healpix_id, data in healpix_dict.items():
        healpix_ids.append(healpix_id)
        geometries.append(data['bounding_polygon'])

        # Convert tile_indices list to string (shapefiles don't support list type)
        tile_indices_str = ','.join(map(str, data['tile_indices']))
        tile_indices_list.append(tile_indices_str)

        trainvaltest_list.append(data['trainvaltest'])

    # Create GeoDataFrame
    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame({
        'healpix_id': healpix_ids,
        'tile_idx': tile_indices_list,  # Shortened name for shapefile compatibility
        'split': trainvaltest_list
    }, geometry=geometries)

    # Set CRS (assuming WGS84 lat/lon coordinates)
    gdf.crs = "EPSG:4326"

    # Write to shapefile
    print(f"Writing shapefile to {output_shapefile}.shp...")
    gdf.to_file(f"{output_shapefile}.shp")

    print("Done!")
    print(f"\nShapefile info:")
    print(f"  - Total features: {len(gdf)}")
    print(f"  - CRS: {gdf.crs}")
    print(f"  - Columns: {list(gdf.columns)}")
    print(f"\nDataset splits:")
    print(gdf['split'].value_counts())

    return gdf


if __name__ == "__main__":
    pickle_path = '/Users/kdoerkse/Documents/ASU/JPL/data/healpix_dict_assigned.pkl'
    output_shp = '/Users/kdoerkse/Documents/ASU/JPL/data/healpix_shapefile'
    gdf = pkl_to_shapefile(pickle_path, output_shp)