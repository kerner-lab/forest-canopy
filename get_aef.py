"""
Grabbing AEF data
"""
import geopandas as gpd
import pandas as pd
import subprocess
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Setup data dir and read parquet
data_dir = '/Users/kdoerkse/Documents/ASU/JPL/data'
index = gpd.read_parquet("{}/aef_index.parquet".format(data_dir))

"""
# Full California Bounding Box
california = index[
    (index["year"] == 2019) &
    (index["wgs84_north"] > 32) &
    (index["wgs84_south"] < 42) &
    (index["wgs84_east"] > -124) &
    (index["wgs84_west"] < -114)
]

print(f"Found {len(california)} tiles")
print(california["utm_zone"].value_counts())

# Save the list of file paths
california.to_csv("{}/california_2019_tiles.csv".format(data_dir), index=False)
"""

california = pd.read_csv("{}/california_2019_tiles.csv".format(data_dir))

# Filtering out for Northern California for first run
california = index[
    (index["year"] == 2019) &
    (index["wgs84_north"] > 37.5) &   # NorCal cutoff ~ SF latitude
    (index["wgs84_south"] < 42) &      # CA northern border
    (index["wgs84_east"] > -124) &
    (index["wgs84_west"] < -114)
]

print(f"Downloading {len(california)} tiles...")

def download_tile(row):
    s3_tiff = row["path"]
    s3_vrt = s3_tiff.replace(".tiff", ".vrt")
    out_dir = os.path.join(data_dir, "norcal_2019", row["utm_zone"])
    os.makedirs(out_dir, exist_ok=True)

    for s3_path in [s3_tiff, s3_vrt]:
        fname = os.path.basename(s3_path)
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Skipping (already exists): {fname}")
            continue

        https_url = s3_path.replace(
            "s3://us-west-2.opendata.source.coop",
            "https://data.source.coop"
        )

        with requests.get(https_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                unit_divisor=1024, desc=fname
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    bar.update(len(chunk))

print(f"Downloading {len(california)} tiles...")
rows = [row for _, row in california.iterrows()]
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(download_tile, rows)

print("Done!")