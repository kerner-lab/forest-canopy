"""
Load parts of AEF embeddings via xarray
"""

import time
import warnings
import numpy as np
import xarray as xr
import zarr
import zarr.storage
from zarr.errors import ZarrUserWarning
import rasterix
from rasterix import RasterIndex
from affine import Affine
from pyproj import CRS
import geopandas as gpd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# gets rid of a bunch of warnings like "ZarrUserWarning: Object at .checkpoint.2018.json is not recognized as a component of a Zarr hierarchy."
warnings.filterwarnings("ignore", category=ZarrUserWarning)


STORE_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"

def open_aef_mosaic():
    tic = time.time()

    store = zarr.storage.FsspecStore.from_url(
        STORE_URL,
        storage_options={"anon": True},
        read_only=True,
    )
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False)

    print(f" Opened in {time.time() - tic:.2f}s")
    return ds

def assign_rasterix_index(ds):
    attrs = ds["embeddings"].attrs
    affine = Affine(*attrs["spatial:transform"])
    crs = CRS.from_user_input(attrs["proj:code"])
    height, width = attrs["spatial:shape"]

    raster_idx = RasterIndex.from_transform(
        affine, width=width, height=height,
        x_dim="x", y_dim="y", crs=crs,
    )

    emb = ds["embeddings"].drop_vars(["x", "y"], errors="ignore")
    emb = emb.assign_coords(raster_idx.create_variables())
    return emb

ds = open_aef_mosaic()
embedding_ds = assign_rasterix_index(ds)

# Load the boundaries from the gedi file
gedi_bound = gpd.read_file('/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2021/tile_index/tile_index.geojson')
minx, miny, maxx, maxy = gedi_bound.total_bounds

t0 = time.time()
subset = embedding_ds.sel(
    x=slice(minx, maxx),
    y=slice(maxy, miny),
    time=2021
)

import ipdb
ipdb.set_trace()

data = subset.compute().values










