from terratorch.registry import BACKBONE_REGISTRY
import rasterio
import numpy as np
from rasterio.plot import show
import torch
import torch.nn as nn


def test_img():
    # Open the GeoTIFF file
    file_path = '/Users/kdoerkse/Documents/JPL/data/gedi_rh98_30.0m_2019.tif'
    with rasterio.open(file_path) as src:
        image_array = src.read(1)

        # Access metadata (e.g., coordinate reference system, geotransform)
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Image bounds: {src.bounds}")
        print(f"Image width, height: {src.width}, {src.height}")

test_img()

# Load model
#model = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True)




