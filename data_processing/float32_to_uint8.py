import rasterio
import numpy as np
import json

filepath_32 = '/Users/kdoerkse/Documents/ASU/JPL/olmo/geotiff-23831-32float.tif'

with rasterio.open(filepath_32) as src:
    dataset = src.read()  # Shape: (bands, height, width)

    # Compute min/max per band (per dimension)
    num_bands = dataset.shape[0]
    scales = []
    offsets = []

    dataset_8 = np.zeros_like(dataset, dtype=np.int8)

    for i in range(num_bands):
        band = dataset[i]

        # Find min/max for this dimension
        vmin = np.nanmin(band)
        vmax = np.nanmax(band)

        # Linear mapping: [vmin, vmax] -> [-128, 127]
        scale = (vmax - vmin) / 255.0
        offset = vmin

        # Quantize to int8 (range -128 to 127)
        band_quantized = np.round((band - offset) / scale - 128).astype(np.int8)
        dataset_8[i] = band_quantized

        scales.append(float(scale))
        offsets.append(float(offset))

    # Save quantized data
    profile = src.profile.copy()
    profile.update(dtype=rasterio.int8)

    output_path = '/Users/kdoerkse/Documents/ASU/JPL/olmo/geotiff-23831-int8.tif'
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dataset_8)

    # Save quantization parameters (CRITICAL!)
    params_path = '/Users/kdoerkse/Documents/ASU/JPL/olmo/geotiff-23831-quantization-params.json'
    with open(params_path, 'w') as f:
        json.dump({
            'scales': scales,
            'offsets': offsets,
            'num_bands': num_bands
        }, f, indent=2)

    print(f"✓ Quantized {num_bands} bands to int8")
    print(f"✓ Saved int8 GeoTIFF: {output_path}")
    print(f"✓ Saved quantization params: {params_path}")
    print(f"✓ Compression ratio: {dataset.nbytes / dataset_8.nbytes:.1f}x")
    print(f"✓ Original size: {dataset.nbytes / (1024 ** 2):.1f} MB")
    print(f"✓ Compressed size: {dataset_8.nbytes / (1024 ** 2):.1f} MB")