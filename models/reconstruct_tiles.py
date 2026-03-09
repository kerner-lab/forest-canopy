"""
Reconstruct Random Forest Predictions to Geographic Map

This script:
1. Loads tiles with their geographic information preserved
2. Runs predictions through Random Forest
3. Reconstructs predictions back into georeferenced tiles
4. Mosaics tiles into a single California map
5. Saves as GeoTIFF
"""

import numpy as np
from pathlib import Path
from osgeo import gdal, osr
import pickle
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TileInfo:
    """Store tile data with geographic information."""
    filepath: Path
    target_filepath: Path
    embedding: np.ndarray  # [64, 224, 224]
    target: np.ndarray  # [224, 224]
    geotransform: tuple
    projection: str
    center_lon: float
    center_lat: float
    split_set: str  # 'train', 'val', or 'test'


class GeoReconstructionPipeline:
    """Pipeline to preserve location info and reconstruct predictions."""

    def __init__(self, input_dir, target_dir, load_all_folders=False):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.load_all_folders = load_all_folders
        self.tiles: List[TileInfo] = []

    def load_tiles_with_geo_info(self, split_lon=None):
        """
        Load all tiles while preserving geographic information.

        Args:
            split_lon: Longitude to split train/val (West) vs test (East)

        Returns:
            train_tiles, val_tiles, test_tiles
        """
        print("Loading tiles with geographic information...")

        # Get all files
        if self.load_all_folders:
            input_files = []
            for subdir in self.input_dir.iterdir():
                if subdir.is_dir():
                    input_files.extend(list(subdir.glob("*.npy")))
            input_files = sorted(input_files)
        else:
            input_files = sorted(list(self.input_dir.glob("*.npy")))

        print(f"Found {len(input_files)} files")

        all_lons = []
        skipped = 0

        for input_file in input_files:
            # Load embedding
            embedding = np.load(input_file).astype(np.float32)

            # Find target file
            tile_name = input_file.stem
            if self.load_all_folders:
                alpha_subdir = input_file.parent.name
                target_file = self.target_dir / alpha_subdir / f"{tile_name}.tif"
            else:
                target_file = self.target_dir / f"{tile_name}.tif"

            if not target_file.exists():
                skipped += 1
                continue

            # Load target and get geotransform
            target_ds = gdal.Open(str(target_file))
            target = target_ds.ReadAsArray().astype(np.float32)

            if len(target.shape) == 3:
                target = target[0]  # Take first channel

            gt = target_ds.GetGeoTransform()
            proj = target_ds.GetProjection()

            if gt is None:
                skipped += 1
                target_ds = None
                continue

            # Calculate center coordinates
            width = target_ds.RasterXSize
            height = target_ds.RasterYSize
            center_lon = gt[0] + (width / 2.0) * gt[1]
            center_lat = gt[3] + (height / 2.0) * gt[5]

            target_ds = None

            all_lons.append(center_lon)

            # Create tile info
            tile_info = TileInfo(
                filepath=input_file,
                target_filepath=target_file,
                embedding=embedding,
                target=target,
                geotransform=gt,
                projection=proj,
                center_lon=center_lon,
                center_lat=center_lat,
                split_set='unknown'  # Will be set later
            )

            self.tiles.append(tile_info)

        if skipped > 0:
            print(f"Skipped {skipped} tiles")

        # Determine split longitude
        if split_lon is None:
            split_lon = np.median(all_lons)

        print(f"\nSplit longitude: {split_lon:.4f}°")
        print(f"Longitude range: [{min(all_lons):.4f}°, {max(all_lons):.4f}°]")

        # Split tiles into train/val/test
        west_tiles = [t for t in self.tiles if t.center_lon < split_lon]
        east_tiles = [t for t in self.tiles if t.center_lon >= split_lon]

        # Mark east as test
        for tile in east_tiles:
            tile.split_set = 'test'

        # Split west into train/val (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(west_tiles))
        train_size = int(0.8 * len(west_tiles))

        train_tiles = [west_tiles[i] for i in indices[:train_size]]
        val_tiles = [west_tiles[i] for i in indices[train_size:]]

        for tile in train_tiles:
            tile.split_set = 'train'
        for tile in val_tiles:
            tile.split_set = 'val'

        print(f"\nSplit:")
        print(f"  Train: {len(train_tiles)} tiles (West)")
        print(f"  Val:   {len(val_tiles)} tiles (West)")
        print(f"  Test:  {len(east_tiles)} tiles (East)")

        return train_tiles, val_tiles, east_tiles

    def tiles_to_arrays(self, tiles: List[TileInfo]):
        """Convert tiles to flattened arrays for Random Forest."""
        X = np.array([tile.embedding.flatten() for tile in tiles])
        y = np.array([tile.target.flatten() for tile in tiles])
        return X, y

    def predict_and_reconstruct(self, model, tiles: List[TileInfo], output_dir: str):
        """
        Run predictions and save as georeferenced tiles.

        Args:
            model: Trained Random Forest model
            tiles: List of tiles to predict
            output_dir: Directory to save reconstructed tiles
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nPredicting and reconstructing {len(tiles)} tiles...")

        # Flatten embeddings for prediction
        X = np.array([tile.embedding.flatten() for tile in tiles])

        # Predict
        predictions_flat = model.predict(X)  # [n_tiles, 224*224]

        # Reshape and save each tile
        for i, tile in enumerate(tiles):
            # Reshape prediction back to 2D
            pred_2d = predictions_flat[i].reshape(224, 224)

            # Save as GeoTIFF
            tile_name = tile.filepath.stem
            output_path = output_dir / f"{tile_name}_prediction.tif"

            self.save_geotiff(
                pred_2d,
                output_path,
                tile.geotransform,
                tile.projection
            )

        print(f"✓ Saved {len(tiles)} prediction tiles to {output_dir}")

        return predictions_flat

    def save_geotiff(self, array, output_path, geotransform, projection):
        """Save array as georeferenced GeoTIFF."""
        driver = gdal.GetDriverByName('GTiff')

        if len(array.shape) == 2:
            height, width = array.shape
            n_bands = 1
        else:
            n_bands, height, width = array.shape

        out_ds = driver.Create(
            str(output_path),
            width,
            height,
            n_bands,
            gdal.GDT_Float32
        )

        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        if n_bands == 1:
            out_ds.GetRasterBand(1).WriteArray(array)
        else:
            for i in range(n_bands):
                out_ds.GetRasterBand(i + 1).WriteArray(array[i])

        out_ds.FlushCache()
        out_ds = None

    def mosaic_tiles_to_map(self, tiles_dir: str, output_path: str):
        """
        Mosaic individual prediction tiles into a single California map.

        Args:
            tiles_dir: Directory with prediction tiles
            output_path: Output path for mosaiced map
        """
        tiles_dir = Path(tiles_dir)
        tile_files = sorted(list(tiles_dir.glob("*_prediction.tif")))

        if len(tile_files) == 0:
            print("No prediction tiles found!")
            return

        print(f"\nMosaicing {len(tile_files)} tiles into single map...")

        # Build VRT (Virtual Dataset) to mosaic tiles
        vrt_path = tiles_dir / "mosaic.vrt"
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
        vrt = gdal.BuildVRT(
            str(vrt_path),
            [str(f) for f in tile_files],
            options=vrt_options
        )
        vrt = None

        # Convert VRT to GeoTIFF
        print(f"Saving mosaiced map to {output_path}")
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'TILED=YES']
        )
        gdal.Translate(
            str(output_path),
            str(vrt_path),
            options=translate_options
        )

        print(f"✓ Mosaiced map saved to {output_path}")


def run_full_pipeline(
        input_dir: str,
        target_dir: str,
        model_path: str,
        output_base_dir: str = "predictions_georeferenced",
        load_all_folders: bool = True,
        split_lon: float = None
):
    """
    Run the full pipeline: load, predict, reconstruct, mosaic.

    Args:
        input_dir: Input tiles directory
        target_dir: Target tiles directory
        model_path: Path to trained Random Forest model (.pkl)
        output_base_dir: Base directory for outputs
        load_all_folders: Load from all subdirectories
        split_lon: Split longitude (None = auto-calculate)
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = GeoReconstructionPipeline(input_dir, target_dir, load_all_folders)

    # Load tiles with geo info
    train_tiles, val_tiles, test_tiles = pipeline.load_tiles_with_geo_info(split_lon)

    # Load model
    print(f"\nLoading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict and reconstruct for each split
    for split_name, tiles in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
        print(f"\n{'=' * 70}")
        print(f"Processing {split_name.upper()} set")
        print('=' * 70)

        # Create output directory for this split
        split_output_dir = output_base_dir / f"{split_name}_tiles"

        # Predict and save tiles
        predictions = pipeline.predict_and_reconstruct(model, tiles, split_output_dir)

        # Mosaic into single map
        mosaic_path = output_base_dir / f"{split_name}_california_map.tif"
        pipeline.mosaic_tiles_to_map(split_output_dir, mosaic_path)

    # Create combined map with all predictions
    print(f"\n{'=' * 70}")
    print("Creating FULL California map (all tiles)")
    print('=' * 70)

    all_tiles = train_tiles + val_tiles + test_tiles
    full_output_dir = output_base_dir / "all_tiles"
    pipeline.predict_and_reconstruct(model, all_tiles, full_output_dir)

    full_map_path = output_base_dir / "full_california_map.tif"
    pipeline.mosaic_tiles_to_map(full_output_dir, full_map_path)

    print(f"\n{'=' * 70}")
    print("COMPLETE!")
    print('=' * 70)
    print(f"Outputs saved to: {output_base_dir}")
    print(f"  - train_california_map.tif")
    print(f"  - val_california_map.tif")
    print(f"  - test_california_map.tif")
    print(f"  - full_california_map.tif")


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "tiles_output/input_tiles"
    TARGET_DIR = "tiles_output/target_tiles"
    MODEL_PATH = "random_forest_model.pkl"  # Your trained model
    OUTPUT_DIR = "predictions_california_map"
    LOAD_ALL_FOLDERS = True
    SPLIT_LON = None  # Auto-calculate or set manually, e.g., -122.0

    # Run full pipeline
    run_full_pipeline(
        input_dir=INPUT_DIR,
        target_dir=TARGET_DIR,
        model_path=MODEL_PATH,
        output_base_dir=OUTPUT_DIR,
        load_all_folders=LOAD_ALL_FOLDERS,
        split_lon=SPLIT_LON
    )

    print("\n✓ You can now open the GeoTIFF files in QGIS to visualize predictions!")
    print("  - Load train/val/test maps separately to compare")
    print("  - Load full_california_map.tif to see all predictions")
    print("  - Compare with your original GEDI data")