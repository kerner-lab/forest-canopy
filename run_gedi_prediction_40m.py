#!/usr/bin/env python3
"""
Random Forest Model for GEDI Canopy Height Prediction using OLMo-Earth Embeddings

QUICK FIX VERSION - Handles CRS Mismatch

This script:
1. Reads healpix shapefile to get train/val/test splits
2. Loads GEDI data (~30m resolution, EPSG:4326) and OLMo embeddings (40m resolution, EPSG:32610)
3. Warps GEDI on-the-fly to match OLMo's UTM grid (handles CRS + resolution mismatch)
4. Transposes OLMo from (C,H,W) to (H,W,C)
5. Filters sparse/invalid GEDI data
6. Trains a Random Forest model
7. Evaluates on validation and test sets

NOTE: This is a temporary solution. For production, regenerate OLMo embeddings at proper
resolution and reproject GEDI to UTM permanently.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
import zarr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime


class GEDIPredictor:
    """Pipeline for GEDI canopy height prediction using OLMo embeddings"""

    def __init__(self, shapefile_path, gedi_dir, embeddings_dir, output_dir):
        """
        Initialize the predictor

        Args:
            shapefile_path: Path to healpix_shapefile.shp
            gedi_dir: Directory containing GEDI files (healpix_*_gedi.tif)
            embeddings_dir: Directory containing embedding zarr files
            output_dir: Directory for outputs (models, metrics, plots)
        """
        self.shapefile_path = Path(shapefile_path)
        self.gedi_dir = Path(gedi_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Track missing tiles
        self.missing_embeddings = []
        self.missing_gedi = []
        self.skipped_tiles = []

        # Load shapefile with splits
        self.logger.info(f"Loading shapefile: {shapefile_path}")
        self.gdf = gpd.read_file(shapefile_path)
        self.logger.info(f"Found {len(self.gdf)} tiles")
        self.logger.info(f"Split distribution:\n{self.gdf['split'].value_counts()}")
        print(f"Loading shapefile: {shapefile_path}")
        print(f"Found {len(self.gdf)} tiles")
        print(f"Split distribution:\n{self.gdf['split'].value_counts()}")

        self.model = None
        self.feature_names = None

        # Track missing tiles
        self.missing_embeddings = []
        self.missing_gedi = []
        self.successfully_loaded = []

    def _setup_logging(self):
        """Setup logging to both file and console"""
        self.logger = logging.getLogger('GEDIPredictor')
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        # File handler
        log_file = self.output_dir / f'gedi_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # Only warnings and errors to console

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def load_gedi_tile(self, healpix_id):
        """
        Load GEDI data for a specific tile

        Args:
            healpix_id: The healpix tile ID

        Returns:
            tuple: (gedi_data, transform, crs) or (None, None, None) if not found
        """
        # Try different possible file extensions
        for ext in ['.tif', '.tiff', '']:
            gedi_path = self.gedi_dir / f"healpix_{healpix_id}_gedi{ext}"
            if gedi_path.exists():
                try:
                    with rasterio.open(gedi_path) as src:
                        gedi_data = src.read(1)  # Read first band
                        transform = src.transform
                        crs = src.crs
                    self.logger.info(f"Loaded GEDI data for tile {healpix_id}: shape {gedi_data.shape}")
                    return gedi_data, transform, crs
                except Exception as e:
                    self.logger.error(f"Error reading GEDI file {gedi_path}: {e}")
                    self.missing_gedi.append(healpix_id)
                    return None, None, None

        self.logger.warning(f"GEDI file not found for healpix_id {healpix_id}")
        self.missing_gedi.append(healpix_id)
        return None, None, None

    def load_embeddings_tile(self, healpix_id):
        """
        Load OLMo embeddings for a specific tile from zarr

        Args:
            healpix_id: The healpix tile ID

        Returns:
            numpy array of shape (height, width, n_channels) or None
        """
        emb_path = self.embeddings_dir / f"healpix_{healpix_id}_embeddings_int8.zarr"

        if not emb_path.exists():
            self.logger.warning(f"Embeddings not found for healpix_id {healpix_id}: {emb_path}")
            self.missing_embeddings.append(healpix_id)
            return None

        # Load zarr array
        try:
            # Try opening as a group first, then as an array
            try:
                z = zarr.open(str(emb_path), mode='r')
            except Exception:
                # If group fails, try as array
                z = zarr.open_array(str(emb_path), mode='r')

            embeddings = np.array(z)

            # Transpose from (C, H, W) to (H, W, C) if needed
            if len(embeddings.shape) == 3 and embeddings.shape[0] == 128:
                self.logger.info(f"Transposing embeddings {healpix_id}: {embeddings.shape}")
                embeddings = np.transpose(embeddings, (1, 2, 0))
                self.logger.info(f"  New shape: {embeddings.shape}")

            # Convert to float and normalize
            embeddings = embeddings.astype(np.float32) / 127.0  # Normalize to [-1, 1]

            return embeddings
        except Exception as e:
            self.logger.error(f"Error loading embeddings for tile {healpix_id}: {e}")
            self.missing_embeddings.append(healpix_id)
            return None

    def load_embeddings_metadata(self, healpix_id):
        """
        Load metadata for OLMo embeddings tile

        Args:
            healpix_id: The healpix tile ID

        Returns:
            dict with transform, crs, shape, bounds or None
        """
        # Try metadata JSON file first
        metadata_path = self.embeddings_dir / f"healpix_{healpix_id}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)

        # Try zarr attributes
        emb_path = self.embeddings_dir / f"healpix_{healpix_id}_embeddings_int8.zarr"
        if emb_path.exists():
            try:
                # Try opening as a group first, then as an array
                try:
                    z = zarr.open(str(emb_path), mode='r')
                except Exception:
                    # If group fails, try as array
                    z = zarr.open_array(str(emb_path), mode='r')

                attrs = dict(z.attrs)
                if 'transform' in attrs and 'crs' in attrs:
                    return attrs
            except:
                pass

        self.logger.warning(f"No metadata found for tile {healpix_id}")
        return None

    def load_gedi_warped_to_olmo(self, healpix_id, target_shape, target_transform, target_crs):
        """
        Load GEDI and warp to match OLMo's grid

        Args:
            healpix_id: The healpix tile ID
            target_shape: Target shape (H, W) from OLMo
            target_transform: Affine transform from OLMo metadata
            target_crs: CRS from OLMo metadata

        Returns:
            Warped GEDI array or None
        """
        # Find GEDI file
        gedi_path = None
        for ext in ['.tif', '.tiff', '']:
            test_path = self.gedi_dir / f"healpix_{healpix_id}_gedi{ext}"
            if test_path.exists():
                gedi_path = test_path
                break

        if gedi_path is None:
            self.logger.warning(f"GEDI file not found for tile {healpix_id}")
            return None

        try:
            from affine import Affine

            with rasterio.open(gedi_path) as src:
                # Create output array at target shape
                gedi_warped = np.empty(target_shape, dtype=np.float32)

                # Build affine transform from target_transform list
                if isinstance(target_transform, list):
                    target_affine = Affine(*target_transform[:6])
                else:
                    target_affine = target_transform

                # Reproject GEDI to match OLMo's grid
                reproject(
                    source=rasterio.band(src, 1),
                    destination=gedi_warped,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_affine,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

                # Debug: Check values before conversion
                valid_before = gedi_warped[np.isfinite(gedi_warped) & (gedi_warped != 0)]
                if len(valid_before) > 0:
                    self.logger.info(f"Before cm→m: min={np.min(valid_before):.1f}, max={np.max(valid_before):.1f}")

                # Convert from centimeters to meters
                gedi_warped = gedi_warped / 100.0

                # Debug: Check values after conversion
                valid_after = gedi_warped[np.isfinite(gedi_warped) & (gedi_warped != 0)]
                if len(valid_after) > 0:
                    self.logger.info(f"After cm→m: min={np.min(valid_after):.2f}, max={np.max(valid_after):.2f}")

                self.logger.info(
                    f"Warped GEDI from {src.crs} {src.shape} to {target_crs} {gedi_warped.shape} (converted cm→m)")
                return gedi_warped

        except Exception as e:
            self.logger.error(f"Error warping GEDI for tile {healpix_id}: {e}")
            return None

    def resample_embeddings_to_gedi(self, embeddings, gedi_shape):
        """
        Resample embeddings from 10m to 30m resolution to match GEDI

        Args:
            embeddings: numpy array of shape (H, W, C) at 10m resolution
            gedi_shape: target shape (H', W') at 30m resolution

        Returns:
            Resampled embeddings of shape (H', W', C)
        """
        # Since 30m/10m = 3, we expect roughly 1/3 the dimensions
        # Use block averaging (mean pooling) to downsample

        h_emb, w_emb, n_channels = embeddings.shape
        h_gedi, w_gedi = gedi_shape

        # Calculate the scaling factors
        scale_h = h_emb / h_gedi
        scale_w = w_emb / w_gedi

        # Use block averaging for downsampling
        resampled = np.zeros((h_gedi, w_gedi, n_channels), dtype=np.float32)

        for i in range(h_gedi):
            for j in range(w_gedi):
                # Define the block in the high-res embeddings
                i_start = int(i * scale_h)
                i_end = int((i + 1) * scale_h)
                j_start = int(j * scale_w)
                j_end = int((j + 1) * scale_w)

                # Average over the block
                resampled[i, j, :] = embeddings[i_start:i_end, j_start:j_end, :].mean(axis=(0, 1))

        return resampled

    def prepare_tile_data(self, healpix_id):
        """
        Prepare training data for a single tile (with CRS handling)

        This version handles the mismatch between:
        - OLMo: 40m UTM (EPSG:32610)
        - GEDI: 30m lat/lon (EPSG:4326)

        Args:
            healpix_id: The healpix tile ID

        Returns:
            tuple: (X, y) where X is features and y is GEDI heights
                   Returns (None, None) if data cannot be loaded
        """
        # Load OLMo embeddings first
        embeddings = self.load_embeddings_tile(healpix_id)
        if embeddings is None:
            self.skipped_tiles.append((healpix_id, 'missing_embeddings'))
            self.logger.info(f"Skipping tile {healpix_id}: OLMo embeddings not available")
            return None, None

        # Load OLMo metadata to get grid parameters
        metadata = self.load_embeddings_metadata(healpix_id)
        if metadata is None:
            self.skipped_tiles.append((healpix_id, 'missing_metadata'))
            self.logger.warning(f"Skipping tile {healpix_id}: No metadata found")
            return None, None

        # Extract grid parameters from metadata
        target_shape = embeddings.shape[:2]  # (H, W)
        target_transform = metadata.get('transform')
        target_crs = metadata.get('crs', 'EPSG:32610')

        self.logger.info(f"Tile {healpix_id} - OLMo grid: {target_shape}, CRS: {target_crs}")
        print(f"  OLMo grid: {target_shape}, CRS: {target_crs}")

        # Load and warp GEDI to match OLMo's grid
        gedi_data = self.load_gedi_warped_to_olmo(healpix_id, target_shape, target_transform, target_crs)
        if gedi_data is None:
            self.skipped_tiles.append((healpix_id, 'missing_gedi'))
            self.logger.info(f"Skipping tile {healpix_id}: GEDI data not available")
            return None, None

        # Verify shapes match
        if gedi_data.shape != embeddings.shape[:2]:
            self.logger.error(
                f"Shape mismatch for tile {healpix_id}: GEDI {gedi_data.shape} vs OLMo {embeddings.shape[:2]}")
            self.skipped_tiles.append((healpix_id, 'shape_mismatch'))
            return None, None

        print(f"  ✓ GEDI warped to match OLMo: {gedi_data.shape}")

        # Debug: Check data range before filtering
        gedi_valid = gedi_data[np.isfinite(gedi_data)]
        if len(gedi_valid) > 0:
            print(f"  GEDI range before filtering: [{np.min(gedi_valid):.2f}, {np.max(gedi_valid):.2f}] meters")

        # Create mask for valid GEDI data (in meters)
        # Reasonable canopy height: 0-100 meters
        valid_mask = np.isfinite(gedi_data) & (gedi_data > 0)

        # Flatten data
        y = gedi_data[valid_mask]

        # Reshape embeddings and filter by mask
        h, w, c = embeddings.shape
        X = embeddings.reshape(-1, c)[valid_mask.flatten()]

        valid_pct = 100 * len(y) / gedi_data.size
        print(f"  Valid pixels: {len(y)} out of {gedi_data.size} ({valid_pct:.2f}%)")
        self.logger.info(f"Tile {healpix_id}: {len(y)} valid pixels ({valid_pct:.2f}%)")

        return X, y

    def prepare_split_data(self, split_name):
        """
        Prepare all data for a given split (train/val/test)

        Args:
            split_name: 'train', 'val', or 'test'

        Returns:
            tuple: (X, y) concatenated from all tiles in the split
        """
        tiles = self.gdf[self.gdf['split'] == split_name]['healpix_id'].values
        print(f"\nPreparing {split_name} data from {len(tiles)} tiles...")
        self.logger.info(f"Preparing {split_name} data from {len(tiles)} tiles")

        # Track tiles before processing this split
        skipped_before = len(self.skipped_tiles)

        X_list = []
        y_list = []

        for healpix_id in tqdm(tiles, desc=f"Loading {split_name} tiles"):
            X_tile, y_tile = self.prepare_tile_data(healpix_id)

            if X_tile is not None and len(X_tile) > 0:
                X_list.append(X_tile)
                y_list.append(y_tile)

        # Calculate how many were skipped in this split
        skipped_this_split = len(self.skipped_tiles) - skipped_before

        if len(X_list) == 0:
            print(f"Warning: No valid data found for {split_name} split!")
            self.logger.warning(f"No valid data found for {split_name} split!")
            return None, None

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        loaded_tiles = len(X_list)
        total_tiles = len(tiles)

        print(f"{split_name} data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Successfully loaded: {loaded_tiles}/{total_tiles} tiles ({100 * loaded_tiles / total_tiles:.1f}%)")
        print(f"  Skipped: {skipped_this_split} tiles")
        print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")

        self.logger.info(f"{split_name} data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Successfully loaded: {loaded_tiles}/{total_tiles} tiles")
        self.logger.info(f"Skipped: {skipped_this_split} tiles")

        return X, y

    def train(self, n_estimators=100, max_depth=None, min_samples_split=2,
              max_features='sqrt', n_jobs=-1, random_state=42):
        """
        Train the Random Forest model

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            max_features: Number of features to consider for best split
            n_jobs: Number of parallel jobs (-1 uses all cores)
            random_state: Random seed
        """
        # Prepare training data
        X_train, y_train = self.prepare_split_data('train')

        if X_train is None:
            raise ValueError("No training data available!")

        # Store feature names
        self.feature_names = [f"embedding_{i}" for i in range(X_train.shape[1])]

        # Initialize and train model
        print(f"\nTraining Random Forest with {n_estimators} trees...")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )

        self.model.fit(X_train, y_train)

        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        print(f"\nTraining Metrics:")
        print(f"  RMSE: {train_rmse:.3f}")
        print(f"  MAE: {train_mae:.3f}")
        print(f"  R²: {train_r2:.3f}")

        # Save training metrics
        metrics = {
            'train': {
                'rmse': float(train_rmse),
                'mae': float(train_mae),
                'r2': float(train_r2),
                'n_samples': len(y_train)
            }
        }

        return metrics

    def evaluate(self, split_name='val'):
        """
        Evaluate the model on validation or test set

        Args:
            split_name: 'val' or 'test'

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Prepare data
        X, y = self.prepare_split_data(split_name)

        if X is None:
            print(f"Warning: No data available for {split_name} split")
            return None

        # Predict
        print(f"\nEvaluating on {split_name} set...")
        y_pred = self.model.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"\n{split_name.capitalize()} Metrics:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")

        # Plot predictions vs actual
        self.plot_predictions(y, y_pred, split_name)

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': len(y)
        }

        return metrics

    def plot_predictions(self, y_true, y_pred, split_name):
        """Create scatter plot of predictions vs actual values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=1)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('Actual GEDI Height (m)')
        ax.set_ylabel('Predicted Height (m)')
        ax.set_title(f'{split_name.capitalize()} Set: Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residual plot
        ax = axes[1]
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.3, s=1)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Height (m)')
        ax.set_ylabel('Residual (m)')
        ax.set_title(f'{split_name.capitalize()} Set: Residual Plot')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f'predictions_{split_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance from the Random Forest model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xticks(range(top_n), indices, rotation=45)
        plt.tight_layout()

        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {output_path}")
        plt.close()

    def save_model(self, filename='gedi_rf_model.pkl'):
        """Save the trained model"""
        import pickle

        if self.model is None:
            raise ValueError("Model not trained yet!")

        model_path = self.output_dir / filename
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

        print(f"Saved model to {model_path}")
        self.logger.info(f"Saved model to {model_path}")

    def print_missing_tiles_summary(self):
        """Print and log summary of missing/skipped tiles"""
        print("\n" + "=" * 60)
        print("MISSING TILES SUMMARY")
        print("=" * 60)

        # Get unique missing embeddings and GEDI
        unique_missing_embeddings = list(set(self.missing_embeddings))
        unique_missing_gedi = list(set(self.missing_gedi))

        # Categorize skipped tiles
        skipped_embeddings = [tile_id for tile_id, reason in self.skipped_tiles if reason == 'missing_embeddings']
        skipped_gedi = [tile_id for tile_id, reason in self.skipped_tiles if reason == 'missing_gedi']

        print(f"\nMissing OLMo Embeddings: {len(unique_missing_embeddings)} tiles")
        if unique_missing_embeddings:
            print(f"  Tile IDs: {sorted(unique_missing_embeddings)}")
            self.logger.warning(
                f"Missing OLMo embeddings for {len(unique_missing_embeddings)} tiles: {sorted(unique_missing_embeddings)}")

        print(f"\nMissing GEDI Data: {len(unique_missing_gedi)} tiles")
        if unique_missing_gedi:
            print(f"  Tile IDs: {sorted(unique_missing_gedi)}")
            self.logger.warning(
                f"Missing GEDI data for {len(unique_missing_gedi)} tiles: {sorted(unique_missing_gedi)}")

        print(f"\nTotal Skipped Tiles: {len(self.skipped_tiles)}")
        print(f"  Skipped due to missing embeddings: {len(skipped_embeddings)}")
        print(f"  Skipped due to missing GEDI: {len(skipped_gedi)}")

        # Save to file
        missing_tiles_file = self.output_dir / 'missing_tiles.json'
        missing_data = {
            'missing_olmo_embeddings': sorted(unique_missing_embeddings),
            'missing_gedi': sorted(unique_missing_gedi),
            'skipped_tiles': [
                {'healpix_id': int(tile_id), 'reason': reason}
                for tile_id, reason in self.skipped_tiles
            ],
            'summary': {
                'n_missing_embeddings': len(unique_missing_embeddings),
                'n_missing_gedi': len(unique_missing_gedi),
                'n_total_skipped': len(self.skipped_tiles)
            }
        }

        with open(missing_tiles_file, 'w') as f:
            json.dump(missing_data, f, indent=2)

        print(f"\nMissing tiles log saved to: {missing_tiles_file}")
        self.logger.info(f"Missing tiles summary saved to {missing_tiles_file}")

        # Also create a simple text file for easy viewing
        missing_tiles_txt = self.output_dir / 'missing_olmo_tiles.txt'
        with open(missing_tiles_txt, 'w') as f:
            f.write("Missing OLMo-Earth Embeddings Tiles\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total missing: {len(unique_missing_embeddings)} tiles\n\n")
            f.write("Tile IDs (one per line):\n")
            for tile_id in sorted(unique_missing_embeddings):
                f.write(f"{tile_id}\n")

        print(f"Missing OLMo tiles list saved to: {missing_tiles_txt}")
        self.logger.info(f"Missing OLMo tiles list saved to {missing_tiles_txt}")

        return missing_data

    def save_metrics(self, metrics, filename='metrics.json'):
        """Save all metrics to JSON"""
        metrics_path = self.output_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {metrics_path}")
        self.logger.info(f"Saved metrics to {metrics_path}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Random Forest for GEDI height prediction')
    parser.add_argument('--shapefile', type=str, required=True,
                        help='Path to healpix_shapefile.shp')
    parser.add_argument('--gedi-dir', type=str, required=True,
                        help='Directory containing GEDI files')
    parser.add_argument('--embeddings-dir', type=str, required=True,
                        help='Directory containing OLMo embedding zarr files')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for models and results')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Maximum depth of trees (None for unlimited)')
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help='Minimum samples required to split a node')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Initialize predictor
    predictor = GEDIPredictor(
        shapefile_path=args.shapefile,
        gedi_dir=args.gedi_dir,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir
    )

    # Train model
    train_metrics = predictor.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )

    # Evaluate on validation set
    val_metrics = predictor.evaluate('val')

    # Evaluate on test set
    test_metrics = predictor.evaluate('test')

    # Combine all metrics
    all_metrics = {
        'train': train_metrics['train'],
        'val': val_metrics,
        'test': test_metrics
    }

    # Plot feature importance
    predictor.plot_feature_importance(top_n=30)

    # Print summary of missing tiles
    predictor.print_missing_tiles_summary()

    # Save everything
    predictor.save_model()
    predictor.save_metrics(all_metrics)

    print("\n" + "=" * 50)
    print("Training and Evaluation Complete!")
    print("=" * 50)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nFinal Metrics:")
    print(f"  Train R²: {train_metrics['train']['r2']:.3f}")
    print(f"  Val R²:   {val_metrics['r2']:.3f}")
    print(f"  Test R²:  {test_metrics['r2']:.3f}")


if __name__ == '__main__':
    main()