#!/usr/bin/env python3
"""
Random Forest Model for GEDI Canopy Height Prediction using OLMo-Earth 30m Embeddings

Embeddings are already reprojected to the healpix tile grid (same CRS/extent as GEDI tifs),
so pixel indices align directly — no coordinate reprojection needed.

Supports float32 and int8 embedding formats via --embedding-format flag.
Predicts non-zero forest height only (GEDI > 0m).
"""

import numpy as np
import geopandas as gpd
import rasterio
import zarr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import logging
import pickle
import time
import os
from datetime import datetime
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class GEDIPredictor30m:
    """Pipeline for GEDI canopy height prediction using OLMo 30m embeddings"""

    def __init__(self, shapefile_path, gedi_dir, embeddings_dir, output_dir,
                 embedding_format='float32'):
        """
        Args:
            shapefile_path: Path to healpix shapefile with 'split' column
            gedi_dir: Directory containing GEDI files (healpix_*_gedi.tif)
            embeddings_dir: Directory containing zarr embedding files
            output_dir: Directory for outputs
            embedding_format: 'float32' or 'int8'
        """
        if embedding_format not in ('float32', 'int8'):
            raise ValueError("embedding_format must be 'float32' or 'int8'")

        self.shapefile_path = Path(shapefile_path)
        self.gedi_dir = Path(gedi_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.embedding_format = embedding_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.missing_embeddings = []
        self.missing_gedi = []

        self.logger.info(f"Loading shapefile: {shapefile_path}")
        self.gdf = gpd.read_file(shapefile_path)
        self.logger.info(f"Found {len(self.gdf)} tiles")
        self.logger.info(f"Split distribution:\n{self.gdf['split'].value_counts().to_string()}")
        print(f"Loaded {len(self.gdf)} tiles")
        print(f"Split distribution:\n{self.gdf['split'].value_counts()}")
        print(f"Embedding format: {embedding_format}")

        self.model = None
        self.feature_names = None

    def _setup_logging(self):
        self.logger = logging.getLogger('GEDIPredictor30m')
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            return

        log_file = self.output_dir / f'gedi_prediction_30m_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(f"Log file: {log_file}")

    def _embedding_path(self, healpix_id):
        return self.embeddings_dir / f"healpix_{healpix_id}_embeddings_{self.embedding_format}_reproj.zarr"

    def load_embeddings_tile(self, healpix_id):
        """
        Load embeddings for a tile. Returns (H, W, C) float32 array or None.
        int8 values are normalized to [-1, 1] by dividing by 127.
        """
        emb_path = self._embedding_path(healpix_id)

        if not emb_path.exists():
            self.logger.warning(f"Embedding not found for tile {healpix_id}: {emb_path}")
            self.missing_embeddings.append(healpix_id)
            return None

        try:
            z = zarr.open(str(emb_path), mode='r')
            embeddings = np.array(z)  # (C, H, W)

            if embeddings.ndim != 3:
                self.logger.error(f"Unexpected embedding shape for tile {healpix_id}: {embeddings.shape}")
                self.missing_embeddings.append(healpix_id)
                return None

            # Transpose (C, H, W) -> (H, W, C)
            embeddings = np.transpose(embeddings, (1, 2, 0))

            if self.embedding_format == 'int8':
                embeddings = embeddings.astype(np.float32) / 127.0
            else:
                embeddings = embeddings.astype(np.float32)

            self.logger.info(f"Loaded embeddings tile {healpix_id}: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error loading embeddings for tile {healpix_id}: {e}")
            self.missing_embeddings.append(healpix_id)
            return None

    def load_gedi_tile(self, healpix_id):
        """
        Load GEDI tif for a tile. Returns (data_array, rasterio_src) or (None, None).
        """
        for ext in ('.tif', '.tiff', ''):
            gedi_path = self.gedi_dir / f"healpix_{healpix_id}_gedi{ext}"
            if gedi_path.exists():
                try:
                    src = rasterio.open(gedi_path)
                    data = src.read(1)
                    return data, src
                except Exception as e:
                    self.logger.error(f"Error reading GEDI file {gedi_path}: {e}")
                    self.missing_gedi.append(healpix_id)
                    return None, None

        self.logger.warning(f"GEDI file not found for tile {healpix_id}")
        self.missing_gedi.append(healpix_id)
        return None, None

    def prepare_tile_data(self, healpix_id):
        """
        Prepare (X, y) for one tile. Returns (None, None) on failure.

        Since embeddings are already reprojected to the tile grid, GEDI pixel
        indices map directly to embedding pixel indices.
        Filters to non-zero forest height only (y > 0m).
        """
        embeddings = self.load_embeddings_tile(healpix_id)
        if embeddings is None:
            return None, None

        gedi_data, src = self.load_gedi_tile(healpix_id)
        if gedi_data is None:
            return None, None

        try:
            emb_h, emb_w, n_channels = embeddings.shape
            gedi_h, gedi_w = gedi_data.shape

            # Valid GEDI mask: exclude nodata and keep non-zero heights
            nodata = src.nodata if src.nodata is not None else -9999
            valid_mask = (
                (gedi_data != nodata) &
                (gedi_data > 0) &       # non-zero forest height only
                (gedi_data < 12000)     # exclude outliers (stored in cm)
            )

            rows, cols = np.where(valid_mask)

            if len(rows) == 0:
                self.logger.info(f"No valid non-zero GEDI pixels for tile {healpix_id}")
                return None, None

            # Map GEDI pixel coords to embedding pixel coords.
            # If resolutions differ, scale indices proportionally.
            if gedi_h != emb_h or gedi_w != emb_w:
                emb_rows = np.clip(
                    np.round(rows * emb_h / gedi_h).astype(int), 0, emb_h - 1
                )
                emb_cols = np.clip(
                    np.round(cols * emb_w / gedi_w).astype(int), 0, emb_w - 1
                )
                self.logger.info(
                    f"Tile {healpix_id}: GEDI shape {gedi_data.shape} vs "
                    f"embedding shape {embeddings.shape[:2]} — scaling indices"
                )
            else:
                emb_rows = rows
                emb_cols = cols

            y = gedi_data[rows, cols] / 100.0  # cm -> m
            X = embeddings[emb_rows, emb_cols, :]

            self.logger.info(
                f"Tile {healpix_id}: {len(y)} non-zero samples, "
                f"height range [{y.min():.2f}, {y.max():.2f}] m"
            )
            return X, y

        finally:
            src.close()

    def _log_progress(self, split_name, i, n_tiles, loaded, skipped, total_samples,
                      split_start, batch_start):
        now = time.time()
        elapsed_total = now - split_start
        elapsed_batch = now - batch_start
        rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
        eta = (n_tiles - i - 1) / rate if rate > 0 else float('nan')
        mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9 if _HAS_PSUTIL else float('nan')
        print(
            f"  [{split_name}] tile {i+1}/{n_tiles} | "
            f"loaded={loaded} skipped={skipped} | "
            f"samples_so_far={total_samples:,} | "
            f"batch_10={elapsed_batch:.1f}s elapsed={elapsed_total:.0f}s ETA={eta:.0f}s | "
            f"mem={mem_gb:.2f}GB"
        )
        self.logger.info(
            f"[{split_name}] Progress {i+1}/{n_tiles}: loaded={loaded} skipped={skipped} "
            f"samples={total_samples} elapsed={elapsed_total:.0f}s ETA={eta:.0f}s mem={mem_gb:.2f}GB"
        )

    def prepare_split_data(self, split_name, max_tiles=None):
        """Load and concatenate all tile data for a split."""
        tiles = self.gdf[self.gdf['split'] == split_name]['healpix_id'].values
        if max_tiles is not None:
            tiles = tiles[:max_tiles]
        n_tiles = len(tiles)
        print(f"\nPreparing '{split_name}' data from {n_tiles} tiles...")
        self.logger.info(f"Preparing '{split_name}' split from {n_tiles} tiles")

        X_list, y_list = [], []
        loaded, skipped = 0, 0
        total_samples = 0
        split_start = time.time()
        batch_start = split_start
        log_interval = 5

        for i, healpix_id in enumerate(tqdm(tiles, desc=f"Loading {split_name} tiles")):
            tile_start = time.time()
            X_tile, y_tile = self.prepare_tile_data(healpix_id)
            tile_elapsed = time.time() - tile_start

            if X_tile is not None and len(X_tile) > 0:
                X_list.append(X_tile)
                y_list.append(y_tile)
                loaded += 1
                total_samples += len(y_tile)
                self.logger.info(
                    f"  Tile {healpix_id}: {len(y_tile):,} samples in {tile_elapsed:.1f}s"
                )
            else:
                skipped += 1
                self.logger.info(
                    f"  Tile {healpix_id}: skipped ({tile_elapsed:.1f}s)"
                )

            if (i + 1) % log_interval == 0 or (i + 1) == n_tiles:
                self._log_progress(split_name, i, n_tiles, loaded, skipped,
                                   total_samples, split_start, batch_start)
                batch_start = time.time()

        if not X_list:
            print(f"Warning: No valid data for '{split_name}' split")
            self.logger.warning(f"No valid data for '{split_name}' split")
            return None, None

        print(f"  Stacking arrays for '{split_name}'...")
        stack_start = time.time()
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        print(f"  Stack took {time.time() - stack_start:.1f}s")

        total_elapsed = time.time() - split_start
        print(f"  {loaded}/{n_tiles} tiles loaded, {skipped} skipped  (total {total_elapsed:.0f}s)")
        print(f"  Samples: {X.shape[0]:,}, Features: {X.shape[1]}")
        print(f"  Height range: [{y.min():.2f}, {y.max():.2f}] m  mean={y.mean():.2f}")
        self.logger.info(
            f"'{split_name}': {X.shape[0]:,} samples from {loaded} tiles "
            f"({skipped} skipped) in {total_elapsed:.0f}s"
        )
        return X, y

    def train(self, n_estimators=100, max_depth=None, min_samples_split=2,
              max_features='sqrt', n_jobs=-1, random_state=42, max_tiles=None):
        """Train the Random Forest on the training split."""
        X_train, y_train = self.prepare_split_data('train', max_tiles=max_tiles)
        if X_train is None:
            raise ValueError("No training data available")

        self.feature_names = [f"emb_{i}" for i in range(X_train.shape[1])]

        print(f"\nTraining Random Forest ({n_estimators} trees)...")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_train)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred))),
            'mae': float(mean_absolute_error(y_train, y_pred)),
            'r2': float(r2_score(y_train, y_pred)),
            'n_samples': int(len(y_train)),
        }
        print(f"Train  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
        self.logger.info(f"Train metrics: {metrics}")
        return {'train': metrics}

    def evaluate(self, split_name='val', max_tiles=None):
        """Evaluate trained model on a split."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X, y = self.prepare_split_data(split_name, max_tiles=max_tiles)
        if X is None:
            return None

        y_pred = self.model.predict(X)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'n_samples': int(len(y)),
        }
        print(f"{split_name.capitalize()}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
        self.logger.info(f"'{split_name}' metrics: {metrics}")

        self._plot_predictions(y, y_pred, split_name)
        return metrics

    def _plot_predictions(self, y_true, y_pred, split_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=1)
        lim = [y_true.min(), y_true.max()]
        ax.plot(lim, lim, 'r--', lw=2, label='1:1')
        ax.set_xlabel('Actual GEDI Height (m)')
        ax.set_ylabel('Predicted Height (m)')
        ax.set_title(f'{split_name.capitalize()}: Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.3, s=1)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Height (m)')
        ax.set_ylabel('Residual (m)')
        ax.set_title(f'{split_name.capitalize()}: Residuals')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / f'predictions_{split_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {out}")

    def plot_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Model not trained yet")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Embedding Channel Index')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (30m embeddings, {self.embedding_format})')
        plt.xticks(range(top_n), indices, rotation=45)
        plt.tight_layout()
        out = self.output_dir / 'feature_importance.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance plot: {out}")

    def save_model(self, filename=None):
        if self.model is None:
            raise ValueError("Model not trained yet")
        if filename is None:
            filename = f'gedi_rf_30m_{self.embedding_format}.pkl'
        model_path = self.output_dir / filename
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_names': self.feature_names,
                         'embedding_format': self.embedding_format}, f)
        print(f"Saved model: {model_path}")
        self.logger.info(f"Saved model: {model_path}")

    def save_metrics(self, metrics, filename='metrics.json'):
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {path}")

    def print_missing_summary(self):
        unique_emb = sorted(set(self.missing_embeddings))
        unique_gedi = sorted(set(self.missing_gedi))
        print("\n" + "=" * 60)
        print("MISSING TILES SUMMARY")
        print("=" * 60)
        print(f"Missing embeddings ({self.embedding_format}): {len(unique_emb)} tiles")
        if unique_emb:
            print(f"  {unique_emb}")
        print(f"Missing GEDI: {len(unique_gedi)} tiles")
        if unique_gedi:
            print(f"  {unique_gedi}")

        self.logger.warning(f"Missing embeddings: {unique_emb}")
        self.logger.warning(f"Missing GEDI: {unique_gedi}")

        out = self.output_dir / 'missing_tiles.json'
        with open(out, 'w') as f:
            json.dump({'missing_embeddings': unique_emb, 'missing_gedi': unique_gedi}, f, indent=2)
        print(f"Missing tiles log: {out}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Random Forest for GEDI height prediction using 30m OLMo embeddings'
    )
    parser.add_argument('--shapefile', required=True,
                        help='Path to healpix shapefile (.shp) with split column')
    parser.add_argument('--gedi-dir', required=True,
                        help='Directory containing GEDI tif files (healpix_*_gedi.tif)')
    parser.add_argument('--embeddings-dir', required=True,
                        help='Directory containing 30m zarr embedding files')
    parser.add_argument('--output-dir', default='./outputs_30m',
                        help='Output directory for models and results')
    parser.add_argument('--embedding-format', choices=['float32', 'int8'], default='float32',
                        help='Embedding format to load (default: float32)')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Limit tiles per split (for pipeline testing)')

    args = parser.parse_args()

    predictor = GEDIPredictor30m(
        shapefile_path=args.shapefile,
        gedi_dir=args.gedi_dir,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        embedding_format=args.embedding_format,
    )

    train_metrics = predictor.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        max_tiles=args.max_tiles,
    )

    val_metrics = predictor.evaluate('val', max_tiles=args.max_tiles)
    test_metrics = predictor.evaluate('test', max_tiles=args.max_tiles)

    all_metrics = {
        'train': train_metrics['train'],
        'val': val_metrics,
        'test': test_metrics,
        'embedding_format': args.embedding_format,
    }

    predictor.plot_feature_importance(top_n=30)
    predictor.print_missing_summary()
    predictor.save_model()
    predictor.save_metrics(all_metrics)

    print("\n" + "=" * 50)
    print("Complete!")
    print(f"  Train R²: {train_metrics['train']['r2']:.3f}")
    if val_metrics:
        print(f"  Val   R²: {val_metrics['r2']:.3f}")
    if test_metrics:
        print(f"  Test  R²: {test_metrics['r2']:.3f}")
    print(f"Results: {args.output_dir}")


if __name__ == '__main__':
    main()
