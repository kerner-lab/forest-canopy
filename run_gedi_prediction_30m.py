#!/usr/bin/env python3
"""
ML Models for GEDI Canopy Height Prediction using OLMo-Earth 30m Embeddings

Supports four model architectures selectable via --model:
  rf  — Random Forest Regressor (default)
  xgb — XGBoost Regressor (requires xgboost package)
  gb  — Gradient Boosting Regressor
  lr  — Logistic Regression (discretises heights into --lr-bin-size bins)

Embeddings are already reprojected to the healpix tile grid (same CRS/extent as GEDI tifs),
so pixel indices align directly — no coordinate reprojection needed.

Supports float32 and int8 embedding formats via --embedding-format flag.
Predicts non-zero forest height only (GEDI > 0m).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import zarr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
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
                 embedding_format='float32', preprocessed_dir=None,
                 subset_fraction=None, subset_seed=42, subset_splits=None,
                 stratified_sample_fraction=None, stratified_sample_seed=42,
                 stratified_splits=None, model_type='rf', embedding_source='olmo',
                 include_zeros=False, tile_allowlist=None):
        """
        Args:
            shapefile_path: Path to healpix shapefile with 'split' column
            gedi_dir: Directory containing GEDI files (healpix_*_gedi.tif)
            embeddings_dir: Directory containing zarr embedding files
            output_dir: Directory for outputs
            embedding_format: 'float32' or 'int8'
            preprocessed_dir: Optional path to .npz cache from preprocess_olmo_to_gedi_tiles.py.
                               When set, zarr/rasterio loading is skipped entirely.
            subset_fraction: If set (0 < float <= 1), randomly sample this fraction of tiles
                             for splits listed in subset_splits. Tile IDs saved to subset_tiles.json.
            subset_seed: Random seed used for subset sampling (default: 42).
            subset_splits: List of split names to apply tile subsetting to (default: ['train']).
            stratified_sample_fraction: If set (any truthy value triggers activation), equalizes
                                        10 m height bins by capping each bin at the smallest
                                        bin's pixel count (stratified by GEDI ground-truth height).
                                        Saves per-bin counts to stratified_sample_info.json.
            stratified_sample_seed: Random seed for stratified pixel sampling (default: 42).
            stratified_splits: List of split names to apply stratified sampling to (default: ['train']).
            model_type: One of 'rf', 'xgb', 'gb', 'lr' (default: 'rf').
            embedding_source: One of 'olmo', 'aef' (default: 'olmo'). Used for
                              output naming and metadata; does not change how .npz
                              files are loaded — point --preprocessed-dir at the
                              correct source directory.
            include_zeros: If True, keep GEDI height == 0 measurements (gedi_data >= 0)
                           and, for the preprocessed path, load the
                           healpix_*_{format}_withzeros.npz cache files written by
                           alphaearth_processing.py --keep-zeros. If False (default),
                           only strictly-positive heights are used and the default
                           zero-filtered .npz cache is loaded.
        """
        _VALID_MODELS = ('rf', 'xgb', 'gb', 'lr')
        _VALID_SOURCES = ('olmo', 'aef', 'tessera', 'presto', 'prithvi')
        if embedding_source not in _VALID_SOURCES:
            raise ValueError(f"embedding_source must be one of {_VALID_SOURCES}")
        if model_type not in _VALID_MODELS:
            raise ValueError(f"model_type must be one of {_VALID_MODELS}")
        if model_type == 'xgb' and not _HAS_XGB:
            raise ImportError("xgboost is required for model_type='xgb'. Install with: pip install xgboost")
        if embedding_format not in ('float32', 'int8'):
            raise ValueError("embedding_format must be 'float32' or 'int8'")
        if subset_fraction is not None and not (0 < subset_fraction <= 1):
            raise ValueError("subset_fraction must be between 0 (exclusive) and 1 (inclusive)")
        if stratified_sample_fraction is not None and not (0 < stratified_sample_fraction <= 1):
            raise ValueError("stratified_sample_fraction must be between 0 (exclusive) and 1 (inclusive)")

        self.shapefile_path = Path(shapefile_path)
        self.gedi_dir = Path(gedi_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.embedding_format = embedding_format
        self.include_zeros = include_zeros
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.subset_splits = set(subset_splits) if subset_splits else {'train'}
        self.stratified_sample_fraction = stratified_sample_fraction
        self.stratified_sample_seed = stratified_sample_seed
        self.stratified_splits = set(stratified_splits) if stratified_splits else {'train'}
        # Restrict every split to these healpix ids. Used to hold the tile set fixed
        # across embedding sources so their metrics are comparable — without it a
        # source with fewer cached tiles is silently scored on a different dataset.
        self.tile_allowlist = set(int(t) for t in tile_allowlist) if tile_allowlist else None
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
        print(f"Embedding source: {embedding_source}  format: {embedding_format}")
        if self.preprocessed_dir:
            print(f"Using preprocessed cache: {self.preprocessed_dir}")
        if self.subset_fraction is not None:
            print(f"Subset fraction: {self.subset_fraction} (seed={self.subset_seed})")
        if self.stratified_sample_fraction is not None:
            print(f"Stratified sample fraction: {self.stratified_sample_fraction} "
                  f"(seed={self.stratified_sample_seed}, splits={sorted(self.stratified_splits)})")

        self.model_type = model_type
        self.embedding_source = embedding_source
        self.model = None
        self.feature_names = None
        self._subset_tile_ids = {}       # populated on first use per split
        self._stratified_sample_info = {}  # populated when stratified sampling is used
        self._lr_bin_edges = None        # set during LR training
        self._lr_bin_centers = None

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

    def _cache_path(self, healpix_id):
        """Preprocessed .npz cache path for a tile. Adds the _withzeros suffix when
        include_zeros is set so the zero-retaining caches are loaded instead of the
        default zero-filtered ones."""
        suffix = "_withzeros" if self.include_zeros else ""
        primary = (self.preprocessed_dir /
                   f"healpix_{healpix_id}_{self.embedding_format}{suffix}.npz")
        if primary.exists():
            return primary
        # Two naming conventions are in use: preprocess_olmo_to_gedi_tiles.py and
        # tessera_processing.py name caches by dtype (healpix_<id>_float32.npz),
        # while olmoearth_download.py names them by source (healpix_<id>_olmo.npz).
        # Accept either so a cache dir does not have to be renamed to be usable.
        alt = (self.preprocessed_dir /
               f"healpix_{healpix_id}_{self.embedding_source}{suffix}.npz")
        return alt if alt.exists() else primary

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

            # Valid GEDI mask: exclude nodata. Keep zero heights when include_zeros
            # is set (gedi_data >= 0), otherwise non-zero heights only (gedi_data > 0).
            nodata = src.nodata if src.nodata is not None else -9999
            valid_mask = (
                (gedi_data != nodata) &
                (gedi_data >= 0 if self.include_zeros else gedi_data > 0) &
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
        """Load and concatenate all tile data for a split.

        When self.preprocessed_dir is set, reads .npz cache files written by
        preprocess_olmo_to_gedi_tiles.py instead of re-loading zarr + rasterio.

        When stratified_sample_fraction is also set and preprocessed_dir is
        available, uses a two-pass approach: pass 1 reads only y to build the
        height index; stratified sampling selects (tile, pixel) pairs; pass 2
        reads X only for those rows. Peak memory stays at ~10% of full dataset.
        """
        tiles = self.gdf[self.gdf['split'] == split_name]['healpix_id'].values
        # Applied BEFORE subset/stratified sampling so those operate on the common
        # tile set, not on tiles that would later be dropped anyway.
        if self.tile_allowlist is not None:
            n_before = len(tiles)
            tiles = np.array([t for t in tiles if int(t) in self.tile_allowlist])
            msg = (f"Tile allowlist '{split_name}': {len(tiles)}/{n_before} tiles kept")
            print(f"  {msg}")
            self.logger.info(msg)
            if len(tiles) == 0:
                raise ValueError(
                    f"Tile allowlist removed every tile from split '{split_name}'. "
                    f"Check that the list matches the shapefile's healpix_id values.")
        if (self.subset_fraction is not None and self.subset_fraction < 1.0
                and split_name in self.subset_splits):
            rng = np.random.default_rng(self.subset_seed)
            n_subset = max(1, int(round(len(tiles) * self.subset_fraction)))
            tiles = rng.choice(tiles, size=n_subset, replace=False)
            tiles = np.sort(tiles)
            self._subset_tile_ids[split_name] = tiles.tolist()
            self.logger.info(
                f"Subset sampling '{split_name}': {n_subset}/{len(self.gdf[self.gdf['split'] == split_name])} "
                f"tiles (fraction={self.subset_fraction}, seed={self.subset_seed})"
            )
        if max_tiles is not None:
            tiles = tiles[:max_tiles]
        n_tiles = len(tiles)
        print(f"\nPreparing '{split_name}' data from {n_tiles} tiles...")
        self.logger.info(f"Preparing '{split_name}' split from {n_tiles} tiles")

        if (self.stratified_sample_fraction is not None
                and split_name in self.stratified_splits
                and self.preprocessed_dir is not None):
            return self._prepare_split_stratified_two_pass(tiles, split_name)

        X_list, y_list = [], []
        loaded, skipped = 0, 0
        total_samples = 0
        split_start = time.time()
        batch_start = split_start
        log_interval = 5

        for i, healpix_id in enumerate(tqdm(tiles, desc=f"Loading {split_name} tiles")):
            tile_start = time.time()

            skip_reason = None
            if self.preprocessed_dir is not None:
                cache_path = self._cache_path(healpix_id)
                if cache_path.exists():
                    npz = np.load(cache_path)
                    X_tile, y_tile = npz['X'], npz['y']
                    if self.embedding_format == 'int8' and X_tile.dtype == np.int8:
                        X_tile = X_tile.astype(np.float32) / 127.0
                else:
                    self.logger.warning(f"Cache missing for tile {healpix_id}: {cache_path}")
                    X_tile, y_tile = None, None
                    skip_reason = "cache_missing"
            else:
                emb_count_before = len(self.missing_embeddings)
                gedi_count_before = len(self.missing_gedi)
                X_tile, y_tile = self.prepare_tile_data(healpix_id)
                if X_tile is None:
                    if len(self.missing_embeddings) > emb_count_before:
                        skip_reason = "missing_embedding"
                    elif len(self.missing_gedi) > gedi_count_before:
                        skip_reason = "missing_gedi"
                    else:
                        skip_reason = "no_valid_pixels"

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
                    f"  Tile {healpix_id}: skipped [{skip_reason}] ({tile_elapsed:.1f}s)"
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

        if self.stratified_sample_fraction is not None and split_name in self.stratified_splits:
            X, y = self._apply_stratified_sample(X, y, split_name)

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
              max_features='sqrt', n_jobs=-1, random_state=42, max_tiles=None,
              learning_rate=0.1, subsample=1.0,
              lr_C=1.0, lr_max_iter=1000, lr_bin_size=5.0):
        """Train the selected model on the training split."""
        X_train, y_train = self.prepare_split_data('train', max_tiles=max_tiles)
        if X_train is None:
            raise ValueError("No training data available")

        self.feature_names = [f"emb_{i}" for i in range(X_train.shape[1])]
        n_samples = X_train.shape[0]

        if self.model_type == 'rf':
            print(f"\nTraining Random Forest ({n_estimators} trees, {n_samples:,} samples)...")
            self.logger.info(f"Training RF: {n_estimators} trees, {n_samples:,} samples, {X_train.shape[1]} features")
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1,
            )
            y_fit = y_train

        elif self.model_type == 'xgb':
            _depth = max_depth if max_depth is not None else 6
            print(f"\nTraining XGBoost ({n_estimators} trees, depth={_depth}, lr={learning_rate}, {n_samples:,} samples)...")
            self.logger.info(f"Training XGBoost: n_estimators={n_estimators}, max_depth={_depth}, lr={learning_rate}")
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                n_jobs=n_jobs,
                random_state=random_state,
                verbosity=1,
            )
            y_fit = y_train

        elif self.model_type == 'gb':
            _depth = max_depth if max_depth is not None else 3
            print(f"\nTraining Gradient Boosting ({n_estimators} trees, depth={_depth}, lr={learning_rate}, {n_samples:,} samples)...")
            self.logger.info(f"Training GB: n_estimators={n_estimators}, max_depth={_depth}, lr={learning_rate}")
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=_depth,
                min_samples_split=min_samples_split,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state,
                verbose=1,
            )
            y_fit = y_train

        elif self.model_type == 'lr':
            bin_edges = np.arange(0, y_train.max() + lr_bin_size, lr_bin_size)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self._lr_bin_edges = bin_edges
            self._lr_bin_centers = bin_centers
            y_fit = np.clip(np.digitize(y_train, bin_edges) - 1, 0, len(bin_centers) - 1)
            n_classes = len(bin_centers)
            print(f"\nTraining Logistic Regression (C={lr_C}, {n_classes} bins of {lr_bin_size}m, {n_samples:,} samples)...")
            self.logger.info(f"Training LR: C={lr_C}, max_iter={lr_max_iter}, n_classes={n_classes}")
            self.model = LogisticRegression(
                C=lr_C,
                max_iter=lr_max_iter,
                n_jobs=n_jobs,
                random_state=random_state,
            )

        train_start = time.time()
        self.model.fit(X_train, y_fit)
        train_elapsed = time.time() - train_start

        label = {'rf': 'RF', 'xgb': 'XGBoost', 'gb': 'GB', 'lr': 'LR'}[self.model_type]
        print(f"{label} training took {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")
        self.logger.info(f"{label} training completed in {train_elapsed:.0f}s")

        y_pred = self._predict(X_train)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred))),
            'mae': float(mean_absolute_error(y_train, y_pred)),
            'r2': float(r2_score(y_train, y_pred)),
            'n_samples': int(len(y_train)),
        }
        print(f"Train  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
        self.logger.info(f"Train metrics: {metrics}")
        self._save_predictions(y_train, y_pred, 'train')
        return {'train': metrics}

    def _predict(self, X):
        """Predict canopy heights; maps LR class indices to bin-centre heights."""
        if self.model_type == 'lr':
            class_preds = self.model.predict(X).astype(int)
            return self._lr_bin_centers[class_preds]
        return self.model.predict(X)

    def evaluate(self, split_name='val', max_tiles=None):
        """Evaluate trained model on a split."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X, y = self.prepare_split_data(split_name, max_tiles=max_tiles)
        if X is None:
            return None

        y_pred = self._predict(X)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'n_samples': int(len(y)),
        }
        print(f"{split_name.capitalize()}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
        self.logger.info(f"'{split_name}' metrics: {metrics}")

        self._save_predictions(y, y_pred, split_name)
        self._plot_predictions(y, y_pred, split_name)
        return metrics

    def _save_predictions(self, y_true, y_pred, split_name):
        df = pd.DataFrame({'y_true': y_true.astype(np.float32), 'y_pred': y_pred.astype(np.float32)})
        out = self.output_dir / f'predictions_{split_name}.parquet'
        df.to_parquet(out, index=False, compression='snappy')
        print(f"Saved predictions: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        self.logger.info(f"Saved {split_name} predictions parquet: {out}")

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
        if self.model_type == 'lr':
            importances = np.mean(np.abs(self.model.coef_), axis=0)
            ylabel = 'Mean |Coefficient|'
            title = f'Top {top_n} Feature Coefficients (LR, {self.embedding_format})'
        else:
            importances = self.model.feature_importances_
            ylabel = 'Importance'
            title = f'Top {top_n} Feature Importances ({self.model_type.upper()}, {self.embedding_source}, {self.embedding_format})'
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Embedding Channel Index')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(range(top_n), indices, rotation=45)
        plt.tight_layout()
        zeros_tag = '_withzeros' if self.include_zeros else ''
        out = self.output_dir / f'feature_importance{zeros_tag}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance plot: {out}")

    def save_model(self, filename=None):
        if self.model is None:
            raise ValueError("Model not trained yet")
        if filename is None:
            zeros_tag = '_withzeros' if self.include_zeros else ''
            filename = f'gedi_{self.model_type}_30m_{self.embedding_source}_{self.embedding_format}{zeros_tag}.pkl'
        model_path = self.output_dir / filename
        payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'embedding_format': self.embedding_format,
            'embedding_source': self.embedding_source,
            'model_type': self.model_type,
            'include_zeros': self.include_zeros,
        }
        if self.model_type == 'lr':
            payload['lr_bin_edges'] = self._lr_bin_edges
            payload['lr_bin_centers'] = self._lr_bin_centers
        with open(model_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Saved model: {model_path}")
        self.logger.info(f"Saved model: {model_path}")

    def save_subset_tiles(self):
        """Save the sampled tile IDs to JSON so the experiment is reproducible."""
        if not self._subset_tile_ids:
            return
        out = self.output_dir / 'subset_tiles.json'
        payload = {
            'subset_fraction': self.subset_fraction,
            'subset_seed': self.subset_seed,
            'splits': {k: [int(v) for v in vs] for k, vs in self._subset_tile_ids.items()},
        }
        with open(out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Saved subset tile IDs: {out}")
        self.logger.info(f"Saved subset tile IDs: {out}")

    def _prepare_split_stratified_two_pass(self, tiles, split_name):
        """
        Stratified sampling for the preprocessed (.npz) path using two passes.

        Pass 1 — read only y per tile to build a global height index; never
                  touches X, so cost is proportional to n_pixels not n_features.
        Sample  — stratified selection of (tile, local_pixel_idx) pairs.
        Pass 2 — read X only for tiles that have selected pixels, extract rows.

        Peak memory: one tile's X at a time + the final selected rows.
        """
        bin_edges  = [0, 10, 20, 30, 40, 50, 60, np.inf]
        bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']

        # ----- Pass 1: scan y only -----
        print(f"  [Pass 1/{split_name}] Scanning heights across {len(tiles)} tiles...")
        pass1_start = time.time()

        tile_ids_per_pixel = []
        local_idxs_per_pixel = []
        y_parts = []

        for healpix_id in tqdm(tiles, desc=f"Pass 1 [{split_name}]"):
            cache_path = self._cache_path(healpix_id)
            if not cache_path.exists():
                self.logger.warning(f"Cache missing for tile {healpix_id}: {cache_path}")
                continue
            npz = np.load(cache_path)   # lazy: only loads arrays on access
            y_tile = npz['y']
            if len(y_tile) == 0:
                continue
            n = len(y_tile)
            tile_ids_per_pixel.append(np.full(n, healpix_id, dtype=np.int64))
            local_idxs_per_pixel.append(np.arange(n, dtype=np.int32))
            y_parts.append(y_tile)

        if not y_parts:
            print(f"Warning: No valid data for '{split_name}' split")
            self.logger.warning(f"No valid data for '{split_name}' split")
            return None, None

        all_tile_ids  = np.concatenate(tile_ids_per_pixel)
        all_local_idx = np.concatenate(local_idxs_per_pixel)
        all_y         = np.concatenate(y_parts)
        print(f"  Pass 1 done: {len(all_y):,} pixels in {time.time() - pass1_start:.1f}s")
        self.logger.info(f"[{split_name}] Pass 1: {len(all_y):,} pixels scanned")

        # ----- Stratified sampling: equalize bins by capping at smallest bin -----
        rng = np.random.default_rng(self.stratified_sample_seed)
        per_bin_before, per_bin_after = {}, {}

        bin_indices = {}
        for (lo, hi), label in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_labels):
            idx = np.where((all_y >= lo) & (all_y < hi))[0]
            per_bin_before[label] = len(idx)
            if len(idx) > 0:
                bin_indices[label] = idx

        cap = min(len(idx) for idx in bin_indices.values()) if bin_indices else 1
        selected = []
        for label in bin_labels:
            if label not in bin_indices:
                per_bin_after[label] = 0
                continue
            chosen = rng.choice(bin_indices[label], size=cap, replace=False)
            selected.append(chosen)
            per_bin_after[label] = len(chosen)

        if not selected:
            print(f"Warning: No samples selected for '{split_name}'")
            return None, None

        sel_global_idx = np.concatenate(selected)
        self._stratified_sample_info[split_name] = {
            'cap_per_bin': cap,
            'seed': self.stratified_sample_seed,
            'n_before': int(len(all_y)),
            'n_after': int(len(sel_global_idx)),
            'per_bin_before': per_bin_before,
            'per_bin_after': per_bin_after,
        }

        print(f"  Stratified sampling '{split_name}': {len(all_y):,} -> {len(sel_global_idx):,} samples (cap={cap:,}/bin)")
        for label in bin_labels:
            b = per_bin_before[label]
            a = per_bin_after.get(label, 0)
            print(f"    [{label:6s}] {b:>10,} -> {a:>10,}")
        self.logger.info(f"Stratified sample '{split_name}': {per_bin_before} -> {per_bin_after}")

        # Group selected indices by tile for sequential I/O in pass 2
        sel_tile_ids  = all_tile_ids[sel_global_idx]
        sel_local_idx = all_local_idx[sel_global_idx]
        sel_y         = all_y[sel_global_idx]

        sort_order    = np.argsort(sel_tile_ids, kind='stable')
        sel_tile_ids  = sel_tile_ids[sort_order]
        sel_local_idx = sel_local_idx[sort_order]
        sel_y         = sel_y[sort_order]

        # ----- Pass 2: load X only for selected rows -----
        unique_tiles, tile_starts = np.unique(sel_tile_ids, return_index=True)
        tile_ends = np.append(tile_starts[1:], len(sel_tile_ids))

        print(f"  [Pass 2/{split_name}] Loading X from {len(unique_tiles)} tiles...")
        pass2_start = time.time()

        X_list, y_list_out = [], []
        for tile_id, start, end in tqdm(
                zip(unique_tiles, tile_starts, tile_ends),
                total=len(unique_tiles),
                desc=f"Pass 2 [{split_name}]"):
            cache_path = self._cache_path(tile_id)
            npz = np.load(cache_path)
            X_tile = npz['X']
            if self.embedding_format == 'int8' and X_tile.dtype == np.int8:
                X_tile = X_tile.astype(np.float32) / 127.0
            local  = sel_local_idx[start:end]
            X_list.append(X_tile[local])
            y_list_out.append(sel_y[start:end])

        X = np.vstack(X_list)
        y = np.concatenate(y_list_out)
        print(f"  Pass 2 done: {X.shape[0]:,} samples, {X.shape[1]} features "
              f"in {time.time() - pass2_start:.1f}s")
        self.logger.info(
            f"[{split_name}] Pass 2: {X.shape[0]:,} samples from {len(unique_tiles)} tiles"
        )

        # Shuffle to remove tile-ordering bias before training
        shuffle_idx = np.random.default_rng(self.stratified_sample_seed + 1).permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        total_elapsed = time.time() - pass1_start
        print(f"  Samples: {X.shape[0]:,}, Features: {X.shape[1]}")
        print(f"  Height range: [{y.min():.2f}, {y.max():.2f}] m  mean={y.mean():.2f}")
        print(f"  Two-pass total: {total_elapsed:.0f}s")
        self.logger.info(
            f"[{split_name}] Two-pass stratified complete: {X.shape[0]:,} samples in {total_elapsed:.0f}s"
        )
        return X, y

    def _apply_stratified_sample(self, X, y, split_name):
        """Equalize bins by capping each 10 m height bin at the smallest bin's count."""
        bin_edges  = [0, 10, 20, 30, 40, 50, 60, np.inf]
        bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']
        rng = np.random.default_rng(self.stratified_sample_seed)

        per_bin_before, per_bin_after = {}, {}
        bin_indices = {}
        for (lo, hi), label in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_labels):
            idx = np.where((y >= lo) & (y < hi))[0]
            per_bin_before[label] = len(idx)
            if len(idx) > 0:
                bin_indices[label] = idx

        cap = min(len(idx) for idx in bin_indices.values()) if bin_indices else 1
        selected = []
        for label in bin_labels:
            if label not in bin_indices:
                per_bin_after[label] = 0
                continue
            chosen = rng.choice(bin_indices[label], size=cap, replace=False)
            selected.append(chosen)
            per_bin_after[label] = len(chosen)

        all_idx = np.concatenate(selected) if selected else np.array([], dtype=int)
        rng.shuffle(all_idx)

        self._stratified_sample_info[split_name] = {
            'cap_per_bin': cap,
            'seed': self.stratified_sample_seed,
            'n_before': int(len(y)),
            'n_after': int(len(all_idx)),
            'per_bin_before': per_bin_before,
            'per_bin_after': per_bin_after,
        }

        print(f"  Stratified sampling '{split_name}': {len(y):,} -> {len(all_idx):,} samples (cap={cap:,}/bin)")
        for label in bin_labels:
            b = per_bin_before[label]
            a = per_bin_after.get(label, 0)
            print(f"    [{label:6s}] {b:>10,} -> {a:>10,}")
        self.logger.info(
            f"Stratified sample '{split_name}': {per_bin_before} -> {per_bin_after}"
        )

        return X[all_idx], y[all_idx]

    def save_stratified_sample_info(self):
        """Write per-bin sample counts to JSON for experiment reproducibility."""
        if not self._stratified_sample_info:
            return
        out = self.output_dir / 'stratified_sample_info.json'
        with open(out, 'w') as f:
            json.dump(self._stratified_sample_info, f, indent=2)
        print(f"Saved stratified sample info: {out}")
        self.logger.info(f"Saved stratified sample info: {out}")

    def save_metrics(self, metrics, filename=None):
        if filename is None:
            zeros_tag = '_withzeros' if self.include_zeros else ''
            filename = f'metrics{zeros_tag}.json'
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
        description='Train ML model for GEDI height prediction using 30m OLMo embeddings'
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
    parser.add_argument('--embedding-source',
                        choices=['olmo', 'aef', 'tessera', 'presto', 'prithvi'], default='olmo',
                        help='Embedding source: olmo, aef, tessera, presto or prithvi '
                             '(default: olmo). '
                             'Used for output naming and metadata. When --year is provided '
                             'and --preprocessed-dir is omitted, auto-constructs path as '
                             '{embeddings-dir}/{year}/{embedding-source}/.')
    parser.add_argument('--model', choices=['rf', 'xgb', 'gb', 'lr'], default='rf',
                        help='Model architecture: rf=RandomForest, xgb=XGBoost, '
                             'gb=GradientBoosting, lr=LogisticRegression (binned) (default: rf)')
    # Shared tree params
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--min-samples-split', type=int, default=2)
    parser.add_argument('--random-state', type=int, default=42)
    # XGBoost / GradientBoosting
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate for xgb or gb (default: 0.1)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Row subsample ratio for xgb or gb (default: 1.0)')
    # Logistic Regression
    parser.add_argument('--lr-C', type=float, default=1.0,
                        help='Inverse regularisation strength for LogisticRegression (default: 1.0)')
    parser.add_argument('--lr-max-iter', type=int, default=1000,
                        help='Max solver iterations for LogisticRegression (default: 1000)')
    parser.add_argument('--lr-bin-size', type=float, default=5.0,
                        help='Height bin size in metres for LR discretisation (default: 5.0)')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Limit tiles per split (for pipeline testing)')
    parser.add_argument('--year', type=int, default=None,
                        help='Year of the GEDI data query (e.g. 2021). When --preprocessed-dir '
                             'is not set, auto-constructs it as '
                             '{embeddings-dir}/{year}/gedi_cache_30m_{embedding-format}.')
    parser.add_argument('--preprocessed-dir', default=None,
                        help='Path to .npz cache from preprocess_olmo_to_gedi_tiles.py. '
                             'When set, skips zarr/rasterio loading entirely. '
                             'If omitted and --year is provided, the path is auto-constructed.')
    parser.add_argument('--subset-fraction', type=float, default=None,
                        help='Randomly sample this fraction (0, 1] of tiles per split. '
                             'Selected tile IDs are saved to subset_tiles.json.')
    parser.add_argument('--subset-seed', type=int, default=42,
                        help='Random seed for subset tile sampling (default: 42)')
    parser.add_argument('--subset-splits', nargs='+', default=['train'],
                        metavar='SPLIT',
                        help='Which splits to apply tile subsetting to (default: train)')
    parser.add_argument('--stratified-sample-fraction', type=float, default=None,
                        help='Enable equalized stratified sampling across 10 m height bins '
                             '(0–10, 10–20, …, 60+). Each bin is capped at the smallest '
                             'bin\'s pixel count so all bins contribute equally. Applied to '
                             'splits in --stratified-splits. Pass any non-zero value to '
                             'activate (the value itself is ignored). Per-bin counts saved '
                             'to stratified_sample_info.json.')
    parser.add_argument('--stratified-sample-seed', type=int, default=42,
                        help='Random seed for stratified pixel sampling (default: 42)')
    parser.add_argument('--stratified-splits', nargs='+', default=['train'],
                        metavar='SPLIT',
                        help='Which splits to apply stratified sampling to (default: train)')
    parser.add_argument('--tile-list', default=None, metavar='PATH',
                        help='File of healpix ids (one per line) restricting EVERY split. '
                             'Use the master list from build_master_tile_list.py so all '
                             'embedding sources train and evaluate on identical tiles — '
                             'otherwise a source with fewer cached tiles is scored on a '
                             'different dataset and the comparison is confounded.')
    parser.add_argument('--include-zeros', action='store_true',
                        help='Include GEDI height == 0 measurements in the analysis '
                             '(gedi_data >= 0). For the preprocessed path this loads the '
                             'healpix_*_{format}_withzeros.npz cache files written by '
                             'alphaearth_processing.py --keep-zeros. Default excludes zeros '
                             'and loads the standard zero-filtered cache.')

    args = parser.parse_args()

    if args.preprocessed_dir is None and args.year is not None:
        args.preprocessed_dir = str(
            Path(args.embeddings_dir) / str(args.year) / args.embedding_source
        )
        print(f"Auto-constructed preprocessed-dir: {args.preprocessed_dir}")

    tile_allowlist = None
    if args.tile_list:
        p = Path(args.tile_list)
        if not p.is_file():
            parser.error(f"--tile-list not found: {p}")
        tile_allowlist = [int(ln) for ln in p.read_text().split()
                          if ln.strip() and not ln.strip().startswith('#')]
        if not tile_allowlist:
            parser.error(f"--tile-list is empty: {p}")
        print(f"Tile allowlist: {len(tile_allowlist)} ids from {p}")

    predictor = GEDIPredictor30m(
        shapefile_path=args.shapefile,
        gedi_dir=args.gedi_dir,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        embedding_format=args.embedding_format,
        preprocessed_dir=args.preprocessed_dir,
        subset_fraction=args.subset_fraction,
        subset_seed=args.subset_seed,
        subset_splits=args.subset_splits,
        stratified_sample_fraction=args.stratified_sample_fraction,
        stratified_sample_seed=args.stratified_sample_seed,
        stratified_splits=args.stratified_splits,
        model_type=args.model,
        embedding_source=args.embedding_source,
        include_zeros=args.include_zeros,
        tile_allowlist=tile_allowlist,
    )

    train_metrics = predictor.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        max_tiles=args.max_tiles,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        lr_C=args.lr_C,
        lr_max_iter=args.lr_max_iter,
        lr_bin_size=args.lr_bin_size,
    )

    val_metrics = predictor.evaluate('val', max_tiles=args.max_tiles)
    test_metrics = predictor.evaluate('test', max_tiles=args.max_tiles)

    all_metrics = {
        'train': train_metrics['train'],
        'val': val_metrics,
        'test': test_metrics,
        'embedding_format': args.embedding_format,
        'embedding_source': args.embedding_source,
        'model': args.model,
        'year': args.year,
        'include_zeros': args.include_zeros,
        # Recorded so a multi-seed study can group runs by seed after the fact.
        'random_state': args.random_state,
        'subset_seed': args.subset_seed,
        'stratified_sample_seed': args.stratified_sample_seed,
        'tile_list': args.tile_list,
        'n_tiles_allowed': len(tile_allowlist) if tile_allowlist else None,
    }

    predictor.plot_feature_importance(top_n=30)
    predictor.print_missing_summary()
    predictor.save_model()
    predictor.save_metrics(all_metrics)
    predictor.save_subset_tiles()
    predictor.save_stratified_sample_info()

    print("\n" + "=" * 50)
    print(f"Complete!  model={args.model.upper()}  source={args.embedding_source}  format={args.embedding_format}  year={args.year}")
    print(f"  Train R²: {train_metrics['train']['r2']:.3f}")
    if val_metrics:
        print(f"  Val   R²: {val_metrics['r2']:.3f}")
    if test_metrics:
        print(f"  Test  R²: {test_metrics['r2']:.3f}")
    print(f"Results: {args.output_dir}")


if __name__ == '__main__':
    main()
