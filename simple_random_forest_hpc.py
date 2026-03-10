"""
Memory-Efficient Random Forest for HPC

This version processes tiles in batches to avoid OOM errors.
Designed for large datasets on HPC clusters.
"""

import numpy as np
from pathlib import Path
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import argparse
import gc
import os


def get_args():
    parser = argparse.ArgumentParser(description='Tiling results into maps')
    parser.add_argument('--root_dir', help='Root directory of data')
    parser.add_argument('--save_dir', help='Directory to save data')
    return parser.parse_args()


def get_tile_list_with_coords(input_dir, target_dir, load_all_folders=False):
    """
    Get list of tile paths with their coordinates without loading data.
    Memory efficient.
    """
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)
    
    # Get all files
    if load_all_folders:
        input_files = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                input_files.extend(list(subdir.glob("*.npy")))
        input_files = sorted(input_files)
    else:
        input_files = sorted(list(input_dir.glob("*.npy")))
    
    print(f"Found {len(input_files)} embedding files")
    
    # Get tile info without loading full data
    tile_info = []
    all_lons = []
    
    for input_file in input_files:
        tile_name = input_file.stem
        
        if load_all_folders:
            alpha_subdir = input_file.parent.name
            target_file = target_dir / alpha_subdir / f"{tile_name}.tif"
        else:
            target_file = target_dir / f"{tile_name}.tif"
        
        if not target_file.exists():
            continue
        
        # Just open to get geotransform, don't load data
        target_ds = gdal.Open(str(target_file))
        gt = target_ds.GetGeoTransform()
        
        if gt is None:
            target_ds = None
            continue
        
        width = target_ds.RasterXSize
        height = target_ds.RasterYSize
        center_lon = gt[0] + (width / 2.0) * gt[1]
        
        target_ds = None
        
        all_lons.append(center_lon)
        tile_info.append({
            'input_path': input_file,
            'target_path': target_file,
            'center_lon': center_lon
        })
    
    return tile_info, all_lons


def split_tiles_by_longitude(tile_info, all_lons, split_lon=None):
    """Split tiles into train/val/test by longitude."""
    if split_lon is None:
        split_lon = np.median(all_lons)
    
    print(f"\nGeographic split:")
    print(f"  Split longitude: {split_lon:.4f}°")
    print(f"  Longitude range: [{min(all_lons):.4f}°, {max(all_lons):.4f}°]")
    
    west_tiles = [t for t in tile_info if t['center_lon'] < split_lon]
    east_tiles = [t for t in tile_info if t['center_lon'] >= split_lon]
    
    print(f"  West tiles: {len(west_tiles)}")
    print(f"  East tiles: {len(east_tiles)}")
    
    # Split west into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(west_tiles))
    train_size = int(0.8 * len(west_tiles))
    
    train_tiles = [west_tiles[i] for i in indices[:train_size]]
    val_tiles = [west_tiles[i] for i in indices[train_size:]]
    test_tiles = east_tiles
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_tiles)} tiles")
    print(f"  Val:   {len(val_tiles)} tiles")
    print(f"  Test:  {len(test_tiles)} tiles")
    
    return train_tiles, val_tiles, test_tiles


def load_tiles_batch(tile_list, batch_size=100):
    """
    Generator to load tiles in batches.
    Yields batches of (X, y) without keeping everything in memory.
    """
    for i in range(0, len(tile_list), batch_size):
        batch = tile_list[i:i+batch_size]
        
        X_batch = []
        y_batch = []
        
        for tile in batch:
            # Load embedding
            embedding = np.load(tile['input_path']).astype(np.float32)
            
            # Load target
            target_ds = gdal.Open(str(tile['target_path']))
            target = target_ds.ReadAsArray().astype(np.float32)
            target_ds = None
            
            if len(target.shape) == 3:
                target = target[0]
            
            # Flatten
            X_batch.append(embedding.flatten())
            y_batch.append(target.flatten())
        
        yield np.array(X_batch), np.array(y_batch)
        
        # Free memory
        del X_batch, y_batch
        gc.collect()


def train_random_forest_incremental(train_tiles, val_tiles, 
                                    n_estimators=100, 
                                    max_depth=20,
                                    batch_size=100,
                                    n_jobs=-1):
    """
    Train Random Forest using batched loading.
    
    Note: sklearn RandomForest doesn't support true incremental learning,
    so we still need to load all training data, but we do it in batches
    to control memory usage.
    """
    print(f"\nLoading training data in batches (batch_size={batch_size})...")
    
    X_train_list = []
    y_train_list = []
    
    # Load training data in batches
    for batch_idx, (X_batch, y_batch) in enumerate(load_tiles_batch(train_tiles, batch_size)):
        print(f"  Loaded batch {batch_idx+1}/{(len(train_tiles)-1)//batch_size + 1}")
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    
    print("  Concatenating batches...")
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    
    # Free memory
    del X_train_list, y_train_list
    gc.collect()
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Training Random Forest...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1,
        max_samples=0.8  # Use 80% of samples per tree to reduce memory
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model


def evaluate_batched(model, tile_list, split_name="", batch_size=100):
    """Evaluate model in batches to save memory."""
    print(f"\n{split_name} Evaluation (batched):")
    
    all_predictions = []
    all_targets = []
    
    for batch_idx, (X_batch, y_batch) in enumerate(load_tiles_batch(tile_list, batch_size)):
        pred_batch = model.predict(X_batch)
        all_predictions.append(pred_batch)
        all_targets.append(y_batch)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx+1}/{(len(tile_list)-1)//batch_size + 1} batches")
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return predictions, mse, rmse, mae, r2


def main():
    args = get_args()
    root_dir = args.root_dir
    save_dir = args.save_dir
    INPUT_DIR = "{}/input_tiles".format(root_dir)
    TARGET_DIR = "{}/target_tiles".format(root_dir)
    LOAD_ALL_FOLDERS = True

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Setup paths
    root_dir = Path(args.root_dir)
    input_dir = root_dir / "input_tiles"
    target_dir = root_dir / "target_tiles"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Memory-Efficient Random Forest for HPC")
    print("="*70)
    print(f"Root dir: {root_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    
    # 1. Get tile list (memory efficient)
    print("\n" + "="*70)
    print("Step 1: Getting tile list")
    print("="*70)
    
    tile_info, all_lons = get_tile_list_with_coords(
        input_dir, target_dir, load_all_folders=True
    )
    
    # 2. Split by geography
    print("\n" + "="*70)
    print("Step 2: Splitting by geography")
    print("="*70)
    
    train_tiles, val_tiles, test_tiles = split_tiles_by_longitude(
        tile_info, all_lons, split_lon=args.split_lon
    )
    
    # 3. Train model
    print("\n" + "="*70)
    print("Step 3: Training Random Forest")
    print("="*70)
    
    model = train_random_forest_incremental(
        train_tiles, val_tiles,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs
    )
    
    # 4. Evaluate
    print("\n" + "="*70)
    print("Step 4: Evaluation")
    print("="*70)
    
    train_preds, train_mse, train_rmse, train_mae, train_r2 = evaluate_batched(
        model, train_tiles, "TRAIN", batch_size=args.batch_size
    )
    
    val_preds, val_mse, val_rmse, val_mae, val_r2 = evaluate_batched(
        model, val_tiles, "VALIDATION", batch_size=args.batch_size
    )
    
    test_preds, test_mse, test_rmse, test_mae, test_r2 = evaluate_batched(
        model, test_tiles, "TEST (East)", batch_size=args.batch_size
    )
    
    # 5. Save results
    print("\n" + "="*70)
    print("Step 5: Saving results")
    print("="*70)
    
    model_path = save_dir / "random_forest_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save predictions
    np.save(save_dir / "train_predictions.npy", train_preds)
    np.save(save_dir / "val_predictions.npy", val_preds)
    np.save(save_dir / "test_predictions.npy", test_preds)
    print(f"✓ Predictions saved")
    
    # Save metrics
    metrics = {
        'train': {'mse': train_mse, 'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
        'val': {'mse': val_mse, 'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2},
        'test': {'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
    }
    
    with open(save_dir / "metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Val R²:   {val_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
