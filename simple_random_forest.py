"""
Super Simple Random Forest Regressor for Alpha Earth + GEDI

Splits tiles by geography (East vs West):
- West tiles: Train + Val
- East tiles: Test

Flattens embeddings and targets, trains sklearn RandomForestRegressor.
"""

import numpy as np
from pathlib import Path
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


def load_all_alpha_files(base_input_dir, base_target_dir, split_lon=None):
    """
    Convenience function to load all alpha earth files.
    Just point to the base directories!
    
    Args:
        base_input_dir: Base input directory with alpha earth subdirectories
        base_target_dir: Base target directory with GEDI subdirectories  
        split_lon: Longitude to split at (West < split_lon, East >= split_lon)
                   If None, uses midpoint
    """
    return load_and_flatten_data(
        input_dir=base_input_dir,
        target_dir=base_target_dir,
        split_lon=split_lon,
        load_all_folders=True
    )


def get_tile_coords_from_name(tile_name):
    """
    Extract tile coordinates from filename.
    Example: 'tile_0000_0001.npy' -> (0, 1)
    """
    parts = tile_name.split('_')
    row = int(parts[1])
    col = int(parts[2].split('.')[0])
    return row, col


def load_and_flatten_data(input_dir, target_dir, split_by='east_west', split_lon=None, load_all_folders=False):
    """
    Load all tiles, flatten them, and split by geography.
    
    Args:
        input_dir: Directory with .npy embedding files (or base directory if load_all_folders=True)
        target_dir: Directory with .tif target files (or base directory if load_all_folders=True)
        split_by: 'east_west' or 'manual'
        split_lon: Longitude to split at (West < split_lon, East >= split_lon). 
                   If None, uses midpoint between bounds.
        load_all_folders: If True, load from all subdirectories (all alpha earth files)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)
    
    # Get all tiles
    if load_all_folders:
        # Load from all subdirectories
        input_files = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                input_files.extend(list(subdir.glob("*.npy")))
        input_files = sorted(input_files)
        print(f"Loading from {len(list(input_dir.iterdir()))} alpha earth folders")
    else:
        # Load from single directory
        input_files = sorted(list(input_dir.glob("*.npy")))
    
    print(f"Found {len(input_files)} embedding files")
    
    # Load all data
    west_X = []
    west_y = []
    east_X = []
    east_y = []
    
    all_lons = []
    skipped_no_target = 0
    skipped_no_geo = 0
    
    for input_file in input_files:
        # Load embedding [64, 224, 224]
        embedding = np.load(input_file).astype(np.float32)
        
        # Load target
        tile_name = input_file.stem
        
        if load_all_folders:
            # Get the alpha file subdirectory name
            alpha_subdir = input_file.parent.name
            target_file = target_dir / alpha_subdir / f"{tile_name}.tif"
        else:
            target_file = target_dir / f"{tile_name}.tif"
        
        if not target_file.exists():
            skipped_no_target += 1
            continue
        
        # Open target to get geotransform
        target_ds = gdal.Open(str(target_file))
        target = target_ds.ReadAsArray().astype(np.float32)
        
        # Get geographic coordinates (center of tile)
        gt = target_ds.GetGeoTransform()
        if gt is None:
            skipped_no_geo += 1
            target_ds = None
            continue
            
        # Calculate center coordinates
        # gt[0] = top left x (longitude)
        # gt[3] = top left y (latitude)
        # gt[1] = pixel width
        # gt[5] = pixel height (negative)
        width = target_ds.RasterXSize
        height = target_ds.RasterYSize
        
        center_lon = gt[0] + (width / 2.0) * gt[1]
        center_lat = gt[3] + (height / 2.0) * gt[5]
        
        target_ds = None
        
        all_lons.append(center_lon)
        
        # Flatten
        # Embedding: [64, 224, 224] -> [64 * 224 * 224]
        X_flat = embedding.flatten()
        
        # Target: [224, 224] -> [224 * 224] or [1, 224, 224] -> [224 * 224]
        if len(target.shape) == 3:
            target = target[0]  # Take first channel if multi-channel
        y_flat = target.flatten()
        
        # Determine split longitude
        if split_lon is None:
            # Will be calculated after loading all tiles
            pass
        
        # Store with longitude for later splitting
        if split_lon is None:
            # Temporarily store all data
            west_X.append((center_lon, X_flat))
            west_y.append((center_lon, y_flat))
        else:
            # Split immediately if split_lon is provided
            if center_lon < split_lon:
                west_X.append(X_flat)
                west_y.append(y_flat)
            else:
                east_X.append(X_flat)
                east_y.append(y_flat)
    
    if skipped_no_target > 0:
        print(f"Warning: Skipped {skipped_no_target} tiles (no matching target)")
    if skipped_no_geo > 0:
        print(f"Warning: Skipped {skipped_no_geo} tiles (no geotransform)")
    
    # Calculate split longitude if not provided
    if split_lon is None:
        split_lon = np.median(all_lons)
        print(f"\nAuto-calculated split longitude: {split_lon:.4f}°")
        
        # Now do the actual split
        temp_west_X = west_X
        temp_west_y = west_y
        west_X = []
        west_y = []
        east_X = []
        east_y = []
        
        for (lon, X_flat) in temp_west_X:
            if lon < split_lon:
                west_X.append(X_flat)
            else:
                east_X.append(X_flat)
        
        for (lon, y_flat) in temp_west_y:
            if lon < split_lon:
                west_y.append(y_flat)
            else:
                east_y.append(y_flat)
    
    print(f"\nGeographic split:")
    print(f"  Split longitude: {split_lon:.4f}°")
    print(f"  Longitude range: [{min(all_lons):.4f}°, {max(all_lons):.4f}°]")
    print(f"  West tiles (lon < {split_lon:.4f}°): {len(west_X)}")
    print(f"  East tiles (lon >= {split_lon:.4f}°): {len(east_X)}")
    
    # Convert to arrays
    west_X = np.array(west_X)
    west_y = np.array(west_y)
    east_X = np.array(east_X)
    east_y = np.array(east_y)
    
    print(f"\nData shapes:")
    print(f"  West X: {west_X.shape}")
    print(f"  West y: {west_y.shape}")
    print(f"  East X: {east_X.shape}")
    print(f"  East y: {east_y.shape}")
    
    # Split west into train and val (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        west_X, west_y, test_size=0.2, random_state=42
    )
    
    # East becomes test
    X_test = east_X
    y_test = east_y
    
    print(f"\nFinal split:")
    print(f"  Train: {X_train.shape[0]} tiles (West)")
    print(f"  Val:   {X_val.shape[0]} tiles (West)")
    print(f"  Test:  {X_test.shape[0]} tiles (East)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, n_jobs=-1):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train: Training features [n_samples, n_features]
        y_train: Training targets [n_samples, n_outputs]
        n_estimators: Number of trees
        max_depth: Maximum depth of trees (None = unlimited)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        
    Returns:
        Trained model
    """
    print(f"\nTraining Random Forest...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  n_jobs: {n_jobs}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model


def evaluate(model, X, y, split_name=""):
    """Evaluate model and print metrics."""
    print(f"\n{split_name} Evaluation:")
    
    predictions = model.predict(X)
    
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return predictions, mse, rmse, mae, r2


def save_model(model, filepath="random_forest_model.pkl"):
    """Save trained model."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {filepath}")


def load_model(filepath="random_forest_model.pkl"):
    """Load trained model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {filepath}")
    return model


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    
    # OPTION 1: Load from a single alpha earth file
    # INPUT_DIR = "tiles_output/input_tiles/your_alpha_file"
    # TARGET_DIR = "tiles_output/target_tiles/your_alpha_file"
    # LOAD_ALL_FOLDERS = False
    
    # OPTION 2: Load from ALL alpha earth files (RECOMMENDED)
    INPUT_DIR = "tiles_output/input_tiles"
    TARGET_DIR = "tiles_output/target_tiles"
    LOAD_ALL_FOLDERS = True
    
    # Geographic bounds of your data:
    # 41.3162° N, 120.1156° W (East) to 41.2778° N, 123.6772° W (West)
    # Longitude ranges from -123.6772 (West) to -120.1156 (East)
    
    # Split longitude: tiles west of this are train/val, tiles east are test
    # Default: None (auto-calculate midpoint = -121.8964°)
    # Or specify manually, e.g., -122.0 to have more tiles in train
    SPLIT_LONGITUDE = None  # Auto-calculate midpoint
    # SPLIT_LONGITUDE = -122.0  # Or set manually
    
    print("="*70)
    print("Random Forest Regressor: Alpha Earth -> GEDI")
    print("="*70)
    print(f"Data bounds: 41.28° to 41.32° N, -123.68° to -120.12° W")
    
    # 1. Load and split data (East vs West by longitude)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_flatten_data(
        input_dir=INPUT_DIR,
        target_dir=TARGET_DIR,
        split_lon=SPLIT_LONGITUDE,
        load_all_folders=LOAD_ALL_FOLDERS
    )
    
    # 2. Train Random Forest
    model = train_random_forest(
        X_train, y_train,
        n_estimators=100,      # Number of trees (increase for better performance)
        max_depth=20,          # Max depth (None = unlimited, may overfit)
        n_jobs=-1              # Use all CPU cores
    )
    
    # 3. Evaluate on all splits
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    
    train_preds, train_mse, train_rmse, train_mae, train_r2 = evaluate(
        model, X_train, y_train, "TRAIN"
    )
    
    val_preds, val_mse, val_rmse, val_mae, val_r2 = evaluate(
        model, X_val, y_val, "VALIDATION"
    )
    
    test_preds, test_mse, test_rmse, test_mae, test_r2 = evaluate(
        model, X_test, y_test, "TEST (East)"
    )
    
    # 4. Save model
    save_model(model, "random_forest_model.pkl")
    
    # 5. Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Val R²:   {val_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print("\n✓ Done!")
    
    # Optional: Save predictions
    print("\nSaving predictions...")
    np.save("train_predictions.npy", train_preds)
    np.save("val_predictions.npy", val_preds)
    np.save("test_predictions.npy", test_preds)
    print("✓ Predictions saved")
