"""
Hybrid Random Forest: Train on Sparse GEDI, Predict Wall-to-Wall

This combines the best of both worlds:
1. TRAINING: Only on pixels with GEDI measurements (fast, accurate)
2. PREDICTION: On ALL pixels including NoData areas (wall-to-wall maps)

Result: Fast training + complete spatial coverage!
"""

import numpy as np
from pathlib import Path
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import argparse
import sys
import json
import wandb


def read_geotiff(filepath):
    """Read GeoTIFF and return array."""
    ds = gdal.Open(str(filepath))
    data = ds.ReadAsArray().astype(np.float32)
    ds = None
    return data


def write_geotiff(data, filepath, reference_file=None):
    """Write array as GeoTIFF with optional georeference from reference file."""
    driver = gdal.GetDriverByName('GTiff')

    if len(data.shape) == 2:
        height, width = data.shape
        n_bands = 1
    else:
        n_bands, height, width = data.shape

    out_ds = driver.Create(
        str(filepath),
        width,
        height,
        n_bands,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW']
    )

    # Copy georeference if available
    if reference_file:
        ref_ds = gdal.Open(str(reference_file))
        if ref_ds:
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            ref_ds = None

    # Write data
    if n_bands == 1:
        out_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(n_bands):
            out_ds.GetRasterBand(i + 1).WriteArray(data[i])

    out_ds.FlushCache()
    out_ds = None


def print_flush(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


def get_tile_info(input_dir, target_dir, load_all_folders=True):
    """Get tile list with geographic info."""
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    if load_all_folders:
        input_files = []
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                input_files.extend(list(subdir.glob("*.npy")))
    else:
        input_files = sorted(list(input_dir.glob("*.npy")))

    print_flush(f"Found {len(input_files)} embedding files")

    tile_info = []
    all_lons = []
    skipped = 0

    for input_file in input_files:
        tile_name = input_file.stem

        if load_all_folders:
            alpha_subdir = input_file.parent.name
            target_file = target_dir / alpha_subdir / f"{tile_name}.tif"
        else:
            target_file = target_dir / f"{tile_name}.tif"

        if not target_file.exists():
            skipped += 1
            continue

        # Get basic info without loading full data
        target_data = read_geotiff(target_file)
        valid_pixels = ((target_data > 0) & (~np.isnan(target_data))).sum()
        coverage = valid_pixels / target_data.size * 100

        # Placeholder for longitude (can extract from geotransform if needed)
        center_lon = 0.0

        all_lons.append(center_lon)
        tile_info.append({
            'input_file': input_file,
            'target_file': target_file,
            'center_lon': center_lon,
            'coverage': coverage,
            'has_data': valid_pixels > 0
        })

    if skipped > 0:
        print_flush(f"Skipped {skipped} files without matching targets")

    return tile_info, all_lons


def extract_training_pixels(tiles, min_coverage=0.01):
    """
    Extract only valid pixels for TRAINING.
    Skip tiles with very low coverage for training efficiency.
    """
    X_pixels = []
    y_pixels = []
    tiles_used = 0

    print_flush(f"\nExtracting training pixels (min coverage: {min_coverage}%)...")

    for i, tile in enumerate(tiles):
        if (i + 1) % 100 == 0:
            print_flush(f"  Processed {i + 1}/{len(tiles)} tiles...")

        # Skip very sparse tiles for training
        if tile['coverage'] < min_coverage:
            continue

        # Load data
        embedding = np.load(tile['input_file']).astype(np.float32)  # [64, 224, 224]
        target = read_geotiff(tile['target_file'])  # [224, 224]

        # Find valid pixels
        valid_mask = (target > 0) & (~np.isnan(target))

        if not valid_mask.any():
            continue

        # Extract features for valid pixels
        embedding_2d = np.transpose(embedding, (1, 2, 0))  # [224, 224, 64]

        X_valid = embedding_2d[valid_mask]  # [n_valid, 64]
        y_valid = target[valid_mask]  # [n_valid]

        X_pixels.append(X_valid)
        y_pixels.append(y_valid)
        tiles_used += 1

    print_flush(f"  Used {tiles_used} tiles with >{min_coverage}% coverage")

    if len(X_pixels) == 0:
        return np.array([]), np.array([])

    X = np.vstack(X_pixels)
    y = np.concatenate(y_pixels)

    print_flush(f"  Extracted {len(y):,} valid pixels for training")

    return X, y


def predict_wall_to_wall(model, tiles, output_dir, split_name):
    """
    Predict on ALL pixels (including NoData areas) to create wall-to-wall maps.

    This creates complete spatial coverage, even where GEDI has no measurements.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_flush(f"\n{split_name}: Predicting wall-to-wall on {len(tiles)} tiles...")

    predictions_flat = []
    targets_flat = []

    for i, tile in enumerate(tiles):
        if (i + 1) % 100 == 0:
            print_flush(f"  Predicted {i + 1}/{len(tiles)} tiles...")

        # Load embedding
        embedding = np.load(tile['input_file']).astype(np.float32)  # [64, 224, 224]

        # Load target
        target = read_geotiff(tile['target_file'])  # [224, 224]

        # Reshape embedding for prediction
        embedding_2d = np.transpose(embedding, (1, 2, 0))  # [224, 224, 64]
        X_all_pixels = embedding_2d.reshape(-1, 64)  # [50176, 64]

        # Predict on ALL pixels (including where GEDI = 0)
        predictions = model.predict(X_all_pixels)  # [50176]

        # Reshape back to 2D
        prediction_map = predictions.reshape(224, 224)

        # Save as GeoTIFF (preserves georeference)
        tile_name = tile['input_file'].stem
        output_path = output_dir / f"{tile_name}_prediction.tif"
        write_geotiff(prediction_map.astype(np.float32), output_path, tile['target_file'])

        # For evaluation: only compare where GEDI has data
        valid_mask = (target > 0) & (~np.isnan(target))
        if valid_mask.any():
            predictions_flat.append(predictions[valid_mask.flatten()])
            targets_flat.append(target[valid_mask])

    print_flush(f"  ✓ Saved {len(tiles)} wall-to-wall prediction maps")

    # Concatenate for evaluation
    if len(predictions_flat) > 0:
        all_preds = np.concatenate(predictions_flat)
        all_targets = np.concatenate(targets_flat)
        return all_preds, all_targets
    else:
        return np.array([]), np.array([])


def evaluate(predictions, targets, split_name=""):
    """Evaluate predictions against ground truth."""
    if len(predictions) == 0:
        print_flush(f"\n{split_name}: No valid data for evaluation")
        return None, None, None, None, None

    print_flush(f"\n{split_name} Evaluation (on pixels with GEDI data):")

    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print_flush(f"  Evaluated on {len(predictions):,} pixels with GEDI measurements")
    print_flush(f"  MSE:  {mse:.4f}")
    print_flush(f"  RMSE: {rmse:.4f}")
    print_flush(f"  MAE:  {mae:.4f}")
    print_flush(f"  R²:   {r2:.4f}")

    return predictions, mse, rmse, mae, r2


def main():
    parser = argparse.ArgumentParser(
        description='Train on sparse GEDI, predict wall-to-wall'
    )
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--min_coverage', type=float, default=0.1,
                        help='Skip tiles with <X%% GEDI coverage for training')
    parser.add_argument('--split_lon', type=float, default=None)

    args = parser.parse_args()

    # Setup
    root_dir = Path(args.root_dir)
    input_dir = root_dir / "input_tiles"
    target_dir = root_dir / "target_tiles"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    experiment = wandb.init(project=args.wandb_project,
                            dir='{}/tmp/wandb'.format(root_dir),
                            resume='allow',
                            anonymous='must')


    print_flush("=" * 70)
    print_flush("Hybrid RF: Train Sparse, Predict Wall-to-Wall")
    print_flush("=" * 70)
    print_flush(f"Root dir: {root_dir}")
    print_flush(f"Min training coverage: {args.min_coverage}%")
    print_flush(f"n_estimators: {args.n_estimators}")
    print_flush(f"max_depth: {args.max_depth}")

    # Get all tiles
    print_flush("\n" + "=" * 70)
    print_flush("Step 1: Loading tile information")
    print_flush("=" * 70)

    tile_info, all_lons = get_tile_info(input_dir, target_dir, load_all_folders=True)

    # Split by geography
    print_flush("\n" + "=" * 70)
    print_flush("Step 2: Splitting tiles by geography")
    print_flush("=" * 70)

    if args.split_lon is None:
        split_lon = np.median(all_lons) if all_lons else 0.0
    else:
        split_lon = args.split_lon

    print_flush(f"Split longitude: {split_lon:.4f}°")

    west_tiles = [t for t in tile_info if t['center_lon'] < split_lon]
    east_tiles = [t for t in tile_info if t['center_lon'] >= split_lon]

    # Split west into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(west_tiles))
    train_size = int(0.8 * len(west_tiles))

    train_tiles = [west_tiles[i] for i in indices[:train_size]]
    val_tiles = [west_tiles[i] for i in indices[train_size:]]
    test_tiles = east_tiles

    print_flush(f"\nTile counts:")
    print_flush(f"  Train: {len(train_tiles)} tiles")
    print_flush(f"  Val:   {len(val_tiles)} tiles")
    print_flush(f"  Test:  {len(test_tiles)} tiles")

    # Extract training pixels (sparse - only where GEDI has data)
    print_flush("\n" + "=" * 70)
    print_flush("Step 3: Extracting training pixels (SPARSE)")
    print_flush("=" * 70)

    X_train, y_train = extract_training_pixels(train_tiles, args.min_coverage)
    X_val, y_val = extract_training_pixels(val_tiles, args.min_coverage)

    if len(X_train) == 0:
        print_flush("ERROR: No training data extracted!")
        return

    # Train model
    print_flush("\n" + "=" * 70)
    print_flush("Step 4: Training Random Forest")
    print_flush("=" * 70)
    print_flush(f"Training on {len(y_train):,} pixels with GEDI measurements")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        random_state=42,
        verbose=1
    )

    model.fit(X_train, y_train)
    print_flush("✓ Training complete!")

    # Predict wall-to-wall (on ALL pixels)
    print_flush("\n" + "=" * 70)
    print_flush("Step 5: Predicting WALL-TO-WALL on all pixels")
    print_flush("=" * 70)
    print_flush("(This creates complete maps, even where GEDI = 0)")

    train_preds, train_targets = predict_wall_to_wall(
        model, train_tiles, save_dir / "train_predictions", "TRAIN"
    )

    val_preds, val_targets = predict_wall_to_wall(
        model, val_tiles, save_dir / "val_predictions", "VAL"
    )

    test_preds, test_targets = predict_wall_to_wall(
        model, test_tiles, save_dir / "test_predictions", "TEST"
    )

    # Evaluate
    print_flush("\n" + "=" * 70)
    print_flush("Step 6: Evaluation")
    print_flush("=" * 70)

    _, train_mse, train_rmse, train_mae, train_r2 = evaluate(
        train_preds, train_targets, "TRAIN"
    )

    _, val_mse, val_rmse, val_mae, val_r2 = evaluate(
        val_preds, val_targets, "VALIDATION"
    )

    _, test_mse, test_rmse, test_mae, test_r2 = evaluate(
        test_preds, test_targets, "TEST"
    )

    # Save
    print_flush("\n" + "=" * 70)
    print_flush("Step 7: Saving")
    print_flush("=" * 70)

    model_path = save_dir / "hybrid_rf_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print_flush(f"✓ Model: {model_path}")

    metrics = {
        'train': {'mse': float(train_mse) if train_mse else None,
                  'rmse': float(train_rmse) if train_rmse else None,
                  'mae': float(train_mae) if train_mae else None,
                  'r2': float(train_r2) if train_r2 else None},
        'val': {'mse': float(val_mse) if val_mse else None,
                'rmse': float(val_rmse) if val_rmse else None,
                'mae': float(val_mae) if val_mae else None,
                'r2': float(val_r2) if val_r2 else None},
        'test': {'mse': float(test_mse) if test_mse else None,
                 'rmse': float(test_rmse) if test_rmse else None,
                 'mae': float(test_mae) if test_mae else None,
                 'r2': float(test_r2) if test_r2 else None}
    }

    experiment.log({
        "train_mse": float(train_mse) if train_mse else None,
        "train_rmse": float(train_rmse) if train_rmse else None,
        "train_mae": float(train_mae) if train_mae else None,
        "train_r2": float(train_r2) if train_r2 else None,
        "val_mae": float(val_mae) if val_mae else None,
        "val_r2": float(val_r2) if val_r2 else None,
        "val_rmse": float(val_rmse) if val_rmse else None,
        "val_mse": float(val_mse) if val_mse else None,
        "test_mae": float(test_mae) if test_mae else None,
        "test_r2": float(test_r2) if test_r2 else None,
        "test_mse": float(test_mse) if test_mse else None,
        "test_rmse": float(test_rmse) if test_rmse else None,
    })

    with open(save_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print_flush(f"✓ Metrics: {save_dir / 'metrics.json'}")

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("Summary")
    print_flush("=" * 70)
    print_flush(f"Trained on: {len(y_train):,} pixels (only GEDI measurements)")
    print_flush(f"Predicted on: {len(train_tiles) + len(val_tiles) + len(test_tiles)} tiles (ALL pixels, wall-to-wall)")
    print_flush(f"\nEvaluation (on GEDI pixels only):")
    print_flush(f"  Train R²: {train_r2:.4f}")
    print_flush(f"  Val R²:   {val_r2:.4f}")
    print_flush(f"  Test R²:  {test_r2:.4f}")
    print_flush(f"\nWall-to-wall predictions saved to:")
    print_flush(f"  {save_dir}/train_predictions/")
    print_flush(f"  {save_dir}/val_predictions/")
    print_flush(f"  {save_dir}/test_predictions/")
    print_flush("\n✓ Complete! You can now mosaic these into California maps.")


if __name__ == "__main__":
    main()