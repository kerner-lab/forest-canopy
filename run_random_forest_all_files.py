"""
SUPER SIMPLE USAGE - Random Forest on ALL Alpha Earth Files

Just point to your base directories and run!

Your data spans: 41.3162° N, 120.1156° W to 41.2778° N, 123.6772° W
Longitude: -123.6772° (West) to -120.1156° (East)

Split: West tiles (lon < split) = Train/Val, East tiles (lon >= split) = Test
"""

from simple_random_forest import (
    load_and_flatten_data, 
    train_random_forest, 
    evaluate, 
    save_model
)


BASE_INPUT_DIR = "/Users/kdoerkse/Documents/ASU/JPL/data/tiles/input_tiles"    # Base folder with all alpha earth subfolders
BASE_TARGET_DIR = "/Users/kdoerkse/Documents/ASU/JPL/data/tiles/target_tiles"  # Base folder with all target subfolders

# Geographic split (longitude)
# None = auto-calculate midpoint (~-121.90°)
# Or set manually, e.g., -122.0
SPLIT_LONGITUDE = None  # Auto-calculate
# SPLIT_LONGITUDE = -122.0  # Or manually set


print("="*70)
print("Loading ALL Alpha Earth tiles...")
print("Geographic split by longitude (East vs West)")
print("="*70)

# Load ALL tiles from ALL alpha earth files
X_train, X_val, X_test, y_train, y_val, y_test = load_and_flatten_data(
    input_dir=BASE_INPUT_DIR,
    target_dir=BASE_TARGET_DIR,
    split_lon=SPLIT_LONGITUDE,
    load_all_folders=True  # This loads from all subdirectories!
)

# Train
print("\n" + "="*70)
print("Training Random Forest...")
print("="*70)

model = train_random_forest(
    X_train, y_train,
    n_estimators=100,
    max_depth=20,
    n_jobs=-1
)

# Evaluate
print("\n" + "="*70)
print("Results")
print("="*70)

train_preds, train_mse, train_rmse, train_mae, train_r2 = evaluate(
    model, X_train, y_train, "TRAIN (West)"
)

val_preds, val_mse, val_rmse, val_mae, val_r2 = evaluate(
    model, X_val, y_val, "VALIDATION (West)"
)

test_preds, test_mse, test_rmse, test_mae, test_r2 = evaluate(
    model, X_test, y_test, "TEST (East)"
)

# Save
save_model(model, "random_forest_all_files.pkl")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"Train R²: {train_r2:.4f}")
print(f"Val R²:   {val_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print("\n✓ Done! Model trained on ALL alpha earth files.")
