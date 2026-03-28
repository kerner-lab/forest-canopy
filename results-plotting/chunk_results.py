"""
Generate scatter plots for GEDI canopy height predictions
Compares observed vs predicted values for validation and test sets
Automatically combines chunked CSV exports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import glob
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# ============================================
# Helper Function: Load and Combine Chunks
# ============================================

def load_and_combine_chunks(base_path, dataset_name):
    """
    Find all chunk files for a dataset and combine them

    Args:
        base_path: Directory containing CSV files
        dataset_name: 'validation' or 'test'

    Returns:
        Combined DataFrame
    """
    # Try multiple naming patterns (including _2021 suffix)
    patterns = [
        f'{dataset_name}_chunk*_2021.csv',      # test_chunk1_10k_2021.csv
        f'{dataset_name}_chunk*.csv',            # test_chunk1_10k.csv
        f'{dataset_name}_*_2021.csv',            # test_sample_2021.csv
        f'{dataset_name}_*.csv',                 # test_sample.csv
        f'{dataset_name}*.csv'                   # test.csv
    ]

    all_files = []
    for pattern in patterns:
        search_path = os.path.join(base_path, pattern)
        files = glob.glob(search_path)
        all_files.extend(files)

    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))

    # Filter to only chunk files (must contain 'chunk' in filename)
    chunk_files = [f for f in all_files if 'chunk' in os.path.basename(f).lower()]

    if not chunk_files:
        # No chunks found, try single file with various patterns
        single_patterns = [
            f'{dataset_name}_sample_california_2021.csv',
            f'{dataset_name}_5k_california_2021.csv',
            f'{dataset_name}_california_2021.csv',
            f'{dataset_name}_sample_california.csv',
            f'{dataset_name}_5k_california.csv',
            f'{dataset_name}_california.csv',
            f'{dataset_name}_2021.csv',
            f'{dataset_name}.csv'
        ]

        for pattern in single_patterns:
            single_file = os.path.join(base_path, pattern)
            if os.path.exists(single_file):
                print(f"✓ Found single {dataset_name} file: {os.path.basename(single_file)}")
                return pd.read_csv(single_file)

        raise FileNotFoundError(f"No {dataset_name} files found in {base_path}")

    print(f"\n✓ Found {len(chunk_files)} {dataset_name} chunk files:")

    # Load and combine all chunks
    chunks = []
    total_rows = 0
    for file in chunk_files:
        df = pd.read_csv(file)
        chunks.append(df)
        total_rows += len(df)
        file_size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  - {os.path.basename(file)}: {len(df):,} rows ({file_size_mb:.1f} MB)")

    combined_df = pd.concat(chunks, ignore_index=True)

    # Remove duplicates if any
    original_size = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    if len(combined_df) < original_size:
        print(f"  ⚠ Removed {original_size - len(combined_df)} duplicate rows")

    print(f"  ✓ Combined into {len(combined_df):,} total samples")

    return combined_df

# ============================================
# Load Data
# ============================================

# Update this path to your directory containing the CSV files
base_path = '/Users/kdoerkse/Documents/ASU/JPL/results/ae-gee'

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

val_df = load_and_combine_chunks(base_path, 'validation')
test_df = load_and_combine_chunks(base_path, 'test')

print(f"\n{'=' * 60}")
print(f"FINAL DATASET SIZES")
print(f"{'=' * 60}")
print(f"Validation samples: {len(val_df):,}")
print(f"Test samples: {len(test_df):,}")

# ============================================
# Compute Metrics
# ============================================

def compute_metrics(df, name):
    """Calculate regression metrics"""
    r2 = r2_score(df['rh98_m'], df['predicted'])
    rmse = np.sqrt(mean_squared_error(df['rh98_m'], df['predicted']))
    mae = mean_absolute_error(df['rh98_m'], df['predicted'])

    # Additional metrics
    bias = np.mean(df['predicted'] - df['rh98_m'])
    relative_rmse = (rmse / df['rh98_m'].mean()) * 100

    print(f"\n{name} Metrics:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f} m")
    print(f"  MAE = {mae:.2f} m")
    print(f"  Bias = {bias:.2f} m")
    print(f"  Relative RMSE = {relative_rmse:.1f}%")

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias}

print(f"\n{'=' * 60}")
print(f"METRICS")
print(f"{'=' * 60}")

val_metrics = compute_metrics(val_df, "Validation")
test_metrics = compute_metrics(test_df, "Test")

# ============================================
# Create Scatter Plots
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Common settings
max_val = max(
    val_df['rh98_m'].max(), val_df['predicted'].max(),
    test_df['rh98_m'].max(), test_df['predicted'].max()
)
min_val = 0
axis_range = [min_val, max_val]

# --- Validation Plot ---
ax1 = axes[0]
ax1.scatter(val_df['rh98_m'], val_df['predicted'],
            alpha=0.4, s=20, color='#2E86AB', edgecolors='none')

# Trendline
z = np.polyfit(val_df['rh98_m'], val_df['predicted'], 1)
p = np.poly1d(z)
ax1.plot(axis_range, p(axis_range), 'r-', linewidth=2, alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

# Labels and metrics
ax1.set_xlabel('Observed GEDI RH98 (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted RH98 (m)', fontsize=12, fontweight='bold')
ax1.set_title(f'Validation Set (n={len(val_df):,})', fontsize=14, fontweight='bold')
ax1.set_xlim(axis_range)
ax1.set_ylim(axis_range)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add metrics text box
metrics_text = f"R² = {val_metrics['r2']:.3f}\n" \
               f"RMSE = {val_metrics['rmse']:.2f} m\n" \
               f"MAE = {val_metrics['mae']:.2f} m\n" \
               f"Bias = {val_metrics['bias']:.2f} m"
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))

# --- Test Plot ---
ax2 = axes[1]
ax2.scatter(test_df['rh98_m'], test_df['predicted'],
            alpha=0.4, s=20, color='#A23B72', edgecolors='none')


# Trendline
z = np.polyfit(test_df['rh98_m'], test_df['predicted'], 1)
p = np.poly1d(z)
ax2.plot(axis_range, p(axis_range), 'r-', linewidth=2, alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

# Labels and metrics
ax2.set_xlabel('Observed GEDI RH98 (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted RH98 (m)', fontsize=12, fontweight='bold')
ax2.set_title(f'Test Set (n={len(test_df):,})', fontsize=14, fontweight='bold')
ax2.set_xlim(axis_range)
ax2.set_ylim(axis_range)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Add metrics text box
metrics_text = f"R² = {test_metrics['r2']:.3f}\n" \
               f"RMSE = {test_metrics['rmse']:.2f} m\n" \
               f"MAE = {test_metrics['mae']:.2f} m\n" \
               f"Bias = {test_metrics['bias']:.2f} m"
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))

plt.tight_layout()
#plt.savefig('gedi_predictions_scatter.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved scatter plot to: gedi_predictions_scatter.png")
plt.show()

# ============================================
# Optional: Combined Plot
# ============================================

fig2, ax = plt.subplots(1, 1, figsize=(8, 8))

# Plot both datasets
ax.scatter(val_df['rh98_m'], val_df['predicted'],
           alpha=0.3, s=20, color='#2E86AB', label=f'Validation (n={len(val_df):,})',
           edgecolors='none')
ax.scatter(test_df['rh98_m'], test_df['predicted'],
           alpha=0.3, s=20, color='#A23B72', label=f'Test (n={len(test_df):,})',
           edgecolors='none')

# Labels
ax.set_xlabel('Observed GEDI RH98 (m)', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted RH98 (m)', fontsize=13, fontweight='bold')
ax.set_title('GEDI Canopy Height Prediction - Combined', fontsize=15, fontweight='bold')
ax.set_xlim(axis_range)
ax.set_ylim(axis_range)
ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('gedi_predictions_combined.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved combined plot to: gedi_predictions_combined.png")
plt.show()

# ============================================
# Residual Analysis (Optional)
# ============================================

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation residuals
val_residuals = val_df['predicted'] - val_df['rh98_m']
axes[0].scatter(val_df['rh98_m'], val_residuals, alpha=0.4, s=20, color='#2E86AB')
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1.5)
axes[0].axhline(y=val_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5, label=f'±RMSE')
axes[0].axhline(y=-val_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Observed RH98 (m)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Residuals (Predicted - Observed) (m)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Validation Residuals (n={len(val_df):,})', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test residuals
test_residuals = test_df['predicted'] - test_df['rh98_m']
axes[1].scatter(test_df['rh98_m'], test_residuals, alpha=0.4, s=20, color='#A23B72')
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1.5)
axes[1].axhline(y=test_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5, label=f'±RMSE')
axes[1].axhline(y=-test_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Observed RH98 (m)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Predicted - Observed) (m)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Test Residuals (n={len(test_df):,})', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('gedi_predictions_residuals.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved residual plot to: gedi_predictions_residuals.png")
plt.show()

print("\n" + "=" * 60)
print("✅ All plots generated successfully!")
print("=" * 60)
print(f"\nFinal sample sizes:")
print(f"  Validation: {len(val_df):,} samples")
print(f"  Test: {len(test_df):,} samples")
print(f"\nPlots saved:")
print(f"  - gedi_predictions_scatter.png")
print(f"  - gedi_predictions_combined.png")
print(f"  - gedi_predictions_residuals.png")