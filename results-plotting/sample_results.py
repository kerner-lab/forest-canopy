"""
Generate scatter plots for GEDI canopy height predictions
Compares observed vs predicted values for validation and test sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# ============================================
# Load Data
# ============================================

# Update these paths to match your downloaded CSV files
val_df = pd.read_csv('/Users/kdoerkse/Documents/ASU/JPL/results/ae-gee-rf/healpix/validation_10k_california_2021_healpix.csv')
test_df = pd.read_csv('/Users/kdoerkse/Documents/ASU/JPL/results/ae-gee-rf/healpix/test_10k_california_2021_healpix.csv')

print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")


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

    print(f"\n{name} Metrics:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f} m")
    print(f"  MAE = {mae:.2f} m")
    print(f"  Bias = {bias:.2f} m")

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias}


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
            alpha=0.6, s=30, color='#2E86AB', edgecolors='white', linewidth=0.5)


# Trendline
z = np.polyfit(val_df['rh98_m'], val_df['predicted'], 1)
p = np.poly1d(z)
ax1.plot(axis_range, p(axis_range), 'r-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

# Labels and metrics
ax1.set_xlabel('Observed GEDI RH98 (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted RH98 (m)', fontsize=12, fontweight='bold')
ax1.set_title('Validation Set', fontsize=14, fontweight='bold')
ax1.set_xlim(axis_range)
ax1.set_ylim(axis_range)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add metrics text box
metrics_text = f"n = {len(val_df)}\n" \
               f"R² = {val_metrics['r2']:.3f}\n" \
               f"RMSE = {val_metrics['rmse']:.2f} m\n" \
               f"MAE = {val_metrics['mae']:.2f} m\n" \
               f"Bias = {val_metrics['bias']:.2f} m"
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Test Plot ---
ax2 = axes[1]
ax2.scatter(test_df['rh98_m'], test_df['predicted'],
            alpha=0.6, s=30, color='#A23B72', edgecolors='white', linewidth=0.5)


# Trendline
z = np.polyfit(test_df['rh98_m'], test_df['predicted'], 1)
p = np.poly1d(z)
ax2.plot(axis_range, p(axis_range), 'r-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

# Labels and metrics
ax2.set_xlabel('Observed GEDI RH98 (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted RH98 (m)', fontsize=12, fontweight='bold')
ax2.set_title('Test Set', fontsize=14, fontweight='bold')
ax2.set_xlim(axis_range)
ax2.set_ylim(axis_range)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Add metrics text box
metrics_text = f"n = {len(test_df)}\n" \
               f"R² = {test_metrics['r2']:.3f}\n" \
               f"RMSE = {test_metrics['rmse']:.2f} m\n" \
               f"MAE = {test_metrics['mae']:.2f} m\n" \
               f"Bias = {test_metrics['bias']:.2f} m"
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
           alpha=0.5, s=30, color='#2E86AB', label=f'Validation (n={len(val_df)})',
           edgecolors='white', linewidth=0.5)
ax.scatter(test_df['rh98_m'], test_df['predicted'],
           alpha=0.5, s=30, color='#A23B72', label=f'Test (n={len(test_df)})',
           edgecolors='white', linewidth=0.5)

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
axes[0].scatter(val_df['rh98_m'], val_residuals, alpha=0.6, s=30, color='#2E86AB')
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1.5)
axes[0].axhline(y=val_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5, label=f'±RMSE')
axes[0].axhline(y=-val_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Observed RH98 (m)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Residuals (Predicted - Observed) (m)', fontsize=12, fontweight='bold')
axes[0].set_title('Validation Residuals', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test residuals
test_residuals = test_df['predicted'] - test_df['rh98_m']
axes[1].scatter(test_df['rh98_m'], test_residuals, alpha=0.6, s=30, color='#A23B72')
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1.5)
axes[1].axhline(y=test_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5, label=f'±RMSE')
axes[1].axhline(y=-test_metrics['rmse'], color='r', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Observed RH98 (m)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Predicted - Observed) (m)', fontsize=12, fontweight='bold')
axes[1].set_title('Test Residuals', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('gedi_predictions_residuals.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved residual plot to: gedi_predictions_residuals.png")
plt.show()

print("\n✅ All plots generated successfully!")