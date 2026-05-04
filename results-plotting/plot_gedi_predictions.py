#!/usr/bin/env python3
"""
Plot density distributions and scatter plots for GEDI canopy height predictions.

Usage:
    python plot_gedi_predictions.py --predictions-dir /path/to/outputs_gedi_rf_30m_float32_...
    python plot_gedi_predictions.py --predictions-dir /path/to/... --splits val test --output-dir ./plots
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

SPLIT_COLORS = {
    'train': '#E07B39',
    'val':   '#2E86AB',
    'test':  '#A23B72',
}


def load_predictions(predictions_dir, split):
    path = Path(predictions_dir) / f'predictions_{split}.parquet'
    if not path.exists():
        print(f"  Warning: {path} not found, skipping {split}")
        return None
    df = pd.read_parquet(path)
    print(f"  Loaded {split}: {len(df):,} samples")
    return df


def compute_metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'r2':   float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
    }


def plot_density_single(split, df, output_path):
    """Overlapping KDE of ground truth and predicted heights for one split."""
    color = SPLIT_COLORS.get(split, '#333333')
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.kdeplot(df['y_true'], ax=ax, color=color, linewidth=2,
                label='Ground truth', fill=True, alpha=0.3)
    sns.kdeplot(df['y_pred'], ax=ax, color=color, linewidth=2,
                label='Predicted', linestyle='--', fill=False)

    ax.set_xlabel('Forest height (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'Forest Height Distribution — {split.capitalize()} (n={len(df):,})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_single(split, df, output_path):
    """Hexbin scatter of predicted vs ground truth with line of best fit and metrics."""
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    metrics = compute_metrics(y_true, y_pred)

    axis_max = float(np.percentile(np.concatenate([y_true, y_pred]), 99))

    fig, ax = plt.subplots(figsize=(6, 6))

    hb = ax.hexbin(y_true, y_pred, gridsize=80, cmap='YlOrRd', mincnt=1,
                   extent=[0, axis_max, 0, axis_max])
    plt.colorbar(hb, ax=ax, label='Count')

    # 1:1 reference
    ax.plot([0, axis_max], [0, axis_max], 'k--', linewidth=1, alpha=0.5, label='1:1')

    # Line of best fit
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot([0, axis_max], p([0, axis_max]), color='#2E86AB', linewidth=2,
            label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    metrics_text = (
        f"R²   = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f} m\n"
        f"MAE  = {metrics['mae']:.2f} m\n"
        f"Bias = {metrics['bias']:.2f} m"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='black', linewidth=1))

    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Ground truth height (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted height (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted vs Ground Truth — {split.capitalize()} (n={len(df):,})',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    print(f"  {split}: R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.2f}m  "
          f"MAE={metrics['mae']:.2f}m  Bias={metrics['bias']:.2f}m")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot density and scatter plots for GEDI RF predictions'
    )
    parser.add_argument('--predictions-dir', required=True,
                        help='Output dir from run_gedi_prediction_30m.py containing predictions_*.parquet')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save plots (default: same as --predictions-dir)')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Which splits to plot (default: val test)')
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from: {predictions_dir}")
    datasets = {}
    for split in args.splits:
        df = load_predictions(predictions_dir, split)
        if df is not None:
            datasets[split] = df

    if not datasets:
        print("No prediction files found.")
        return

    for split, df in datasets.items():
        print(f"\nGenerating plots for {split}...")
        plot_density_single(split, df, output_dir / f'density_{split}.png')
        plot_scatter_single(split, df, output_dir / f'scatter_{split}.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
