#!/usr/bin/env python
# Evaluate model predictions broken down by 10 m canopy-height bins.
# Supports the same inputs as evaluate.py (parquet or tif-based).

import os
import sys
import rasterio
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


BINS = [
    (0,  10,  '0–10 m'),
    (10, 20,  '10–20 m'),
    (20, 30,  '20–30 m'),
    (30, 40,  '30–40 m'),
    (40, 50,  '40–50 m'),
    (50, 60,  '50–60 m'),
    (60, np.inf, '60+ m'),
]


def _load_from_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    gedi_arr = df['y_true'].values.astype(np.float64)
    pred_arr = df['y_pred'].values.astype(np.float64)
    return pred_arr, gedi_arr


def _load_from_tifs(preds_dir, gedi_dir, data_file, ignore_above):
    with open(data_file, 'r') as f:
        tile_indices = [t.strip() for t in f.readlines()]

    pred_arr = np.array([])
    gedi_arr = np.array([])
    for tile_index in tqdm(tile_indices, desc='Loading tiles'):
        pred_file = os.path.join(
            preds_dir,
            f'prediction_rh98_30.0m_{tile_index.replace("_v6", "")}.tif')
        pred_src = rasterio.open(pred_file, 'r')
        pred_data = np.squeeze(pred_src.read(), axis=0)
        pred_src.close()

        gedi_file = os.path.join(
            gedi_dir,
            f'gedi_rh98_30.0m_{tile_index.replace("_v6", "")}.tif')
        gedi_src = rasterio.open(gedi_file, 'r')
        gedi_data = np.squeeze(gedi_src.read(), axis=0)
        gedi_src.close()

        mask = np.where(gedi_data != 0)
        pred_data = pred_data[mask]
        gedi_data = gedi_data[mask]

        pred_arr = np.hstack((pred_arr, pred_data.flatten()))
        gedi_arr = np.hstack((gedi_arr, gedi_data.flatten()))

    # GEDI tif values are in cm; convert to m
    gedi_arr = gedi_arr / 100.

    if ignore_above > 0:
        mask = gedi_arr <= ignore_above
        pred_arr = pred_arr[mask]
        gedi_arr = gedi_arr[mask]

    return pred_arr, gedi_arr


def _bin_metrics(gedi_arr, pred_arr, lo, hi):
    mask = (gedi_arr >= lo) & (gedi_arr < hi)
    g = gedi_arr[mask]
    p = pred_arr[mask]
    n = len(g)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(g, p))
    r, _ = stats.pearsonr(g, p)
    bias = np.mean(p - g)
    return n, rmse, r, bias


def _scatter_subplot(ax, g, p, label, subsample_factor):
    if len(g) < 2:
        ax.set_title(f'{label}  (n={len(g)}, insufficient data)', fontsize=8)
        ax.set_xlabel('Ground Truth GEDI rh98 (unit: m)', fontsize=7)
        ax.set_ylabel('Predicted GEDI rh98 (unit: m)', fontsize=7)
        return

    # Compute metrics on full data before any subsampling
    rmse = np.sqrt(mean_squared_error(g, p))
    r, _ = stats.pearsonr(g, p)

    min_value = min(g.min(), p.min())
    max_value = max(g.max(), p.max())

    # Perfect prediction line
    ax.plot([min_value, max_value], [min_value, max_value], '--', color='gray', linewidth=1)

    # Subsample for plotting (matches evaluate.py threshold and factor)
    if len(g) > 10000:
        g = g[::subsample_factor]
        p = p[::subsample_factor]

    # KDE density scatter
    try:
        xy = np.vstack([g, p])
        z = stats.gaussian_kde(xy)(xy)
        idx = z.argsort()
        sc = ax.scatter(g[idx], p[idx], c=z[idx], cmap='jet',
                        label='RMSE = %.3f\nPearson correlation = %.3f' % (rmse, r),
                        s=0.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_ticks([])
    except Exception:
        ax.scatter(g, p, s=0.5, alpha=0.3,
                   label='RMSE = %.3f\nPearson correlation = %.3f' % (rmse, r))

    # Regression line
    coef = np.polyfit(g, p, 1)
    poly1d = np.poly1d(coef)
    ax.plot(g, poly1d(g), '--', color='black', linewidth=1,
            label=f'Regression line (slope={coef[0]:.2f})')

    # Binned mean ± std (1 m bins, matching evaluate.py)
    bins = np.linspace(min_value, max_value,
                       int(np.ceil((max_value - min_value) / 1)))
    if len(bins) > 1:
        digitized = np.digitize(g, bins)
        x_means = np.array([g[digitized == i].mean() for i in range(1, len(bins))])
        y_means = np.array([p[digitized == i].mean() for i in range(1, len(bins))])
        y_stds  = np.array([p[digitized == i].std()  for i in range(1, len(bins))])
        ax.plot(x_means, y_means, '--', color='mediumvioletred', linewidth=1,
                label='Binned prediction mean and std')
        ax.fill_between(x_means, y_means - y_stds, y_means + y_stds,
                        alpha=0.3, edgecolor='deeppink', facecolor='hotpink')

    ax.set_xlabel('Ground Truth GEDI rh98 (unit: m)', fontsize=7)
    ax.set_ylabel('Predicted GEDI rh98 (unit: m)', fontsize=7)
    ax.set_xlim((min_value, max_value))
    ax.set_ylim((min_value, max_value))
    ax.legend(loc='upper right', fontsize=6)
    ax.set_title(label, fontsize=8)
    ax.tick_params(labelsize=7)


def main(preds_dir, gedi_dir, data_file, out_plot, subsample_factor, ignore_above,
         parquet=None):
    if parquet is not None:
        if not os.path.exists(parquet):
            print(f'Parquet file not found: {parquet}')
            sys.exit(1)
        pred_arr, gedi_arr = _load_from_parquet(parquet)
    else:
        for f in [preds_dir, gedi_dir, data_file]:
            if not os.path.exists(f):
                print(f'Input dir/file not found: {f}')
                sys.exit(1)
        pred_arr, gedi_arr = _load_from_tifs(preds_dir, gedi_dir, data_file, ignore_above)

    # --- Print per-bin metrics table ---
    header = f"{'Bin':<12} {'N':>8} {'RMSE (m)':>10} {'r':>8} {'Bias (m)':>10}"
    print(header)
    print('-' * len(header))
    for lo, hi, label in BINS:
        n, rmse, r, bias = _bin_metrics(gedi_arr, pred_arr, lo, hi)
        rmse_s = f'{rmse:.3f}' if not np.isnan(rmse) else 'N/A'
        r_s    = f'{r:.3f}'    if not np.isnan(r)    else 'N/A'
        bias_s = f'{bias:+.3f}' if not np.isnan(bias) else 'N/A'
        print(f'{label:<12} {n:>8,} {rmse_s:>10} {r_s:>8} {bias_s:>10}')

    # --- Multi-panel scatter figure ---
    n_bins = len(BINS)
    ncols = 4
    nrows = (n_bins + ncols - 1) // ncols   # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (lo, hi, label) in enumerate(BINS):
        mask = (gedi_arr >= lo) & (gedi_arr < hi)
        _scatter_subplot(axes[i], gedi_arr[mask], pred_arr[mask], label, subsample_factor)

    # Hide any unused subplot slots
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Per-bin evaluation (ground truth binned by GEDI rh98)', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f'\nFigure saved to: {out_plot}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate model predictions by 10 m canopy-height bins.')
    parser.add_argument('preds_dir', type=str, nargs='?', default=None)
    parser.add_argument('gedi_dir',  type=str, nargs='?', default=None)
    parser.add_argument('data_file', type=str, nargs='?', default=None)
    parser.add_argument('out_plot',  type=str, nargs='?', default=None)
    parser.add_argument('--subsample_factor', type=int, default=10)
    parser.add_argument('--ignore_above', type=float, default=70,
                        help='Exclude GEDI labels above this value (metres). '
                             'Default 70. Set to 0 to include all. '
                             'Only applies to tif-based loading.')
    parser.add_argument('--parquet', type=str, default=None,
                        help='Path to predictions parquet file (y_true, y_pred '
                             'columns, values in metres). When set, '
                             'preds_dir/gedi_dir/data_file are not required.')

    args = parser.parse_args()

    if args.parquet is None and any(
            v is None for v in [args.preds_dir, args.gedi_dir, args.data_file]):
        parser.error('preds_dir, gedi_dir, and data_file are required unless --parquet is provided')

    if args.out_plot is None:
        if args.parquet is not None:
            p = os.path.splitext(args.parquet)
            args.out_plot = p[0] + '_evaluate_bins.png'
        else:
            parser.error('out_plot is required when not using --parquet')

    main(**vars(args))
