#!/usr/bin/env python3
"""
Height-binned ("stratified") regression metrics, shared by the training pipeline
and the post-hoc backfill so the two cannot drift apart.

Kept separate from run_gedi_prediction_30m.py so that tools which only need the
metric maths (stratify_from_predictions.py) do not have to import geopandas,
rasterio and the rest of the training stack.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def stratified_metrics(y_true, y_pred, bin_size=10.0, max_height=60.0,
                       min_bin_n=100, balance_seed=42, overflow_bin=False):
    """Height-binned metrics, plus two ways of summarising them.

    GEDI height distributions are heavily skewed toward short/mid canopy, so the
    overall (micro) RMSE is dominated by whichever bin holds the most pixels and
    says little about tall-canopy performance. Two stratified summaries are
    returned, and they answer different questions:

    macro_*   unweighted mean of the per-bin metrics. Every height class counts
              equally regardless of how rare it is. macro_rmse is the honest
              "how does this model do across the whole height range" number.

    balanced_*  metrics recomputed on a bin-balanced subsample (every bin cut to
              the smallest bin's count). Preserves the ACROSS-bin variance, so
              balanced_r2 is meaningful in a way macro_r2 is not — see below.

    CAVEAT on per-bin R^2: within a single 10 m bin the true heights barely vary,
    so the denominator of R^2 (the within-bin variance of y) is tiny by
    construction. Per-bin R^2 therefore collapses toward 0 or goes negative even
    for a good model. This is range restriction, not a model failure. Report
    per-bin RMSE and balanced_r2; do not headline macro_r2.

    overflow_bin: by default everything above max_height is CLIPPED into the top
    closed bin, so a bin labelled 50-60 silently also contains the 70 m pixels.
    Set True to add a final open-ended bin [max_height, inf) instead, reported
    with hi=inf and labelled ">60" — which is what you want for a per-bin plot.
    """
    edges = np.arange(0.0, max_height + bin_size, bin_size)
    if overflow_bin:
        edges = np.append(edges, np.inf)
    idx = np.clip(np.digitize(y_true, edges) - 1, 0, len(edges) - 2)

    bins, rmses, maes, r2s = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        lo, hi = float(edges[b]), float(edges[b + 1])
        if n < min_bin_n:
            bins.append({'lo': lo, 'hi': hi, 'n': n, 'rmse': None,
                         'mae': None, 'r2': None})
            continue
        yt, yp = y_true[m], y_pred[m]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        # Undefined when every true value in the bin is identical.
        r2 = float(r2_score(yt, yp)) if np.ptp(yt) > 0 else None
        bins.append({'lo': lo, 'hi': hi, 'n': n, 'rmse': rmse,
                     'mae': mae, 'r2': r2, 'bias': float(np.mean(yp - yt))})
        rmses.append(rmse)
        maes.append(mae)
        if r2 is not None:
            r2s.append(r2)

    out = {
        'bin_size': bin_size,
        'bins': bins,
        'n_bins_used': len(rmses),
        'macro_rmse': float(np.mean(rmses)) if rmses else None,
        'macro_mae': float(np.mean(maes)) if maes else None,
        'macro_r2': float(np.mean(r2s)) if r2s else None,
    }

    # Bin-balanced resample: equalise bin counts, then compute ordinary metrics.
    # Seeded off the model seed so the sweep's seed-to-seed spread includes this.
    counts = [b['n'] for b in bins if b['n'] >= min_bin_n]
    if len(counts) >= 2:
        k = min(counts)
        rng = np.random.default_rng(balance_seed)
        take = []
        for b in range(len(edges) - 1):
            pos = np.flatnonzero(idx == b)
            if len(pos) >= min_bin_n:
                take.append(rng.choice(pos, size=k, replace=False))
        sel = np.concatenate(take)
        yt, yp = y_true[sel], y_pred[sel]
        out.update({
            'balanced_n_per_bin': int(k),
            'balanced_n_samples': int(len(sel)),
            'balanced_rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'balanced_mae': float(mean_absolute_error(yt, yp)),
            'balanced_r2': float(r2_score(yt, yp)),
            'balanced_bias': float(np.mean(yp - yt)),
        })
    return out


def overall_metrics(y_true, y_pred):
    """RMSE / MAE / R^2 / Pearson r for a whole prediction set.

    Both r and R^2 are returned because they answer different questions, and for a
    BASELINE product (Potapov, ETH) they can diverge sharply:

      pearson_r  correlation only — invariant to any linear rescaling of y_pred.
                 A baseline that is systematically 30% too short still scores high.
      r2         agreement with the 1:1 line. Bias and scale errors are penalised,
                 so it can go NEGATIVE (worse than predicting the mean).

    The RF models are scored with r2, so r2 is the number to compare against them.
    Reporting r alone would flatter a biased baseline.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        'n_samples': int(len(yt)),
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mae': float(mean_absolute_error(yt, yp)),
        'r2': float(r2_score(yt, yp)) if len(yt) > 1 else float('nan'),
        'pearson_r': float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float('nan'),
        'bias': float(np.mean(yp - yt)),
    }


def print_stratified(strat, indent='  '):
    """Per-bin table + macro/balanced summary for a stratified_metrics() result."""
    print(f"{indent}stratified ({strat['bin_size']:.0f} m bins):")
    print(f"{indent}  {'bin (m)':>12} {'n':>10} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'bias':>8}")
    for b in strat['bins']:
        if b['rmse'] is None:
            print(f"{indent}  {b['lo']:5.0f}-{b['hi']:<6.0f} {b['n']:>10,}   (too few)")
            continue
        r2 = f"{b['r2']:8.3f}" if b['r2'] is not None else f"{'n/a':>8}"
        print(f"{indent}  {b['lo']:5.0f}-{b['hi']:<6.0f} {b['n']:>10,} {b['rmse']:8.3f} "
              f"{b['mae']:8.3f} {r2} {b['bias']:8.3f}")
    macro = strat['macro_rmse']
    print(f"{indent}  macro RMSE={macro:.3f}" if macro is not None
          else f"{indent}  macro RMSE=n/a", end="")
    if strat.get('balanced_rmse') is not None:
        print(f"   balanced RMSE={strat['balanced_rmse']:.3f}"
              f"  balanced R2={strat['balanced_r2']:.3f}"
              f"  (n={strat['balanced_n_per_bin']:,}/bin)")
    else:
        print()


def evaluate_and_report(y_true, y_pred, label, bin_size=10.0, max_height=60.0,
                        min_bin_n=100, balance_seed=42, indent='  '):
    """Overall + stratified metrics for one prediction set: print them, return them.

    The returned dict matches the shape run_gedi_prediction_30m.py writes into
    metrics.json, so baseline results can sit alongside model results.
    """
    m = overall_metrics(y_true, y_pred)
    print(f"{label:<24}: n={m['n_samples']:>9,}  RMSE={m['rmse']:6.3f} m  "
          f"MAE={m['mae']:6.3f}  R2={m['r2']:7.3f}  r={m['pearson_r']:.3f}  "
          f"bias={m['bias']:+.3f}")
    m['stratified'] = stratified_metrics(
        y_true, y_pred, bin_size=bin_size, max_height=max_height,
        min_bin_n=min_bin_n, balance_seed=balance_seed)
    print_stratified(m['stratified'], indent=indent)
    return m


# Student-t 97.5th percentile by degrees of freedom, for a two-sided 95% CI.
# Hardcoded so this module keeps working without scipy. At n=5 (df=4) t=2.776,
# NOT 1.96 — using the normal approximation would understate the interval by
# ~40%, which matters a lot at the seed counts used here.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    """Two-sided 95% t multiplier for `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        return 1.96
    return _T975[min(k for k in keys if k >= df)]


def mean_ci95(values):
    """(mean, half-width of the two-sided 95% CI of the MEAN, n).

    Half-width = t(0.975, n-1) * s / sqrt(n). This is the interval on the mean —
    the right thing when the reported number IS the mean. It is NOT the spread of
    the seed values themselves; that would be t * s, wider by sqrt(n).
    """
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return v[0], 0.0, 1
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    return mean, t975(n - 1) * sd / np.sqrt(n), n
