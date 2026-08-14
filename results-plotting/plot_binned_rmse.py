#!/usr/bin/env python3
"""
Per-height-bin RMSE, aggregated ACROSS SEEDS, one panel per year.

For every (year, source, seed) this reads predictions_<split>.parquet, computes
RMSE within each height bin, and then takes the mean and sample std ACROSS THE
SEEDS. That is the correct aggregation for reporting: it is the same quantity
aggregate_seed_sweep.py reports, just resolved per height class instead of
summarised into macro_rmse.

NOTE this is deliberately NOT computed from predictions_<split>_seedmean.parquet.
Averaging predictions first would give an ensemble whose per-bin RMSE is better
than any individual seed, and it would have no error bar.

Bins are 0-10, 10-20, ... 50-60 and a final open-ended >60 — the last one is a
real bin here, not the clipped top bin that stratified_metrics uses by default.

Usage:
    python plot_binned_rmse.py \\
        --root 2019=/ocean/.../outputs_seed_sweep_2019 \\
        --root 2020=/ocean/.../outputs_seed_sweep_2020 \\
        --sources aef olmo tessera prithvi \\
        --out /ocean/.../figures/binned_rmse.png --csv binned_rmse.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_stratified import stratified_metrics, t975  # noqa: E402

# Categorical slots in fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Secondary encoding: identity is never carried by colour alone. Required here
# because a multi-panel comparison is read pairwise across all series, not just
# adjacent ones, and marker shape survives CVD, greyscale print and photocopying.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK = "#1a1a19"
INK_2 = "#5c5b55"
INK_MUTED = "#8a8981"
GRID = "#e6e5e0"

LABELS = {"aef": "AlphaEarth", "olmo": "OlmoEarth", "tessera": "TESSERA",
          "presto": "Presto", "prithvi": "Prithvi"}


def bin_label(lo, hi):
    return f">{lo:.0f}" if not np.isfinite(hi) else f"{lo:.0f}-{hi:.0f}"


def per_seed_bin_rmse(src_dir, split, bin_size, max_height, min_bin_n):
    """(labels, rmse[n_seeds, n_bins], n_seeds) for one source, or None."""
    paths = sorted(src_dir.glob(f"seed_*/predictions_{split}.parquet"))
    if not paths:
        return None
    labels, rows = None, []
    for p in paths:
        df = pd.read_parquet(p, columns=["y_true", "y_pred"])
        s = stratified_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(),
                               bin_size=bin_size, max_height=max_height,
                               min_bin_n=min_bin_n, overflow_bin=True)
        lab = [bin_label(b["lo"], b["hi"]) for b in s["bins"]]
        if labels is None:
            labels = lab
        elif lab != labels:
            raise SystemExit(f"ERROR: inconsistent bin edges in {p}")
        rows.append([b["rmse"] if b["rmse"] is not None else np.nan
                     for b in s["bins"]])
    return labels, np.array(rows, dtype=float), len(paths)


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, metavar="YEAR=PATH",
                    help="sweep root per year; repeat for each year")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sources", nargs="+", default=None,
                    help="subset + ORDER of sources; order fixes colour/marker")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--csv", default=None, help="also write the numbers here")
    ap.add_argument("--bin-size", type=float, default=10.0)
    ap.add_argument("--max-height", type=float, default=60.0,
                    help="last closed edge; above it becomes the >N bin")
    ap.add_argument("--min-bin-n", type=int, default=100)
    ap.add_argument("--interval", default="ci95", choices=["ci95", "std"],
                    help="error bars: 'ci95' (default) = two-sided 95%% CI of the "
                         "mean over seeds, t(0.975, n-1)*s/sqrt(n); 'std' = sample "
                         "standard deviation across seeds")
    args = ap.parse_args()

    roots = {}
    for spec in args.root:
        if "=" not in spec:
            raise SystemExit(f"ERROR: --root must be YEAR=PATH, got: {spec}")
        y, pth = spec.split("=", 1)
        p = Path(pth)
        if not p.is_dir():
            raise SystemExit(f"ERROR: no such directory: {p}")
        roots[y.strip()] = p

    # Source order is shared across years so a model keeps one colour+marker.
    if args.sources:
        sources = list(args.sources)
    else:
        seen = []
        for p in roots.values():
            for d in sorted(p.iterdir()):
                if d.is_dir() and d.name not in seen:
                    seen.append(d.name)
        sources = seen

    data, labels = {}, None
    for year, root in roots.items():
        for src in sources:
            d = root / src
            if not d.is_dir():
                print(f"SKIP {year}/{src}: no directory")
                continue
            got = per_seed_bin_rmse(d, args.split, args.bin_size,
                                    args.max_height, args.min_bin_n)
            if got is None:
                print(f"SKIP {year}/{src}: no predictions_{args.split}.parquet")
                continue
            lab, arr, n_seeds = got
            labels = labels or lab
            mean = np.nanmean(arr, axis=0)
            if n_seeds > 1:
                sd = np.nanstd(arr, axis=0, ddof=1)
                # Interval on the MEAN, since the mean is what is reported.
                # t(0.975, n-1), not 1.96 — at n=5 that is 2.776.
                err = (sd if args.interval == "std"
                       else t975(n_seeds - 1) * sd / np.sqrt(n_seeds))
            else:
                sd = err = np.zeros(arr.shape[1])
            data[(year, src)] = (mean, err, sd, n_seeds)
            print(f"{year} {src:<10} {n_seeds} seeds  " +
                  "  ".join(f"{l}={m:.2f}" for l, m in zip(lab, mean)))

    if not data:
        raise SystemExit("Nothing to plot.")

    years = list(roots)
    x = np.arange(len(labels))
    # Dodge series within each bin slot so error bars do not overlap.
    n_src = len(sources)
    span = 0.62
    offsets = (np.arange(n_src) - (n_src - 1) / 2) * (span / max(n_src, 1))

    fig, axes = plt.subplots(1, len(years), figsize=(6.4 * len(years), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, year in zip(axes, years):
        style_axes(ax)
        for i, src in enumerate(sources):
            if (year, src) not in data:
                continue
            mean, err, _sd, _n = data[(year, src)]
            ax.errorbar(x + offsets[i], mean, yerr=err,
                        color=SERIES[i % len(SERIES)],
                        marker=MARKERS[i % len(MARKERS)], markersize=7,
                        markeredgecolor="white", markeredgewidth=1.4,
                        linewidth=2.0, elinewidth=1.4, capsize=3,
                        capthick=1.4, label=LABELS.get(src, src), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("GEDI height bin (m)", color=INK_2, fontsize=10)
        ax.set_title(year, color=INK, fontsize=12, fontweight="600",
                     loc="left", pad=8)
        ax.margins(y=0.10)

    lab_y = "RMSE (m)" if args.interval == "std" else "RMSE (m)"
    axes[0].set_ylabel(lab_y, color=INK_2, fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper left",
                         ncol=min(n_src, 2))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("RMSE by canopy-height class", color=INK, fontsize=14,
                 fontweight="600", x=0.02, ha="left", y=1.02)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {args.out}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "source", "n_seeds", "bin", "rmse_mean",
                        "rmse_std", "rmse_ci95_halfwidth"])
            for (year, src), (mean, err, sd, n) in data.items():
                ci = (err if args.interval == "ci95"
                      else t975(n - 1) * sd / np.sqrt(n) if n > 1 else sd)
                for lab, m, sdv, c in zip(labels, mean, sd, ci):
                    w.writerow([year, src, n, lab, f"{m:.6f}",
                                f"{sdv:.6f}", f"{c:.6f}"])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
