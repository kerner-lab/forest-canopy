#!/usr/bin/env python3
"""
Seed-averaged prediction-vs-GEDI figures for comparing embedding sources.

Two things happen here:

1. AVERAGE the per-seed prediction parquets for each source into one
   predictions_<split>_seedmean.parquet (mean of y_pred across seeds).

   This is only valid because every seed of a source sees the SAME rows in the
   SAME order: prepare_split_data() walks tiles in shapefile order and pixels in
   np.where(valid_mask) order, and the seed changes only the RF's internals. The
   script REFUSES to average unless y_true matches row-for-row across all seeds,
   so a misaligned run cannot be silently blended.

   !! The seed-mean is an ENSEMBLE. Averaging predictions cancels independent
   error, so metrics computed on it are BETTER than any single seed and better
   than the mean of the per-seed metrics. Use it for FIGURES, and report numbers
   from aggregate_seed_sweep.py (mean +/- std over seeds). The panel annotations
   are labelled "seed-mean" for exactly this reason.

2. PLOT, in two figures:

   <out>_density.png   small multiples, one panel per source: 2-D density of
                       predicted vs GEDI height with the 1:1 line and the
                       per-bin mean profile. Density is a magnitude, so it uses a
                       single-hue sequential ramp, identical in every panel;
                       model identity is carried by the panel title, not colour.

   <out>_bins.png      per-height-bin RMSE (top) and bias (bottom), one line per
                       source. This is the direct "which model degrades where"
                       comparison; bias is separated from RMSE because a model
                       can have acceptable RMSE while being systematically short.

Usage:
    python plot_prediction_scatter.py \\
        --root /ocean/.../outputs_seed_sweep_2019 --split test \\
        --out /ocean/.../figures/prediction_comparison
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_stratified import stratified_metrics, overall_metrics  # noqa: E402

# Categorical slots 1..8 in fixed order, never cycled. Used only by the per-bin
# line figure (adjacent pairlist). A 9th source would fold into "Other" rather
# than getting a generated hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Single-hue sequential ramp (blue 100 -> 600) for the density panels.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#b7d3f6", "#86b6ef", "#6da7ec",
                 "#3987e5", "#2a78d6", "#1c5cab", "#184f95"])

INK = "#1a1a19"
INK_2 = "#5c5b55"
INK_MUTED = "#8a8981"
GRID = "#e6e5e0"

# Nicer display names; anything not listed falls back to the directory name.
LABELS = {"aef": "AlphaEarth", "olmo": "OlmoEarth", "tessera": "TESSERA",
          "presto": "Presto", "prithvi": "Prithvi"}


def load_and_average(src_dir, split):
    """Mean y_pred across seeds. Returns (df, n_seeds) or (None, reason)."""
    paths = sorted(src_dir.glob(f"seed_*/predictions_{split}.parquet"))
    if not paths:
        return None, f"no predictions_{split}.parquet under {src_dir}"

    y_true = None
    preds = []
    for p in paths:
        df = pd.read_parquet(p, columns=["y_true", "y_pred"])
        if y_true is None:
            y_true = df["y_true"].to_numpy()
        elif len(df) != len(y_true):
            return None, (f"row-count mismatch: {p.parent.name} has {len(df):,} "
                          f"rows vs {len(y_true):,} in {paths[0].parent.name} "
                          f"— seeds are not aligned, refusing to average")
        elif not np.allclose(df["y_true"].to_numpy(), y_true, rtol=0, atol=1e-4):
            return None, (f"y_true differs in {p.parent.name} — the seeds did not "
                          f"see the same pixels, refusing to average")
        preds.append(df["y_pred"].to_numpy())

    return pd.DataFrame({"y_true": y_true,
                         "y_pred": np.mean(preds, axis=0)}), len(paths)


def per_bin_series(y_true, y_pred, bin_size, max_height, min_bin_n):
    """(centres, rmse, bias) over the height bins, with gaps for sparse bins."""
    s = stratified_metrics(y_true, y_pred, bin_size=bin_size,
                           max_height=max_height, min_bin_n=min_bin_n)
    c = np.array([(b["lo"] + b["hi"]) / 2 for b in s["bins"]])
    rmse = np.array([b["rmse"] if b["rmse"] is not None else np.nan
                     for b in s["bins"]])
    bias = np.array([b.get("bias") if b["rmse"] is not None else np.nan
                     for b in s["bins"]], dtype=float)
    return c, rmse, bias, s


def point_density(x, y, hi, nbins):
    """Local point count for each (x, y), from a 2-D histogram lookup.

    A KDE is O(n^2)-ish and hopeless at these row counts; binning the plane and
    reading each point's own cell is O(n) and gives the same "how many points are
    here" read, which is all the colour needs to convey.
    """
    edges = np.linspace(0, hi, nbins + 1)
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    ix = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(y, edges) - 1, 0, nbins - 1)
    return H[ix, iy]


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def figure_density(results, args, out_path):
    """Small multiples: predicted vs GEDI density, one panel per source."""
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows),
                             squeeze=False)
    hi = args.max_height + args.bin_size

    hb = None
    mappable = None
    rng = np.random.default_rng(0)

    # Pre-pass: subsample once and find the global max density, so the colour
    # scale means the same count in every panel. Without this a sparse model
    # would look as dense as a well-populated one.
    panel_pts, dens_max = {}, 1.0
    if args.style == "scatter":
        for src, r in results:
            yt = r["df"]["y_true"].to_numpy()
            yp = r["df"]["y_pred"].to_numpy()
            if args.max_points and len(yt) > args.max_points:
                sel = rng.choice(len(yt), size=args.max_points, replace=False)
                yt, yp = yt[sel], yp[sel]
            panel_pts[src] = (yt, yp)
            if args.density_cmap != "none":
                dens_max = max(dens_max,
                               float(point_density(yt, yp, hi,
                                                   args.density_bins).max()))

    for ax, (src, r) in zip(axes.ravel(), results):
        style_axes(ax)
        yt, yp = r["df"]["y_true"].to_numpy(), r["df"]["y_pred"].to_numpy()

        if args.style == "hexbin":
            hb = ax.hexbin(yt, yp, gridsize=args.gridsize, extent=(0, hi, 0, hi),
                           cmap=DENSITY_CMAP, norm=LogNorm(vmin=1), mincnt=1,
                           linewidths=0, zorder=1)
        else:
            sy, sp = panel_pts[src]
            if args.density_cmap == "none":
                ax.scatter(sy, sp, s=args.point_size, alpha=args.point_alpha,
                           color=SERIES[0], linewidths=0, zorder=1,
                           rasterized=True)
            else:
                # Colour each point by local point count, via a 2-D histogram
                # lookup (fast for millions of rows, unlike a KDE). Shared vmin/
                # vmax across panels so colour means the same count everywhere.
                dens = point_density(sy, sp, hi, args.density_bins)
                order = np.argsort(dens)          # dense points drawn on top
                sc = ax.scatter(sy[order], sp[order], c=dens[order],
                                s=args.point_size, cmap=args.density_cmap,
                                norm=LogNorm(vmin=1, vmax=dens_max),
                                linewidths=0, zorder=1, rasterized=True)
                mappable = sc

        # Bin cut-offs, so the per-bin segments and the binned table can be read
        # off the same picture.
        if args.bin_lines:
            for e in np.arange(args.bin_size, args.max_height + args.bin_size,
                               args.bin_size):
                if e < hi:
                    ax.axvline(e, color=INK_MUTED, linewidth=0.9,
                               linestyle=(0, (1, 3)), zorder=2)

        # 1:1 reference, recessive so it does not read as a data series.
        ax.plot([0, hi], [0, hi], color=INK_MUTED, linewidth=1.5,
                linestyle=(0, (5, 4)), zorder=3)

        # Per-bin least-squares fit, drawn as one segment per height bin and
        # deliberately NOT joined between bins — the breaks show where the
        # relationship changes. No markers: the segments are the message.
        if args.fit_bins:
            edges = np.arange(0.0, args.max_height + args.bin_size, args.bin_size)
            edges = np.append(edges, max(hi, float(yt.max())))
            for lo, up in zip(edges[:-1], edges[1:]):
                m = (yt >= lo) & (yt < up)
                # NB: not `n` — that name holds the panel count used below to
                # hide the unused axes in a partly-filled grid.
                n_in_bin = int(m.sum())
                if n_in_bin < args.min_bin_n or np.ptp(yt[m]) <= 0:
                    continue
                slope, intercept = np.polyfit(yt[m], yp[m], 1)
                xs = np.array([lo, min(up, hi)])
                ax.plot(xs, slope * xs + intercept, color=INK,
                        linewidth=2.0, solid_capstyle="round", zorder=4)

        # No metrics annotation inside the axes — the numbers belong in the table
        # from aggregate_seed_sweep.py, not printed over the data.
        ax.set_title(LABELS.get(src, src), color=INK, fontsize=12,
                     fontweight="600", loc="left", pad=8)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    # Label the first panel of each row and the LAST VISIBLE panel of each column —
    # with a partly-filled final row, axes[-1][c] is hidden for the trailing columns
    # and its label would never render.
    for r_i in range(nrows):
        if axes[r_i][0].get_visible():
            axes[r_i][0].set_ylabel("Predicted height (m)", color=INK_2, fontsize=10)
    for c_i in range(ncols):
        for r_i in range(nrows - 1, -1, -1):
            if axes[r_i][c_i].get_visible():
                axes[r_i][c_i].set_xlabel("GEDI height (m)", color=INK_2, fontsize=10)
                break

    bar_src = hb if hb is not None else mappable
    if bar_src is not None:
        cbar = fig.colorbar(bar_src, ax=axes.ravel().tolist(),
                            fraction=0.02, pad=0.02)
        cbar.set_label("pixels per hexagon" if hb is not None
                       else "pixels per cell", color=INK_2, fontsize=9)
        cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
        cbar.outline.set_visible(False)

    # Title above; the explanatory note goes BELOW the panels as a caption, which
    # is the paper convention and avoids colliding with the title in figure coords.
    fig.suptitle("Predicted vs GEDI canopy height, seed-averaged",
                 color=INK, fontsize=14, fontweight="600", x=0.02, ha="left",
                 y=1.005)
    note = "Dashed line 1:1."
    if args.fit_bins:
        note += f" Solid segments: least-squares fit within each {args.bin_size:.0f} m bin."
    if args.style == "scatter" and args.max_points:
        note += f" Up to {args.max_points:,} points per panel."
    fig.text(0.02, -0.01,
             note + "\nMetrics are for the seed-mean ensemble, not the "
             "reported per-seed scores.",
             color=INK_MUTED, fontsize=9, ha="left", va="top", linespacing=1.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_bins(results, args, out_path):
    """Per-bin RMSE and bias vs height, one line per source."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    style_axes(ax1)
    style_axes(ax2)

    for i, (src, r) in enumerate(results):
        colour = SERIES[i % len(SERIES)]
        c, rmse, bias, _s = r["bins"]
        label = LABELS.get(src, src)
        ax1.plot(c, rmse, color=colour, linewidth=2.0, marker="o",
                 markersize=8, markerfacecolor=colour,
                 markeredgecolor="white", markeredgewidth=2.0,
                 label=label, zorder=3)
        ax2.plot(c, bias, color=colour, linewidth=2.0, marker="o",
                 markersize=8, markerfacecolor=colour,
                 markeredgecolor="white", markeredgewidth=2.0,
                 label=label, zorder=3)

    ax2.axhline(0, color=INK_MUTED, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    # Headroom so the 8px markers at the extremes are not clipped by the spines.
    ax1.margins(y=0.10)
    ax2.margins(y=0.10)

    ax1.set_ylabel("RMSE (m)", color=INK_2, fontsize=10)
    ax2.set_ylabel("Bias (m), predicted − GEDI", color=INK_2, fontsize=10)
    ax2.set_xlabel("GEDI height bin centre (m)", color=INK_2, fontsize=10)

    leg = ax1.legend(frameon=False, fontsize=9, loc="upper left",
                     ncol=min(len(results), 3))
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax1.set_title("Error by canopy-height class", color=INK, fontsize=13,
                  fontweight="600", loc="left", pad=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="sweep root with <source>/seed_<n>/")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True,
                    help="output path stem; _density.png and _bins.png are appended")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="subset + ORDER of sources (default: all, alphabetical). "
                         "Order fixes the colour assignment.")
    ap.add_argument("--bin-size", type=float, default=10.0)
    ap.add_argument("--max-height", type=float, default=60.0)
    ap.add_argument("--min-bin-n", type=int, default=100)
    ap.add_argument("--fit-bins", action=argparse.BooleanOptionalAction, default=True,
                    help="draw a per-bin least-squares fit segment in each panel "
                         "(--no-fit-bins to omit)")
    ap.add_argument("--style", default="scatter", choices=["scatter", "hexbin"],
                    help="panel style: plain scatter (default, matches the existing "
                         "prediction plots) or a 2-D density hexbin")
    ap.add_argument("--max-points", type=int, default=200000,
                    help="scatter only: subsample to at most this many points per "
                         "panel (0 = plot all; default 200000)")
    ap.add_argument("--point-size", type=float, default=1.5,
                    help="scatter marker size (default: 1.5)")
    ap.add_argument("--point-alpha", type=float, default=0.25,
                    help="scatter marker alpha (only used when --density-cmap none)")
    ap.add_argument("--density-cmap", default="turbo",
                    choices=["turbo", "jet", "rainbow", "viridis", "magma", "none"],
                    help="colour scatter points by local point count. 'turbo' "
                         "(default) is the well-behaved rainbow; 'viridis' is "
                         "perceptually uniform and greyscale/CVD-safe; 'none' "
                         "gives flat single-colour points")
    ap.add_argument("--density-bins", type=int, default=140,
                    help="grid resolution for the density lookup (default: 140)")
    ap.add_argument("--bin-lines", action=argparse.BooleanOptionalAction, default=True,
                    help="dotted grey verticals at the height-bin cut-offs")
    ap.add_argument("--gridsize", type=int, default=70,
                    help="hexbin resolution (default: 70)")
    ap.add_argument("--no-save-parquet", action="store_true",
                    help="skip writing predictions_<split>_seedmean.parquet")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"ERROR: no such directory: {root}")

    src_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if args.sources:
        by_name = {d.name: d for d in src_dirs}
        missing = [s for s in args.sources if s not in by_name]
        if missing:
            raise SystemExit(f"ERROR: no such source dir(s): {missing}")
        src_dirs = [by_name[s] for s in args.sources]

    results, problems = [], []
    for d in src_dirs:
        df, info = load_and_average(d, args.split)
        if df is None:
            problems.append((d.name, info))
            continue
        n_seeds = info
        yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
        c, rmse, bias, s = per_bin_series(yt, yp, args.bin_size,
                                          args.max_height, args.min_bin_n)
        results.append((d.name, {
            "df": df, "n_seeds": n_seeds, "bins": (c, rmse, bias, s),
            "overall": overall_metrics(yt, yp), "strat": s,
        }))
        m = results[-1][1]["overall"]
        print(f"{d.name:<10} {n_seeds} seeds  n={m['n_samples']:>10,}  "
              f"seed-mean RMSE={m['rmse']:6.3f}  R2={m['r2']:6.3f}  "
              f"macro_rmse={s['macro_rmse']:6.3f}")
        if not args.no_save_parquet:
            out_pq = d / f"predictions_{args.split}_seedmean.parquet"
            df.to_parquet(out_pq, index=False, compression="snappy")

    for name, why in problems:
        print(f"SKIP {name}: {why}")
    if not results:
        raise SystemExit("Nothing to plot.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_density(results, args, out.with_name(out.name + "_density.png"))
    figure_bins(results, args, out.with_name(out.name + "_bins.png"))

    print("\nNOTE: the seed-mean is an ENSEMBLE — averaging predictions cancels\n"
          "      independent error, so its RMSE/R2 beat any single seed. These\n"
          "      figures are for visual comparison; report the numbers from\n"
          "      aggregate_seed_sweep.py (mean +/- std over seeds).")


if __name__ == "__main__":
    main()
