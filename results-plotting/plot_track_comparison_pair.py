#!/usr/bin/env python3
"""
Paper figure: two GEDI transects side by side, each a Sentinel-2 map above its
stacked canopy-height profiles.

Column (a) is the 2019 open-woodland transect (tile 31330), column (b) the 2020
tall coastal-forest transect (tile 33360). Each column is

  - a Sentinel-2 L2A true-colour map with the transect drawn on it,
  - one panel per embedding model (solid lines),
  - a small gap, then one panel per existing canopy-height product (dashed).

GEDI RH98 is drawn as discrete footprints (black dot on a faint stem), never
as a filled or connected profile, so gaps between shots stay visible.

The embedding-model prediction rasters only hold values AT GEDI pixels (they
are rebuilt from the test-split predictions), so there is nothing between shots
to draw. To keep every panel comparable, all models, including the wall-to-wall
products, are sampled at the same GEDI shots. Their lines join consecutive
shots and break wherever the along-track gap exceeds --max-gap-m (default
100 m). GEDI fires every ~60 m; snapped to the 30 m grid, consecutive shots sit
18-72 m apart, while a dropped shot leaves >= ~105 m. So a missing shot is
never bridged by a made-up segment.

Track selection, raster sampling and the Sentinel-2 chip all come from
plot_track_comparison.py, so the transects match the single-tile figures.

Example:
  python results-plotting/plot_track_comparison_pair.py \\
      --out ~/Documents/ASU/JPL/results/wacv_geocv/track_comparison_pair.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from plot_track_comparison import (fetch_s2_rgb, load_raster_on_gedi_grid,
                                   select_track)

DATA = "/Users/kdoerkse/Documents/ASU/JPL/data"
RESULTS = "/Users/kdoerkse/Documents/ASU/JPL/results/wacv_geocv"

# (label, raster path template, colour). Templates may use {gid}, {year},
# {data}, {results}. Order is panel order, top to bottom.
EMBEDDING_MODELS = [
    ("AlphaEarth", "{results}/pred_rasters_{year}/aef/healpix_{gid}_aef.tif", "#7b3fb0"),
    ("OlmoEarth", "{results}/pred_rasters_{year}/olmo/healpix_{gid}_olmo.tif", "#d1495b"),
    ("Prithvi", "{results}/pred_rasters_{year}/prithvi/healpix_{gid}_prithvi.tif", "#2a9d8f"),
    ("Tessera", "{results}/pred_rasters_{year}/tessera/healpix_{gid}_tessera.tif", "#3a6fb4"),
    ("Presto", "{results}/pred_rasters_{year}/presto/healpix_{gid}_presto.tif", "#e08214"),
]
ECHOSAT = ("ECHOSAT", "{data}/echosat/echosat_tiles/healpix_{gid}_echosat.tif", "#8c564b")
REFERENCE_GREEN = "#2e8b57"   # Potapov / Lang: green so they never read as GEDI

# One dict per column, left to right. `products` are the existing canopy-height
# products for that year: ECHOSAT plus the matching reference map.
TRANSECTS = [
    dict(gid=31330, year=2019, desc="Crater Lake National Park, Oregon",
         gedi_dir=f"{DATA}/gedi_tiles_2019_healpix", rank=1, near=None, eps_m=120.0,
         s2_item="S2B_10TEN_20190727_1_L2A", s2_pct=(1, 99), s2_gamma=1.4,
         products=[ECHOSAT,
                   ("Potapov", "{results}/potapov_baseline/potapov_tiles/healpix_{gid}_potapov.tif",
                    REFERENCE_GREEN)]),
    dict(gid=33360, year=2020, desc="California Redwoods\nNational and State Parks",
         gedi_dir=f"{DATA}/gedi_tiles_2020_healpix", rank=0, near=(41.4082, -124.0615),
         eps_m=240.0,
         # Bright surf and beach set a 99th percentile far above the forest;
         # clip lower and lift the shadows so the canopy texture shows.
         s2_item="S2A_10TDL_20200927_1_L2A", s2_pct=(1, 96), s2_gamma=1.8,
         products=[ECHOSAT,
                   ("Lang", "{results}/eth_baseline/eth_tiles/healpix_{gid}_eth.tif",
                    REFERENCE_GREEN)]),
]

# Nodata values the baseline rasters use besides their header nodata
# (Potapov writes 0 for "no data"; 0 m is not a valid Potapov height).
EXTRA_NODATA = {"Potapov": 0.0}


def sample_on_track(path, trk, year, label):
    """Model values at the track's GEDI pixels; NaN where the raster has none."""
    arr, rnod = load_raster_on_gedi_grid(path, trk["shape"], year)
    vals = arr[trk["rows"], trk["cols"]]
    mask = np.isfinite(vals)
    for nd in (rnod, EXTRA_NODATA.get(label)):
        if nd is not None:
            mask &= vals != nd
    return np.where(mask, vals, np.nan)


def break_at_gaps(t, y, max_gap):
    """Insert NaNs where consecutive shots are more than `max_gap` apart, so
    matplotlib breaks the line instead of drawing across the gap."""
    gaps = np.where(np.diff(t) > max_gap)[0]
    return (np.insert(t.astype(float), gaps + 1, np.nan),
            np.insert(y.astype(float), gaps + 1, np.nan))


def nice_step(ymax):
    """Tick step giving three ticks (0, s, 2s) that cover most of the data."""
    for s in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100):
        if 2 * s >= 0.75 * ymax:
            return s
    return 100


def draw_map(ax, trk, cfg, letter, pad_m, bar_m):
    lon, lat = trk["transform"] * (trk["cols"] + 0.5, trk["rows"] + 0.5)
    rgb, extent, tx, ty, item_id, date = fetch_s2_rgb(
        np.asarray(lon), np.asarray(lat), pad_m, cfg["year"], (6, 9), 20.0, 8,
        cfg["s2_item"], cfg["s2_pct"], cfg["s2_gamma"])
    ax.imshow(rgb, extent=extent, interpolation="nearest")
    halo = [pe.Stroke(linewidth=3, foreground="black"), pe.Normal()]
    ax.plot(tx, ty, color="#ffd60a", lw=1.6, solid_capstyle="round",
            path_effects=halo, zorder=3)
    t = trk["t"]
    for i, txt in ((0, "0 m"), (-1, f"{t[-1]:,.0f} m")):
        ax.plot(tx[i], ty[i], "o", ms=4, mfc="white", mec="black", mew=0.6, zorder=4)
        ax.annotate(txt, (tx[i], ty[i]), xytext=(4, 4), textcoords="offset points",
                    color="white", fontsize=6.5, fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    l, r, b, tp = extent
    w, h = r - l, tp - b
    # Scale bar on a white box so it reads on dark forest as well as on pumice.
    x0, y0 = l + 0.05 * w, b + 0.05 * h
    ax.add_patch(plt.Rectangle((x0 - 0.03 * w, y0 - 0.03 * h), bar_m + 0.06 * w,
                               0.12 * h, fc="white", ec="none", alpha=0.85, zorder=5))
    ax.add_patch(plt.Rectangle((x0, y0), bar_m, 0.015 * h, fc="black", ec="none",
                               zorder=6))
    ax.text(x0 + bar_m / 2, y0 + 0.03 * h,
            f"{bar_m / 1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m",
            ha="center", va="bottom", fontsize=8, zorder=6)
    ax.annotate("N", xy=(0.93, 0.95), xytext=(0.93, 0.85), xycoords="axes fraction",
                ha="center", va="center", color="white", fontsize=7, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2,
                                path_effects=[pe.Stroke(linewidth=2.6, foreground="black"),
                                              pe.Normal()]),
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
    # Panel letter just outside the map's top-left corner.
    ax.text(-0.03, 1.0, f"({letter})", transform=ax.transAxes, ha="right",
            va="top", color="black", fontsize=11, fontweight="bold")
    ax.text(0.985, 0.015, f"Sentinel-2 L2A, {date}\nContains modified "
            "Copernicus Sentinel data",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=4.5,
            color="white", linespacing=1.1,
            path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])
    ax.set_xlim(l, r)
    ax.set_ylim(b, tp)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    print(f"  map ({letter}): scene {item_id}")


def draw_profile(ax, t, gedi_m, y, label, color, dashed, ylim, step, max_gap,
                 show_metrics):
    ax.vlines(t, 0, gedi_m, color="0.72", lw=0.6, zorder=1)
    ax.plot(t, gedi_m, "o", ms=2.6, color="black", zorder=4)
    tl, yl = break_at_gaps(t, y, max_gap)
    ax.plot(tl, yl, color=color, lw=1.5, ls=(0, (4, 2)) if dashed else "-",
            zorder=3, solid_capstyle="round")
    ok = np.isfinite(y)
    ax.text(0.01, 0.97, label, transform=ax.transAxes, ha="left", va="top",
            color=color, fontsize=9, fontweight="bold")
    if show_metrics and ok.any():
        d = y[ok] - gedi_m[ok]
        rmse, bias = np.sqrt(np.mean(d ** 2)), np.mean(d)
        bias_s = f"{bias:+.1f}".replace("-", "−")
        ax.text(0.99, 0.97, f"RMSE {rmse:.1f} m   bias {bias_s} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color="0.3")
        print(f"    {label:10s} {ok.sum():2d}/{len(y)} shots  RMSE {rmse:5.2f}  "
              f"bias {bias:+6.2f}")
    ax.set_ylim(0, ylim)
    ax.set_yticks([0, step, 2 * step])
    ax.set_xlim(0, t.max())
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{RESULTS}/track_comparison_pair.png",
                    help="Output PNG; a PDF with the same stem is written too")
    ap.add_argument("--max-gap-m", type=float, default=100.0,
                    help="Break model lines where consecutive GEDI shots are more "
                         "than this far apart, i.e. a shot is missing (default 100)")
    ap.add_argument("--s2-pad-m", type=float, default=1500.0)
    ap.add_argument("--s2-scalebar-m", type=float, default=1000.0)
    ap.add_argument("--embedding-group-label",
                    default="Embedding models (random forest)",
                    help="Label above the embedding-model panels")
    ap.add_argument("--product-group-label",
                    default="Baseline canopy height products")
    ap.add_argument("--panel-metrics", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "pdf.fonttype": 42})

    n_emb = len(EMBEDDING_MODELS)
    n_prod = max(len(c["products"]) for c in TRANSECTS)
    # Rows: map, label strip, embedding panels, gap + label strip, product panels.
    heights = [3.3, 0.12] + [1.0] * n_emb + [0.3] + [1.0] * n_prod
    fig = plt.figure(figsize=(7.2, 10.2))
    gs = GridSpec(len(heights), len(TRANSECTS), figure=fig, height_ratios=heights,
                  hspace=0.16, wspace=0.16, left=0.08, right=0.99, top=0.945,
                  bottom=0.075)

    for col, cfg in enumerate(TRANSECTS):
        letter = "ab"[col]
        print(f"\n({letter}) tile {cfg['gid']} {cfg['year']}")
        trk = select_track(Path(cfg["gedi_dir"]) / f"healpix_{cfg['gid']}_gedi.tif",
                           eps_m=cfg["eps_m"], rank=cfg["rank"], near=cfg["near"])
        t, gedi_m = trk["t"], trk["gedi_m"]
        max_gap = args.max_gap_m
        fill = dict(gid=cfg["gid"], year=cfg["year"], data=DATA, results=RESULTS)

        ax_map = fig.add_subplot(gs[0, col])
        draw_map(ax_map, trk, cfg, letter, args.s2_pad_m, args.s2_scalebar_m)
        # Site name as the title; tile / year move to the subtitle so a long
        # park name still fits the column.
        ax_map.set_title(cfg["desc"], loc="left", fontsize=9, fontweight="bold",
                         pad=14)
        ax_map.text(0, 1.012, f"Tile {cfg['gid']} · {cfg['year']} · "
                    f"{t.max() / 1000:.1f} km \u00b7 {len(t)} GEDI shots",
                    transform=ax_map.transAxes, ha="left", va="bottom",
                    fontsize=8, color="0.4")

        groups = [(EMBEDDING_MODELS, False, 2), (cfg["products"], True, 3 + n_emb)]
        series = []
        for models, dashed, row0 in groups:
            for i, (label, tmpl, color) in enumerate(models):
                p = tmpl.format(**fill)
                if not Path(p).exists():
                    raise SystemExit(f"({letter}) {label}: raster not found: {p}")
                series.append((row0 + i, label, color, dashed,
                               sample_on_track(p, trk, cfg["year"], label)))
        ymax = max([gedi_m.max()] + [np.nanmax(s[4]) for s in series])
        step = nice_step(ymax)
        # Headroom so the name / metrics line never sits on the profile.
        ylim = max(2 * step, ymax) * 1.28

        axes = []
        for row, label, color, dashed, y in series:
            ax = fig.add_subplot(gs[row, col], sharex=axes[0] if axes else None)
            draw_profile(ax, t, gedi_m, y, label, color, dashed, ylim, step,
                         max_gap, args.panel_metrics)
            axes.append(ax)
            if row != series[-1][0]:
                ax.tick_params(labelbottom=False)
        for row, text in ((2, args.embedding_group_label),
                          (3 + n_emb, args.product_group_label)):
            first = next(a for a, s in zip(axes, series) if s[0] == row)
            first.text(0, 1.06, text, transform=first.transAxes, ha="left",
                       va="bottom", fontsize=8, style="italic", color="0.4")
        axes[-1].set_xlabel("Distance along transect (m)", fontsize=8.5)

    fig.text(0.012, (0.075 + 0.62) / 2, "Canopy height (m)", rotation=90,
             ha="center", va="center", fontsize=9)
    handles = [
        Line2D([], [], color="black", marker="o", ms=3.5, lw=0.6, mec="black",
               markerfacecolor="black"),
        Line2D([], [], color="0.3", lw=1.5),
        Line2D([], [], color="0.3", lw=1.5, ls=(0, (4, 2))),
    ]
    handles[0].set_color("0.72")
    fig.legend(handles, ["GEDI RH98 (footprint)", "GFM embedding model prediction",
                         "Baseline model"], loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.53, 0.0))

    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nwrote {out}\nwrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
