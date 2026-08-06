#!/usr/bin/env python3
"""
Along-track canopy-height comparison figure for a paper.

Picks one GEDI ground track inside a HEALPix tile and plots, in stacked panels
(one per model), the GEDI RH98 "ground truth" profile with each model's
prediction overlaid along the same transect -- so the per-track difference
between models is directly visible. Modelled on the ALS-transect style figure
performance_comparison_chm.png, adapted for GEDI-as-ground-truth (there is no ALS
point cloud): GEDI is drawn as a black line with light grey shading in every
panel.

Predictions are read from per-tile rasters ALREADY on the GEDI grid (same CRS /
transform / shape), so they are sampled at the track pixels directly. The
baselines produced by potapov_baseline.py and eth_canopy_baseline.py are exactly
such rasters.

Track extraction:
  - valid GEDI pixels (gedi != nodata, 0 < gedi < 12000 cm) are clustered with
    DBSCAN in metric space; each thin, elongated cluster is one orbit-track
    segment. The chosen track (default: the longest elongated one) is ordered
    along its principal axis, giving distance-along-transect in metres.

Outputs:
  <out>                      the stacked comparison figure.
  <out stem>_overview.png    the tile's GEDI pixels with the selected track
                             highlighted, to verify / choose a different track.

Example:
  python plot_track_comparison.py --healpix-id 31330 --year 2019
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

GEDI_NODATA = -9999.0
GEDI_MAX_CM = 12000

DEFAULT_PRED_ROOT = "/Users/kdoerkse/Documents/ASU/JPL/data"

# Default panels, TOP to BOTTOM. Each entry is
#   (label, raster path template, colour, nodata)
# and the template may use {root} (--pred-root), {year} and {gid}.
#
# Panel order is the list order, so the learned encoders come first and the
# Potapov baseline is last -- it is the reference every encoder is being judged
# against, and putting it at the foot of the stack is the convention in the
# ALS-transect figures this is modelled on. Models whose raster is missing are
# dropped from the figure but reported at the end, so a silently absent model
# cannot be mistaken for one that simply performed badly.
#
# The bracketed resolution is the encoder's NATIVE support, not the prediction
# grid -- every raster here is already on the GEDI grid. Worth keeping visible,
# since spatial support differs by ~50x across these encoders.
DEFAULT_MODELS = [
    ("OlmoEarth RF (40m)",
     "{root}/olmo_pred_rasters/{year}/healpix_{gid}_olmo.tif",
     "#d1495b", -9999.0),
    ("AlphaEarth RF (10m)",
     "{root}/aef_pred_rasters/healpix_{gid}_aef.tif",
     "#7b3fb0", -9999.0),
    ("TESSERA RF (10m)",
     "{root}/tessera_pred_rasters/{year}/healpix_{gid}_tessera.tif",
     "#1f6fb4", -9999.0),
    ("Presto RF (10m)",
     "{root}/presto_pred_rasters/{year}/healpix_{gid}_presto.tif",
     "#e08214", -9999.0),
    ("Prithvi RF (30m)",
     "{root}/prithvi_pred_rasters/{year}/healpix_{gid}_prithvi.tif",
     "#2a9d8f", -9999.0),
    # Baseline last => bottom panel.
    ("Potapov (30m)",
     "{root}/potapov_baseline_{year}/potapov_tiles/healpix_{gid}_potapov.tif",
     "#6c757d", 0.0),
]


def load_raster_on_gedi_grid(path, ref_shape):
    """Read a single-band raster and confirm it matches the GEDI grid shape."""
    with rasterio.open(path) as s:
        arr = s.read(1).astype(np.float64)
        nod = s.nodata
    if arr.shape != ref_shape:
        raise ValueError(f"{path}: shape {arr.shape} != GEDI grid {ref_shape}")
    return arr, nod


def metric_coords(rows, cols, transform, n_rows):
    """Convert pixel (row, col) to local metres (for distances / clustering)."""
    px_x = abs(transform.a)
    px_y = abs(transform.e)
    center_lat = transform.f + transform.e * n_rows / 2.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
    x_m = cols * px_x * m_per_deg_lon
    y_m = rows * px_y * m_per_deg_lat
    return np.column_stack([x_m, y_m])


def extract_tracks(rows, cols, xy_m, eps_m, min_samples, min_aspect, min_len_m):
    """Cluster valid pixels into elongated track segments.

    Returns a list of dicts sorted best-first (most points, then longest),
    each: {idx, n, length_m, aspect, t (along-track metres), order}.
    """
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(xy_m)
    tracks = []
    for lab in np.unique(labels):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        pts = xy_m[idx]
        c = pts - pts.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        t = c @ vt[0]                    # along-axis coordinate (m)
        w = c @ vt[1]                    # across-axis coordinate (m)
        length_m = float(t.max() - t.min())
        width_m = float(w.max() - w.min())
        aspect = length_m / max(width_m, 1.0)
        if aspect < min_aspect or length_m < min_len_m:
            continue
        order = np.argsort(t)
        tracks.append({
            "idx": idx[order],
            "n": len(idx),
            "length_m": length_m,
            "aspect": aspect,
            "width_m": width_m,
            "t": (t[order] - t[order].min()),
        })
    # Rank by length: a longer single-beam transect makes the clearer figure.
    tracks.sort(key=lambda d: d["length_m"], reverse=True)
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--healpix-id", type=int, required=True)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--gedi-dir",
                    default="/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2019_healpix")
    ap.add_argument("--out", default=None,
                    help="Output figure path (default: track_comparison_<gid>.png in cwd)")
    ap.add_argument("--eps-m", type=float, default=120.0,
                    help="DBSCAN neighbourhood radius in metres (bridges along-track gaps)")
    ap.add_argument("--min-samples", type=int, default=4)
    ap.add_argument("--min-aspect", type=float, default=8.0,
                    help="Minimum length/width ratio for a cluster to count as a track")
    ap.add_argument("--min-len-m", type=float, default=300.0)
    ap.add_argument("--track-rank", type=int, default=0,
                    help="Which track to plot: 0 = best (most points), 1 = next, ...")
    ap.add_argument("--model", action="append", default=None, metavar="SPEC",
                    help="Model panel spec 'label|path_template|color|nodata' "
                         "({gid} is filled with the healpix id). Repeat for each "
                         "model; overrides DEFAULT_MODELS. Example: "
                         "'AlphaEarth (2019)|/data/aef_pred_rasters/healpix_{gid}_aef2019.tif|#7b3fb0|-9999'")
    ap.add_argument("--max-length-m", type=float, default=None,
                    help="Crop the transect to the first N metres (for a tighter figure)")
    ap.add_argument("--pred-root", default=DEFAULT_PRED_ROOT,
                    help="Base dir filled into {root} in the default path templates "
                         "(default: %(default)s)")
    ap.add_argument("--panel-metrics", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Annotate each panel with along-track RMSE and bias "
                         "vs GEDI (default: on)")
    args = ap.parse_args()

    gid = args.healpix_id
    gedi_path = Path(args.gedi_dir) / f"healpix_{gid}_gedi.tif"
    with rasterio.open(gedi_path) as s:
        gedi = s.read(1).astype(np.float64)
        transform = s.transform
        nod = s.nodata if s.nodata is not None else GEDI_NODATA
    ref_shape = gedi.shape

    valid = (gedi != nod) & (gedi > 0) & (gedi < GEDI_MAX_CM)
    rows, cols = np.where(valid)
    print(f"tile {gid}: {valid.sum()} valid GEDI pixels")

    xy_m = metric_coords(rows, cols, transform, ref_shape[0])
    tracks = extract_tracks(rows, cols, xy_m, args.eps_m, args.min_samples,
                            args.min_aspect, args.min_len_m)
    if not tracks:
        raise SystemExit("No elongated GEDI track found; try larger --eps-m or "
                         "lower --min-aspect / --min-len-m.")
    print(f"found {len(tracks)} elongated track segment(s); top candidates:")
    for i, d in enumerate(tracks[:8]):
        h = gedi[rows[d["idx"]], cols[d["idx"]]] / 100.0
        print(f"  rank {i}: len={d['length_m']:6.0f}m  n={d['n']:3d}  "
              f"width={d['width_m']:3.0f}m  aspect={d['aspect']:5.0f}  "
              f"GEDI H[{h.min():4.1f},{h.max():4.1f}] range={h.max()-h.min():4.1f}m")
    if args.track_rank >= len(tracks):
        raise SystemExit(f"--track-rank {args.track_rank} out of range "
                         f"(only {len(tracks)} tracks)")
    tr = tracks[args.track_rank]

    sel = tr["idx"]
    t = tr["t"]
    if args.max_length_m is not None:
        keep = t <= args.max_length_m
        sel, t = sel[keep], t[keep]
    trk_rows, trk_cols = rows[sel], cols[sel]
    gedi_m = gedi[trk_rows, trk_cols] / 100.0   # cm -> m
    print(f"selected track rank {args.track_rank}: {len(t)} shots over "
          f"{t.max():.0f} m; GEDI height {gedi_m.min():.1f}-{gedi_m.max():.1f} m")

    # Model panels: from --model specs if given, else DEFAULT_MODELS.
    if args.model:
        models = []
        for spec in args.model:
            label, tmpl, color, nod = spec.split("|")
            models.append((label, tmpl, color, float(nod)))
    else:
        models = DEFAULT_MODELS

    # Sample each model raster at the track pixels.
    model_series, missing = [], []
    for label, tmpl, color, mnod in models:
        p = tmpl.format(gid=gid, year=args.year, root=args.pred_root)
        if not Path(p).exists():
            print(f"  skip {label}: raster not found ({p})")
            missing.append(label)
            continue
        arr, rnod = load_raster_on_gedi_grid(p, ref_shape)
        vals = arr[trk_rows, trk_cols]
        mask = np.isfinite(vals)
        for nd in (mnod, rnod):
            if nd is not None:
                mask &= (vals != nd)
        y = np.where(mask, vals, np.nan)
        # Per-panel skill against GEDI on THIS track only -- indicative, not the
        # paper's headline metric (that comes from the full test split).
        if mask.any():
            d = y[mask] - gedi_m[mask]
            rmse = float(np.sqrt(np.mean(d ** 2)))
            bias = float(np.mean(d))
        else:
            rmse = bias = float("nan")
        model_series.append((label, color, y, int(mask.sum()), rmse, bias))
        print(f"  {label}: {int(mask.sum())}/{len(vals)} valid on track  "
              f"RMSE={rmse:.2f} m  bias={bias:+.2f} m")

    if not model_series:
        raise SystemExit("No model rasters available to plot.")
    if missing:
        print(f"\nWARNING: {len(missing)} model(s) absent from the figure: "
              f"{', '.join(missing)}")

    out = Path(args.out) if args.out else Path(f"track_comparison_{gid}.png")
    ymax = np.nanmax([gedi_m.max()] +
                     [np.nanmax(y) for _, _, y, n, _, _ in model_series if n]) * 1.1

    n_panels = len(model_series)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.1 * n_panels + 0.5),
                             sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, color, y, n, rmse, bias) in zip(axes, model_series):
        # GEDI ground truth: light shading + black line, repeated in each panel.
        ax.fill_between(t, 0, gedi_m, color="0.75", alpha=0.6, linewidth=0, zorder=1)
        ax.plot(t, gedi_m, color="black", linewidth=1.2, zorder=2,
                label="GEDI RH98")
        # Model prediction line.
        ax.plot(t, y, color=color, linewidth=2.2, zorder=3, label=label)
        ax.set_ylim(0, ymax)
        ax.set_xlim(t.min(), t.max())
        ax.text(0.015, 0.90, label, transform=ax.transAxes, ha="left", va="top",
                color=color, fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        if args.panel_metrics and n:
            ax.text(0.985, 0.90, f"RMSE {rmse:.1f} m   bias {bias:+.1f} m",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="0.25",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.7))
        ax.set_ylabel("Height (m)")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Distance along transect (m)")
    axes[0].set_title(
        f"Canopy height along GEDI track — tile {gid}  ({args.year}, "
        f"{len(t)} shots, {t.max():.0f} m)  |  black = GEDI RH98",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Overview map: all GEDI pixels + selected track highlighted.
    ov = out.with_name(out.stem + "_overview.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(cols, rows, s=1, c="0.7", label="GEDI valid pixels")
    ax.scatter(trk_cols, trk_rows, s=8, c="crimson", label="selected track")
    ax.scatter([trk_cols[0]], [trk_rows[0]], s=40, c="lime",
               edgecolor="black", zorder=5, label="track start")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"tile {gid}: GEDI pixels & selected track (rank {args.track_rank})")
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(ov, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ov}")


if __name__ == "__main__":
    main()
