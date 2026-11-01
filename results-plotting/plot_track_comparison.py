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
baselines produced by potapov_baseline.py, eth_canopy_baseline.py and
echosat_baseline.py are exactly such rasters.

Multi-year rasters (ECHOSAT's cached tiles hold 2018-2024 as 7 bands, each band
described by its year) are sampled at the band whose description equals --year,
so the baseline year always matches the GEDI epoch being plotted.

Track extraction:
  - valid GEDI pixels (gedi != nodata, 0 < gedi < 12000 cm) are clustered with
    DBSCAN in metric space; each thin, elongated cluster is one orbit-track
    segment. The chosen track (default: the longest elongated one) is ordered
    along its principal axis, giving distance-along-transect in metres.

Outputs:
  <out>                      the stacked comparison figure.
  <out stem>_s2.png          a Sentinel-2 L2A true-colour image of the area
                             (growing season of --year, least-cloudy scene over
                             the transect) with the selected transect overlaid.
                             Imagery is read windowed from the public Element84
                             Earth Search COGs, so no credentials are needed.

Example:
  python plot_track_comparison.py --healpix-id 31330 --year 2019
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pyproj import Transformer
from rasterio.windows import from_bounds
from sklearn.cluster import DBSCAN

GEDI_NODATA = -9999.0
GEDI_MAX_CM = 12000

S2_STAC = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
# SCL classes that make a scene unusable for the backdrop: no-data, saturated,
# cloud shadow, medium/high-probability cloud, thin cirrus.
S2_BAD_SCL = (0, 1, 3, 8, 9, 10)

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
    ("OlmoEarth",
     "{root}/olmo_pred_rasters/{year}/healpix_{gid}_olmo.tif",
     "#d1495b", -9999.0),
    ("AlphaEarth",
     "{root}/aef_pred_rasters/healpix_{gid}_aef.tif",
     "#7b3fb0", -9999.0),
    ("TESSERA",
     "{root}/tessera_pred_rasters/{year}/healpix_{gid}_tessera.tif",
     "#1f6fb4", -9999.0),
    ("Presto",
     "{root}/presto_pred_rasters/{year}/healpix_{gid}_presto.tif",
     "#e08214", -9999.0),
    ("Prithvi",
     "{root}/prithvi_pred_rasters/{year}/healpix_{gid}_prithvi.tif",
     "#2a9d8f", -9999.0),
    # Baselines last => bottom panels.
    # ECHOSAT: 7-band cache from echosat_baseline.py; the --year band is picked
    # by its description (see load_raster_on_gedi_grid).
    ("ECHOSAT",
     "{root}/echosat/echosat_tiles/healpix_{gid}_echosat.tif",
     "#8c564b", -9999.0),
    ("Potapov",
     "{root}/potapov_baseline_{year}/potapov_tiles/healpix_{gid}_potapov.tif",
     "#6c757d", 0.0),
]


def load_raster_on_gedi_grid(path, ref_shape, year):
    """Read a raster on the GEDI grid; for multi-band rasters take the `year` band.

    A multi-band raster must label its bands with years (echosat_baseline.py sets
    the descriptions to '2018'..'2024'). Picking by description, never by a fixed
    index, means a wrong year cannot slip into the figure silently.
    """
    with rasterio.open(path) as s:
        band = 1
        if s.count > 1:
            descs = list(s.descriptions)
            if str(year) not in descs:
                raise ValueError(f"{path}: {s.count} bands described {descs}, "
                                 f"none is {year}")
            band = descs.index(str(year)) + 1
        arr = s.read(band).astype(np.float64)
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


def select_track(gedi_path, eps_m=120.0, min_samples=4, min_aspect=8.0,
                 min_len_m=300.0, rank=0, near=None, max_length_m=None):
    """Find the GEDI transect to plot in one tile (see module docstring).

    `near` = (lat, lon) overrides `rank` with the track closest to that point.
    Returns a dict: rows, cols (track pixels, ordered along track), t
    (along-track metres from the start), gedi_m (RH98, m), rank, width_m, plus
    the tile's transform, shape and epsg.
    """
    with rasterio.open(gedi_path) as s:
        gedi = s.read(1).astype(np.float64)
        transform = s.transform
        nod = s.nodata if s.nodata is not None else GEDI_NODATA
        gedi_epsg = s.crs.to_epsg() if s.crs else None
    ref_shape = gedi.shape

    valid = (gedi != nod) & (gedi > 0) & (gedi < GEDI_MAX_CM)
    rows, cols = np.where(valid)
    print(f"{Path(gedi_path).name}: {valid.sum()} valid GEDI pixels")

    xy_m = metric_coords(rows, cols, transform, ref_shape[0])
    tracks = extract_tracks(rows, cols, xy_m, eps_m, min_samples,
                            min_aspect, min_len_m)
    if not tracks:
        raise SystemExit("No elongated GEDI track found; try larger --eps-m or "
                         "lower --min-aspect / --min-len-m.")
    print(f"found {len(tracks)} elongated track segment(s); top candidates:")
    for i, d in enumerate(tracks[:8]):
        h = gedi[rows[d["idx"]], cols[d["idx"]]] / 100.0
        print(f"  rank {i}: len={d['length_m']:6.0f}m  n={d['n']:3d}  "
              f"width={d['width_m']:3.0f}m  aspect={d['aspect']:5.0f}  "
              f"GEDI H[{h.min():4.1f},{h.max():4.1f}] range={h.max()-h.min():4.1f}m")
    if near is not None:
        if gedi_epsg != 4326:
            raise SystemExit(f"--near needs an EPSG:4326 GEDI tile, got EPSG:{gedi_epsg}")
        lat, lon = near
        col_f, row_f = ~transform * (lon, lat)
        if not (0 <= row_f < ref_shape[0] and 0 <= col_f < ref_shape[1]):
            raise SystemExit(f"--near {lat} {lon} is outside {gedi_path}")
        # Same pixel-to-metre mapping the tracks were clustered in, so the
        # distance is in metres and comparable across tracks.
        pt = metric_coords(np.array([row_f]), np.array([col_f]), transform,
                           ref_shape[0])[0]
        dists = [float(np.min(np.hypot(*(xy_m[d["idx"]] - pt).T))) for d in tracks]
        rank = int(np.argmin(dists))
        print(f"--near {lat} {lon}: closest track is rank {rank}, "
              f"{dists[rank]:.0f} m away (next closest "
              f"{sorted(dists)[1]:.0f} m)" if len(dists) > 1 else "")
    if rank >= len(tracks):
        raise SystemExit(f"--track-rank {rank} out of range "
                         f"(only {len(tracks)} tracks)")
    tr = tracks[rank]

    sel = tr["idx"]
    t = tr["t"]
    if max_length_m is not None:
        keep = t <= max_length_m
        sel, t = sel[keep], t[keep]
    trk_rows, trk_cols = rows[sel], cols[sel]
    gedi_m = gedi[trk_rows, trk_cols] / 100.0   # cm -> m
    print(f"selected track rank {rank}: {len(t)} shots over "
          f"{t.max():.0f} m (width {tr['width_m']:.0f} m); "
          f"GEDI height {gedi_m.min():.1f}-{gedi_m.max():.1f} m")
    return {"rows": trk_rows, "cols": trk_cols, "t": t, "gedi_m": gedi_m,
            "rank": rank, "width_m": tr["width_m"], "transform": transform,
            "shape": ref_shape, "epsg": gedi_epsg}


def fetch_s2_rgb(lon, lat, pad_m, year, months, max_cloud, n_candidates,
                 item_id=None, pct=(1, 99), gamma=1.4):
    """Least-cloudy Sentinel-2 L2A true-colour chip around a transect.

    Scenes from the `months` window of `year` are ranked by scene cloud cover;
    the top `n_candidates` are then scored on the SCL cloud / no-data fraction
    INSIDE the chip (scene-level cloud cover says nothing about the few km we
    actually show), and the cleanest one is read at 10 m.

    Returns (rgb float HxWx3 in [0,1], extent (l, r, b, t) in the scene's UTM
    metres, track x, track y in the same CRS, item id, acquisition date).
    """
    from pystac_client import Client

    pad_deg = pad_m / 111320.0 / max(np.cos(np.radians(np.mean(lat))), 0.1)
    bbox = [lon.min() - pad_deg, lat.min() - pad_m / 111320.0,
            lon.max() + pad_deg, lat.max() + pad_m / 111320.0]
    cat = Client.open(S2_STAC)
    if item_id:
        items = list(cat.search(collections=[S2_COLLECTION], ids=[item_id]).items())
        if not items:
            raise SystemExit(f"Sentinel-2 item {item_id} not found in {S2_STAC}")
    else:
        m0, m1 = months
        last_day = 31 if m1 in (1, 3, 5, 7, 8, 10, 12) else 30
        items = list(cat.search(
            collections=[S2_COLLECTION], bbox=bbox,
            datetime=f"{year}-{m0:02d}-01/{year}-{m1:02d}-{last_day}",
            query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=200).items())
        if not items:
            raise SystemExit(f"No Sentinel-2 scenes < {max_cloud}% cloud over the "
                             f"transect in {year} months {m0}-{m1}")
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        items = items[:n_candidates]

    def chip_window(src):
        to_utm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = to_utm.transform(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        # Square chip so the panel does not depend on the track's orientation.
        half = max(x.max() - x.min(), y.max() - y.min()) / 2 + pad_m
        bounds = (cx - half, cy - half, cx + half, cy + half)
        return x, y, bounds

    best = None
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
        for it in items:
            with rasterio.open(it.assets["scl"].href) as src:
                _, _, bounds = chip_window(src)
                scl = src.read(1, window=from_bounds(*bounds, src.transform),
                               boundless=True, fill_value=0)
            bad = float(np.isin(scl, S2_BAD_SCL).mean())
            print(f"  S2 {it.id}: scene cloud {it.properties.get('eo:cloud_cover', float('nan')):5.1f}%"
                  f"  chip cloud/no-data {100 * bad:5.1f}%")
            if best is None or bad < best[0]:
                best = (bad, it)
            if bad == 0.0:
                break
        bad, it = best
        bands = []
        for key in ("red", "green", "blue"):
            with rasterio.open(it.assets[key].href) as src:
                x, y, bounds = chip_window(src)
                win = from_bounds(*bounds, src.transform)
                bands.append(src.read(1, window=win, boundless=True,
                                      fill_value=0).astype(np.float64))
                win_tf = src.window_transform(win)
                h, w = bands[-1].shape
                extent = (win_tf.c, win_tf.c + win_tf.a * w,
                          win_tf.f + win_tf.e * h, win_tf.f)
    if bad > 0.01:
        print(f"  WARNING: best scene still has {100 * bad:.1f}% cloud/no-data "
              f"in the chip; pass --s2-item to choose another")

    # One `pct` stretch shared by all three bands (robust to the L2A +1000
    # offset change; a per-band stretch shifts the colour balance, e.g. bright
    # pumice turns magenta) and a mild gamma so conifer canopy is not crushed.
    rgb = np.dstack(bands)
    valid = rgb.sum(-1) > 0
    lo, hi = np.percentile(rgb[valid], pct)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / gamma)
    rgb[~valid] = 1.0
    date = it.datetime.strftime("%Y-%m-%d") if it.datetime else "?"
    return rgb, extent, x, y, it.id, date


def plot_s2_context(out, rgb, extent, tx, ty, t, item_id, date, gid, year,
                    bar_m=1000.0):
    """Sentinel-2 chip with the transect drawn on it, start/end labelled to
    match the profile figure's distance axis."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(rgb, extent=extent, interpolation="nearest")
    halo = [pe.Stroke(linewidth=4.5, foreground="black"), pe.Normal()]
    ax.plot(tx, ty, color="#ffd60a", linewidth=2.5, solid_capstyle="round",
            path_effects=halo, zorder=3, label="GEDI transect")
    for i, txt in ((0, "0 m"), (-1, f"{t[-1]:,.0f} m")):
        ax.plot(tx[i], ty[i], "o", ms=7, mfc="white", mec="black", zorder=4)
        ax.annotate(txt, (tx[i], ty[i]), xytext=(6, 6), textcoords="offset points",
                    color="white", fontsize=10, fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    # Scale bar, bottom-left. Fixed length (not scaled to the chip) so every
    # context map in the paper carries the same bar.
    l, r, b, tp = extent
    width = r - l
    bar = float(bar_m)
    x0, y0 = l + 0.05 * width, b + 0.05 * (tp - b)
    ax.add_patch(plt.Rectangle((x0, y0), bar, 0.012 * (tp - b), color="white",
                               ec="black", lw=0.8, zorder=5))
    ax.text(x0 + bar / 2, y0 + 0.025 * (tp - b),
            f"{bar / 1000:g} km" if bar >= 1000 else f"{bar:g} m",
            ha="center", va="bottom", color="white", fontsize=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")], zorder=5)
    # North arrow (UTM grid north), top-right.
    ax.annotate("N", xy=(0.93, 0.95), xytext=(0.93, 0.85), xycoords="axes fraction",
                ha="center", va="center", color="white", fontsize=12,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2,
                                path_effects=[pe.Stroke(linewidth=4,
                                                        foreground="black"),
                                              pe.Normal()]),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    ax.set_xlim(l, r)
    ax.set_ylim(b, tp)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Sentinel-2 L2A true colour, {date} — tile {gid} ({year})",
                 fontsize=11)
    ax.text(0.99, 0.01, "Contains modified Copernicus Sentinel data",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="white", path_effects=[pe.withStroke(linewidth=1.5,
                                                       foreground="black")])
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (scene {item_id})")


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
    ap.add_argument("--near", type=float, nargs=2, default=None, metavar=("LAT", "LON"),
                    help="Plot the track passing closest to this point (WGS84 lat lon) "
                         "instead of picking by --track-rank. The GEDI tile must be in "
                         "EPSG:4326 (the HEALPix tiles are).")
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
    ap.add_argument("--s2", action=argparse.BooleanOptionalAction, default=True,
                    help="Write the Sentinel-2 context map <out stem>_s2.png "
                         "(needs network; default: on)")
    ap.add_argument("--s2-year", type=int, default=None,
                    help="Year of the Sentinel-2 backdrop (default: --year)")
    ap.add_argument("--s2-months", type=int, nargs=2, default=(6, 9),
                    metavar=("FIRST", "LAST"),
                    help="Month window to search for scenes (default: 6 9 = Jun-Sep)")
    ap.add_argument("--s2-max-cloud", type=float, default=20.0,
                    help="Max scene-level cloud cover %% to consider (default: 20)")
    ap.add_argument("--s2-candidates", type=int, default=8,
                    help="How many of the least-cloudy scenes to score on the "
                         "chip's own SCL cloud fraction (default: 8)")
    ap.add_argument("--s2-item", default=None,
                    help="Force a specific Earth Search item id, e.g. "
                         "S2B_10TEN_20190925_1_L2A")
    ap.add_argument("--s2-pad-m", type=float, default=1500.0,
                    help="Margin around the transect in the S2 map, metres "
                         "(default: 1500)")
    ap.add_argument("--s2-scalebar-m", type=float, default=1000.0,
                    help="Scale-bar length in the S2 map, metres; keep it the "
                         "same across figures (default: 1000)")
    args = ap.parse_args()

    gid = args.healpix_id
    trk = select_track(Path(args.gedi_dir) / f"healpix_{gid}_gedi.tif",
                       args.eps_m, args.min_samples, args.min_aspect,
                       args.min_len_m, args.track_rank, args.near,
                       args.max_length_m)
    gedi_epsg, transform, ref_shape = trk["epsg"], trk["transform"], trk["shape"]
    trk_rows, trk_cols, t, gedi_m = trk["rows"], trk["cols"], trk["t"], trk["gedi_m"]
    args.track_rank = trk["rank"]

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
        arr, rnod = load_raster_on_gedi_grid(p, ref_shape, args.year)
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
                     [np.nanmax(y) for _, _, y, n, _, _ in model_series if n]) * 1.25   # headroom so the panel labels never cover the profile

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

    # Context map: transect over a Sentinel-2 true-colour image of the same year.
    if args.s2:
        if gedi_epsg != 4326:
            raise SystemExit(f"S2 context map needs an EPSG:4326 GEDI tile, "
                             f"got EPSG:{gedi_epsg} (or pass --no-s2)")
        lon, lat = transform * (trk_cols + 0.5, trk_rows + 0.5)
        rgb, extent, tx, ty, item_id, date = fetch_s2_rgb(
            np.asarray(lon), np.asarray(lat), args.s2_pad_m,
            args.s2_year or args.year, args.s2_months, args.s2_max_cloud,
            args.s2_candidates, args.s2_item)
        plot_s2_context(out.with_name(out.stem + "_s2.png"), rgb, extent, tx, ty,
                        t, item_id, date, gid, args.s2_year or args.year,
                        args.s2_scalebar_m)

if __name__ == "__main__":
    main()
