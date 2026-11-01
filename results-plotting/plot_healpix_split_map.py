#!/usr/bin/env python3
"""
Map of the HEALPix tiles coloured by data split, over a basemap with a
lat/lon frame, so the study area's location is clear.

  train -> green, val -> yellow, test -> red

The split comes from the shapefile's `split` attribute (--split-col). With
--master-dir (e.g. wacv_geocv/master_tiles_2019), train / test are instead taken
from master_train_tile_ids.txt / master_test_tile_ids.txt: shapefile train or test
tiles absent from those lists were dropped for missing / corrupt embeddings (see
coverage_report.json) and are drawn black as "Missing embedding coverage".
Validation tiles are not part of the master lists (coverage was only checked for
train and test), so they keep their shapefile colour.

--inset-tile <healpix id> adds a zoomed inset (in the empty area east of the
tiles) of that tile's GEDI RH98 footprints from --gedi-dir, over satellite
imagery, with the tile outlined on the main map and joined to the inset.

The map is
drawn in EPSG:4326 so the frame ticks are plain degrees. By default one degree
of longitude and one of latitude get the same length on the page (plate carree,
as QGIS / GIS tools show EPSG:4326), which is how the HEALPix tiles are usually
seen. --local-aspect instead stretches y by 1/cos(mid-latitude) so ground
distances are equal in x and y near the map centre. The basemap
(contextily web tiles, reprojected to EPSG:4326) needs network access; pass
--no-basemap to draw the tiles alone.

Example:
  python results-plotting/plot_healpix_split_map.py \\
      --out ~/Documents/ASU/JPL/results/wacv_geocv/healpix_split_map.png
"""

import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import FuncFormatter

DEFAULT_SHP = "/Users/kdoerkse/Documents/ASU/JPL/data/shapefiles/healpix_shapefile.shp"
DEFAULT_OUT = "/Users/kdoerkse/Documents/ASU/JPL/results/wacv_geocv/healpix_split_map.png"

# split value -> (legend label, fill colour). Legend order is this order.
SPLITS = {
    "train": ("Train", "#2ca02c"),
    "val": ("Validation", "#ffd60a"),
    "test": ("Test", "#d62728"),
    # Only with --master-dir: train/test tiles dropped from the master list.
    # Two lines so the legend is no wider than the plain split map's.
    "missing": ("Missing embedding\ncoverage", "#000000"),
}


def fmt_lon(x, _):
    return f"{abs(x):g}°{'W' if x < 0 else 'E' if x > 0 else ''}"


def fmt_lat(y, _):
    return f"{abs(y):g}°{'S' if y < 0 else 'N' if y > 0 else ''}"


def apply_master_lists(gdf, splits, master_dir):
    """Relabel shapefile train/test tiles missing from the master lists as
    'missing'. Fails if a master tile disagrees with the shapefile split, so a
    list from the wrong year or split scheme cannot be drawn silently."""
    def read_ids(name):
        path = master_dir / name
        if not path.exists():
            raise SystemExit(f"master list not found: {path}")
        return {int(l) for l in path.read_text().split() if l.strip()}

    master = {"train": read_ids("master_train_tile_ids.txt"),
              "test": read_ids("master_test_tile_ids.txt")}
    ids = gdf["healpix_id"].astype(int)
    shp_split = dict(zip(ids, splits))
    for name, tiles in master.items():
        wrong = sorted(t for t in tiles if shp_split.get(t) != name)
        if wrong:
            raise SystemExit(f"{len(wrong)} master {name} tiles are not shapefile "
                             f"{name} tiles (e.g. {wrong[:5]})")
    out = splits.copy()
    for name, tiles in master.items():
        out[(splits == name) & ~ids.isin(tiles)] = "missing"
    dropped = sorted(ids[out == "missing"])
    print(f"master lists {master_dir}: train {len(master['train'])}, test "
          f"{len(master['test'])}; missing embedding coverage: {dropped}")
    return out


def add_gedi_inset(ax, gdf, splits, args):
    """Zoomed inset of one tile's GEDI RH98 footprints over satellite imagery."""
    ids = gdf["healpix_id"].astype(int)
    if args.inset_tile not in set(ids):
        raise SystemExit(f"--inset-tile {args.inset_tile} is not in the shapefile")
    tile = gdf[ids == args.inset_tile]
    split = splits[ids == args.inset_tile].iloc[0]
    gedi_path = Path(args.gedi_dir).expanduser() / f"healpix_{args.inset_tile}_gedi.tif"
    with rasterio.open(gedi_path) as s:
        gedi = s.read(1).astype(np.float64)
        nod = s.nodata
        tf = s.transform
    valid = (gedi > 0) & (gedi < 12000)
    if nod is not None:
        valid &= gedi != nod
    rows, cols = np.where(valid)
    lon, lat = tf * (cols + 0.5, rows + 0.5)
    h = gedi[rows, cols] / 100.0    # cm -> m
    print(f"inset tile {args.inset_tile} ({split}): {len(h)} GEDI footprints, "
          f"RH98 median {np.median(h):.1f} m, p95 {np.percentile(h, 95):.1f} m")

    # Square window (in degrees, same 1:1 aspect as the main map) around the tile.
    tx0, ty0, tx1, ty1 = tile.total_bounds
    cx_, cy_ = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    half = max(tx1 - tx0, ty1 - ty0) / 2 * 1.06
    ins = ax.inset_axes(args.inset_rect)
    ins.set_xlim(cx_ - half, cx_ + half)
    ins.set_ylim(cy_ - half, cy_ + half)
    ins.set_aspect(ax.get_aspect())
    if not args.no_basemap:
        import contextily as cx
        import xyzservices.providers as xyz
        try:
            cx.add_basemap(ins, crs="EPSG:4326", source=xyz.Esri.WorldImagery,
                           attribution=False, zorder=0)
        except Exception as e:
            print(f"WARNING: inset basemap failed ({e})")
    order = np.argsort(h)   # tallest drawn last so they stay visible
    sc = ins.scatter(lon[order], lat[order], c=h[order], s=0.6, lw=0,
                     cmap="viridis", vmin=0, vmax=60, zorder=2, rasterized=True)
    # aspect=None: GeoPandas would otherwise reset the axes to its own
    # 1/cos(lat) aspect and stretch the inset relative to the main map.
    tile.boundary.plot(ax=ins, color="white", lw=1.0, zorder=3, aspect=None)
    ins.set_xlim(cx_ - half, cx_ + half)
    ins.set_ylim(cy_ - half, cy_ + half)
    ins.set_aspect(ax.get_aspect())
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.8)

    # Tile outlined on the main map and joined to the inset.
    tile.boundary.plot(ax=ax, color="black", lw=1.4, zorder=4, aspect=None)
    vx, vy = np.asarray(tile.geometry.iloc[0].exterior.coords.xy)
    top, bottom = np.argmax(vy), np.argmin(vy)
    for i, corner in ((top, (0, 1)), (bottom, (0, 0))):
        ax.add_artist(ConnectionPatch(
            xyA=(vx[i], vy[i]), coordsA=ax.transData,
            xyB=corner, coordsB=ins.transAxes,
            color="black", lw=0.7, alpha=0.8, zorder=5))

    label = {"train": "training", "val": "validation", "test": "test"}.get(split, split)
    ins.set_title(f"Example {label} tile {args.inset_tile}\n"
                  f"GEDI RH98 footprints (n={len(h):,})", fontsize=7.5, pad=3)
    cax = ins.inset_axes([0.0, -0.075, 1.0, 0.04])
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("Canopy height (m)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cb.outline.set_linewidth(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile", default=DEFAULT_SHP)
    ap.add_argument("--split-col", default="split",
                    help="Attribute holding train / val / test (default: split)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output PNG; a PDF with the same stem is written too")
    ap.add_argument("--pad-deg", type=float, default=1.0,
                    help="Margin around the tiles, degrees (default: 1)")
    ap.add_argument("--basemap", default="Esri.WorldGrayCanvas",
                    help="xyzservices provider needing no API key, e.g. "
                         "Esri.WorldGrayCanvas, Esri.WorldTerrain, Esri.WorldImagery "
                         "(CARTO now needs an API key; default: Esri.WorldGrayCanvas)")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.75, help="Tile fill opacity")
    ap.add_argument("--title", default=None)
    ap.add_argument("--master-dir", default=None,
                    help="Dir with master_train_tile_ids.txt / master_test_tile_ids.txt; "
                         "train/test tiles not listed are drawn black as missing "
                         "embedding coverage")
    ap.add_argument("--inset-tile", type=int, default=None,
                    help="HEALPix id to show zoomed in with its GEDI footprints "
                         "(e.g. 31329, a 2019 training tile)")
    ap.add_argument("--gedi-dir",
                    default="/Users/kdoerkse/Documents/ASU/JPL/data/gedi_tiles_2019_healpix",
                    help="GEDI tiles (healpix_<id>_gedi.tif, RH98 in cm) for the inset")
    ap.add_argument("--inset-rect", type=float, nargs=4,
                    default=(0.625, 0.47, 0.36, 0.25), metavar=("X", "Y", "W", "H"),
                    help="Inset position in main-axes fractions (default sits in the "
                         "empty area east of the tiles)")
    ap.add_argument("--local-aspect", action="store_true",
                    help="Stretch y by 1/cos(mid-latitude) (equal ground distance "
                         "in x and y) instead of plain 1:1 degrees")
    args = ap.parse_args()

    gdf = gpd.read_file(args.shapefile)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    if args.split_col not in gdf.columns:
        raise SystemExit(f"no '{args.split_col}' column; columns are {list(gdf.columns)}")
    splits = gdf[args.split_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(splits) - {"train", "val", "test"})
    if unknown:
        raise SystemExit(f"unexpected split values {unknown}; expected train/val/test")
    if args.master_dir:
        splits = apply_master_lists(gdf, splits, Path(args.master_dir).expanduser())
    counts = splits.value_counts()
    print(f"{len(gdf)} tiles: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in SPLITS))

    x0, y0, x1, y1 = gdf.total_bounds
    x0, y0, x1, y1 = x0 - args.pad_deg, y0 - args.pad_deg, x1 + args.pad_deg, y1 + args.pad_deg
    mid_lat = (y0 + y1) / 2
    aspect = 1 / np.cos(np.radians(mid_lat)) if args.local_aspect else 1.0

    fig_w = 6.0
    fig_h = fig_w * (y1 - y0) * aspect / (x1 - x0) + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(aspect)

    if not args.no_basemap:
        import contextily as cx
        import xyzservices.providers as xyz
        provider = xyz
        for part in args.basemap.split("."):
            provider = provider[part]
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=provider, attribution_size=5,
                           zorder=0)
        except Exception as e:   # offline / tile server down: still make the map
            print(f"WARNING: basemap failed ({e}); drawing tiles without it")

    for key, (label, color) in SPLITS.items():
        sub = gdf[splits == key]
        if len(sub):
            sub.plot(ax=ax, facecolor=color, edgecolor="0.15", linewidth=0.4,
                     alpha=1.0 if key == "missing" else args.alpha, zorder=2)
    # Re-apply limits/aspect: GeoDataFrame.plot may reset them.
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(aspect)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.tick_params(labelsize=8, top=True, right=True, labeltop=True,
                   labelright=True, direction="out")
    ax.grid(color="0.4", lw=0.3, alpha=0.4, ls=":", zorder=1)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    handles = [Patch(facecolor=c, edgecolor="0.15", lw=0.4,
                     alpha=1.0 if k == "missing" else args.alpha,
                     label=f"{lab} ({counts.get(k, 0)})")
               for k, (lab, c) in SPLITS.items()
               if k != "missing" or args.master_dir]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
              title=f"HEALPix tiles (n={len(gdf)})", title_fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=10, pad=22)
    if args.inset_tile is not None:
        add_gedi_inset(ax, gdf, splits, args)

    out = Path(args.out).expanduser()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}\nwrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
