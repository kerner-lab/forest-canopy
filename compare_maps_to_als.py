#!/usr/bin/env python3
"""
Score wall-to-wall canopy-height maps against ALS RH98 ground truth over the ALS extent.

Compares any number of gridded height products (Alpha Earth RF prediction, LANDFIRE CH,
...) to the ALS RH98 band, on the SAME pixels, and reports RMSE / R^2 / bias so the
products can be ranked fairly.

Pipeline
--------
1. GRID: the ALS raster defines the analysis grid (CRS, transform, size). Nothing is
   resampled onto anything else — every map is brought onto the ALS grid.
2. ALIGN: each map is read through a WarpedVRT onto that grid (windowed, so a CONUS-wide
   LANDFIRE tif is never read whole). A map already on the ALS grid is read directly,
   with no resampling at all.
3. SCALE / MASK: per-map unit scaling (LANDFIRE is metres x 10) and nodata handling.
4. COMMON MASK: metrics are computed only on pixels valid in the ALS *and in every map*,
   so the products are scored on an identical pixel set. This is the whole point — a map
   must not look better merely by declining to predict the hard pixels.
5. METRICS: overall RMSE / MAE / R^2 / Pearson r / bias, plus height-stratified metrics
   (reusing metrics_stratified.py, so these numbers are directly comparable to the RF
   training/backfill numbers).
6. OUTPUT: printed tables, a JSON + CSV sidecar, a comparison figure, and optionally a
   band-stacked GeoTIFF of the aligned layers for inspection in QGIS.

Usage
-----
    python compare_maps_to_als.py \
        --als-tif  /path/WA_PierceCounty_1_2020_4326_ALS.tif \
        --aef-tif  /path/wall-to-wall-pred-WA-2020-alphaearth.tif \
        --evh-tif  /path/LF2025_EVH_CONUS/Tif/LF2025_EVH_CONUS.tif \
        --out-dir  /path/results/als_comparison

Notes
-----
- ALS band: RH98 is band 2 in the Pierce County stack (band descriptions confirm it).
  --als-band overrides; the script warns if the band's description is not "rh98".
- Two LANDFIRE products are supported and they are NOT interchangeable:
    --evh-tif       EVH (Existing Vegetation Height), class-coded: 101-199 = tree height at
                    1 m steps, 201-230 = shrub (0.1 m steps), 301-310 = herb (0.1 m steps),
                    and 11-100 = land-cover classes with no height. --evh-classes picks how
                    much of that counts (tree / veg / all). This is the better product for
                    a height comparison: 1 m tree resolution vs CH's 4 m bins.
    --landfire-tif  CH (Canopy Height), binned forest height, metres x 10 — see below.
- LANDFIRE CH is a BINNED product, not a continuous one. Over this AOI it takes only the
  14 values from LF2022_CH.csv (0, 30, 70, 110, ... 510) = metres x 10, where each value
  labels a ~4 m wide class ("190" = 17-21 m). Dividing by 10 gives the class midpoint, so
  LANDFIRE carries an irreducible ~+/-2 m quantisation error before any model error. Keep
  that in mind when reading its RMSE against continuous ALS.
- LANDFIRE value 0 is "Non-Forested", NOT nodata. That is a definitional difference, not a
  prediction, and it is ~48% of this AOI, so it dominates the comparison:
      --landfire-nonforest zero  (default) treat it as a real 0 m prediction
      --landfire-nonforest drop            exclude those pixels from the common mask
  Either way the choice applies to the common mask, so every map is still scored on the
  same pixels. Run both — if the ranking flips, say so in the write-up.
- Year mismatch is recorded in the JSON sidecar as a caveat but is NOT corrected. Check it:
  a 2020 ALS flight vs an LF2025 EVH raster is five years of growth and disturbance charged
  to LANDFIRE that the year-matched AEF map does not pay.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_stratified import overall_metrics, stratified_metrics, print_stratified

# LANDFIRE CH: -9999 = "Fill-NoData" per LF2022_CH.csv; the tif itself declares 32767.
LANDFIRE_NODATA = (32767, -9999)
LANDFIRE_SCALE = 0.1  # stored as metres x 10


def decode_landfire_evh(a, mode):
    """LANDFIRE EVH class values -> height in metres; NaN where the class is not a height.

      101-199  Tree  Height = (value - 100) m       (1 m steps; 199 = ">= 99 m")
      201-230  Shrub Height = (value - 200) / 10 m  (230 = ">= 3.0 m")
      301-310  Herb  Height = (value - 300) / 10 m  (310 = ">= 1 m")

    Everything else (11 Open Water, 21-25 Developed, 31 Barren, 61-82 crops, 100 Sparse
    Vegetation Canopy, ...) is a LAND-COVER class carrying no height at all. `mode` decides
    what happens to the non-tree pixels:

      tree  only 101-199 is a height; shrub, herb and land cover are dropped. The strictest
            read — scores EVH only where it claims to be measuring trees.
      veg   tree + shrub + herb heights; land cover dropped.
      all   tree + shrub + herb heights, and land cover pinned to 0 m — the wall-to-wall
            read, where every pixel gets a number as the AEF map does.
    """
    out = np.full(a.shape, np.nan, dtype="float64")
    tree = (a >= 101) & (a <= 199)
    shrub = (a >= 201) & (a <= 230)
    herb = (a >= 301) & (a <= 310)
    out[tree] = a[tree] - 100.0
    if mode in ("veg", "all"):
        out[shrub] = (a[shrub] - 200.0) / 10.0
        out[herb] = (a[herb] - 300.0) / 10.0
    if mode == "all":
        out[np.isfinite(a) & ~tree & ~shrub & ~herb] = 0.0
    return out


@dataclass
class MapSpec:
    """One map to score against the ALS."""
    label: str
    path: str
    scale: float = 1.0
    resampling: str = "bilinear"
    band: int = 1
    extra_nodata: tuple = ()
    drop_zero: bool = False  # treat 0 as nodata rather than a 0 m prediction
    decode: object = None    # callable(raw)->metres, for class-coded products (EVH)
    note: str = ""           # shown in the log/sidecar (e.g. the EVH class mode)
    # filled in by align():
    values: np.ndarray = field(default=None, repr=False)
    regridded: bool = None


@dataclass
class Grid:
    crs: object
    transform: object
    width: int
    height: int
    profile: dict = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Reading / alignment
# ---------------------------------------------------------------------------

def read_als(als_tif, band, expect="rh98"):
    """ALS band as float64 with NaN nodata, plus the grid it defines."""
    with rasterio.open(als_tif) as src:
        if band > src.count:
            sys.exit(f"ERROR: --als-band {band} but {als_tif} has only {src.count} bands")
        desc = src.descriptions[band - 1]
        if desc and expect and desc.lower() != expect.lower():
            print(f"WARNING: ALS band {band} is described as '{desc}', not '{expect}'. "
                  f"Check --als-band.", file=sys.stderr)
        print(f"ALS: {als_tif}\n  band {band} = '{desc}'  {src.width}x{src.height}  "
              f"{src.crs}  res={src.res}")
        arr = src.read(band, masked=True).astype("float64").filled(np.nan)
        grid = Grid(src.crs, src.transform, src.width, src.height, src.profile.copy())
    return arr, grid


def same_grid(src, grid, tol=1e-9):
    """True when src already sits exactly on the analysis grid (no resampling needed)."""
    return (src.crs == grid.crs
            and src.width == grid.width and src.height == grid.height
            and all(abs(a - b) < tol for a, b in
                    zip(src.transform.to_gdal(), grid.transform.to_gdal())))


def align(spec, grid):
    """Bring one map onto the analysis grid, scaled to metres, NaN outside valid data."""
    p = Path(spec.path)
    if not p.exists():
        sys.exit(f"ERROR: {spec.label} raster not found: {spec.path}")

    with rasterio.open(spec.path) as src:
        if spec.band > src.count:
            sys.exit(f"ERROR: {spec.label} has {src.count} band(s), --band {spec.band} requested")
        spec.regridded = not same_grid(src, grid)
        if not spec.regridded:
            raw = src.read(spec.band, masked=True)
            how = "already on the ALS grid (no resampling)"
        else:
            # WarpedVRT reads only the source windows the destination grid touches, so a
            # CONUS-wide source costs a subset, not a full read.
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=Resampling[spec.resampling]) as vrt:
                raw = vrt.read(spec.band, masked=True)
            how = f"warped {src.crs} -> {grid.crs} ({spec.resampling})"

    a = raw.astype("float64").filled(np.nan)
    n_read = int(np.isfinite(a).sum())

    # Sentinel values first, so the thresholds are still in native units.
    for nd in spec.extra_nodata:
        a[a == nd] = np.nan

    if spec.decode is not None:
        a = spec.decode(a)          # class codes -> metres (already scaled)
    else:
        if spec.drop_zero:
            a[a == 0] = np.nan
        a *= spec.scale

    n_valid = int(np.isfinite(a).sum())
    n_zero = int(np.nansum(a == 0))
    spec.values = a
    print(f"{spec.label}: {spec.path}\n  {how}"
          + (f"  [{spec.note}]" if spec.note else "") + "\n"
          f"  valid {n_valid:,}/{a.size:,} px ({100*n_valid/a.size:.1f}%)"
          + (f"  [{n_read - n_valid:,} dropped as nodata/non-height class]"
             if n_read != n_valid else "")
          + (f"  zeros={n_zero:,}" if n_zero else ""))
    return spec


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def common_mask(als, specs, als_min, als_max):
    """Pixels valid in the ALS and in every map — the identical set all maps are scored on."""
    m = np.isfinite(als)
    print(f"\nCommon-pixel mask:\n  ALS finite               {m.sum():>12,}")
    if als_min is not None:
        m &= als >= als_min
        print(f"  & ALS >= {als_min:<15g} {m.sum():>12,}")
    if als_max is not None:
        m &= als <= als_max
        print(f"  & ALS <= {als_max:<15g} {m.sum():>12,}")
    for s in specs:
        m &= np.isfinite(s.values)
        print(f"  & {s.label + ' valid':<22} {m.sum():>12,}")
    if not m.any():
        sys.exit("ERROR: no pixels are valid in the ALS and every map at once — check "
                 "extents, nodata and --landfire-nonforest")
    print(f"  -> scoring on {m.sum():,} px "
          f"({100*m.sum()/als.size:.1f}% of the ALS raster)")
    return m


def score(als, specs, mask, bin_size, max_height, min_bin_n, balance_seed):
    y_true = als[mask]
    results = []
    for s in specs:
        y_pred = s.values[mask]
        res = {
            "label": s.label,
            "path": s.path,
            "regridded": bool(s.regridded),
            "note": s.note,
            "resampling": s.resampling if s.regridded else "none",
            "scale": s.scale,
            "overall": overall_metrics(y_true, y_pred),
            "stratified": stratified_metrics(
                y_true, y_pred, bin_size=bin_size, max_height=max_height,
                min_bin_n=min_bin_n, balance_seed=balance_seed, overflow_bin=True),
        }
        results.append(res)
    return y_true, results


def print_report(y_true, results):
    print(f"\n{'='*78}\nScored on {len(y_true):,} common pixels  "
          f"(ALS RH98: mean={y_true.mean():.2f} m  sd={y_true.std():.2f} m  "
          f"range={y_true.min():.2f}-{y_true.max():.2f} m)\n{'='*78}")
    hdr = f"{'map':<14}{'n':>12}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'r':>8}{'bias':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in results:
        o = r["overall"]
        print(f"{r['label']:<14}{o['n_samples']:>12,}{o['rmse']:>9.3f}{o['mae']:>9.3f}"
              f"{o['r2']:>9.3f}{o['pearson_r']:>8.3f}{o['bias']:>+9.3f}")
    print("-" * len(hdr))
    print("R2 is agreement with the 1:1 line (can go negative); r is correlation only.")
    print("bias = mean(map - ALS); negative = the map is too SHORT.\n")

    for r in results:
        print(f"{r['label']}:")
        print_stratified(r["stratified"], indent="  ")
        print()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_sidecars(out_dir, stem, meta, results):
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"{stem}.json"
    with open(jpath, "w") as f:
        json.dump({**meta, "results": results}, f, indent=2)

    cpath = out_dir / f"{stem}.csv"
    cols = ["label", "n_samples", "rmse", "mae", "r2", "pearson_r", "bias",
            "macro_rmse", "balanced_rmse", "balanced_r2", "path"]
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            s = r["stratified"]
            w.writerow({"label": r["label"], "path": r["path"],
                        **{k: r["overall"][k] for k in
                           ("n_samples", "rmse", "mae", "r2", "pearson_r", "bias")},
                        "macro_rmse": s.get("macro_rmse"),
                        "balanced_rmse": s.get("balanced_rmse"),
                        "balanced_r2": s.get("balanced_r2")})
    print(f"Metrics: {jpath}\n         {cpath}")
    return jpath, cpath


def write_aligned_stack(out_path, grid, als, specs):
    """Band-stacked GeoTIFF (ALS + each aligned map) on the analysis grid, for QGIS."""
    profile = dict(grid.profile)
    profile.update(driver="GTiff", count=1 + len(specs), dtype="float32",
                   nodata=np.float32("nan"), compress="lzw", tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    profile.pop("predictor", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(als.astype("float32"), 1)
        dst.set_band_description(1, "als_rh98_m")
        for i, s in enumerate(specs, start=2):
            dst.write(s.values.astype("float32"), i)
            dst.set_band_description(i, f"{s.label}_m")
    print(f"Aligned stack: {out_path}")


def make_figure(out_path, y_true, specs, results, mask, hi, gridsize, bin_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(results)
    fig, axes = plt.subplots(2, max(n, 2), figsize=(6.0 * max(n, 2), 11.0),
                             squeeze=False)

    # Row 1: density scatter vs ALS, one panel per map.
    for i, (s, r) in enumerate(zip(specs, results)):
        ax = axes[0][i]
        y_pred = s.values[mask]
        hb = ax.hexbin(y_true, y_pred, gridsize=gridsize, extent=(0, hi, 0, hi),
                       cmap="viridis", norm=LogNorm(), mincnt=1, linewidths=0)
        ax.plot([0, hi], [0, hi], "r--", lw=1.2, label="1:1")
        o = r["overall"]
        ax.text(0.03, 0.97,
                f"RMSE {o['rmse']:.2f} m\nR$^2$ {o['r2']:.3f}\n"
                f"bias {o['bias']:+.2f} m\nn {o['n_samples']:,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="0.7"))
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel("ALS RH98 (m)")
        ax.set_ylabel(f"{s.label} height (m)")
        ax.set_title(f"{s.label} vs ALS RH98")
        ax.legend(loc="lower right", fontsize=9)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="pixels")
    for j in range(n, axes.shape[1]):
        axes[0][j].axis("off")

    # Row 2: per-height-bin RMSE and bias, all maps overlaid.
    ax_r, ax_b = axes[1][0], axes[1][1]
    for r in results:
        bins = r["stratified"]["bins"]
        # The final bin is open-ended (>max_height); centre it half a bin above its
        # lower edge rather than clamping it to `hi`, which would plot it to the LEFT
        # of its own lower edge.
        c = [(b["lo"] + (b["hi"] if np.isfinite(b["hi"]) else b["lo"] + bin_size)) / 2
             for b in bins]
        rm = [b["rmse"] if b["rmse"] is not None else np.nan for b in bins]
        bi = [b.get("bias") if b["rmse"] is not None else np.nan for b in bins]
        ax_r.plot(c, rm, "o-", label=r["label"])
        ax_b.plot(c, bi, "o-", label=r["label"])
    ax_r.set_xlabel(f"ALS RH98 bin centre (m, {bin_size:g} m bins)")
    ax_r.set_ylabel("RMSE (m)"); ax_r.set_title("RMSE by true height")
    ax_b.axhline(0, color="k", lw=0.8, ls="--")
    ax_b.set_xlabel(f"ALS RH98 bin centre (m, {bin_size:g} m bins)")
    ax_b.set_ylabel("bias (m)"); ax_b.set_title("Bias by true height (map - ALS)")
    for a in (ax_r, ax_b):
        a.grid(alpha=0.3); a.legend(fontsize=9)
    for j in range(2, axes.shape[1]):
        axes[1][j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_extra_map(token):
    """LABEL=PATH[:scale=S][:resampling=R][:band=B] -> MapSpec."""
    if "=" not in token:
        sys.exit(f"ERROR: --extra-map wants LABEL=PATH[,scale=..][,resampling=..], got {token!r}")
    label, rest = token.split("=", 1)
    parts = rest.split(",")
    spec = MapSpec(label=label, path=parts[0])
    for kv in parts[1:]:
        k, _, v = kv.partition("=")
        if k == "scale":
            spec.scale = float(v)
        elif k == "resampling":
            spec.resampling = v
        elif k == "band":
            spec.band = int(v)
        else:
            sys.exit(f"ERROR: unknown --extra-map option {k!r} in {token!r}")
    return spec


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-tif", required=True, help="ALS ground-truth raster")
    ap.add_argument("--als-band", type=int, default=2,
                    help="1-based band holding RH98 (default 2, as in the Pierce County stack)")
    ap.add_argument("--als-band-name", default="rh98",
                    help="expected band description; a mismatch warns (empty to skip)")
    ap.add_argument("--aef-tif", default=None, help="Alpha Earth RF wall-to-wall prediction (m)")
    ap.add_argument("--landfire-tif", default=None,
                    help="LANDFIRE CH (Canopy Height) tif — the BINNED forest-height "
                         "product (CONUS is fine, it is windowed not read whole)")
    ap.add_argument("--evh-tif", default=None,
                    help="LANDFIRE EVH (Existing Vegetation Height) tif — class-coded, "
                         "1 m tree resolution; preferred over CH for height comparison")
    ap.add_argument("--evh-classes", choices=["tree", "veg", "all"], default="all",
                    help="which EVH classes count as a height: 'tree' = 101-199 only "
                         "(strictest), 'veg' = + shrub/herb, 'all' = + land cover pinned "
                         "to 0 m (default; the wall-to-wall read matching the AEF map)")
    ap.add_argument("--evh-resampling", default="nearest",
                    help="EVH is CLASS-CODED — nearest (default) is the only correct "
                         "choice; interpolating class numbers is meaningless")
    ap.add_argument("--extra-map", action="append", default=[],
                    help="additional map as LABEL=PATH[,scale=S][,resampling=R][,band=B]; "
                         "repeatable (e.g. ETH=/path/eth.tif,resampling=bilinear)")
    ap.add_argument("--landfire-nonforest", choices=["zero", "drop"], default="zero",
                    help="LANDFIRE 0 = 'Non-Forested': keep as a 0 m prediction (default) "
                         "or drop those pixels from the common mask")
    ap.add_argument("--landfire-resampling", default="nearest",
                    help="LANDFIRE CH is a BINNED product — nearest (default) preserves its "
                         "class values; bilinear would invent between-class heights")
    ap.add_argument("--aef-resampling", default="bilinear",
                    help="only used if the AEF tif is not already on the ALS grid")
    ap.add_argument("--als-min", type=float, default=None,
                    help="drop common pixels with ALS RH98 below this (m)")
    ap.add_argument("--als-max", type=float, default=None,
                    help="drop common pixels with ALS RH98 above this (m)")
    # stratified metrics
    ap.add_argument("--bin-size", type=float, default=10.0, help="height bin width (m)")
    ap.add_argument("--max-height", type=float, default=60.0,
                    help="top closed bin edge; taller pixels go in an open >max bin")
    ap.add_argument("--min-bin-n", type=int, default=100, help="min pixels for a bin to report")
    ap.add_argument("--balance-seed", type=int, default=42)
    # outputs
    ap.add_argument("--out-dir", required=True, help="directory for metrics/figure outputs")
    ap.add_argument("--stem", default="als_comparison", help="basename for the outputs")
    ap.add_argument("--plot-max", type=float, default=None,
                    help="scatter axis max (m); default = 99.5th pct of ALS, rounded up")
    ap.add_argument("--gridsize", type=int, default=90, help="hexbin resolution")
    ap.add_argument("--save-aligned", action="store_true",
                    help="also write a band-stacked GeoTIFF of ALS + aligned maps")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    specs = []
    if args.aef_tif:
        specs.append(MapSpec("alphaearth", args.aef_tif, scale=1.0,
                             resampling=args.aef_resampling))
    if args.landfire_tif:
        specs.append(MapSpec("landfire", args.landfire_tif, scale=LANDFIRE_SCALE,
                             resampling=args.landfire_resampling,
                             extra_nodata=LANDFIRE_NODATA,
                             drop_zero=(args.landfire_nonforest == "drop")))
    if args.evh_tif:
        specs.append(MapSpec("landfire_evh", args.evh_tif,
                             resampling=args.evh_resampling,
                             extra_nodata=LANDFIRE_NODATA,
                             decode=lambda a, m=args.evh_classes: decode_landfire_evh(a, m),
                             note=f"EVH classes={args.evh_classes}"))
    specs += [parse_extra_map(t) for t in args.extra_map]
    if not specs:
        sys.exit("ERROR: give at least one of --aef-tif / --landfire-tif / --evh-tif "
                 "/ --extra-map")

    als, grid = read_als(args.als_tif, args.als_band, args.als_band_name)
    print()
    for s in specs:
        align(s, grid)

    mask = common_mask(als, specs, args.als_min, args.als_max)
    y_true, results = score(als, specs, mask, args.bin_size, args.max_height,
                            args.min_bin_n, args.balance_seed)
    print_report(y_true, results)

    out_dir = Path(args.out_dir)
    meta = {
        "als_tif": args.als_tif, "als_band": args.als_band,
        "als_band_name": read_band_desc(args.als_tif, args.als_band),
        "grid": {"crs": str(grid.crs), "width": grid.width, "height": grid.height,
                 "transform": list(grid.transform)[:6]},
        "landfire_nonforest": args.landfire_nonforest,
        "evh_classes": args.evh_classes if args.evh_tif else None,
        "als_min": args.als_min, "als_max": args.als_max,
        "n_common_px": int(mask.sum()),
        "als_stats": {"mean": float(y_true.mean()), "std": float(y_true.std()),
                      "min": float(y_true.min()), "max": float(y_true.max())},
        "caveats": [
            "Year mismatch: the products are not all from the same year (see the paths in "
            "'results'). Whichever map is furthest from the ALS acquisition year is charged "
            "for the growth/disturbance in between; that is a real error source, and it is "
            "NOT corrected here. LANDFIRE EVH in particular is released as LF20xx well "
            "after the ALS flight.",
            "LANDFIRE CH is binned (~4 m classes, values = metres x 10 from LF2022_CH.csv), "
            "so it carries ~+/-2 m quantisation error before any model error.",
            f"LANDFIRE CH 'Non-Forested' (0) handling: {args.landfire_nonforest}.",
        ] + ([f"LANDFIRE EVH class mode: {args.evh_classes}. EVH tree height (101-199) is "
              f"1 m resolution and tops out at '>= 99 m' (199); shrub (201-230) and herb "
              f"(301-310) cap at >=3.0 m and >=1 m respectively."] if args.evh_tif else []),
    }
    write_sidecars(out_dir, args.stem, meta, results)

    if args.save_aligned:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_aligned_stack(out_dir / f"{args.stem}_aligned.tif", grid, als, specs)

    if not args.no_plot:
        hi = args.plot_max or float(np.ceil(np.percentile(y_true, 99.5) / 5) * 5)
        make_figure(out_dir / f"{args.stem}.png", y_true, specs, results, mask,
                    hi, args.gridsize, args.bin_size)
    print("\nDone.")


def read_band_desc(path, band):
    with rasterio.open(path) as s:
        return s.descriptions[band - 1]


if __name__ == "__main__":
    main()
