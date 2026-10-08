"""
Shared helpers for the ALS cal/val pipeline (train_healpix_rf.py, predict_als_aoi.py,
finetune_als_rf.py).

  - tile lists         read_tile_ids
  - GEDI (X, y)        load_tile_xy
  - RF                 make_rf, fit_rf, save_model_bundle, load_model_bundle
  - AEF <-> ALS        qa_alignment, read_aoi_pairs, aoi_footprint, report_tile_overlap
  - outputs            predict_to_tif, plot_map, evaluate, plot_scatter_panels

Conventions (same as run_aef_rf_wall_to_wall.py / compare_maps_to_als.py):
  - heights in METRES; 0 m is a valid height everywhere, nodata is NaN.
  - an AEF pixel is valid when ALL bands are finite.
  - metrics come from metrics_stratified.py so numbers stay comparable to the rest of
    the repo — quote balanced_rmse / balanced_r2 as the stratified pair.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from metrics_stratified import overall_metrics, stratified_metrics, print_stratified  # noqa: E402

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

EQUAL_AREA_CRS = "EPSG:6933"
AEF_BAND_RE = re.compile(r"^aef_ch_(\d{3})$")


# ---------------------------------------------------------------------------
# Tile lists
# ---------------------------------------------------------------------------

def read_tile_ids(path):
    """Healpix ids from a tile-list file, as strings, order kept, duplicates dropped.

    Accepts:
      - .txt (or anything non-CSV): ids separated by newlines/commas/whitespace; '#'
        starts a comment; 'healpix_123' and '123.0' are normalised to '123'.
      - .csv: a column named healpix_id / tile_id / tile / id (first match).
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: tile list not found: {path}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in ("healpix_id", "tile_id", "tile", "id") if c in df.columns), None)
        if col is None:
            sys.exit(f"ERROR: {path} has no healpix_id/tile_id/tile/id column "
                     f"(columns: {list(df.columns)})")
        raw = df[col].dropna().astype(str).tolist()
    else:
        raw = []
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0]
            raw += [t for t in re.split(r"[,\s]+", line) if t]

    ids = []
    for t in raw:
        t = t.strip().removeprefix("healpix_")
        try:
            t = str(int(float(t)))
        except ValueError:
            sys.exit(f"ERROR: {path}: cannot parse healpix id {t!r}")
        ids.append(t)
    ids = list(dict.fromkeys(ids))
    if not ids:
        sys.exit(f"ERROR: {path} contains no tile ids")
    return ids


def check_disjoint(train_ids, test_ids, allow_overlap=False):
    both = sorted(set(train_ids) & set(test_ids), key=int)
    if both:
        msg = (f"{len(both)} tile(s) are in BOTH the train and test lists: "
               f"{both[:10]}{' ...' if len(both) > 10 else ''}")
        if not allow_overlap:
            sys.exit(f"ERROR: {msg}\n       Fix the lists, or pass --allow-tile-overlap "
                     f"(test metrics are then optimistic).")
        print(f"WARNING: {msg}", file=sys.stderr)
    return both


# ---------------------------------------------------------------------------
# GEDI (X, y) per healpix tile
# ---------------------------------------------------------------------------

def load_tile_xy(tile_ids, data_dirs, include_zeros=False, embedding_format="float32",
                 max_samples=None, seed=42, label=""):
    """Concatenate (X, y) over tile_ids x data_dirs (one dir per AEF year).

    Returns X, y, info where info lists the per-dir pixel counts and missing tiles.
    """
    suffix = "_withzeros" if include_zeros else ""
    Xs, ys, per_dir = [], [], {}
    for d in data_dirs:
        d = Path(d)
        if not d.is_dir():
            sys.exit(f"ERROR: data dir not found: {d}")
        n_px, found, missing = 0, [], []
        for tid in tile_ids:
            p = d / f"healpix_{tid}_{embedding_format}{suffix}.npz"
            if not p.exists():
                missing.append(tid)
                continue
            with np.load(p) as npz:
                X, y = npz["X"], npz["y"]
            found.append(tid)
            if X.size == 0:
                continue
            Xs.append(X.astype(np.float32))
            ys.append(y.astype(np.float32))
            n_px += len(y)
        per_dir[str(d)] = {"n_tiles_found": len(found), "n_tiles_missing": len(missing),
                           "missing": missing, "n_pixels": n_px}
        print(f"  [{label}] {d}: {len(found)}/{len(tile_ids)} tiles, {n_px:,} px"
              + (f"  (missing {len(missing)}: {missing[:8]}{' ...' if len(missing) > 8 else ''})"
                 if missing else ""))

    if not Xs:
        sys.exit(f"ERROR: no {label} .npz found — check --data-dirs / tile list / "
                 f"--include-zeros / --embedding-format")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print(f"  [{label}] X={X.shape}  height (m): min={y.min():.2f} max={y.max():.2f} "
          f"mean={y.mean():.2f}")
    if max_samples and len(y) > max_samples:
        idx = np.random.default_rng(seed).choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        print(f"  [{label}] subsampled to {max_samples:,} px (seed={seed})")
    return X, y, {"per_dir": per_dir, "n_pixels_used": int(len(y))}


# ---------------------------------------------------------------------------
# Random forest + model bundle
# ---------------------------------------------------------------------------

def add_rf_args(ap, prefix="", defaults=None):
    """RF hyperparameter flags (optionally prefixed, e.g. 'ft-' for the fine-tune RF)."""
    d = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5,
         "max_features": "sqrt", **(defaults or {})}
    ap.add_argument(f"--{prefix}n-estimators", type=int, default=d["n_estimators"])
    ap.add_argument(f"--{prefix}max-depth", type=int, default=d["max_depth"])
    ap.add_argument(f"--{prefix}min-samples-leaf", type=int, default=d["min_samples_leaf"])
    ap.add_argument(f"--{prefix}max-features", default=d["max_features"],
                    help="RF max_features (sqrt, 1.0, ...)")


def rf_params_from_args(args, prefix=""):
    a = prefix.replace("-", "_")
    mf = getattr(args, f"{a}max_features")
    try:
        mf = float(mf) if mf not in ("sqrt", "log2") else mf
    except (TypeError, ValueError):
        pass
    return {"n_estimators": getattr(args, f"{a}n_estimators"),
            "max_depth": getattr(args, f"{a}max_depth"),
            "min_samples_leaf": getattr(args, f"{a}min_samples_leaf"),
            "max_features": mf}


def fit_rf(X, y, params, n_jobs=-1, random_state=42, sample_weight=None, label="RF"):
    print(f"\nTraining {label} {params} on {len(y):,} px ...")
    t = time.time()
    rf = RandomForestRegressor(**params, n_jobs=n_jobs, random_state=random_state)
    rf.fit(X, y, sample_weight=sample_weight)
    print(f"  fit in {time.time() - t:.1f}s")
    return rf


class ResidualCorrectedModel:
    """base(X) + correction(X), clipped at `clip_min` (residuals can push below 0 m).

    Defined here (not in a script) so a joblib-dumped instance can be re-loaded by any
    script that imports common.
    """

    def __init__(self, base, correction, clip_min=0.0):
        self.base = base
        self.correction = correction
        self.clip_min = clip_min
        self.n_features_in_ = base.n_features_in_

    def predict(self, X):
        out = self.base.predict(X) + self.correction.predict(X)
        if self.clip_min is not None:
            out = np.maximum(out, self.clip_min)
        return out


def save_model_bundle(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)
    print(f"Saved model: {path}")


def load_model_bundle(path):
    """(model, meta). Also accepts a bare estimator (e.g. run_aef_rf_wall_to_wall --model-out)."""
    if not Path(path).exists():
        sys.exit(f"ERROR: model not found: {path}")
    print(f"Loading model: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("meta", {})
    return obj, {}


def model_n_features(model):
    n = getattr(model, "n_features_in_", None)
    if n is None:
        sys.exit("ERROR: cannot tell how many features the model expects")
    return int(n)


# ---------------------------------------------------------------------------
# AEF <-> ALS alignment QA
# ---------------------------------------------------------------------------

def _grid_mismatches(a, b, tol_frac):
    """Human-readable differences between two open datasets' grids (empty = aligned)."""
    out = []
    if a.crs != b.crs:
        out.append(f"CRS differs: AEF {a.crs} vs ALS {b.crs}")
    if (a.width, a.height) != (b.width, b.height):
        out.append(f"shape differs: AEF {a.width}x{a.height} vs ALS {b.width}x{b.height}")
    ta, tb = a.transform, b.transform
    px = min(abs(ta.a), abs(ta.e))
    for name, va, vb in (("x pixel size", ta.a, tb.a), ("y pixel size", ta.e, tb.e),
                         ("row rotation", ta.b, tb.b), ("col rotation", ta.d, tb.d),
                         ("x origin", ta.c, tb.c), ("y origin", ta.f, tb.f)):
        if abs(va - vb) > tol_frac * px:
            out.append(f"{name} differs: AEF {va:.10g} vs ALS {vb:.10g} "
                       f"(diff {abs(va - vb) / px:.3g} px)")
    return out


def _bounds_4326(src):
    return rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def qa_alignment(aef_tif, als_tif, als_band, als_band_name, n_features,
                 regrid_als=False, tol_frac=1e-3):
    """Metadata-level QA. Exits on a hard failure; returns a report dict otherwise.

    Hard failures: missing CRS, AEF band count != model features, AEF band names out of
    order, ALS band missing, grids differ (unless regrid_als), footprints disjoint.
    """
    rep = {"aef_tif": str(aef_tif), "als_tif": str(als_tif), "als_band": als_band,
           "warnings": [], "grid_mismatches": [], "regrid_als": False}
    errors = []
    for p in (aef_tif, als_tif):
        if not Path(p).exists():
            sys.exit(f"ERROR: not found: {p}")

    with rasterio.open(aef_tif) as aef, rasterio.open(als_tif) as als:
        print(f"QA  AEF: {aef_tif}\n      {aef.width}x{aef.height}  {aef.count} bands  "
              f"{aef.dtypes[0]}  {aef.crs}  res={tuple(round(r, 10) for r in aef.res)}")
        print(f"QA  ALS: {als_tif}\n      {als.width}x{als.height}  {als.count} bands  "
              f"{als.dtypes[0]}  {als.crs}  res={tuple(round(r, 10) for r in als.res)}")
        if aef.crs is None or als.crs is None:
            errors.append("a raster has no CRS")

        # AEF bands: count, dtype, order
        if aef.count != n_features:
            errors.append(f"AEF tif has {aef.count} bands but the model expects {n_features}")
        if not np.issubdtype(np.dtype(aef.dtypes[0]), np.floating):
            errors.append(f"AEF dtype is {aef.dtypes[0]} — expected dequantized float "
                          f"(int8 means the tif was never dequantized)")
        descs = list(aef.descriptions)
        if any(descs):
            nums = [int(m.group(1)) if (m := AEF_BAND_RE.match(d or "")) else None
                    for d in descs]
            if None in nums:
                rep["warnings"].append(f"AEF band names are not all aef_ch_###: {descs[:3]}...")
            elif nums != list(range(1, len(nums) + 1)):
                errors.append("AEF bands are not in aef_ch_001..N order — feature order "
                              "would not match training X")
        else:
            rep["warnings"].append("AEF bands have no descriptions; cannot verify band order")

        # ALS band
        if als_band > als.count:
            errors.append(f"--als-band {als_band} but ALS has only {als.count} bands")
        else:
            d = als.descriptions[als_band - 1]
            rep["als_band_desc"] = d
            if als_band_name and d and d.lower() != als_band_name.lower():
                rep["warnings"].append(f"ALS band {als_band} is '{d}', not '{als_band_name}'"
                                       f" — check --als-band")
            elif als_band_name and not d:
                rep["warnings"].append(f"ALS band {als_band} has no description; cannot "
                                       f"confirm it is {als_band_name}")

        if not errors:
            mism = _grid_mismatches(aef, als, tol_frac)
            rep["grid_mismatches"] = mism
            if mism:
                print("QA  grid: AEF and ALS are NOT on the same grid:")
                for m in mism:
                    print(f"      - {m}")
                if not regrid_als:
                    errors.append(
                        "AEF and ALS grids differ. Either regenerate the AEF tif on the ALS "
                        "grid:\n         python data_processing/alphaearth_wall_to_wall.py "
                        "--like <ALS tif> ...\n       or pass --regrid-als to warp the ALS "
                        "band onto the AEF grid.")
                else:
                    rep["regrid_als"] = True
                    print("QA  grid: --regrid-als set -> ALS will be warped onto the AEF grid")
            else:
                print("QA  grid: identical (CRS, shape, transform) — no resampling")

            # footprint overlap (catches 'right grid, wrong AOI')
            ab, lb = _bounds_4326(aef), _bounds_4326(als)
            ix = max(0.0, min(ab[2], lb[2]) - max(ab[0], lb[0]))
            iy = max(0.0, min(ab[3], lb[3]) - max(ab[1], lb[1]))
            als_area = (lb[2] - lb[0]) * (lb[3] - lb[1])
            frac = ix * iy / als_area if als_area > 0 else 0.0
            rep["als_bbox_covered_by_aef_bbox"] = frac
            print(f"QA  footprint: {100 * frac:.1f}% of the ALS bbox is inside the AEF bbox")
            if frac == 0:
                errors.append("AEF and ALS footprints do not overlap at all — wrong pair of files?")
            elif frac < 0.95:
                rep["warnings"].append(f"only {100 * frac:.1f}% of the ALS bbox is covered by "
                                       f"the AEF bbox")

    if errors:
        sys.exit("ERROR: AEF/ALS QA failed:\n  - " + "\n  - ".join(errors))
    return rep


def qa_pixels(rep, stats, min_coverage, n_norm_sample=200_000, norm_range=(0.85, 1.1)):
    """Pixel-level QA on the counts/norms gathered by read_aoi_pairs. Exits on failure."""
    rep.update({k: v for k, v in stats.items() if k != "norm_sample"})
    n_als, n_both = stats["n_als_valid"], stats["n_both_valid"]
    cov = n_both / n_als if n_als else 0.0
    rep["als_pixels_with_aef"] = cov
    print(f"QA  pixels: AEF valid {stats['n_aef_valid']:,}  ALS valid {n_als:,}  "
          f"both {n_both:,}  -> {100 * cov:.1f}% of ALS pixels have an embedding")
    if stats["n_aef_partial"]:
        rep["warnings"].append(f"{stats['n_aef_partial']:,} AEF pixels have SOME but not all "
                               f"bands finite (dropped) — possible band/tiling corruption")
    if stats["n_aef_valid"] == 0:
        sys.exit("ERROR: AEF tif has no valid (all-bands-finite) pixels — is it an "
                 "all-NaN output from a failed alphaearth_wall_to_wall.py run?")
    if n_both == 0:
        sys.exit("ERROR: no pixel is valid in both AEF and ALS")
    if cov < min_coverage:
        sys.exit(f"ERROR: only {100 * cov:.1f}% of valid ALS pixels have an AEF embedding "
                 f"(< --min-coverage {min_coverage:g}). The AEF tif probably does not cover "
                 f"this ALS AOI (or was masked to a different polygon).")

    norms = stats.get("norm_sample")
    if norms is not None and len(norms):
        q = np.percentile(norms, [1, 50, 99])
        rep["aef_norm_p1_p50_p99"] = [float(v) for v in q]
        print(f"QA  AEF L2 norm p1/p50/p99 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} "
              f"(dequantized AEF is ~unit length)")
        if not (norm_range[0] <= q[1] <= norm_range[1]):
            rep["warnings"].append(f"median AEF norm {q[1]:.3f} is outside {norm_range} — the "
                                   f"tif may be quantized/scaled differently from training X")
    for w in rep["warnings"]:
        print(f"QA  WARNING: {w}", file=sys.stderr)
    print("QA  passed" + (f" with {len(rep['warnings'])} warning(s)" if rep["warnings"] else ""))
    return rep


def _open_als_on_aef_grid(als_src, aef_src, resampling):
    return WarpedVRT(als_src, crs=aef_src.crs, transform=aef_src.transform,
                     width=aef_src.width, height=aef_src.height,
                     resampling=Resampling[resampling], src_nodata=als_src.nodata,
                     nodata=np.nan, dtype="float32")


def read_aoi_pairs(aef_tif, als_tif, als_band, regrid_als=False, resampling="bilinear",
                   als_min=None, als_max=None, block_rows=256, seed=0):
    """(X, y, rows, cols, stats) for every pixel valid in BOTH the AEF and the ALS.

    Reads in row strips on the AEF grid. When regrid_als, the ALS band is warped onto the
    AEF grid (the embeddings are never resampled). als_min/als_max filter y.
    """
    Xs, ys, rs, cs, norms = [], [], [], [], []
    stats = {"n_aef_valid": 0, "n_aef_partial": 0, "n_als_valid": 0, "n_both_valid": 0,
             "n_dropped_by_als_range": 0}
    rng = np.random.default_rng(seed)
    with rasterio.open(aef_tif) as aef, rasterio.open(als_tif) as als_src:
        als = _open_als_on_aef_grid(als_src, aef, resampling) if regrid_als else als_src
        try:
            H, W = aef.height, aef.width
            rows = range(0, H, block_rows)
            for r0 in (tqdm(list(rows), desc="read AEF+ALS") if _HAS_TQDM else rows):
                bh = min(block_rows, H - r0)
                win = Window(0, r0, W, bh)
                feats = aef.read(window=win).astype(np.float32).reshape(aef.count, -1).T
                fin = np.isfinite(feats)
                v_aef = fin.all(axis=1)
                stats["n_aef_valid"] += int(v_aef.sum())
                stats["n_aef_partial"] += int((fin.any(axis=1) & ~v_aef).sum())

                h = als.read(als_band, window=win, masked=True)
                h = h.astype(np.float32).filled(np.nan).ravel()
                v_als = np.isfinite(h)
                stats["n_als_valid"] += int(v_als.sum())
                v = v_aef & v_als
                stats["n_both_valid"] += int(v.sum())
                if als_min is not None or als_max is not None:
                    keep = np.ones_like(v)
                    if als_min is not None:
                        keep &= ~(h < als_min)
                    if als_max is not None:
                        keep &= ~(h > als_max)
                    stats["n_dropped_by_als_range"] += int((v & ~keep).sum())
                    v &= keep
                if not v.any():
                    continue
                idx = np.flatnonzero(v)
                Xs.append(feats[idx])
                ys.append(h[idx])
                rs.append((r0 + idx // W).astype(np.int32))
                cs.append((idx % W).astype(np.int32))
                take = idx if len(idx) <= 2000 else rng.choice(idx, 2000, replace=False)
                norms.append(np.linalg.norm(feats[take], axis=1))
        finally:
            if regrid_als:
                als.close()

    stats["norm_sample"] = np.concatenate(norms) if norms else np.array([])
    if not Xs:
        return (np.empty((0, 0), np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32), np.empty(0, np.int32), stats)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs),
            np.concatenate(cs), stats)


def load_aoi(aef_tif, als_tif, args, n_features, label):
    """QA (metadata + pixels) and read (X, y, rows, cols) for one AEF/ALS AOI."""
    print(f"\n=== {label}: AEF/ALS alignment QA ===")
    rep = qa_alignment(aef_tif, als_tif, args.als_band, args.als_band_name, n_features,
                       regrid_als=args.regrid_als, tol_frac=args.grid_tol)
    X, y, r, c, stats = read_aoi_pairs(
        aef_tif, als_tif, args.als_band, regrid_als=rep["regrid_als"],
        resampling=args.als_resampling, als_min=args.als_min, als_max=args.als_max,
        block_rows=args.block_rows)
    rep = qa_pixels(rep, stats, args.min_coverage)
    print(f"{label}: {len(y):,} paired px  ALS height mean={y.mean():.2f} "
          f"sd={y.std():.2f} range={y.min():.2f}-{y.max():.2f} m")
    return X, y, r, c, rep


def add_aoi_args(ap):
    """ALS band / QA / regrid flags shared by predict_als_aoi.py and finetune_als_rf.py."""
    g = ap.add_argument_group("ALS + alignment QA")
    g.add_argument("--als-band", type=int, default=2,
                   help="1-based ALS band holding RH98 (default 2, as in the county stacks)")
    g.add_argument("--als-band-name", default="rh98",
                   help="expected ALS band description; mismatch warns ('' to skip)")
    g.add_argument("--regrid-als", action="store_true",
                   help="if the AEF and ALS grids differ, warp the ALS band onto the AEF grid "
                        "instead of failing QA (embeddings are never resampled)")
    g.add_argument("--als-resampling", default="bilinear",
                   help="resampling for --regrid-als (bilinear, average, nearest, ...)")
    g.add_argument("--grid-tol", type=float, default=1e-3,
                   help="max transform difference, as a fraction of a pixel, to call the "
                        "grids identical")
    g.add_argument("--min-coverage", type=float, default=0.8,
                   help="fail QA if fewer than this fraction of valid ALS pixels have a valid "
                        "AEF embedding (0 disables)")
    g.add_argument("--als-min", type=float, default=None,
                   help="drop pixels with ALS height below this (m)")
    g.add_argument("--als-max", type=float, default=None,
                   help="drop pixels with ALS height above this (m)")
    g.add_argument("--block-rows", type=int, default=256, help="rows per read/predict strip")


# ---------------------------------------------------------------------------
# AOI footprints / overlap guards
# ---------------------------------------------------------------------------

def aoi_footprint(tif):
    """Bounding box of a raster as a shapely polygon in EPSG:4326."""
    from shapely.geometry import box
    with rasterio.open(tif) as src:
        return box(*_bounds_4326(src))


def report_tile_overlap(shapefile, tile_ids, aoi_geom, label):
    """Warn about train tiles whose footprint overlaps an AOI (spatial leakage)."""
    import geopandas as gpd
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    ids = gdf["healpix_id"]
    if np.issubdtype(ids.dtype, np.floating):
        ids = ids.astype("int64")
    gdf["tile_id"] = ids.astype(str)
    sel = gdf[gdf["tile_id"].isin(set(tile_ids))].to_crs(EQUAL_AREA_CRS)
    aoi = gpd.GeoSeries([aoi_geom], crs=4326).to_crs(EQUAL_AREA_CRS).iloc[0]
    frac = (sel.geometry.intersection(aoi).area / sel.geometry.area).to_numpy()
    hits = [{"tile_id": t, "overlap_frac": float(f)}
            for t, f in zip(sel["tile_id"], frac) if f > 0]
    hits.sort(key=lambda d: -d["overlap_frac"])
    if hits:
        print(f"WARNING: {len(hits)} GEDI training tile(s) overlap the {label} bbox — "
              f"metrics there are not fully independent:", file=sys.stderr)
        for h in hits[:10]:
            print(f"    healpix {h['tile_id']}: {100 * h['overlap_frac']:.1f}% inside",
                  file=sys.stderr)
    else:
        print(f"Leakage check: no GEDI training tile overlaps the {label} bbox")
    return hits


# ---------------------------------------------------------------------------
# Prediction -> GeoTIFF, map plot
# ---------------------------------------------------------------------------

def predict_to_tif(model, aef_tif, output_tif, block_rows=256, compress="lzw"):
    """Wall-to-wall prediction over the AEF tif -> single-band float32 GeoTIFF (NaN nodata).

    LZW without predictor by default (see run_aef_rf_wall_to_wall.py for why deflate+
    predictor=3 is avoided).
    """
    n_valid = 0
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(aef_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.float32("nan"), tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", num_threads=1)
        profile.pop("compress", None)
        profile.pop("predictor", None)
        if compress and compress.lower() != "none":
            profile["compress"] = compress
        H, W = src.height, src.width
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.set_band_description(1, "predicted_canopy_height_m")
            rows = range(0, H, block_rows)
            for r0 in (tqdm(list(rows), desc="predict") if _HAS_TQDM else rows):
                bh = min(block_rows, H - r0)
                win = Window(0, r0, W, bh)
                feats = src.read(window=win).astype(np.float32).reshape(src.count, -1).T
                valid = np.all(np.isfinite(feats), axis=1)
                out = np.full(feats.shape[0], np.nan, dtype=np.float32)
                if valid.any():
                    out[valid] = model.predict(feats[valid]).astype(np.float32)
                    n_valid += int(valid.sum())
                dst.write(out.reshape(bh, W), 1, window=win)
    print(f"Predicted {n_valid:,} px -> {output_tif}")
    return n_valid


def plot_map(tif, out_png, title, vmin=0.0, vmax=None, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rasterio.open(tif) as src:
        arr = np.ma.masked_invalid(src.read(1))
        left, bottom, right, top = array_bounds(src.height, src.width, src.transform)
        crs = src.crs
    if vmax is None and arr.count():
        vmax = float(np.percentile(arr.compressed(), 98))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, extent=[left, right, bottom, top],
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Canopy height (m)")
    ax.set_title(title)
    ax.set_xlabel(f"Easting ({crs.to_string()})" if crs else "Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved map: {out_png}")


# ---------------------------------------------------------------------------
# Metrics + figures
# ---------------------------------------------------------------------------

def add_metric_args(ap):
    g = ap.add_argument_group("stratified metrics")
    g.add_argument("--bin-size", type=float, default=10.0, help="height bin width (m)")
    g.add_argument("--max-height", type=float, default=60.0,
                   help="top closed bin edge; taller pixels go in an open >max bin")
    g.add_argument("--min-bin-n", type=int, default=100, help="min pixels for a bin to count")


def evaluate(y_true, y_pred, label, args, seed):
    """overall + stratified metrics, printed and returned."""
    o = overall_metrics(y_true, y_pred)
    s = stratified_metrics(y_true, y_pred, bin_size=args.bin_size, max_height=args.max_height,
                           min_bin_n=args.min_bin_n, balance_seed=seed, overflow_bin=True)
    bal = (f"  bal.RMSE={s['balanced_rmse']:.3f} bal.R2={s['balanced_r2']:.3f}"
           if s.get("balanced_rmse") is not None else "")
    print(f"[{label}] n={o['n_samples']:,}  RMSE={o['rmse']:.3f}  MAE={o['mae']:.3f}  "
          f"R2={o['r2']:.3f}  r={o['pearson_r']:.3f}  bias={o['bias']:+.3f}{bal}")
    print_stratified(s, indent="    ")
    return {"overall": o, "stratified": s}


def summary_row(label, res):
    o, s = res["overall"], res["stratified"]
    return {"label": label, **{k: o[k] for k in ("n_samples", "rmse", "mae", "r2",
                                                 "pearson_r", "bias")},
            "balanced_rmse": s.get("balanced_rmse"), "balanced_r2": s.get("balanced_r2"),
            "balanced_bias": s.get("balanced_bias")}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Metrics CSV: {path}")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Sidecar: {path}")


def plot_scatter_panels(out_png, y_true, preds, results, xlabel, bin_size, hi=None,
                        gridsize=90):
    """Row 1: hexbin of each prediction vs truth. Row 2: per-bin RMSE and bias overlay."""
    matplotlib.use("Agg")

    if hi is None:
        hi = float(np.ceil(np.percentile(y_true, 99.5) / 5) * 5) or 1.0
    n = len(preds)
    ncol = max(n, 2)
    fig, axes = plt.subplots(2, ncol, figsize=(5.5 * ncol, 10.5), squeeze=False)
    for i, (label, yp) in enumerate(preds.items()):
        ax = axes[0][i]
        hb = ax.hexbin(y_true, yp, gridsize=gridsize, extent=(0, hi, 0, hi), cmap="viridis",
                       norm=LogNorm(), mincnt=1, linewidths=0)
        ax.plot([0, hi], [0, hi], "r--", lw=1.2)
        o = results[label]["overall"]
        s = results[label]["stratified"]
        txt = (f"RMSE {o['rmse']:.2f} m\nR$^2$ {o['r2']:.3f}\nbias {o['bias']:+.2f} m\n"
               f"n {o['n_samples']:,}")
        if s.get("balanced_rmse") is not None:
            txt += f"\nbal. RMSE {s['balanced_rmse']:.2f}\nbal. R$^2$ {s['balanced_r2']:.3f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="0.7"))
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel(xlabel); ax.set_ylabel("predicted height (m)"); ax.set_title(label)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="pixels")
    for j in range(n, ncol):
        axes[0][j].axis("off")

    ax_r, ax_b = axes[1][0], axes[1][1]
    for label in preds:
        bins = results[label]["stratified"]["bins"]
        c = [(b["lo"] + (b["hi"] if np.isfinite(b["hi"]) else b["lo"] + bin_size)) / 2
             for b in bins]
        ax_r.plot(c, [b["rmse"] if b["rmse"] is not None else np.nan for b in bins],
                  "o-", label=label)
        ax_b.plot(c, [b.get("bias") if b["rmse"] is not None else np.nan for b in bins],
                  "o-", label=label)
    ax_r.set_ylabel("RMSE (m)"); ax_r.set_title("RMSE by true height")
    ax_b.axhline(0, color="k", lw=0.8, ls="--")
    ax_b.set_ylabel("bias (m)"); ax_b.set_title("Bias by true height (pred - truth)")
    for a in (ax_r, ax_b):
        a.set_xlabel(f"true height bin centre (m, {bin_size:g} m bins)")
        a.grid(alpha=0.3); a.legend(fontsize=9)
    for j in range(2, ncol):
        axes[1][j].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {out_png}")
