#!/usr/bin/env python3
"""
Write per-tile prediction rasters (on the GEDI tile grid) for an als_calval model, so it
can be plotted along a GEDI transect with results-plotting/plot_track_comparison.py.

For each healpix tile:
  1. Load the AEF (X, y) .npz from alphaearth_processing.py.
  2. Recover each sample's pixel position. alphaearth_processing.py builds X/y from
         rows, cols = np.where(valid_mask)   (row-major)
     with valid_mask = (gedi != nodata) & (gedi > 0, or >= 0 with zeros) & (gedi < 12000 cm),
     so recomputing the mask on the GEDI tile gives the positions in the same order.
  3. QA: the recomputed GEDI heights must equal the .npz y (cm -> m) — proves the
     positions line up; a mismatching tile is skipped, never written.
  4. Predict and write healpix_<id>_<name>.tif: float32, prediction at the valid GEDI
     pixels, nodata -9999, same CRS/transform/shape as the GEDI tile — the format
     plot_track_comparison.py reads.

Tiles in the model's TRAIN list are flagged: a transect there is in-sample and flatters
the model. Default tile list = the model bundle's test tiles.

Usage
-----
    python als_calval/predict_tiles_to_rasters.py \\
        --model    /scratch/kdoerkse/aef/calval/combined/combined_rf.joblib \\
        --data-dir /data/aef/2020 \\
        --gedi-dir /data/gedi_tiles_2020_healpix \\
        --out-dir  /results/aef_combined_pred_rasters/2020 \\
        --name     aef_combined \\
        --tiles    33360
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GEDI_MAX_HEIGHT_CM = 12000      # as in alphaearth_processing.py
GEDI_NODATA_DEFAULT = -9999.0
OUT_NODATA = -9999.0            # what plot_track_comparison.py expects


def tile_ids_from_args(tokens, meta):
    """--tiles accepts ids and/or tile-list files; default = the bundle's test tiles."""
    if not tokens:
        ids = meta.get("test_tile_ids")
        if not ids:
            sys.exit("ERROR: no --tiles given and the model bundle stores no test tiles")
        return list(ids)
    ids = []
    for t in tokens:
        ids += C.read_tile_ids(t) if Path(t).exists() else [str(int(t))]
    return list(dict.fromkeys(ids))


def gedi_tile_path(gedi_dir, tid):
    for ext in (".tif", ".tiff", ""):
        p = Path(gedi_dir) / f"healpix_{tid}_gedi{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="als_calval model bundle (.joblib)")
    ap.add_argument("--data-dir", required=True,
                    help="AEF npz dir for ONE year (must match the --gedi-dir year)")
    ap.add_argument("--gedi-dir", required=True,
                    help="GEDI tiles for the same year (healpix_<id>_gedi.tif)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="aef_combined",
                    help="model token in the output filename healpix_<id>_<name>.tif")
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="healpix ids and/or tile-list files (default: the bundle's test tiles)")
    ap.add_argument("--include-zeros", action=argparse.BooleanOptionalAction, default=None,
                    help="use *_withzeros.npz (default: as the model was trained)")
    ap.add_argument("--embedding-format", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model, meta = C.load_model_bundle(args.model)
    include_zeros = args.include_zeros if args.include_zeros is not None \
        else meta.get("include_zeros", False)
    fmt = args.embedding_format or meta.get("embedding_format", "float32")
    suffix = "_withzeros" if include_zeros else ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = tile_ids_from_args(args.tiles, meta)
    train = set(meta.get("train_tile_ids") or [])
    in_train = [t for t in tiles if t in train]
    if in_train:
        print(f"WARNING: {len(in_train)} tile(s) are in the model's TRAIN list — transects "
              f"there are in-sample: {in_train[:10]}", file=sys.stderr)
    print(f"{len(tiles)} tile(s) -> {out_dir}/healpix_<id>_{args.name}.tif "
          f"(include_zeros={include_zeros})")

    status = {"written": [], "skipped_exists": [], "missing": [], "qa_fail": []}
    for tid in tiles:
        out = out_dir / f"healpix_{tid}_{args.name}.tif"
        if out.exists() and not args.overwrite:
            status["skipped_exists"].append(tid)
            continue
        npz = Path(args.data_dir) / f"healpix_{tid}_{fmt}{suffix}.npz"
        gp = gedi_tile_path(args.gedi_dir, tid)
        if not npz.exists() or gp is None:
            print(f"  {tid}: missing {'npz' if not npz.exists() else 'GEDI tile'} — skipped")
            status["missing"].append(tid)
            continue

        with np.load(npz) as z:
            X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
        with rasterio.open(gp) as src:
            gedi = src.read(1)
            nodata = src.nodata if src.nodata is not None else GEDI_NODATA_DEFAULT
            profile = src.profile.copy()
        valid = ((gedi != nodata) & (gedi >= 0 if include_zeros else gedi > 0)
                 & (gedi < GEDI_MAX_HEIGHT_CM))
        rows, cols = np.where(valid)

        # QA: positions must reproduce the stored y exactly (same tile, year, zeros rule).
        y_grid = (gedi[rows, cols] / 100.0).astype(np.float32)
        if len(y_grid) != len(y) or not np.allclose(y_grid, y, atol=1e-3):
            n_bad = "count" if len(y_grid) != len(y) else \
                f"{int((~np.isclose(y_grid, y, atol=1e-3)).sum())} values"
            print(f"  {tid}: QA FAIL — GEDI tile does not reproduce npz y ({len(y_grid)} vs "
                  f"{len(y)} px, mismatch in {n_bad}). Wrong year / --include-zeros? skipped",
                  file=sys.stderr)
            status["qa_fail"].append(tid)
            continue

        arr = np.full(gedi.shape, OUT_NODATA, dtype=np.float32)
        if len(y):
            arr[rows, cols] = model.predict(X).astype(np.float32)
        profile.update(count=1, dtype="float32", nodata=OUT_NODATA, compress="lzw")
        profile.pop("predictor", None)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, f"{args.name}_predicted_height_m")
        print(f"  {tid}: {len(y):,} px  QA ok  -> {out.name}"
              + ("  [IN TRAIN SET]" if tid in train else ""))
        status["written"].append(tid)

    print(f"\nwritten {len(status['written'])}, already existed "
          f"{len(status['skipped_exists'])} (--overwrite to redo), missing "
          f"{len(status['missing'])}, QA fail {len(status['qa_fail'])}")
    if status["qa_fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
