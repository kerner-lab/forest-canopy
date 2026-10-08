#!/usr/bin/env python3
"""
PART 2 — Run a trained AEF RF on an ALS AOI's wall-to-wall AEF embeddings, with an
AEF <-> ALS alignment QA, and score the prediction against the ALS heights.

Pipeline
--------
1. LOAD the model bundle from train_healpix_rf.py (or a bare joblib'd RF).
2. QA (fails loudly, before any prediction):
     metadata  AEF band count == model features; bands in aef_ch_001..N order; float
               dtype; ALS band exists and is described as --als-band-name (warn);
               CRS / shape / transform identical within --grid-tol pixels;
               footprints overlap.
     pixels    AEF tif not all-NaN; partially-NaN AEF pixels (warn); >= --min-coverage
               of valid ALS pixels have an embedding; AEF L2 norm ~1 (warn otherwise —
               dequantized AEF is unit length, so a far-off norm means a scale mismatch
               with training X).
   If the grids differ, QA fails unless --regrid-als, which warps the ALS band onto the
   AEF grid (the embeddings are never resampled).
3. PREDICT wall-to-wall over the AEF tif -> float32 GeoTIFF (NaN nodata) + map PNG.
4. SCORE vs ALS on every pixel valid in both: overall + stratified metrics, figure.
5. Optional leakage report: with --shapefile, list GEDI training tiles (from the bundle)
   that overlap this AOI.

Usage
-----
    python als_calval/predict_als_aoi.py \\
        --model   /results/als_calval/healpix_rf/healpix_rf.joblib \\
        --aef-tif /data/aef_2020_wall_to_wall_wa.tif \\
        --als-tif /data/WA_site-PierceCounty_1_2020/WA_PierceCounty_1_2020_4326_ALS.tif \\
        --out-dir /results/als_calval/predict_WA \\
        --shapefile /data/shapefiles/healpix_shapefile.shp
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def write_pair_tif(path, aef_tif, rows, cols, y, pred):
    """2-band GeoTIFF on the AEF grid (ALS used for scoring, prediction) for QGIS checks."""
    with rasterio.open(aef_tif) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
    profile.update(count=2, dtype="float32", nodata=np.float32("nan"), compress="lzw",
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    profile.pop("predictor", None)
    with rasterio.open(path, "w", **profile) as dst:
        for b, (vals, name) in enumerate(((y, "als_height_m"), (pred, "predicted_m")), 1):
            a = np.full((H, W), np.nan, np.float32)
            a[rows, cols] = vals
            dst.write(a, b)
            dst.set_band_description(b, name)
    print(f"ALS/prediction pair: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model bundle from train_healpix_rf.py")
    ap.add_argument("--aef-tif", required=True,
                    help="wall-to-wall AEF GeoTIFF for the AOI (alphaearth_wall_to_wall.py)")
    ap.add_argument("--als-tif", required=True, help="ALS raster for the same AOI")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default=None,
                    help="basename for outputs (default: <aef-tif stem>_pred)")
    ap.add_argument("--shapefile", default=None,
                    help="healpix shapefile; if given, report GEDI train tiles overlapping "
                         "this AOI")
    C.add_aoi_args(ap)
    C.add_metric_args(ap)
    ap.add_argument("--seed", type=int, default=42, help="balanced-metric resample seed")
    ap.add_argument("--qa-only", action="store_true", help="run the QA and stop")
    ap.add_argument("--save-pair-tif", action="store_true",
                    help="also write a 2-band (ALS, prediction) tif on the AEF grid")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{Path(args.aef_tif).stem}_pred"

    model, meta = C.load_model_bundle(args.model)
    n_features = C.model_n_features(model)

    X, y, rows, cols, qa = C.load_aoi(args.aef_tif, args.als_tif, args, n_features, "AOI")
    C.write_json(out_dir / f"{stem}_qa.json", qa)
    if args.qa_only:
        print("\n--qa-only: stopping after QA.")
        return

    leak = None
    if args.shapefile and meta.get("train_tile_ids"):
        leak = C.report_tile_overlap(args.shapefile, meta["train_tile_ids"],
                                     C.aoi_footprint(args.aef_tif), "AOI")
    elif args.shapefile:
        print("NOTE: model has no train_tile_ids metadata; skipping leakage report")

    # Wall-to-wall map
    print(f"\nPredicting wall-to-wall over {args.aef_tif}")
    out_tif = out_dir / f"{stem}.tif"
    n_pred = C.predict_to_tif(model, args.aef_tif, out_tif, args.block_rows)
    if not args.no_plot:
        C.plot_map(out_tif, out_dir / f"{stem}_map.png",
                   args.title or f"Predicted canopy height — {Path(args.aef_tif).stem}")

    # Score vs ALS on the paired pixels
    print("\nScoring against ALS")
    pred = model.predict(X)
    res = C.evaluate(y, pred, "GEDI RF vs ALS", args, args.seed)
    if args.save_pair_tif:
        write_pair_tif(out_dir / f"{stem}_als_pair.tif", args.aef_tif, rows, cols, y, pred)

    C.write_json(out_dir / f"{stem}_metrics.json", {
        "model": args.model, "model_meta": {k: v for k, v in meta.items()
                                            if k not in ("train_tile_ids", "test_tile_ids")},
        "aef_tif": args.aef_tif, "als_tif": args.als_tif, "prediction_tif": str(out_tif),
        "n_predicted_px": n_pred, "qa": qa, "train_tiles_overlapping_aoi": leak,
        "results": res,
    })
    C.write_csv(out_dir / f"{stem}_metrics.csv", [C.summary_row("gedi_rf", res)])
    if not args.no_plot:
        C.plot_scatter_panels(out_dir / f"{stem}_vs_als.png", y, {"GEDI RF": pred},
                              {"GEDI RF": res}, "ALS height (m)", args.bin_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
