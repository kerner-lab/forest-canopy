#!/usr/bin/env python3
"""
PART 3 — Fine-tune the GEDI-trained AEF RF on one ALS AOI, then test on a DIFFERENT ALS AOI.

A random forest has no gradient fine-tuning, so "fine-tune" is implemented as several
standard adaptation strategies, all scored on the SAME test-AOI pixels so they compare
directly (--modes picks which to run):

  base        the GEDI-trained model unchanged (the reference every mode must beat).
  warm_start  keep all GEDI trees and ADD --ws-new-trees trees grown on the fine-tune ALS
              pixels (sklearn warm_start). Prediction averages all trees, so ALS gets
              weight n_new / (n_old + n_new) — 0.5 by default.
  residual    keep the GEDI model and fit a second RF on the fine-tune AOI to predict
              (ALS - base prediction); output = base + correction, clipped at 0 m.
              Corrects the GEDI->ALS bias without discarding what GEDI learned.
  als_only    an RF trained on the fine-tune ALS pixels alone (no GEDI). Tells you how
              much the GEDI pre-training is actually worth on the test AOI.
  combined    retrain from scratch on GEDI train tiles + fine-tune ALS pixels, with ALS
              given --combined-als-frac of the total sample weight. Needs --data-dirs
              (train tiles default to those stored in the model bundle).

Pipeline
--------
1. Load the base bundle (train_healpix_rf.py).
2. QA + read the fine-tune AOI (AEF + ALS) and the test AOI (AEF + ALS) — the same
   alignment QA as predict_als_aoi.py. The two AOIs must not overlap (bbox check;
   --allow-aoi-overlap to override), or the test is not independent.
3. Subsample the fine-tune pixels (--max-finetune-samples; dense ALS pixels are highly
   spatially autocorrelated, so more pixels add little information).
4. Build each mode, predict the test-AOI pixels, report overall + stratified metrics
   (quote balanced_rmse / balanced_r2), comparison CSV/JSON and figure.
   The base model is also scored on the fine-tune AOI, to show the GEDI->ALS gap there.
5. Optional: --write-maps (wall-to-wall test-AOI tif per mode), --save-models.

Usage
-----
    python als_calval/finetune_als_rf.py \\
        --model         /results/als_calval/healpix_rf/healpix_rf.joblib \\
        --ft-aef-tif    /data/aef_2020_wall_to_wall_wa.tif \\
        --ft-als-tif    /data/WA_site-PierceCounty_1_2020/WA_PierceCounty_1_2020_4326_ALS.tif \\
        --test-aef-tif  /data/aef_2020_wall_to_wall_cal.tif \\
        --test-als-tif  /data/CA_site-SantaClaraCounty_2020/2020_Lidar_Santa_Clara_County_CA_4326.tif \\
        --out-dir       /results/als_calval/ft_WA_test_CA \\
        --modes base warm_start residual als_only
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ALL_MODES = ["base", "warm_start", "residual", "als_only", "combined"]


def check_aoi_independence(args):
    if Path(args.ft_aef_tif).resolve() == Path(args.test_aef_tif).resolve() or \
       Path(args.ft_als_tif).resolve() == Path(args.test_als_tif).resolve():
        sys.exit("ERROR: the fine-tune and test AOIs use the same file — test on a different AOI")
    ft, te = C.aoi_footprint(args.ft_aef_tif), C.aoi_footprint(args.test_aef_tif)
    inter = ft.intersection(te).area
    if inter > 0:
        frac = inter / te.area
        msg = (f"fine-tune and test AOI bounding boxes overlap ({100 * frac:.1f}% of the "
               f"test bbox) — test pixels may sit next to fine-tune pixels")
        if not args.allow_aoi_overlap:
            sys.exit(f"ERROR: {msg}. Pass --allow-aoi-overlap if the polygons themselves "
                     f"are disjoint (bboxes of irregular AOIs can overlap).")
        print(f"WARNING: {msg}", file=sys.stderr)
        return frac
    print("AOI independence: fine-tune and test AOI bounding boxes do not overlap")
    return 0.0


def build_warm_start(base, X, y, n_new, n_jobs):
    if not isinstance(base, RandomForestRegressor):
        print("  warm_start: base is not a RandomForestRegressor — skipping", file=sys.stderr)
        return None, None
    m = copy.deepcopy(base)
    n_old = len(m.estimators_)
    m.set_params(warm_start=True, n_estimators=n_old + n_new, n_jobs=n_jobs)
    print(f"\nwarm_start: +{n_new} ALS trees on top of {n_old} GEDI trees "
          f"(ALS weight {n_new / (n_old + n_new):.2f}) on {len(y):,} px ...")
    m.fit(X, y)
    m.set_params(warm_start=False)
    return m, {"n_gedi_trees": n_old, "n_als_trees": n_new,
               "als_tree_weight": n_new / (n_old + n_new)}


def build_residual(base, X, y, params, n_jobs, seed):
    resid = y - base.predict(X)
    print(f"\nresidual: ALS - base on fine-tune px: mean={resid.mean():+.2f} m "
          f"sd={resid.std():.2f} m")
    corr = C.fit_rf(X, resid, params, n_jobs, seed, label="residual RF")
    return C.ResidualCorrectedModel(base, corr, clip_min=0.0), {
        "finetune_residual_mean": float(resid.mean()),
        "finetune_residual_sd": float(resid.std()), "rf_params": params}


def build_combined(base, meta, X_ft, y_ft, args):
    tile_ids = C.read_tile_ids(args.train_tiles) if args.train_tiles else meta.get("train_tile_ids")
    data_dirs = args.data_dirs or meta.get("data_dirs")
    if not tile_ids or not data_dirs:
        sys.exit("ERROR: mode 'combined' needs GEDI training data: pass --data-dirs (and "
                 "--train-tiles if the bundle does not store them)")
    include_zeros = meta.get("include_zeros", False) if args.include_zeros is None \
        else args.include_zeros
    print("\ncombined: loading GEDI train tiles")
    X_g, y_g, _ = C.load_tile_xy(tile_ids, data_dirs, include_zeros,
                                 meta.get("embedding_format", "float32"),
                                 args.max_train_samples or None, args.seed, label="GEDI train")
    f = args.combined_als_frac
    w_als = f * len(y_g) / ((1 - f) * len(y_ft))
    print(f"  ALS weight per px = {w_als:.3f}  (ALS share of total weight = {f:.2f})")
    X = np.concatenate([X_g, X_ft])
    y = np.concatenate([y_g, y_ft])
    w = np.concatenate([np.ones(len(y_g)), np.full(len(y_ft), w_als)])
    del X_g
    if isinstance(base, RandomForestRegressor):
        p = base.get_params()
        params = {k: p[k] for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                    "max_features")}
    else:
        params = meta.get("rf_params") or C.rf_params_from_args(args, "ft-")
    m = C.fit_rf(X, y, params, args.n_jobs, args.seed, sample_weight=w, label="combined RF")
    return m, {"n_gedi_px": int(len(y) - len(y_ft)), "n_als_px": int(len(y_ft)),
               "als_weight_per_px": w_als, "als_weight_frac": f, "rf_params": params,
               "data_dirs": data_dirs}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base bundle from train_healpix_rf.py")
    ap.add_argument("--ft-aef-tif", required=True, help="fine-tune AOI: wall-to-wall AEF tif")
    ap.add_argument("--ft-als-tif", required=True, help="fine-tune AOI: ALS tif")
    ap.add_argument("--test-aef-tif", required=True, help="test AOI: wall-to-wall AEF tif")
    ap.add_argument("--test-als-tif", required=True, help="test AOI: ALS tif")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="finetune")
    ap.add_argument("--modes", nargs="+", default=["base", "warm_start", "residual", "als_only"],
                    choices=ALL_MODES, help="adaptation strategies to run (see top of file)")
    ap.add_argument("--allow-aoi-overlap", action="store_true",
                    help="allow overlapping fine-tune/test AOI bounding boxes")
    ap.add_argument("--shapefile", default=None,
                    help="healpix shapefile; report GEDI train tiles overlapping the test AOI")
    C.add_aoi_args(ap)

    g = ap.add_argument_group("fine-tuning")
    g.add_argument("--max-finetune-samples", type=int, default=500_000,
                   help="random cap on fine-tune ALS pixels; 0 = all")
    g.add_argument("--ws-new-trees", type=int, default=None,
                   help="warm_start: trees to add (default = as many as the base has -> "
                        "50/50 GEDI/ALS)")
    C.add_rf_args(g, prefix="ft-")  # residual + als_only RF
    g.add_argument("--combined-als-frac", type=float, default=0.5,
                   help="combined: share of total sample weight given to the ALS pixels")
    g.add_argument("--data-dirs", nargs="+", default=None,
                   help="combined: GEDI npz dirs (default: from the model bundle)")
    g.add_argument("--train-tiles", default=None,
                   help="combined: GEDI train tile list (default: from the model bundle)")
    g.add_argument("--include-zeros", action=argparse.BooleanOptionalAction, default=None,
                   help="combined: *_withzeros.npz (default: as the base model was trained)")
    g.add_argument("--max-train-samples", type=int, default=3_000_000,
                   help="combined: cap on GEDI train pixels; 0 = no cap")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    C.add_metric_args(ap)
    ap.add_argument("--write-maps", action="store_true",
                    help="write a wall-to-wall test-AOI prediction tif + PNG per mode")
    ap.add_argument("--save-models", action="store_true",
                    help="joblib-dump each fine-tuned model (RF bundles can be GBs)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if not 0 < args.combined_als_frac < 1:
        sys.exit("ERROR: --combined-als-frac must be in (0, 1)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base, meta = C.load_model_bundle(args.model)
    n_features = C.model_n_features(base)
    aoi_overlap = check_aoi_independence(args)

    X_ft, y_ft, _, _, qa_ft = C.load_aoi(args.ft_aef_tif, args.ft_als_tif, args, n_features,
                                         "fine-tune AOI")
    X_te, y_te, _, _, qa_te = C.load_aoi(args.test_aef_tif, args.test_als_tif, args,
                                         n_features, "test AOI")
    C.write_json(out_dir / f"{args.stem}_qa.json", {"finetune_aoi": qa_ft, "test_aoi": qa_te})

    leak = None
    if args.shapefile and meta.get("train_tile_ids"):
        leak = {lab: C.report_tile_overlap(args.shapefile, meta["train_tile_ids"],
                                           C.aoi_footprint(t), lab)
                for lab, t in (("fine-tune AOI", args.ft_aef_tif),
                               ("test AOI", args.test_aef_tif))}

    # Base model on the fine-tune AOI: the GEDI->ALS gap the fine-tuning has to close.
    print("\n=== base model on the FINE-TUNE AOI (all paired px, before fine-tuning) ===")
    base_on_ft = C.evaluate(y_ft, base.predict(X_ft), "base @ fine-tune AOI", args, args.seed)

    n_ft_all = len(y_ft)
    if args.max_finetune_samples and n_ft_all > args.max_finetune_samples:
        idx = np.random.default_rng(args.seed).choice(n_ft_all, args.max_finetune_samples,
                                                      replace=False)
        X_ft, y_ft = X_ft[idx], y_ft[idx]
        print(f"\nFine-tune pixels subsampled {n_ft_all:,} -> {len(y_ft):,} (seed={args.seed})")

    ft_params = C.rf_params_from_args(args, "ft-")
    models, info = {}, {}
    for mode in args.modes:
        if mode == "base":
            models[mode], info[mode] = base, {}
        elif mode == "warm_start":
            n_new = args.ws_new_trees or len(getattr(base, "estimators_", [])) or 200
            m, i = build_warm_start(base, X_ft, y_ft, n_new, args.n_jobs)
            if m is not None:
                models[mode], info[mode] = m, i
        elif mode == "residual":
            models[mode], info[mode] = build_residual(base, X_ft, y_ft, ft_params,
                                                      args.n_jobs, args.seed)
        elif mode == "als_only":
            models[mode] = C.fit_rf(X_ft, y_ft, ft_params, args.n_jobs, args.seed,
                                    label="ALS-only RF")
            info[mode] = {"rf_params": ft_params}
        elif mode == "combined":
            models[mode], info[mode] = build_combined(base, meta, X_ft, y_ft, args)

    # Score every mode on the identical test-AOI pixel set.
    print(f"\n=== TEST AOI: {len(y_te):,} paired px (identical set for every mode) ===")
    preds, results, rows = {}, {}, []
    for mode, m in models.items():
        preds[mode] = m.predict(X_te)
        results[mode] = C.evaluate(y_te, preds[mode], mode, args, args.seed)
        rows.append(C.summary_row(mode, results[mode]))

    print(f"\n{'mode':<12}{'RMSE':>9}{'R2':>9}{'bias':>9}{'bal.RMSE':>10}{'bal.R2':>9}")
    for r in rows:
        br = f"{r['balanced_rmse']:>10.3f}{r['balanced_r2']:>9.3f}" \
            if r["balanced_rmse"] is not None else f"{'-':>10}{'-':>9}"
        print(f"{r['label']:<12}{r['rmse']:>9.3f}{r['r2']:>9.3f}{r['bias']:>+9.3f}{br}")

    C.write_csv(out_dir / f"{args.stem}_test_metrics.csv", rows)
    C.write_json(out_dir / f"{args.stem}_test_metrics.json", {
        "model": args.model,
        "base_meta": {k: v for k, v in meta.items()
                      if k not in ("train_tile_ids", "test_tile_ids")},
        "finetune_aoi": {"aef_tif": args.ft_aef_tif, "als_tif": args.ft_als_tif,
                         "n_paired_px": n_ft_all, "n_used_px": int(len(y_ft))},
        "test_aoi": {"aef_tif": args.test_aef_tif, "als_tif": args.test_als_tif,
                     "n_paired_px": int(len(y_te))},
        "aoi_bbox_overlap_frac_of_test": aoi_overlap,
        "train_tiles_overlapping_aois": leak,
        "qa": {"finetune_aoi": qa_ft, "test_aoi": qa_te},
        "base_on_finetune_aoi": base_on_ft,
        "modes": info, "results": results,
    })
    if not args.no_plot:
        C.plot_scatter_panels(out_dir / f"{args.stem}_test.png", y_te, preds, results,
                              "test-AOI ALS height (m)", args.bin_size)

    if args.write_maps:
        for mode, m in models.items():
            tif = out_dir / f"{args.stem}_test_{mode}.tif"
            C.predict_to_tif(m, args.test_aef_tif, tif, args.block_rows)
            if not args.no_plot:
                C.plot_map(tif, tif.with_suffix(".png"), f"Test AOI — {mode}")
    if args.save_models:
        for mode, m in models.items():
            if mode != "base":
                C.save_model_bundle(out_dir / f"{args.stem}_{mode}.joblib", m,
                                    {"mode": mode, "base_model": args.model, **info[mode],
                                     "finetune_aef_tif": args.ft_aef_tif,
                                     "finetune_als_tif": args.ft_als_tif})
    print("\nDone.")


if __name__ == "__main__":
    main()
