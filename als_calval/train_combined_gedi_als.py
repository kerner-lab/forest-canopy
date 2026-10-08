#!/usr/bin/env python3
"""
COMBINED — Train the AEF RF on GEDI train tiles + one or more ALS AOIs, and test it on
the GEDI TEST tile list.

  train  = GEDI (X, y) from --train-tiles  +  every paired AEF/ALS pixel of each ALS AOI
  test   = GEDI (X, y) from --test-tiles   (the same held-out tiles as train_healpix_rf.py)

So the question answered is: does adding ALS supervision improve GEDI-tile performance?
Pass --base-model (the GEDI-only bundle from train_healpix_rf.py) to score it on the
identical test pixels as the reference; its tile lists / npz dirs / zeros setting are then
the defaults here, so both models use exactly the same GEDI split.

Pipeline
--------
1. Tile lists (--train-tiles/--test-tiles, else from --base-model); must be disjoint.
2. Each ALS AOI (--als-aef-tifs[i] paired with --als-tifs[i]) goes through the same
   AEF<->ALS alignment QA as predict_als_aoi.py, then is capped at
   --max-als-samples-per-aoi pixels (dense ALS pixels are strongly autocorrelated, and
   the cap stops one big AOI from dominating the others).
3. Leakage: ALS pixels are TRAINING data, so a GEDI test tile inside an ALS AOI is no
   longer independent. With --shapefile those tiles are reported, and dropped from the
   test set if --drop-test-tiles-in-als.
4. Fit one RF on GEDI + ALS. Weighting: by default every pixel counts once; with
   --als-weight-frac F the ALS pixels together get share F of the total sample weight.
5. Score on the GEDI test tiles (pooled + per data dir), overall + stratified metrics
   (quote balanced_rmse / balanced_r2). Save a model bundle usable by predict_als_aoi.py.

Usage
-----
    python als_calval/train_combined_gedi_als.py \\
        --base-model    /results/als_calval/healpix_rf/healpix_rf.joblib \\
        --als-aef-tifs  /data/aef_2020_wall_to_wall_wa.tif  /data/aef_2020_wall_to_wall_cal.tif \\
        --als-tifs      /data/WA_PierceCounty_1_2020_4326_ALS.tif  /data/2020_Lidar_Santa_Clara_County_CA_4326.tif \\
        --shapefile     /data/shapefiles/healpix_shapefile.shp \\
        --out-dir       /results/als_calval/combined_WACA

    # without a base bundle, give the GEDI inputs explicitly:
    ... --train-tiles train.txt --test-tiles test.txt --data-dirs /data/aef/2020
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

AEF_DIM = 64  # AEF embedding channels; checked against each ALS AOI's AEF tif in QA


def print_table(title, rows):
    print(f"\n{title:<14}{'RMSE':>9}{'R2':>9}{'bias':>9}{'bal.RMSE':>10}{'bal.R2':>9}")
    for r in rows:
        br = f"{r['balanced_rmse']:>10.3f}{r['balanced_r2']:>9.3f}" \
            if r["balanced_rmse"] is not None else f"{'-':>10}{'-':>9}"
        print(f"{r['label']:<14}{r['rmse']:>9.3f}{r['r2']:>9.3f}{r['bias']:>+9.3f}{br}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("GEDI (defaults come from --base-model when given)")
    g.add_argument("--base-model", default=None,
                   help="GEDI-only bundle from train_healpix_rf.py: scored as the reference, "
                        "and its tile lists / dirs / settings become the defaults")
    g.add_argument("--train-tiles", default=None, help="GEDI train healpix id list")
    g.add_argument("--test-tiles", default=None, help="GEDI test healpix id list")
    g.add_argument("--data-dirs", nargs="+", default=None,
                   help="dirs of healpix_<id>_<fmt>[_withzeros].npz, one per AEF year")
    g.add_argument("--test-data-dirs", nargs="+", default=None,
                   help="dirs for the TEST tiles (default: --data-dirs)")
    g.add_argument("--include-zeros", action=argparse.BooleanOptionalAction, default=None,
                   help="load *_withzeros.npz (default: as the base model, else off)")
    g.add_argument("--embedding-format", default=None, help="default: base model's, else float32")
    g.add_argument("--max-train-samples", type=int, default=3_000_000,
                   help="cap on GEDI train pixels; 0 = no cap")

    a = ap.add_argument_group("ALS AOIs (training data)")
    a.add_argument("--als-aef-tifs", nargs="+", required=True,
                   help="wall-to-wall AEF tif per ALS AOI")
    a.add_argument("--als-tifs", nargs="+", required=True,
                   help="ALS tif per AOI, in the same order as --als-aef-tifs")
    a.add_argument("--max-als-samples-per-aoi", type=int, default=500_000,
                   help="random cap on ALS pixels per AOI; 0 = all")
    a.add_argument("--als-weight-frac", type=float, default=None,
                   help="give the ALS pixels this share of the total sample weight "
                        "(default: no reweighting, every pixel counts once)")
    a.add_argument("--shapefile", default=None,
                   help="healpix shapefile: report GEDI test tiles that overlap an ALS AOI")
    a.add_argument("--drop-test-tiles-in-als", action="store_true",
                   help="drop GEDI test tiles overlapping an ALS AOI (needs --shapefile)")
    C.add_aoi_args(ap)

    C.add_rf_args(ap)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--random-state", type=int, default=42)
    C.add_metric_args(ap)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="combined_rf")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if len(args.als_aef_tifs) != len(args.als_tifs):
        sys.exit(f"ERROR: {len(args.als_aef_tifs)} --als-aef-tifs but {len(args.als_tifs)} "
                 f"--als-tifs — give one ALS tif per AEF tif, in the same order")
    if args.als_weight_frac is not None and not 0 < args.als_weight_frac < 1:
        sys.exit("ERROR: --als-weight-frac must be in (0, 1)")
    if args.drop_test_tiles_in_als and not args.shapefile:
        sys.exit("ERROR: --drop-test-tiles-in-als needs --shapefile")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- GEDI inputs (CLI wins, else the base bundle) ---
    base, meta = (C.load_model_bundle(args.base_model) if args.base_model else (None, {}))
    train_ids = C.read_tile_ids(args.train_tiles) if args.train_tiles \
        else meta.get("train_tile_ids")
    test_ids = C.read_tile_ids(args.test_tiles) if args.test_tiles \
        else meta.get("test_tile_ids")
    data_dirs = args.data_dirs or meta.get("data_dirs")
    test_dirs = args.test_data_dirs or (meta.get("test_data_dirs") if not args.data_dirs
                                        else None) or data_dirs
    include_zeros = args.include_zeros if args.include_zeros is not None \
        else meta.get("include_zeros", False)
    emb_fmt = args.embedding_format or meta.get("embedding_format", "float32")
    if not (train_ids and test_ids and data_dirs):
        sys.exit("ERROR: need GEDI --train-tiles, --test-tiles and --data-dirs "
                 "(or a --base-model bundle that stores them)")
    print(f"GEDI tiles: {len(train_ids)} train, {len(test_ids)} test")
    C.check_disjoint(train_ids, test_ids)

    # --- ALS AOIs: QA + read ---
    aois = list(zip(args.als_aef_tifs, args.als_tifs))
    if len({Path(p).resolve() for p, _ in aois}) != len(aois):
        sys.exit("ERROR: the same AEF tif is listed twice in --als-aef-tifs")
    n_features = C.model_n_features(base) if base is not None else AEF_DIM
    Xa, ya, aoi_info, rng = [], [], [], np.random.default_rng(args.random_state)
    for i, (aef_tif, als_tif) in enumerate(aois, 1):
        X, y, _, _, qa = C.load_aoi(aef_tif, als_tif, args, n_features,
                                    f"ALS AOI {i} ({Path(als_tif).stem})")
        n_all = len(y)
        cap = args.max_als_samples_per_aoi
        if cap and n_all > cap:
            idx = rng.choice(n_all, cap, replace=False)
            X, y = X[idx], y[idx]
            print(f"  subsampled {n_all:,} -> {len(y):,} px")
        Xa.append(X)
        ya.append(y)
        aoi_info.append({"aef_tif": aef_tif, "als_tif": als_tif, "n_paired_px": n_all,
                         "n_used_px": int(len(y)), "qa": qa})
    X_als, y_als = np.concatenate(Xa), np.concatenate(ya)
    del Xa, ya

    # --- leakage: GEDI test tiles inside an ALS AOI ---
    leak, dropped = {}, []
    if args.shapefile:
        for info in aoi_info:
            hits = C.report_tile_overlap(args.shapefile, test_ids,
                                         C.aoi_footprint(info["aef_tif"]),
                                         f"ALS AOI {Path(info['als_tif']).stem} (GEDI TEST tiles)")
            leak[info["als_tif"]] = hits
            dropped += [h["tile_id"] for h in hits]
        dropped = sorted(set(dropped), key=int)
        if args.drop_test_tiles_in_als and dropped:
            test_ids = [t for t in test_ids if t not in set(dropped)]
            print(f"Dropped {len(dropped)} GEDI test tile(s) overlapping an ALS AOI -> "
                  f"{len(test_ids)} test tiles")
            if not test_ids:
                sys.exit("ERROR: every GEDI test tile overlaps an ALS AOI")
    else:
        print("NOTE: no --shapefile, so GEDI test tiles inside the ALS AOIs are not "
              "checked — those pixels would make the test score optimistic")

    # --- train on GEDI + ALS ---
    print("\nLoading GEDI train tiles")
    X_g, y_g, g_info = C.load_tile_xy(train_ids, data_dirs, include_zeros, emb_fmt,
                                      args.max_train_samples or None, args.random_state,
                                      label="GEDI train")
    if args.als_weight_frac is None:
        w_als = 1.0
        w = None
    else:
        f = args.als_weight_frac
        w_als = f * len(y_g) / ((1 - f) * len(y_als))
        w = np.concatenate([np.ones(len(y_g)), np.full(len(y_als), w_als)])
    als_share = w_als * len(y_als) / (w_als * len(y_als) + len(y_g))
    print(f"\nTraining set: {len(y_g):,} GEDI px + {len(y_als):,} ALS px "
          f"(ALS weight/px {w_als:.3f} -> ALS share of total weight {als_share:.2f})")
    X = np.concatenate([X_g, X_als])
    y = np.concatenate([y_g, y_als])
    del X_g, X_als
    rf_keys = ("n_estimators", "max_depth", "min_samples_leaf", "max_features")
    rf_flags_default = all(getattr(args, k) == ap.get_default(k) for k in rf_keys)
    if isinstance(base, RandomForestRegressor) and rf_flags_default:
        p = base.get_params()
        params = {k: p[k] for k in rf_keys}
        print("RF hyperparameters: copied from --base-model (pass RF flags to override)")
    else:
        params = C.rf_params_from_args(args)
    model = C.fit_rf(X, y, params, args.n_jobs, args.random_state, sample_weight=w,
                     label="combined GEDI+ALS RF")
    del X, y

    # --- test on the GEDI test tiles ---
    print("\nLoading GEDI test tiles")
    models = {"combined": model}
    if base is not None:
        models = {"gedi_only": base, **models}
    results, rows, y_all, p_all = {}, [], [], {m: [] for m in models}
    for d in test_dirs:
        X_te, y_te, _ = C.load_tile_xy(test_ids, [d], include_zeros, emb_fmt, None,
                                       label=f"test {Path(d).name}")
        y_all.append(y_te)
        for name, m in models.items():
            p = m.predict(X_te)
            p_all[name].append(p)
            if len(test_dirs) > 1:
                lab = f"{name}_{Path(d).name}"
                results[lab] = C.evaluate(y_te, p, lab, args, args.random_state)
                rows.append(C.summary_row(lab, results[lab]))
    y_te = np.concatenate(y_all)
    preds = {name: np.concatenate(v) for name, v in p_all.items()}
    print(f"\n=== GEDI TEST TILES: {len(y_te):,} px (identical set for every model) ===")
    pooled = []
    for name, p in preds.items():
        results[name] = C.evaluate(y_te, p, name, args, args.random_state)
        pooled.append(C.summary_row(name, results[name]))
    rows = pooled + rows
    print_table("GEDI test", pooled)

    bundle_meta = {
        "script": "als_calval/train_combined_gedi_als.py",
        "base_model": args.base_model,
        "train_tile_ids": train_ids, "test_tile_ids": test_ids,
        "test_tiles_dropped_for_als_overlap": dropped if args.drop_test_tiles_in_als else [],
        "data_dirs": data_dirs, "test_data_dirs": test_dirs,
        "include_zeros": include_zeros, "embedding_format": emb_fmt,
        "n_features": int(model.n_features_in_), "rf_params": params,
        "random_state": args.random_state,
        "als_aois": [{k: v for k, v in a.items() if k != "qa"} for a in aoi_info],
        "als_weight_frac_arg": args.als_weight_frac, "als_weight_per_px": w_als,
        "als_share_of_weight": als_share, "n_gedi_train_px": int(g_info["n_pixels_used"]),
    }
    C.save_model_bundle(out_dir / f"{args.stem}.joblib", model, bundle_meta)
    C.write_json(out_dir / f"{args.stem}_metrics.json", {
        **bundle_meta, "gedi_train_data": g_info, "als_qa": [a["qa"] for a in aoi_info],
        "gedi_test_tiles_overlapping_als_aois": leak, "results": results})
    C.write_csv(out_dir / f"{args.stem}_metrics.csv", rows)
    if not args.no_plot:
        C.plot_scatter_panels(out_dir / f"{args.stem}_gedi_test.png", y_te, preds,
                              {n: results[n] for n in preds}, "GEDI RH98, test tiles (m)",
                              args.bin_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
