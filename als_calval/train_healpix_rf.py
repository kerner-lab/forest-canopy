#!/usr/bin/env python3
"""
PART 1 — Train the AEF Random Forest on GEDI from an explicit healpix train/test tile list,
and score it on the held-out GEDI test tiles.

Same model as run_aef_rf_wall_to_wall.py (RF on the 64 AEF channels -> GEDI RH98 in m),
but the tiles come from specific lists that hold out ALS regions.
Tiles are evaluated over GEDI test tiles.

Pipeline
--------
1. Read --train-tiles / --test-tiles (one healpix id per line, or a CSV with a
   healpix_id column). The two lists must be disjoint (--allow-tile-overlap to override).
2. Load per-tile (X, y) .npz from alphaearth_processing.py across --data-dirs (one dir per
   AEF year; tiles missing in a year are skipped and reported).
3. Fit the RF on the train pixels; predict the test pixels.
4. Report overall + height-stratified metrics (balanced_rmse / balanced_r2 are the
   stratified pair to quote), pooled and per data dir (year).
5. Save a model bundle (.joblib: model + metadata incl. tile lists) for
   predict_als_aoi.py and finetune_als_rf.py, plus JSON/CSV metrics and a figure.

Usage
-----
    python als_calval/train_healpix_rf.py \\
        --train-tiles /path/train_tile_ids.txt \\
        --test-tiles  /path/test_tile_ids.txt \\
        --data-dirs   /data/aef/2019 /data/aef/2020 \\
        --out-dir     /results/als_calval/healpix_rf

    # different years for test than train:
    ... --data-dirs /data/aef/2019 /data/aef/2020 --test-data-dirs /data/aef/2020
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-tiles", required=True, help="train healpix id list (.txt/.csv)")
    ap.add_argument("--test-tiles", required=True, help="test healpix id list (.txt/.csv)")
    ap.add_argument("--data-dirs", nargs="+", required=True,
                    help="dirs of healpix_<id>_<fmt>[_withzeros].npz, one per AEF year")
    ap.add_argument("--test-data-dirs", nargs="+", default=None,
                    help="dirs to read the TEST tiles from (default: --data-dirs)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="healpix_rf", help="basename for outputs")
    ap.add_argument("--allow-tile-overlap", action="store_true",
                    help="allow tiles in both lists (test metrics become optimistic)")
    ap.add_argument("--include-zeros", action="store_true",
                    help="load *_withzeros.npz (keep 0 m GEDI heights)")
    ap.add_argument("--embedding-format", default="float32")
    ap.add_argument("--max-train-samples", type=int, default=3_000_000,
                    help="random cap on train pixels for memory; 0 = no cap")
    C.add_rf_args(ap)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--random-state", type=int, default=42)
    C.add_metric_args(ap)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dirs = args.test_data_dirs or args.data_dirs

    train_ids = C.read_tile_ids(args.train_tiles)
    test_ids = C.read_tile_ids(args.test_tiles)
    print(f"Tile lists: {len(train_ids)} train ({args.train_tiles}), "
          f"{len(test_ids)} test ({args.test_tiles})")
    shared = C.check_disjoint(train_ids, test_ids, args.allow_tile_overlap)

    print("\nLoading train tiles")
    X_tr, y_tr, tr_info = C.load_tile_xy(
        train_ids, args.data_dirs, args.include_zeros, args.embedding_format,
        args.max_train_samples or None, args.random_state, label="train")
    params = C.rf_params_from_args(args)
    rf = C.fit_rf(X_tr, y_tr, params, args.n_jobs, args.random_state, label="GEDI RF")
    train_res = C.overall_metrics(y_tr, rf.predict(X_tr))
    print(f"  train (in-sample) R2={train_res['r2']:.3f} RMSE={train_res['rmse']:.2f} m")
    n_features = X_tr.shape[1]
    del X_tr, y_tr

    # Test: pooled over all test dirs, plus one block per dir (year).
    print("\nLoading test tiles")
    results, rows, preds = {}, [], {}
    y_all, p_all = [], []
    for d in test_dirs:
        X_te, y_te, _ = C.load_tile_xy(test_ids, [d], args.include_zeros,
                                       args.embedding_format, None, label=f"test {Path(d).name}")
        p = rf.predict(X_te)
        y_all.append(y_te)
        p_all.append(p)
        if len(test_dirs) > 1:
            lab = f"test_{Path(d).name}"
            results[lab] = C.evaluate(y_te, p, lab, args, args.random_state)
            rows.append(C.summary_row(lab, results[lab]))
    y_te, p_te = np.concatenate(y_all), np.concatenate(p_all)
    print()
    results["test_pooled"] = C.evaluate(y_te, p_te, "test_pooled", args, args.random_state)
    rows.insert(0, C.summary_row("test_pooled", results["test_pooled"]))
    preds["GEDI test tiles"] = p_te

    meta = {
        "script": "als_calval/train_healpix_rf.py",
        "train_tiles_file": args.train_tiles, "test_tiles_file": args.test_tiles,
        "train_tile_ids": train_ids, "test_tile_ids": test_ids,
        "tiles_in_both_lists": shared,
        "data_dirs": args.data_dirs, "test_data_dirs": test_dirs,
        "include_zeros": args.include_zeros, "embedding_format": args.embedding_format,
        "n_features": int(n_features), "rf_params": params, "random_state": args.random_state,
        "train_data": tr_info, "train_metrics_in_sample": train_res,
    }
    C.save_model_bundle(out_dir / f"{args.stem}.joblib", rf, meta)
    C.write_json(out_dir / f"{args.stem}_metrics.json", {**meta, "results": results})
    C.write_csv(out_dir / f"{args.stem}_metrics.csv", rows)
    if not args.no_plot:
        C.plot_scatter_panels(out_dir / f"{args.stem}_test.png", y_te, preds,
                              {"GEDI test tiles": results["test_pooled"]},
                              "GEDI RH98 (m)", args.bin_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
