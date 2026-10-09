#!/usr/bin/env python3
"""
Per-height-bin comparison table from an als_calval *_metrics.json — no re-running.

Every als_calval script stores stratified metrics per model in the JSON:
    results[<model>]["stratified"]["bins"]  ->  [{lo, hi, n, rmse, mae, r2, bias}, ...]
This prints RMSE and bias per bin for each model, and the change vs the FIRST model
(the reference), so the effect of adding ALS is read off directly.

  train_combined_gedi_als.py  models: gedi_only (needs --base-model), combined
                              (+ per-year keys like combined_2020)
  finetune_als_rf.py          models: base, warm_start, residual, als_only
                              (in <stem>_test_metrics.json)
  train_healpix_rf.py         models: test_pooled (+ test_<year>)
  predict_als_aoi.py          one model, shown as gedi_rf (scored against the ALS)

Usage
-----
    python als_calval/compare_bins.py /scratch/.../combined/combined_rf_metrics.json
    python als_calval/compare_bins.py <json> --models gedi_only_2020 combined_2020
    python als_calval/compare_bins.py <json> --csv bins.csv
"""

import argparse
import csv
import json
import math
import sys


def bin_label(b):
    return f">{b['lo']:.0f}" if math.isinf(b["hi"]) else f"{b['lo']:.0f}-{b['hi']:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="a *_metrics.json from an als_calval script")
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to compare, reference first (default: the pooled models)")
    ap.add_argument("--csv", default=None, help="also write the long-format table here")
    args = ap.parse_args()

    res = json.load(open(args.json))["results"]
    if "stratified" in res:  # predict_als_aoi.py: one model, results = {overall, stratified}
        res = {"gedi_rf": res}
    if args.models:
        models = args.models
    elif "combined" in res:
        models = [m for m in ("gedi_only", "combined") if m in res]
    else:  # finetune modes / train_healpix test_pooled: keys without a per-year suffix
        models = [k for k in res if not k.split("_")[-1].isdigit()]
    missing = [m for m in models if m not in res]
    if missing:
        sys.exit(f"ERROR: {missing} not in this JSON; available: {list(res)}")
    if models == ["combined"]:
        print("NOTE: no gedi_only results — rerun train_combined_gedi_als.py with --base-model "
              "to get the GEDI-only reference on the same pixels", file=sys.stderr)

    ref = models[0]
    bins = {m: res[m]["stratified"]["bins"] for m in models}
    hdr = f"{'bin (m)':<9}{'n':>11}" + "".join(f"{m[:16] + ' RMSE':>22}{'bias':>9}"
                                               for m in models)
    hdr += "".join(f"{'dRMSE ' + m[:10]:>19}{'dbias':>9}" for m in models[1:])
    print(f"reference = {ref}   (d = model - reference; negative dRMSE = better)\n")
    print(hdr + "\n" + "-" * len(hdr))

    rows = []
    for i, b in enumerate(bins[ref]):
        line = f"{bin_label(b):<9}{b['n']:>11,}"
        for m in models:
            bb = bins[m][i]
            line += (f"{bb['rmse']:>22.2f}{bb['bias']:>+9.2f}" if bb["rmse"] is not None
                     else f"{'-':>22}{'-':>9}")
            rows.append({"model": m, "bin": bin_label(bb), "n": bb["n"], "rmse": bb["rmse"],
                         "mae": bb["mae"], "bias": bb.get("bias")})
        for m in models[1:]:
            a, bb = bins[ref][i], bins[m][i]
            line += (f"{bb['rmse'] - a['rmse']:>+19.2f}{bb['bias'] - a['bias']:>+9.2f}"
                     if a["rmse"] is not None and bb["rmse"] is not None
                     else f"{'-':>19}{'-':>9}")
        print(line)

    print()
    for m in models:
        s, o = res[m]["stratified"], res[m]["overall"]
        bal = (f"balanced RMSE={s['balanced_rmse']:.3f}  balanced R2={s['balanced_r2']:.3f}"
               if s.get("balanced_rmse") is not None else "balanced: n/a")
        print(f"{m:<20} overall RMSE={o['rmse']:.3f}  R2={o['r2']:.3f}  bias={o['bias']:+.3f}"
              f"  |  {bal}  macro RMSE={s['macro_rmse']:.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
