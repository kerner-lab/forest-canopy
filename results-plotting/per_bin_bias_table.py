#!/usr/bin/env python3
"""
Per-height-bin bias (mean of prediction - GEDI, m) for every model, 2019 and 2020,
as a manuscript table.

Embedding models: computed from the seed-averaged test predictions
(seed_avg_<year>/<src>/predictions_test.parquet) with the pipeline's own
metrics_stratified.stratified_metrics(), so the bins are exactly the ones used
everywhere else (10 m bins, 0-60 m; heights above 60 m are clipped into 50-60).
Bias is linear, so the bias of the 5-seed mean prediction equals the mean of the
five per-seed biases -- the overall column reproduces summary_test.csv bias_mean.
Tessera drops pixels whose embedding is all zeros (no coverage), ~0.1% of the test
set, so its n is slightly smaller; every other model scores the full GEDI-valid set.

Baselines: read from their metrics.json "stratified" block, using the variant
whose n equals the embedding models' test-pixel count (same pixels):
  ECHOSAT -> echosat_valid, Potapov -> all_valid, Lang (ETH) -> masked_filled_zero.

Writes <out>.csv and prints Markdown and LaTeX versions.

Example:
  python results-plotting/per_bin_bias_table.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_stratified import stratified_metrics  # noqa: E402

RESULTS = "/Users/kdoerkse/Documents/ASU/JPL/results/wacv_geocv"

EMBEDDINGS = [("aef", "AlphaEarth"), ("olmo", "OlmoEarth"), ("prithvi", "Prithvi"),
              ("tessera", "Tessera"), ("presto", "Presto")]
# year -> [(label, metrics.json path relative to RESULTS, block)]
BASELINES = {
    2019: [("ECHOSAT", "echosat_2019/metrics.json", "echosat_valid"),
           ("Potapov", "potapov_baseline/metrics.json", "all_valid")],
    2020: [("ECHOSAT", "echosat_2020/metrics.json", "echosat_valid"),
           ("Lang", "eth_baseline/metrics.json", "masked_filled_zero")],
}


def bin_label(b):
    return f"{b['lo']:.0f}-{b['hi']:.0f}"


def row_from_strat(year, model, group, n, bias, strat):
    row = {"year": year, "model": model, "group": group, "n": n, "overall": bias}
    for b in strat["bins"]:
        row[bin_label(b)] = b["bias"]
    row["balanced"] = strat.get("balanced_bias")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default=f"{RESULTS}/per_bin_bias.csv")
    args = ap.parse_args()
    R = Path(args.results)

    rows = []
    for year in (2019, 2020):
        summary = pd.read_csv(R / f"seed_avg_{year}" / "summary_test.csv").set_index("source")
        # Reference n = the full GEDI-valid test set (AlphaEarth applies no extra
        # pixel filter); the baselines must match it exactly.
        n_ref = len(pd.read_parquet(R / f"seed_avg_{year}" / "aef" /
                                    "predictions_test.parquet", columns=["y_true"]))
        for src, label in EMBEDDINGS:
            df = pd.read_parquet(R / f"seed_avg_{year}" / src / "predictions_test.parquet",
                                 columns=["y_true", "y_pred"])
            yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
            strat = stratified_metrics(yt, yp)
            bias = float((yp - yt).mean())
            want = summary.loc[src, "bias_mean"]
            if abs(bias - want) > 1e-3:
                raise SystemExit(f"{year} {src}: bias {bias:.4f} != summary_test.csv "
                                 f"bias_mean {want:.4f}; parquet / summary mismatch")
            if len(yt) != n_ref:
                print(f"note: {year} {label} scored on n={len(yt):,} "
                      f"({n_ref - len(yt):,} fewer px than the full test set)")
            rows.append(row_from_strat(year, label, "GFM embedding", len(yt), bias, strat))
        for label, rel, block in BASELINES[year]:
            m = json.load(open(R / rel))[block]
            if m["n_samples"] != n_ref:
                raise SystemExit(f"{year} {label} [{block}]: n={m['n_samples']} != "
                                 f"embedding test n={n_ref}")
            rows.append(row_from_strat(year, label, "Baseline", m["n_samples"],
                                       m["bias"], m["stratified"]))

    table = pd.DataFrame(rows)
    out = Path(args.out).expanduser()
    table.to_csv(out, index=False, float_format="%.3f")

    bins = [c for c in table.columns if c[0].isdigit()]
    counts = {}
    for year in (2019, 2020):
        # Per-bin pixel counts are the same for every model (same pixels).
        e = json.load(open(R / BASELINES[year][0][1]))[BASELINES[year][0][2]]["stratified"]
        counts[year] = {bin_label(b): b["n"] for b in e["bins"]}

    def fmt(v):
        return "–" if pd.isna(v) else f"{v:+.2f}".replace("-", "−")

    print("Per-bin bias (m), prediction − GEDI RH98; 10 m bins of GEDI height, "
          "50-60 includes >60 m\n")
    head = ["Year", "Model"] + [f"{b} m" for b in bins] + ["Overall", "Balanced", "n"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|")
    for year in (2019, 2020):
        for _, r in table[table.year == year].iterrows():
            print(f"| {year} | {r.model} | " + " | ".join(fmt(r[b]) for b in bins)
                  + f" | {fmt(r.overall)} | {fmt(r.balanced)} | {r.n:,} |")
        print(f"| {year} | *n per bin* | "
              + " | ".join(f"{counts[year][b]:,}" for b in bins)
              + " | | | |")

    print("\n% LaTeX (booktabs)")
    print("\\begin{tabular}{ll" + "r" * (len(bins) + 2) + "}\n\\toprule")
    print("Year & Model & " + " & ".join(f"{b}" for b in bins)
          + " & Overall & Balanced \\\\\n\\midrule")
    for year in (2019, 2020):
        sub = table[table.year == year]
        for i, (_, r) in enumerate(sub.iterrows()):
            if i == len(EMBEDDINGS):
                print("\\cmidrule(l){2-" + str(len(bins) + 4) + "}")
            cells = [f"{r[b]:+.2f}" for b in bins] + [f"{r.overall:+.2f}",
                                                      f"{r.balanced:+.2f}"]
            print(f"{year if i == 0 else ''} & {r.model} & "
                  + " & ".join(c.replace("-", "$-$") for c in cells) + " \\\\")
        print("\\midrule" if year == 2019 else "\\bottomrule")
    print("\\end{tabular}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
