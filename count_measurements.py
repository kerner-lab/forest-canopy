#!/usr/bin/env python3
"""
Count GEDI measurements per split per embedding source, and verify that the sources
actually sampled the SAME pixels.

Why this is separate from build_master_tile_list.py
---------------------------------------------------
build_master_tile_list.py intersects sources at TILE granularity: it checks that a
cache file exists, opens cleanly, and has len(X) == len(y) within itself. It never
compares one source against another, so two sources can both "pass" for a tile while
holding different pixel sets.

They genuinely can differ. save_xy_npz() applies two masks
(unified_embedding_pipeline.py:523-527):

    mask   = np.isfinite(y_grid)              # GEDI-valid  -- identical for all sources
    finite = np.all(np.isfinite(Xv), axis=1)  # embedding-valid -- SOURCE-DEPENDENT

The second drops pixels where that encoder's embedding vector has a non-finite
component (no imagery coverage), which Prithvi and Presto do not lose in the same
places. AlphaEarth / Tessera / OLMo build (X,y) through their own code paths.

The (X,y) npz files carry no coordinates, but they do not need to: every path writes
y as the GEDI heights in row-major order over its mask, so y IS the identity key.
Equal-length, equal-valued y vectors mean the same pixels in the same order.

Comparison is by value with a tolerance, not by hash: GEDI is stored in cm and the
paths differ in where they cast (float32 division vs float64 then cast), so the last
ULP can disagree on identical pixels. --atol defaults to 1e-4 m, far below GEDI's 1 cm
storage resolution, so it accepts that noise and nothing else.

Usage:
    python count_measurements.py \\
        --source aef=/ocean/.../data/aef/2019 \\
        --source tessera=/ocean/.../data/tessera/2019 \\
        --source olmo=/ocean/.../data/olmo_xy/2019 \\
        --source prithvi=/ocean/.../prithvi_presto/2019/out/prithvi_xy \\
        --source presto=/ocean/.../prithvi_presto/2019/out/presto_xy \\
        --tile-list train=./master_tiles_2019/master_train_tile_ids.txt \\
        --tile-list test=./master_tiles_2019/master_test_tile_ids.txt \\
        --out-csv counts_2019.csv --out-md counts_2019.md

Only the `y` array is read from each npz (np.load on an npz is lazy per-member), so
this never decompresses the embedding matrices and stays fast on the full tile set.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the exact cache-naming rules so "what this counts" and "what the master list
# selected" cannot drift apart.
from build_master_tile_list import parse_kv, tiles_in_dir  # noqa: E402


def load_y(path: Path):
    """Return the y vector from an (X,y) npz, or None if it cannot be read."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if "y" not in z.files:
                return None
            return z["y"]
    except Exception:
        return None


def fmt_int(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="NAME=PATH",
                    help="embedding source and its (X,y) cache dir; repeat per source")
    ap.add_argument("--tile-list", action="append", required=True, metavar="SPLIT=PATH",
                    help="split name and its tile id file; repeat per split. Use the "
                         "master_<split>_tile_ids.txt files from build_master_tile_list.py")
    ap.add_argument("--include-zeros", action="store_true",
                    help="match the _withzeros caches instead of the default ones")
    ap.add_argument("--reference", default=None, metavar="NAME",
                    help="source to treat as the pixel-set reference for the alignment "
                         "check (default: the first --source given)")
    ap.add_argument("--atol", type=float, default=1e-4,
                    help="tolerance in metres when comparing y vectors (default: 1e-4, "
                         "well below GEDI's 1 cm storage resolution)")
    ap.add_argument("--max-pixel-diff-frac", type=float, default=0.0, metavar="F",
                    help="tolerate a tile whose per-source pixel COUNT differs from the "
                         "reference by <= F of the reference count (e.g. 0.10 for 10%%). "
                         "Default 0.0 = exact match required. A tolerated tile is kept in "
                         "the aligned lists even though the sources hold slightly "
                         "different pixels — see the caveat printed with the summary.")
    ap.add_argument("--no-check-alignment", action="store_true",
                    help="only count; skip the cross-source pixel-identity check")
    ap.add_argument("--max-report", type=int, default=20,
                    help="max mismatching tiles to print per source (default: 20)")
    ap.add_argument("--out-aligned-dir", default=None, metavar="DIR",
                    help="write aligned_<split>_tile_ids.txt / aligned_tile_ids.txt "
                         "holding ONLY the tiles where every source is present AND "
                         "sampled identical pixels. Feed these to "
                         "run_gedi_prediction_30m.py --tile-list and to the Potapov / "
                         "ETH baselines' --tile-list so models and baselines share one "
                         "pixel set. Ignored with --no-check-alignment.")
    ap.add_argument("--out-csv", default=None, help="write the per-split table as CSV")
    ap.add_argument("--out-md", default=None,
                    help="write the per-split table as a markdown table for the manuscript")
    ap.add_argument("--out-json", default=None,
                    help="write full detail (per-tile counts, mismatches) as JSON")
    args = ap.parse_args()

    sources = parse_kv(args.source, "source")
    splits = parse_kv(args.tile_list, "tile-list")

    for name, d in sources.items():
        if not d.is_dir():
            raise SystemExit(f"ERROR: source '{name}' dir does not exist: {d}")
    for name, f in splits.items():
        if not f.is_file():
            raise SystemExit(f"ERROR: tile list '{name}' not found: {f}")

    reference = args.reference or next(iter(sources))
    if reference not in sources:
        raise SystemExit(f"ERROR: --reference '{reference}' is not a declared source")

    # Resolve every source's cache files once.
    found = {name: tiles_in_dir(d, name, args.include_zeros)
             for name, d in sources.items()}

    wanted = {}
    for split, f in splits.items():
        ids = [int(ln) for ln in f.read_text().split()
               if ln.strip() and not ln.strip().startswith("#")]
        wanted[split] = sorted(set(ids))

    # counts[split][source] = {tile_id: n_measurements}
    counts = {s: {name: {} for name in sources} for s in wanted}
    missing = {s: {name: [] for name in sources} for s in wanted}
    mismatch = {s: {name: [] for name in sources} for s in wanted}

    for split, ids in wanted.items():
        n = len(ids)
        for i, tid in enumerate(ids, 1):
            if sys.stderr.isatty():
                print(f"\r  {split}: {i}/{n}", end="", file=sys.stderr, flush=True)

            ys = {}
            for name in sources:
                p = found[name].get(tid)
                if p is None:
                    missing[split][name].append(tid)
                    continue
                y = load_y(p)
                if y is None:
                    missing[split][name].append(tid)
                    continue
                ys[name] = y
                counts[split][name][tid] = int(len(y))

            if args.no_check_alignment or reference not in ys:
                continue
            ref = ys[reference]
            for name, y in ys.items():
                if name == reference:
                    continue
                if len(y) != len(ref):
                    frac = abs(len(y) - len(ref)) / len(ref) if len(ref) else 1.0
                    mismatch[split][name].append(
                        {"tile": tid, "reason": "length",
                         "n": int(len(y)), "n_ref": int(len(ref)),
                         "frac": frac,
                         "tolerated": frac <= args.max_pixel_diff_frac})
                elif not np.allclose(y, ref, rtol=0, atol=args.atol, equal_nan=True):
                    # Equal length but different values means the sources dropped the
                    # SAME NUMBER of different pixels. Under the subsequence structure
                    # that offsets every later row, so an element-wise differing count
                    # overstates the real divergence and cannot be thresholded
                    # meaningfully. Never tolerated — it is structurally different from
                    # a plain count gap and warrants a look.
                    bad = int(np.sum(~np.isclose(y, ref, rtol=0, atol=args.atol,
                                                 equal_nan=True)))
                    mismatch[split][name].append(
                        {"tile": tid, "reason": "values",
                         "n_differing": bad, "n": int(len(y)),
                         "frac": 1.0, "tolerated": False})
        if sys.stderr.isatty():
            print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    # ---- the manuscript table ----------------------------------------------
    rows = []
    for split in wanted:
        for name in sources:
            c = counts[split][name]
            per_tile = np.array(sorted(c.values()), dtype=np.int64)
            rows.append({
                "split": split,
                "source": name,
                "tiles_requested": len(wanted[split]),
                "tiles_with_data": len(c),
                "measurements": int(per_tile.sum()) if len(per_tile) else 0,
                "mean_per_tile": float(per_tile.mean()) if len(per_tile) else 0.0,
                "median_per_tile": float(np.median(per_tile)) if len(per_tile) else 0.0,
                "min_per_tile": int(per_tile.min()) if len(per_tile) else 0,
                "max_per_tile": int(per_tile.max()) if len(per_tile) else 0,
            })

    hdr = (f"{'split':<8} {'source':<10} {'tiles':>11} {'measurements':>14} "
           f"{'mean/tile':>10} {'median':>9} {'min':>8} {'max':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['split']:<8} {r['source']:<10} "
              f"{r['tiles_with_data']:>4}/{r['tiles_requested']:<6} "
              f"{fmt_int(r['measurements']):>14} "
              f"{r['mean_per_tile']:>10,.0f} {r['median_per_tile']:>9,.0f} "
              f"{fmt_int(r['min_per_tile']):>8} {fmt_int(r['max_per_tile']):>9}")

    # ---- alignment verdict --------------------------------------------------
    if not args.no_check_alignment:
        print(f"\nPixel alignment (reference = '{reference}', atol = {args.atol} m)")
        print("-" * 72)
        if args.max_pixel_diff_frac > 0:
            print(f"  tolerating count gaps <= {100*args.max_pixel_diff_frac:.1f}% "
                  f"of the reference pixel count")
        clean = True
        worst = 0.0
        for split in wanted:
            for name in sources:
                if name == reference:
                    continue
                bad = mismatch[split][name]
                hard = [m for m in bad if not m["tolerated"]]
                soft = [m for m in bad if m["tolerated"]]
                worst = max([worst] + [m["frac"] for m in soft])
                if not bad:
                    print(f"  {split:<8} {name:<10} OK — identical pixel sets")
                    continue
                if not hard:
                    biggest = max(m["frac"] for m in soft)
                    print(f"  {split:<8} {name:<10} OK — {len(soft)} tile(s) within "
                          f"tolerance (worst {100*biggest:.2f}%)")
                    continue
                clean = False
                print(f"  {split:<8} {name:<10} {len(hard)} tile(s) EXCEED tolerance "
                      f"({len(soft)} within it):")
                for m in hard[:args.max_report]:
                    if m["reason"] == "length":
                        print(f"      tile {m['tile']}: {m['n']} px vs "
                              f"{m['n_ref']} px (diff {m['n'] - m['n_ref']:+d}, "
                              f"{100*m['frac']:.2f}%)")
                    else:
                        print(f"      tile {m['tile']}: same length, "
                              f"{m['n_differing']}/{m['n']} y values differ "
                              f"— different pixels, not a count gap")
                if len(hard) > args.max_report:
                    print(f"      ... and {len(hard) - args.max_report} more")
        if worst > 0:
            print(f"\n  CAVEAT: tiles kept under --max-pixel-diff-frac hold slightly\n"
                  f"  DIFFERENT pixels per encoder (worst gap {100*worst:.2f}% of the\n"
                  f"  reference count). The comparison is no longer pixel-identical —\n"
                  f"  report N per encoder, or the reference N with the max gap stated.")
        if clean and worst == 0.0:
            print("\n  All sources sampled identical pixels on every tile — the "
                  "measurement\n  counts above are directly comparable across encoders.")
        elif clean:
            print("\n  Every remaining gap is within tolerance; no tile is excluded for\n"
                  "  misalignment. See the CAVEAT above for how to report N.")
        else:
            print("\n  WARNING: sources do NOT share a pixel set on every tile, so the\n"
                  "  encoders are not being scored on identical data. A 'length' diff is\n"
                  "  the save_xy_npz() finite-embedding filter dropping different\n"
                  "  no-coverage pixels per encoder (unified_embedding_pipeline.py:526).\n"
                  "  To report one honest N, intersect at pixel level, or state the\n"
                  "  per-encoder N and the spread in the table caption.")

    # ---- aligned tile lists -------------------------------------------------
    # Tile-level exclusion, not pixel-level intersection. On a tile every source
    # agreed on, the shared pixel set IS the full GEDI-valid set (AlphaEarth applies
    # no second filter), which is exactly what potapov_baseline.py:280 and
    # eth_canopy_baseline.py:324 sample — so models and baselines line up with no
    # change to any of them, and no pixel index needs to be recovered.
    aligned = {}
    if args.out_aligned_dir and not args.no_check_alignment:
        ad = Path(args.out_aligned_dir)
        ad.mkdir(parents=True, exist_ok=True)
        print(f"\nAligned tile lists -> {ad}")
        for split, ids in wanted.items():
            bad = {m["tile"] for name in sources for m in mismatch[split][name]
                   if not m["tolerated"]}
            absent = {t for name in sources for t in missing[split][name]}
            keep = [t for t in ids if t not in bad and t not in absent]
            aligned[split] = keep
            path = ad / f"aligned_{split}_tile_ids.txt"
            path.write_text("\n".join(str(t) for t in keep) + "\n" if keep else "")
            pct = 100.0 * len(keep) / len(ids) if ids else 0.0
            print(f"  {split:<8} {len(keep):>4}/{len(ids):<4} ({pct:5.1f}%)  "
                  f"dropped {len(bad)} misaligned, {len(absent)} absent  -> {path}")
            if not keep:
                print(f"    WARNING: split '{split}' is EMPTY — training on it will fail.")
        every = sorted({t for v in aligned.values() for t in v})
        master = ad / "aligned_tile_ids.txt"
        master.write_text("\n".join(str(t) for t in every) + "\n" if every else "")
        print(f"  {'all':<8} {len(every):>4}                            -> {master}")
        print("\n  Measurements on the aligned set:")
        for s in aligned:
            per_src = {n: sum(counts[s][n].get(t, 0) for t in aligned[s])
                       for n in sources}
            lo, hi = min(per_src.values()), max(per_src.values())
            ref_n = per_src[reference]
            if lo == hi:
                print(f"    {s:<8} {len(aligned[s]):>4} tiles  {fmt_int(ref_n):>14} px "
                      f"(identical for every source)")
            else:
                spread = 100.0 * (hi - lo) / hi if hi else 0.0
                print(f"    {s:<8} {len(aligned[s]):>4} tiles  {fmt_int(ref_n):>14} px "
                      f"({reference}); per-source range {fmt_int(lo)}–{fmt_int(hi)}, "
                      f"spread {spread:.2f}%")
                for n in sources:
                    print(f"        {n:<10} {fmt_int(per_src[n]):>14}")
        print("\n  Use for BOTH models and baselines:")
        print(f"    run_gedi_prediction_30m.py ... --tile-list {master}")
        print(f"    potapov_baseline.py        ... --tile-list {master}")
        print(f"    eth_canopy_baseline.py     ... --tile-list {master}")

    # ---- outputs ------------------------------------------------------------
    fields = ["split", "source", "tiles_requested", "tiles_with_data", "measurements",
              "mean_per_tile", "median_per_tile", "min_per_tile", "max_per_tile"]
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out_csv}")

    if args.out_md:
        lines = ["| Split | Source | Tiles | Measurements | Mean/tile | Median/tile |",
                 "|---|---|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['split']} | {r['source']} | "
                f"{r['tiles_with_data']}/{r['tiles_requested']} | "
                f"{fmt_int(r['measurements'])} | {r['mean_per_tile']:,.0f} | "
                f"{r['median_per_tile']:,.0f} |")
        Path(args.out_md).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out_md}")

    if args.out_json:
        detail = {
            "reference": reference, "atol": args.atol,
            "sources": {k: str(v) for k, v in sources.items()},
            "table": rows,
            "per_tile_counts": {s: {n: counts[s][n] for n in sources} for s in wanted},
            "missing": {s: {n: missing[s][n] for n in sources} for s in wanted},
            "mismatch": {s: {n: mismatch[s][n] for n in sources} for s in wanted},
        }
        Path(args.out_json).write_text(json.dumps(detail, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
