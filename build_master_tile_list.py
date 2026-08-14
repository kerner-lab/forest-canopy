#!/usr/bin/env python3
"""
Build a MASTER tile list: the healpix tiles that every embedding source actually has.

Comparing encoders is only fair if they are trained and evaluated on identical tiles.
In practice each source has a different set on disk — AlphaEarth and Tessera are
downloaded and near-complete, while Presto/Prithvi/OlmoEarth lose tiles to failed
materializes, walltime kills and truncated npz writes. Scoring them on whatever each
happens to have confounds encoder quality with coverage.

This script intersects the per-source caches against your split lists and writes the
common tile ids, ready to hand to run_gedi_prediction_30m.py --tile-list.

Cache naming is not uniform, so each source dir is searched for ALL of:
    healpix_<id>_float32.npz / healpix_<id>_int8.npz     (dtype-named)
    healpix_<id>_<source>.npz                            (source-named)
plus the _withzeros variants when --include-zeros is set.

Usage:
    python build_master_tile_list.py \\
        --source aef=/ocean/.../data/aef/2019 \\
        --source tessera=/ocean/.../data/tessera/2019 \\
        --source olmo=/ocean/.../data/olmo/2019 \\
        --source presto=/ocean/.../prithvi_presto/2019/out/presto_xy \\
        --source prithvi=/ocean/.../prithvi_presto/2019/out/prithvi_xy \\
        --tile-list train=/ocean/.../data/train_tile_ids.txt \\
        --tile-list test=/ocean/.../data/test_tile_ids.txt \\
        --out-dir ./master_tiles --validate

Writes into --out-dir:
    master_<split>_tile_ids.txt   intersection per split (feed to --tile-list)
    master_tile_ids.txt           all splits combined
    coverage_report.json          per-source present/missing/corrupt, for the paper

--validate opens each npz to catch the truncated writes that existence checks miss.
The (X,y) caches are small, so this is cheap — recommended.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ID_RE = re.compile(r"healpix_(\d+)_")


def parse_kv(pairs, what):
    """['name=path', ...] -> {name: Path}, preserving order."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"ERROR: --{what} must be name=path, got: {p}")
        name, path = p.split("=", 1)
        name = name.strip()
        if name in out:
            raise SystemExit(f"ERROR: duplicate --{what} name: {name}")
        out[name] = Path(path.strip()).expanduser()
    return out


def tiles_in_dir(d: Path, source: str, include_zeros: bool):
    """{tile_id: path} for every cache file in `d`, across both naming conventions."""
    suffix = "_withzeros" if include_zeros else ""
    patterns = [f"healpix_*_float32{suffix}.npz",
                f"healpix_*_int8{suffix}.npz",
                f"healpix_*_{source}{suffix}.npz"]
    found = {}
    for pat in patterns:
        for p in d.glob(pat):
            m = _ID_RE.match(p.name)
            if m:
                # First match wins; float32 is listed first so it is preferred.
                found.setdefault(int(m.group(1)), p)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="NAME=PATH",
                    help="embedding source and its cache dir; repeat per source")
    ap.add_argument("--tile-list", action="append", required=True, metavar="SPLIT=PATH",
                    help="split name and its tile id file; repeat per split")
    ap.add_argument("--out-dir", default="./master_tiles")
    ap.add_argument("--include-zeros", action="store_true",
                    help="match the _withzeros caches instead of the default ones")
    ap.add_argument("--validate", action="store_true",
                    help="open each npz to reject truncated/corrupt files (recommended)")
    ap.add_argument("--require", nargs="+", default=None, metavar="NAME",
                    help="intersect only these sources (default: all). Lets you drop a "
                         "lagging source rather than shrink the whole comparison.")
    args = ap.parse_args()

    sources = parse_kv(args.source, "source")
    splits = parse_kv(args.tile_list, "tile-list")

    for name, d in sources.items():
        if not d.is_dir():
            raise SystemExit(f"ERROR: source '{name}' dir does not exist: {d}")
    for name, f in splits.items():
        if not f.is_file():
            raise SystemExit(f"ERROR: tile list '{name}' not found: {f}")

    required = list(args.require) if args.require else list(sources)
    unknown = [r for r in required if r not in sources]
    if unknown:
        raise SystemExit(f"ERROR: --require names an undeclared source: {unknown}")

    validate = None
    if args.validate:
        from check_missing_embeddings import validate_npz as validate

    wanted = {}
    for split, f in splits.items():
        ids = [int(ln) for ln in f.read_text().split()
               if ln.strip() and not ln.strip().startswith("#")]
        wanted[split] = sorted(set(ids))
        print(f"{split:<8} {len(wanted[split])} tiles requested  ({f})")
    all_wanted = sorted({t for v in wanted.values() for t in v})
    print(f"{'total':<8} {len(all_wanted)} unique tiles across splits\n")

    # ---- per-source availability -------------------------------------------
    have = {}
    report = {"sources": {}, "splits": {}, "required": required,
              "validated": bool(args.validate)}

    for name, d in sources.items():
        found = tiles_in_dir(d, name, args.include_zeros)
        present, corrupt = [], {}
        n = len(all_wanted)
        for i, tid in enumerate(all_wanted, 1):
            p = found.get(tid)
            if p is None:
                continue
            if validate is not None:
                if sys.stderr.isatty():   # keep slurm .err files free of progress spam
                    print(f"\r  validating {name}: {i}/{n}", end="",
                          file=sys.stderr, flush=True)
                err = validate(p)
                if err:
                    corrupt[tid] = err
                    continue
            present.append(tid)
        if validate is not None and sys.stderr.isatty():
            print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

        have[name] = set(present)
        missing = [t for t in all_wanted if t not in have[name] and t not in corrupt]
        report["sources"][name] = {
            "dir": str(d), "n_files_in_dir": len(found),
            "n_present": len(present), "n_missing": len(missing),
            "n_corrupt": len(corrupt),
            "missing": missing, "corrupt": corrupt,
        }
        flag = "" if name in required else "  (not required)"
        print(f"{name:<10} {len(present):>4}/{len(all_wanted)} usable   "
              f"missing={len(missing):<4} corrupt={len(corrupt)}{flag}")

    # ---- intersection -------------------------------------------------------
    common = set(all_wanted)
    for name in required:
        common &= have[name]

    print(f"\nIntersection over {', '.join(required)}: "
          f"{len(common)}/{len(all_wanted)} tiles\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, ids in wanted.items():
        keep = sorted(t for t in ids if t in common)
        path = out_dir / f"master_{split}_tile_ids.txt"
        path.write_text("\n".join(str(t) for t in keep) + "\n" if keep else "")
        pct = 100.0 * len(keep) / len(ids) if ids else 0.0
        print(f"  {split:<8} {len(keep):>4}/{len(ids):<4} ({pct:5.1f}%)  -> {path}")
        report["splits"][split] = {"requested": len(ids), "kept": len(keep),
                                   "dropped": sorted(set(ids) - set(keep))}
        if not keep:
            print(f"    WARNING: split '{split}' is EMPTY after intersection — "
                  f"training on it will fail.")

    master = out_dir / "master_tile_ids.txt"
    master.write_text("\n".join(str(t) for t in sorted(common)) + "\n" if common else "")
    print(f"  {'all':<8} {len(common):>4}{'':<5}         -> {master}")

    rpt = out_dir / "coverage_report.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"\nPer-source detail: {rpt}")

    # Which source is costing you the most tiles? That is the one to fix (or drop).
    if len(common) < len(all_wanted):
        print("\nTiles lost per source (relative to the requested set):")
        for name in required:
            lost = len(all_wanted) - len(have[name])
            uniq = len((set(all_wanted) - have[name])
                       - set().union(*[set(all_wanted) - have[o]
                                       for o in required if o != name]) )
            print(f"  {name:<10} blocks {lost:>4} tiles ({uniq} of them uniquely)")
        print("\nA source whose 'uniquely' count is large is the sole reason those tiles "
              "are excluded —\ndropping it via --require, or repairing it, recovers them.")

    print(f"\nUse it with:\n"
          f"  python run_gedi_prediction_30m.py ... --tile-list {out_dir}/master_tile_ids.txt")


if __name__ == "__main__":
    main()
