#!/usr/bin/env python3
"""
List the rslearn/healpix_<id> folders that are safe to delete, i.e. tiles whose BOTH
embeddings have already been generated.

Cross-references three locations:
    <out>/prithvi_xy/healpix_<id>_float32.npz   Prithvi (X,y)
    <out>/healpix_<id>_presto.npz               Presto dense
    <rslearn>/healpix_<id>/                     materialized Sentinel-2 (the space hog)

A tile is reported as RECLAIMABLE only when both embedding files exist. Anything missing
one of them is listed separately with the reason, so nothing gets removed while it is
still needed as an input.

This script NEVER deletes anything. It writes a list of paths for you to review and
remove yourself.

Usage:
    # default paths are the 2019 prithvi_presto tree on /ocean
    python find_reclaimable_tiles.py

    # write the removable folder paths, one per line
    python find_reclaimable_tiles.py --out reclaimable.txt

    # ALSO open every npz to prove it is not truncated (slow: the dense Presto files are
    # GB-scale). Strongly recommended before deleting, since the imagery is the only way
    # to regenerate a tile whose npz turns out to be corrupt.
    python find_reclaimable_tiles.py --validate --out reclaimable.txt

Then, after reading the list:
    xargs -a reclaimable.txt -d '\\n' du -sh      # confirm the payoff
    xargs -a reclaimable.txt -d '\\n' rm -rf      # remove
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_BASE = "/ocean/projects/agr260002p/kdoerkse/data/prithvi_presto"


def dir_size(path: Path) -> int:
    """Total bytes under `path` (follows the tree, ignores unreadable entries)."""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"default: {DEFAULT_BASE}")
    ap.add_argument("--year", default="2019")
    ap.add_argument("--out-dir", default=None, help="override <base>/<year>/out")
    ap.add_argument("--rslearn-dir", default=None, help="override <base>/<year>/rslearn")
    ap.add_argument("--validate", action="store_true",
                    help="open each npz to confirm it is fully readable (slow but safer)")
    ap.add_argument("--no-size", action="store_true",
                    help="skip the du walk (faster on a busy filesystem)")
    ap.add_argument("--out", default=None,
                    help="write reclaimable folder paths here, one per line")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.base) / args.year / "out"
    rslearn_dir = Path(args.rslearn_dir) if args.rslearn_dir else Path(args.base) / args.year / "rslearn"
    prithvi_dir = out_dir / "prithvi_xy"

    for label, p in (("out", out_dir), ("rslearn", rslearn_dir)):
        if not p.is_dir():
            sys.exit(f"ERROR: {label} dir does not exist: {p}")

    validate = None
    if args.validate:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_missing_embeddings import validate_npz as validate

    print(f"prithvi_xy : {prithvi_dir}")
    print(f"presto     : {out_dir}/healpix_<id>_presto.npz")
    print(f"rslearn    : {rslearn_dir}")
    print(f"validate   : {'ON (opening every npz)' if args.validate else 'OFF (existence only)'}\n")

    tiles = sorted(
        (int(d.name.split("_", 1)[1]), d)
        for d in rslearn_dir.glob("healpix_*")
        if d.is_dir() and d.name.split("_", 1)[1].isdigit()
    )
    if not tiles:
        sys.exit(f"No healpix_* directories found under {rslearn_dir}")

    reclaimable, blocked = [], []
    for tid, tile_dir in tiles:
        reasons = []
        for label, path in (("prithvi", prithvi_dir / f"healpix_{tid}_float32.npz"),
                            ("presto", out_dir / f"healpix_{tid}_presto.npz")):
            if not path.exists():
                reasons.append(f"{label} missing")
            elif validate is not None:
                err = validate(path)
                if err:
                    reasons.append(f"{label} CORRUPT ({err})")
        size = 0 if args.no_size else dir_size(tile_dir)
        (blocked if reasons else reclaimable).append((tid, tile_dir, size, reasons))

    print(f"{'tile':>10}  {'size':>8}  status")
    print("-" * 72)
    for tid, _d, size, _r in reclaimable:
        print(f"{tid:>10}  {human(size):>8}  RECLAIMABLE (prithvi + presto both present)")
    for tid, _d, size, reasons in blocked:
        print(f"{tid:>10}  {human(size):>8}  KEEP — {', '.join(reasons)}")

    total = sum(s for _t, _d, s, _r in reclaimable)
    kept = sum(s for _t, _d, s, _r in blocked)
    print("-" * 72)
    print(f"rslearn tiles       : {len(tiles)}")
    print(f"RECLAIMABLE         : {len(reclaimable)}  ({human(total)})")
    print(f"keep                : {len(blocked)}  ({human(kept)})")

    # Tiles with embeddings but no rslearn dir — already cleaned, listed for completeness.
    have_ids = {t for t, _ in tiles}
    done_only = sorted(
        int(p.name.split("_")[1]) for p in prithvi_dir.glob("healpix_*_float32.npz")
    ) if prithvi_dir.is_dir() else []
    already = [i for i in done_only if i not in have_ids]
    if already:
        print(f"already cleaned     : {len(already)} tiles have prithvi (X,y) and no rslearn dir")

    if args.out:
        if reclaimable:
            Path(args.out).write_text("\n".join(str(d) for _t, d, _s, _r in reclaimable) + "\n")
            print(f"\nWrote {len(reclaimable)} paths -> {args.out}")
            print(f"Review it, then:  xargs -a {args.out} -d '\\n' rm -rf")
        else:
            print(f"\nNothing reclaimable; {args.out} not written.")


if __name__ == "__main__":
    main()
