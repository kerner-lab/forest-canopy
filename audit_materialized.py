#!/usr/bin/env python3
"""
Audit the Phase-1 materialized Sentinel-2 imagery before spending GPU hours on Phase 2.

Motivation: when /ocean hits its quota, `rslearn dataset materialize` fails per-tile and
leaves windows with missing or TRUNCATED geotiffs. unified_embedding_pipeline.py does not
treat that as fatal — it logs "no S2 imagery for Presto" (or writes garbage embeddings for
a partially-readable stack) and moves on. This script finds those tiles up front.

It deliberately reuses find_olmo_s2_geotiffs() from the pipeline, so "what the audit
checks" and "what the pipeline will actually find at run time" cannot drift apart.

Verdicts:
    ok           all expected monthly geotiffs present and readable
    no_dataset   the per-tile rslearn dataset / window dir does not exist
    incomplete   fewer than --expect geotiffs found (materialize did not finish)
    corrupt      a geotiff exists but fails to open or decode (truncated write)

Usage:
    # fast structural check (open headers only)
    python audit_materialized.py \\
        --dataset-root /ocean/projects/agr260002p/kdoerkse/data/prithvi_presto/2019/rslearn \\
        --tile-ids /ocean/projects/agr260002p/kdoerkse/data/shapefiles/tile_ids.txt

    # also decode pixels — catches a geotiff truncated mid-raster, which a header-only
    # check happily accepts. Slower; this is the one to trust after a quota event.
    python audit_materialized.py --dataset-root ... --tile-ids ... --deep

    # write the tiles needing a Phase-1 re-materialize, one per line (tile_ids.txt format,
    # so it can be used directly as the TILE_IDS of a re-run array job)
    python audit_materialized.py --dataset-root ... --tile-ids ... --deep --out remat_ids.txt
"""

import argparse
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing.unified_embedding_pipeline import (  # noqa: E402
    OLMO_S2_LAYER,
    UNIFIED_S1_LAYER,
    find_olmo_s2_geotiffs,
    find_unified_s1_geotiffs,
    window_dir,
)
from data_processing.presto_processing import PRESTO_NUM_TIMESTEPS  # noqa: E402

# Worst-verdict ranking: a tile is reported at the severity of its worst layer.
SEVERITY = {"ok": 0, "incomplete": 1, "corrupt": 2}


def check_geotiff(path: Path, deep: bool) -> str:
    """Return an error string if `path` is not a usable geotiff, else ""."""
    try:
        with rasterio.open(path) as src:
            if src.count == 0 or src.width == 0 or src.height == 0:
                return f"{path.parent.parent.name}: empty raster ({src.count}x{src.height}x{src.width})"
            if deep:
                # Opening only parses the header — a file truncated inside the pixel data
                # still opens cleanly. Force an actual decode of the last block, which is
                # the part a truncated write loses first.
                h = min(src.height, 256)
                src.read(1, window=rasterio.windows.Window(0, src.height - h, min(src.width, 256), h))
    except Exception as exc:
        return f"{path.parent.parent.name}: {type(exc).__name__}: {exc}"
    return ""


def audit_tile(dataset_root: Path, tid: int, expect: int, deep: bool, layers: list) -> tuple:
    """Return (verdict, detail) for one healpix tile, worst verdict over `layers`."""
    win_dir = window_dir(dataset_root / f"healpix_{tid}", tid)
    if not win_dir.is_dir():
        return "no_dataset", str(win_dir)

    verdict, details = "ok", []
    for layer_name, finder in layers:
        paths = finder(win_dir)
        if len(paths) < expect:
            layer_verdict = "incomplete"
            detail = f"{len(paths)}/{expect} geotiffs"
        else:
            errors = [e for e in (check_geotiff(p, deep) for p in paths) if e]
            if errors:
                layer_verdict = "corrupt"
                detail = "; ".join(errors[:3]) + (" ..." if len(errors) > 3 else "")
            else:
                layer_verdict = "ok"
                detail = f"{len(paths)}/{expect} geotiffs"
        if SEVERITY[layer_verdict] > SEVERITY[verdict]:
            verdict = layer_verdict
        details.append(f"{layer_name}: {detail}")
    return verdict, " | ".join(details)


def clear_completed(dataset_root: Path, tid: int, expect: int, layers: list) -> int:
    """Delete the `completed` markers for `tid` so Phase 1 re-materializes it.

    rslearn skips a layer whose group-0 `completed` marker exists (dataset/manage.py
    materialize_window), so a layer that finished materializing and was TRUNCATED
    afterwards is skipped forever. Removing the markers is what makes a Phase-1 re-run
    actually redo the tile; the geotiffs themselves are overwritten in place.
    """
    win_dir = window_dir(dataset_root / f"healpix_{tid}", tid)
    removed = 0
    for layer_name, _ in layers:
        for i in range(expect):
            layer = layer_name if i == 0 else f"{layer_name}.{i}"
            marker = win_dir / "layers" / layer / "completed"
            if marker.exists():
                marker.unlink()
                removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True,
                    help="dir holding the per-tile rslearn datasets (…/rslearn, "
                         "containing healpix_<id>/ subdirs)")
    ap.add_argument("--tile-ids", default=None,
                    help="file with one healpix_id per line (the array job's TILE_IDS). "
                         "Omit and pass --discover to audit whatever is on disk instead.")
    ap.add_argument("--discover", action="store_true",
                    help="ignore --tile-ids and audit every healpix_<id> dir found under "
                         "--dataset-root. Use this when you know imagery exists but not "
                         "which array indices it corresponds to.")
    ap.add_argument("--out-ok", default=None,
                    help="write the tile ids that PASSED (one per line). This is the run "
                         "list: point the array job's TILE_IDS at it and use "
                         "--array=1-<line count> so index N maps to line N.")
    ap.add_argument("--expect", type=int, default=PRESTO_NUM_TIMESTEPS,
                    help=f"monthly geotiffs expected per tile (default {PRESTO_NUM_TIMESTEPS})")
    ap.add_argument("--deep", action="store_true",
                    help="decode pixels, not just headers — catches mid-raster truncation")
    ap.add_argument("--with-s1", action="store_true",
                    help=f"also audit the {UNIFIED_S1_LAYER} VV/VH layer (Presto --with-s1). "
                         "Without this only Sentinel-2 is checked, so a tile missing ALL "
                         "of its S1 still reports ok.")
    ap.add_argument("--clear-completed", action="store_true",
                    help="delete the rslearn `completed` markers of every non-ok tile. "
                         "REQUIRED for 'corrupt' tiles: without it a Phase-1 re-run skips "
                         "them (the marker says they are done). Only the markers are "
                         "deleted; the geotiffs are overwritten by the re-run.")
    ap.add_argument("--out", default=None,
                    help="write tile ids needing a Phase-1 re-materialize (one per line)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        ap.error(f"--dataset-root is not a directory: {dataset_root}")

    if args.discover:
        # The per-tile dataset dirs are the ground truth for "what imagery exists".
        tile_ids, source = [], f"discovered under {dataset_root}"
        for d in sorted(dataset_root.glob("healpix_*")):
            if d.is_dir():
                try:
                    tile_ids.append(int(d.name.split("_", 1)[1]))
                except ValueError:
                    print(f"  (skipping unparseable dir name: {d.name})")
        tile_ids.sort()
    elif args.tile_ids:
        tile_ids = [int(ln) for ln in Path(args.tile_ids).read_text().split() if ln.strip()]
        source = args.tile_ids
    else:
        ap.error("pass either --tile-ids <file> or --discover")

    layers = [(OLMO_S2_LAYER, find_olmo_s2_geotiffs)]
    if args.with_s1:
        layers.append((UNIFIED_S1_LAYER, find_unified_s1_geotiffs))

    print(f"Dataset root : {dataset_root}")
    print(f"Tiles        : {len(tile_ids)} from {source}")
    print(f"Layers       : {', '.join(name for name, _ in layers)}")
    print(f"Mode         : {'deep (decodes pixels)' if args.deep else 'structural (headers only)'}")
    print()

    buckets = {"ok": [], "no_dataset": [], "incomplete": [], "corrupt": []}
    for tid in tile_ids:
        verdict, detail = audit_tile(dataset_root, tid, args.expect, args.deep, layers)
        buckets[verdict].append(tid)
        if verdict != "ok":
            print(f"  {verdict:11s} healpix_{tid}: {detail}")

    print()
    for verdict in ("ok", "no_dataset", "incomplete", "corrupt"):
        print(f"{verdict:11s}: {len(buckets[verdict])}")

    # The pass list — tiles that can be embedded right now.
    if args.out_ok:
        Path(args.out_ok).write_text("\n".join(str(t) for t in sorted(buckets["ok"])) + "\n")
        print(f"\nWrote {len(buckets['ok'])} ready tile ids -> {args.out_ok}")
        print(f"Run the embedding array with TILE_IDS={args.out_ok} "
              f"and --array=1-{len(buckets['ok'])}")

    # Everything that is not "ok" needs Phase 1 re-run for that tile.
    remat = sorted(buckets["no_dataset"] + buckets["incomplete"] + buckets["corrupt"])

    if args.clear_completed and remat:
        total = sum(clear_completed(dataset_root, tid, args.expect, layers) for tid in remat)
        print(f"\nCleared {total} `completed` marker(s) across {len(remat)} tile(s) — "
              f"Phase 1 will now re-materialize them.")
    elif args.clear_completed:
        print("\nNothing to clear; every tile is ok.")
    elif buckets["corrupt"]:
        print(f"\nWARNING: {len(buckets['corrupt'])} corrupt tile(s) are still marked "
              f"completed — a Phase-1 re-run will SKIP them. Re-run with --clear-completed.")

    if args.out and remat:
        Path(args.out).write_text("\n".join(str(t) for t in remat) + "\n")
        print(f"\nWrote {len(remat)} tile ids to re-materialize -> {args.out}")
        print(f"Re-run Phase 1 with TILE_IDS={args.out} and --array=1-{len(remat)}")
    elif args.out:
        print(f"\nNothing to re-materialize; {args.out} not written.")


if __name__ == "__main__":
    main()
