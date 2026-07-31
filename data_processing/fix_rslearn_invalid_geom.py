#!/usr/bin/env python3
"""Patch the installed rslearn so invalid STAC footprints don't kill a whole layer.

Symptom (Phase 1 with --with-s1):

    shapely.errors.GEOSException: TopologyException: side location conflict at
    -118.92514736712955 44.376983527204914. This can occur if the input geometry
    is invalid.

raised from rslearn/utils/geometry.py safely_reproject_within_valid_area(), reached via
get_items() -> match_candidate_items_to_window() -> _filter_and_project_items(). Some
Sentinel-1 GRD swath polygons served by Planetary Computer are self-intersecting, and
GEOS refuses to intersect them. Sentinel-2 (Element84) footprints are simple boxes, so
only the sentinel1 layer trips this.

Impact without the fix: the exception escapes get_items() for the WHOLE window, so
`rslearn dataset prepare` (which defaults to --ignore-errors) drops the entire sentinel1
layer from items.json -- all 12 monthly groups, not just the offending scene. materialize
then logs "Not materializing layer sentinel1 ... because it is not prepared" and the tile
silently ends up S2-only. Retrying does not help: the same invalid polygon comes back from
the STAC API on every attempt.

The fix mirrors the fallback shp_intersects() already applies ~30 lines above in the same
file -- it catches shapely.GEOSException there, but not on the .intersection() call that
actually throws.

This is a string-level edit rather than a diff so it needs no line numbers and survives
minor upstream reformatting. It is idempotent: re-running on a patched file is a no-op.
Re-run it after any rebuild of the conda env, which silently reverts site-packages.

    python data_processing/fix_rslearn_invalid_geom.py            # apply
    python data_processing/fix_rslearn_invalid_geom.py --check    # report status only
    python data_processing/fix_rslearn_invalid_geom.py --revert   # restore the .bak
"""

import argparse
import shutil
import sys
from pathlib import Path

TARGET = "        clipped_shp = src_wgs84.shp.intersection(get_clip_region())\n"

REPLACEMENT = '''        try:
            clipped_shp = src_wgs84.shp.intersection(get_clip_region())
        except shapely.GEOSException:
            # Self-intersecting source footprint (seen on Sentinel-1 GRD swaths):
            # GEOS raises "side location conflict" instead of returning a result.
            # Repair the rings and retry, as shp_intersects() does above.
            logger.warning(
                "repairing invalid source geometry before clipping (%s)",
                shapely.is_valid_reason(src_wgs84.shp),
            )
            clipped_shp = shapely.make_valid(src_wgs84.shp).intersection(
                get_clip_region()
            )
            # make_valid can emit a GeometryCollection mixing polygons with the
            # degenerate lines/points it split out. Downstream expects an areal
            # geometry, so keep only the polygonal parts.
            if clipped_shp.geom_type == "GeometryCollection":
                polys = [
                    g
                    for g in clipped_shp.geoms
                    if g.geom_type in ("Polygon", "MultiPolygon")
                ]
                clipped_shp = shapely.unary_union(polys) if polys else shapely.Polygon()
'''

MARKER = "repairing invalid source geometry before clipping"


def locate() -> Path:
    """Find the installed rslearn/utils/geometry.py."""
    try:
        from rslearn.utils import geometry
    except ImportError as exc:
        sys.exit(f"cannot import rslearn ({exc}) -- activate the env first")
    return Path(geometry.__file__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the .bak backup")
    ap.add_argument("--path", default=None, help="geometry.py to patch (default: the "
                                                 "installed rslearn's)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else locate()
    backup = path.with_suffix(".py.bak")
    src = path.read_text()
    applied = MARKER in src

    print(f"file    : {path}")
    print(f"status  : {'PATCHED' if applied else 'unpatched'}")

    if args.check:
        return

    if args.revert:
        if not backup.exists():
            sys.exit(f"no backup to restore: {backup}")
        shutil.copy2(backup, path)
        print(f"reverted from {backup}")
        return

    if applied:
        print("nothing to do")
        return

    count = src.count(TARGET)
    if count != 1:
        sys.exit(f"expected exactly 1 occurrence of the clip line, found {count}. "
                 f"rslearn has changed -- re-derive the patch from:\n"
                 f"  python -c \"import inspect; from rslearn.utils import geometry; "
                 f"print(inspect.getsource(geometry.safely_reproject_within_valid_area))\"")

    patched = src.replace(TARGET, REPLACEMENT)
    try:
        compile(patched, str(path), "exec")
    except SyntaxError as exc:
        sys.exit(f"patched source does not compile ({exc}) -- aborting, file untouched")

    shutil.copy2(path, backup)
    path.write_text(patched)
    print(f"backup  : {backup}")
    print("applied : ok")


if __name__ == "__main__":
    main()
