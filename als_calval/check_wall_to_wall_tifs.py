#!/usr/bin/env python3
"""
Check wall-to-wall embedding tifs (AEF / OlmoEarth / TESSERA) against their ALS tifs.

Pairs <emb-dir>/<stem>_<encoder>_<year>.tif with <als-dir>/<stem>.tif and checks:
  grid      same CRS, shape and transform as the ALS tif (pixel-for-pixel pairing)
  bands     float dtype, band count as expected for the encoder, names <prefix>_ch_001..N
  data      the file finalized (not all-NaN) and the fraction of valid ALS pixels that
            have an embedding (band 1 finite), read in row strips
  values    on a random sample of windows: pixels with only SOME bands finite, and the
            median L2 norm (AEF is ~unit length)
  sidecar   the .json year/size agree with the tif

    python als_calval/check_wall_to_wall_tifs.py                  # Bridges-2 defaults
    python als_calval/check_wall_to_wall_tifs.py --emb-dir /path/olmo/als_aois
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DATA = "/ocean/projects/agr260002p/kdoerkse/data"
NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<enc>aef|olmo|tessera)_(?P<year>\d{4})\.tif$")
EXPECTED_BANDS = {"aef": 64, "olmo": (64, 128), "tessera": 128}


def grid_problems(e, a, tol=1e-3):
    out = []
    if e.crs != a.crs:
        out.append(f"CRS {e.crs} vs ALS {a.crs}")
    if e.shape != a.shape:
        out.append(f"shape {e.shape} vs ALS {a.shape}")
    px = abs(a.transform.a)
    diff = max(abs(x - y) for x, y in zip(e.transform[:6], a.transform[:6])) / px
    if diff > tol:
        out.append(f"transform differs by {diff:.3g} px")
    return out


def check(emb_path, als_dir, als_band, n_windows, rng):
    m = NAME_RE.match(emb_path.name)
    stem, enc, year = m["stem"], m["enc"], int(m["year"])
    als_path = als_dir / f"{stem}.tif"
    fails, warns, info = [], [], {}
    if not als_path.exists():
        return [f"no ALS tif {als_path.name}"], warns, info

    with rasterio.open(emb_path) as e, rasterio.open(als_path) as a:
        info["size"] = f"{e.width}x{e.height}"
        info["bands"] = e.count
        fails += [f"grid: {p}" for p in grid_problems(e, a)]

        exp = EXPECTED_BANDS[enc]
        if e.count not in (exp if isinstance(exp, tuple) else (exp,)):
            fails.append(f"{e.count} bands, expected {exp}")
        if not np.issubdtype(np.dtype(e.dtypes[0]), np.floating):
            fails.append(f"dtype {e.dtypes[0]} (not dequantized float)")
        want = [f"{enc}_ch_{i:03d}" for i in range(1, e.count + 1)]
        if list(e.descriptions) != want:
            warns.append(f"band names {e.descriptions[:2]}... not {want[0]}..{want[-1]}")
        if a.descriptions[als_band - 1] != "rh98":
            warns.append(f"ALS band {als_band} is {a.descriptions[als_band - 1]!r}, not rh98")

        # Coverage: band 1 finite vs ALS valid, in row strips (bounded memory).
        n_als = n_both = n_emb = 0
        if not any(f.startswith("grid") for f in fails):
            for r0 in range(0, e.height, 512):
                win = Window(0, r0, e.width, min(512, e.height - r0))
                fe = np.isfinite(e.read(1, window=win))
                fa = np.isfinite(a.read(als_band, window=win, masked=True).filled(np.nan))
                n_emb += int(fe.sum())
                n_als += int(fa.sum())
                n_both += int((fe & fa).sum())
            cov = n_both / n_als if n_als else 0.0
            info["ALS_px"] = f"{n_als:,}"
            info["coverage"] = f"{100 * cov:.1f}%"
            if n_emb == 0:
                fails.append("all-NaN embedding (did not finalize / failed run)")
            elif cov < 0.8:
                fails.append(f"only {100 * cov:.1f}% of ALS pixels have an embedding")
            elif cov < 0.99:
                warns.append(f"{100 * (1 - cov):.1f}% of ALS pixels have no embedding")

        # Values: random windows, all bands.
        norms, partial = [], 0
        ws = 128
        for _ in range(n_windows if n_emb else 0):
            r = int(rng.integers(0, max(1, e.height - ws)))
            c = int(rng.integers(0, max(1, e.width - ws)))
            x = e.read(window=Window(c, r, min(ws, e.width), min(ws, e.height))).reshape(e.count, -1)
            fin = np.isfinite(x)
            allf = fin.all(axis=0)
            partial += int((fin.any(axis=0) & ~allf).sum())
            if allf.any():
                norms.append(np.linalg.norm(x[:, allf], axis=0))
        if norms:
            med = float(np.median(np.concatenate(norms)))
            info["norm_p50"] = f"{med:.3f}"
            if enc == "aef" and not 0.85 <= med <= 1.1:
                warns.append(f"median AEF norm {med:.3f}, expected ~1")
        if partial:
            fails.append(f"{partial:,} sampled px have only some bands finite")

    side = emb_path.with_suffix(".json")
    if side.exists():
        s = json.loads(side.read_text())
        if s.get("year") != year:
            fails.append(f"sidecar year {s.get('year')} != filename year {year}")
        if (s.get("width"), s.get("height")) != tuple(map(int, info["size"].split("x"))):
            warns.append("sidecar width/height differ from the tif")
        if "resampling" in s:
            info["resampling"] = s["resampling"]
    else:
        warns.append("no .json sidecar")
    return fails, warns, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", default=f"{DATA}/aef/als_aois")
    ap.add_argument("--als-dir", default=f"{DATA}/als")
    ap.add_argument("--als-band", type=int, default=2, help="ALS RH98 band (1-based)")
    ap.add_argument("--n-windows", type=int, default=20,
                    help="random 128 px windows read with all bands for the value checks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb_dir, als_dir = Path(args.emb_dir), Path(args.als_dir)
    tifs = sorted(p for p in emb_dir.glob("*.tif") if NAME_RE.match(p.name))
    if not tifs:
        sys.exit(f"No <stem>_<aef|olmo|tessera>_<year>.tif files in {emb_dir}")
    rng = np.random.default_rng(args.seed)

    n_fail = 0
    for p in tifs:
        print(f"\n{p.name}", flush=True)
        try:
            fails, warns, info = check(p, als_dir, args.als_band, args.n_windows, rng)
        except Exception as exc:                       # unreadable / corrupt file
            fails, warns, info = [f"{type(exc).__name__}: {exc}"], [], {}
        print("  " + "  ".join(f"{k}={v}" for k, v in info.items()))
        for w in warns:
            print(f"  WARN  {w}")
        for f in fails:
            print(f"  FAIL  {f}")
        print("  -> " + ("FAIL" if fails else "OK"))
        n_fail += bool(fails)

    print(f"\n{len(tifs) - n_fail}/{len(tifs)} OK")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
