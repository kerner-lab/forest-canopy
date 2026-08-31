#!/usr/bin/env python3
"""
Measure the USER-SIDE ACQUISITION COST of each embedding source, per tile.

The five encoders are not acquired the same way, so a single "compute cost" number
would be misleading. This measures what it actually costs YOU to go from nothing to
an (X, y) cache for one tile, per stage:

    stage                 serves            what it does
    --------------------  ----------------  ------------------------------------------
    materialize           presto, prithvi   rslearn prepare+materialize S2/S1 (the big one)
    presto_encode         presto            Presto forward pass (CPU)
    prithvi_encode        prithvi           Prithvi-EO-2.0 forward pass (GPU)
    aef_fetch             aef               subset + dequantize AlphaEarth mosaic
    tessera_fetch         tessera           fetch + dequantize GeoTessera tiles
    olmo_preprocess       olmo              downloaded COGs -> (X, y) cache

`materialize` is reported ONCE and shared by presto+prithvi — do not add it to both
columns of a paper table or you double-count the dominant cost.

For each (stage, tile) it records wall-clock, CPU seconds, peak RSS, and bytes written
to the watched directories, then reports mean +/- std per tile and an extrapolation to
your full tile count.

Run it on 3-5 tiles. It does not need the whole dataset to give a usable estimate.

Usage:
    # see the exact commands without running anything
    python benchmark_acquisition.py --tiles 25156 34901 35925 --year 2019 \\
        --shapefile /path/healpix_shapefile.shp --gedi-dir /path/2019/gedi \\
        --base /path/prithvi_presto --dry-run

    # run a subset of stages
    python benchmark_acquisition.py --tiles 34901 35925 --year 2019 ... \\
        --only aef_fetch tessera_fetch

    # summarize a previous run without re-running
    python benchmark_acquisition.py --report bench_results.jsonl --n-tiles-total 311

Results append to --out (JSONL), so an interrupted sweep resumes without losing work.
"""

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Stages are ordered: `materialize` must precede the two encoders for a given tile.
STAGE_ORDER = ["materialize", "presto_encode", "prithvi_encode",
               "aef_fetch", "tessera_fetch", "olmo_preprocess"]

SERVES = {
    "materialize": ["presto", "prithvi"],
    "presto_encode": ["presto"],
    "prithvi_encode": ["prithvi"],
    "aef_fetch": ["aef"],
    "tessera_fetch": ["tessera"],
    "olmo_preprocess": ["olmo"],
}


def dir_size(path: Path) -> int:
    """Total bytes under `path`; 0 if it does not exist."""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}T"


def build_stages(a):
    """Command + watched dirs per stage. Edit here if your tree differs."""
    base = Path(a.base)
    ds = base / str(a.year) / "rslearn" / "healpix_{tile}"
    out = base / str(a.year) / "out"
    py = sys.executable

    common = ["--year", str(a.year), "--shapefile", a.shapefile, "--gedi-dir", a.gedi_dir]

    return {
        "materialize": {
            "cmd": [py, "-u", "data_processing/unified_embedding_pipeline.py",
                    "--healpix-id", "{tile}", *common,
                    "--dataset-path", str(ds), "--output-dir", str(out),
                    "--materialize-only", "--workers", str(a.workers), "--verbose"]
                   + (["--dataset-config", a.dataset_config] if a.dataset_config else []),
            "watch": [str(ds)],
        },
        "presto_encode": {
            "cmd": [py, "-u", "data_processing/unified_embedding_pipeline.py",
                    "--healpix-id", "{tile}", *common,
                    "--dataset-path", str(ds), "--output-dir", str(out),
                    "--skip-olmo", "--skip-download", "--presto-sampling", "aggregate30",
                    "--with-s1", "--emit-xy", "--batch-size", "4096", "--verbose"]
                   + (["--dataset-config", a.dataset_config] if a.dataset_config else []),
            "watch": [str(out / "healpix_{tile}_presto.npz"), str(out / "presto_xy")],
        },
        "prithvi_encode": {
            "cmd": [py, "-u", "data_processing/unified_embedding_pipeline.py",
                    "--healpix-id", "{tile}", *common,
                    "--dataset-path", str(ds), "--output-dir", str(out),
                    "--skip-olmo", "--skip-presto", "--do-prithvi",
                    "--prithvi-model-config", a.prithvi_config,
                    "--emit-xy", "--skip-download", "--overwrite", "--verbose"]
                   + (["--dataset-config", a.dataset_config] if a.dataset_config else []),
            "watch": [str(out / "prithvi_xy")],
        },
        "aef_fetch": {
            "cmd": [py, "-u", "data_processing/alphaearth_processing.py",
                    "--healpix-id", "{tile}", *common,
                    "--output-dir", str(a.aef_out), "--overwrite"],
            "watch": [str(a.aef_out)],
        },
        "tessera_fetch": {
            "cmd": [py, "-u", "data_processing/tessera_processing.py",
                    "--healpix-id", "{tile}", *common,
                    "--output-dir", str(a.tessera_out), "--overwrite"],
            "watch": [str(a.tessera_out)] + ([a.tessera_cache] if a.tessera_cache else []),
        },
        # OlmoEarth embeddings arrive as COGs from the Ai2 API (olmoearth_download.py);
        # this stage measures only the COG -> (X, y) conversion. Time the download
        # separately with --olmo-download-seconds / --olmo-download-bytes, since it is
        # a one-shot bulk transfer rather than a per-tile operation.
        "olmo_preprocess": {
            "cmd": [py, "-u", "preprocess_olmo_to_gedi_tiles.py",
                    "--shapefile", a.shapefile, "--gedi-dir", a.gedi_dir,
                    "--embeddings-dir", a.olmo_emb_dir, "--output-dir", str(a.olmo_out),
                    "--tiles", "{tile}", "--overwrite"],
            "watch": [str(a.olmo_out)],
        },
    }


_GNU_TIME = None


def gnu_time_cmd():
    """Path to a GNU `time` supporting -v, or None.

    BSD time (macOS) is at the same path but rejects -v, which would make every stage
    exit non-zero. Probe once instead of assuming.
    """
    global _GNU_TIME
    if _GNU_TIME is None:
        _GNU_TIME = False
        for cand in ("/usr/bin/time", shutil.which("gtime")):
            if cand and os.path.exists(cand):
                try:
                    p = subprocess.run([cand, "-v", "true"], capture_output=True,
                                       text=True, timeout=10)
                    if p.returncode == 0 and "Maximum resident set size" in p.stderr:
                        _GNU_TIME = cand
                        break
                except (OSError, subprocess.SubprocessError):
                    pass
    return _GNU_TIME or None


def run_stage(name, spec, tile, timeout):
    """Run one stage for one tile, returning a metrics dict."""
    cmd = [c.replace("{tile}", str(tile)) for c in spec["cmd"]]
    watch = [Path(w.replace("{tile}", str(tile))) for w in spec["watch"]]

    before = {str(w): dir_size(w) if w.is_dir() else (w.stat().st_size if w.exists() else 0)
              for w in watch}

    # /usr/bin/time -v gives a per-invocation peak RSS. getrusage(RUSAGE_CHILDREN) is
    # monotonic across the whole parent lifetime, so it cannot attribute peak memory to
    # one stage; use it only as a fallback (flagged in the output).
    gnu_time = gnu_time_cmd()
    wrapped = [gnu_time, "-v"] + cmd if gnu_time else cmd

    t0 = time.perf_counter()
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        proc = subprocess.run(wrapped, capture_output=True, text=True, timeout=timeout)
        rc, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        rc, stderr = -1, f"TIMEOUT after {timeout}s"
    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)

    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)

    peak_rss_bytes, rss_source = None, "none"
    if gnu_time:
        for line in (stderr or "").splitlines():
            if "Maximum resident set size" in line:
                try:  # GNU time reports kbytes
                    peak_rss_bytes = int(line.split(":")[-1].strip()) * 1024
                    rss_source = "gnu_time"
                except ValueError:
                    pass
    if peak_rss_bytes is None:
        # Linux ru_maxrss is kB, macOS is bytes.
        mult = 1 if sys.platform == "darwin" else 1024
        peak_rss_bytes = ru1.ru_maxrss * mult
        rss_source = "rusage_children_highwater"

    after = {str(w): dir_size(w) if w.is_dir() else (w.stat().st_size if w.exists() else 0)
             for w in watch}
    disk_delta = sum(after[k] - before[k] for k in before)

    return {
        "stage": name, "tile": tile, "serves": SERVES[name],
        "returncode": rc, "ok": rc == 0,
        "wall_s": round(wall, 2), "cpu_s": round(cpu_s, 2),
        "peak_rss_bytes": peak_rss_bytes, "peak_rss_source": rss_source,
        "disk_delta_bytes": disk_delta,
        "cmd": " ".join(cmd),
        "stderr_tail": (stderr or "")[-2000:] if rc != 0 else "",
    }


def summarize(rows, n_total, split_failures=True):
    """Print the per-stage table. Only successful runs feed the statistics."""
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)

    print(f"\n{'stage':<16} {'serves':<16} {'n':>2} {'wall s/tile':>16} "
          f"{'CPU s/tile':>16} {'peak RSS':>10} {'disk/tile':>10}")
    print("-" * 92)

    for stage in STAGE_ORDER:
        rs = by_stage.get(stage)
        if not rs:
            continue
        ok = [r for r in rs if r["ok"]]
        if not ok:
            print(f"{stage:<16} {','.join(SERVES[stage]):<16} "
                  f"{len(rs):>2}  ALL {len(rs)} RUNS FAILED")
            continue

        def ms(key):
            v = [r[key] for r in ok]
            return (v[0], 0.0) if len(v) == 1 else (statistics.mean(v), statistics.stdev(v))

        w_m, w_s = ms("wall_s")
        c_m, c_s = ms("cpu_s")
        d_m, _ = ms("disk_delta_bytes")
        rss = max(r["peak_rss_bytes"] for r in ok)

        print(f"{stage:<16} {','.join(SERVES[stage]):<16} {len(ok):>2} "
              f"{w_m:>8.1f} +/-{w_s:>5.1f} {c_m:>8.1f} +/-{c_s:>5.1f} "
              f"{human(rss):>10} {human(d_m):>10}")
        if split_failures and len(ok) < len(rs):
            print(f"{'':<16} {'':<16}    ({len(rs) - len(ok)} failed run(s) excluded)")

    print("-" * 92)
    print(f"\nExtrapolated to {n_total} tiles:\n")
    print(f"{'stage':<16} {'CPU-hours':>12} {'wall-hours':>12} {'disk':>12}")
    print("-" * 56)
    for stage in STAGE_ORDER:
        ok = [r for r in by_stage.get(stage, []) if r["ok"]]
        if not ok:
            continue
        cpu_h = statistics.mean([r["cpu_s"] for r in ok]) * n_total / 3600
        wall_h = statistics.mean([r["wall_s"] for r in ok]) * n_total / 3600
        disk = statistics.mean([r["disk_delta_bytes"] for r in ok]) * n_total
        print(f"{stage:<16} {cpu_h:>12.1f} {wall_h:>12.1f} {human(disk):>12}")

    print("\nNOTE: `materialize` is shared by presto+prithvi. Report it as its own row;\n"
          "      adding it to both encoder columns double-counts the dominant cost.")
    if any(r.get("peak_rss_source") == "rusage_children_highwater" for r in rows):
        print("NOTE: peak RSS came from getrusage high-water mark (no /usr/bin/time)\n"
              "      — it is an upper bound across stages, not per-stage.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=None,
                    help="summarize an existing JSONL and exit (no runs)")
    ap.add_argument("--tiles", nargs="+", type=int, help="tiles to benchmark (3-5 is plenty)")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--shapefile")
    ap.add_argument("--gedi-dir")
    ap.add_argument("--base", help="prithvi_presto root (holds <year>/rslearn and <year>/out)")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--prithvi-config", default="data_processing/prithvi_predict.yaml")
    ap.add_argument("--aef-out", default="./bench_cache/aef")
    ap.add_argument("--tessera-out", default="./bench_cache/tessera")
    ap.add_argument("--tessera-cache", default=None,
                    help="GeoTessera download cache, so fetched bytes are counted")
    ap.add_argument("--olmo-emb-dir", default=None, help="dir of downloaded OlmoEarth COGs")
    ap.add_argument("--olmo-out", default="./bench_cache/olmo")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="+", default=None, help=f"subset of {STAGE_ORDER}")
    ap.add_argument("--skip", nargs="+", default=None)
    ap.add_argument("--timeout", type=int, default=14400, help="per-stage seconds")
    ap.add_argument("--n-tiles-total", type=int, default=311,
                    help="tile count to extrapolate to (default: 311)")
    ap.add_argument("--out", default="bench_results.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(l) for l in Path(args.report).read_text().splitlines() if l.strip()]
        summarize(rows, args.n_tiles_total)
        return

    for req in ("tiles", "shapefile", "gedi_dir", "base"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or use --report)")

    stages = build_stages(args)
    names = [s for s in STAGE_ORDER
             if (not args.only or s in args.only) and (not args.skip or s not in args.skip)]
    if args.olmo_emb_dir is None and "olmo_preprocess" in names:
        print("Skipping olmo_preprocess: --olmo-emb-dir not set")
        names.remove("olmo_preprocess")

    if args.dry_run:
        for tile in args.tiles:
            for n in names:
                cmd = " ".join(c.replace("{tile}", str(tile)) for c in stages[n]["cmd"])
                print(f"\n# [{n}] tile {tile}\n{cmd}")
        return

    out_path = Path(args.out)
    rows = []
    if out_path.exists():
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        print(f"Resuming: {len(rows)} result(s) already in {out_path}")
    done = {(r["stage"], r["tile"]) for r in rows if r["ok"]}

    for tile in args.tiles:
        for n in names:
            if (n, tile) in done:
                print(f"SKIP [{n}] tile {tile} — already recorded")
                continue
            print(f"\n=== [{n}] tile {tile} ===", flush=True)
            r = run_stage(n, stages[n], tile, args.timeout)
            rows.append(r)
            with out_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            status = "ok" if r["ok"] else f"FAILED rc={r['returncode']}"
            print(f"    {status}  wall={r['wall_s']:.1f}s  cpu={r['cpu_s']:.1f}s  "
                  f"disk={human(r['disk_delta_bytes'])}", flush=True)
            if not r["ok"]:
                print(f"    stderr tail:\n{r['stderr_tail'][-600:]}")

    summarize(rows, args.n_tiles_total)
    print(f"\nRaw results: {out_path}")


if __name__ == "__main__":
    main()
