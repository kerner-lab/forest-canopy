#!/usr/bin/env python3
"""
Measure the energy / carbon cost of embedding ONE tile, then extrapolate to the full run.

Three subcommands:

  measure      wrap a real single-tile command, record what it actually cost, write JSON
  extrapolate  scale one (or more) measurements up to N tiles -> per-model totals
  report       render the measurements/totals as a markdown, CSV or LaTeX table

The energy model is Green Algorithms (Lannelongue, Grealey & Inouye 2021, Adv. Sci. 7:2100707):

    E_kWh = t_h * (n_cores * P_core * u_core + n_GB * 0.3725 W/GB + n_gpu * P_gpu * u_gpu) * PUE / 1000
    CO2e  = E_kWh * carbon_intensity

with two improvements over the web calculator, because we are wrapping the real process:
  * u_core is MEASURED (getrusage CPU-seconds / (elapsed * allocated cores)), not assumed 1.0.
  * GPU energy is MEASURED by integrating `nvidia-smi --query-gpu=power.draw`, not TDP * time.

Everything else (PUE, grid carbon intensity, per-core TDP) is a constant you must verify --
see VERIFY_BEFORE_PUBLISHING below. The script prints a loud banner listing every unverified
constant it used, so an un-checked number cannot quietly reach a manuscript.

--------------------------------------------------------------------------------
TYPICAL SESSION
--------------------------------------------------------------------------------
Run each of these as a normal single-tile batch job (or interactively in an salloc), with
carbon_accounting.py wrapping the python call. Pick a mid-sized tile, not your smallest.

  # Presto, one tile, Sol htc  (CPU)
  python carbon_accounting.py measure \
      --model presto --cluster sol-htc --tile 23833 \
      --out-dir /scratch/kdoerkse/presto_prithvi/carbon \
      --pixels-from /scratch/kdoerkse/presto_prithvi/2019/out/healpix_23833_presto.npz \
      -- python -u data_processing/unified_embedding_pipeline.py \
             --healpix-id 23833 --year 2019 ... --overwrite

  # Prithvi, one tile, Bridges-2 GPU  (add --gpus 1 to turn on NVML power sampling)
  python carbon_accounting.py measure \
      --model prithvi --cluster bridges2-gpu --tile 23833 --gpus 1 \
      --out-dir .../carbon \
      --pixels-from .../out/prithvi_xy/healpix_23833_float32.npz \
      -- python -u data_processing/unified_embedding_pipeline.py --do-prithvi ...

  # Phase 1 is shared by Presto AND Prithvi -- measure it separately so you can attribute
  # it once instead of double counting it in both rows.
  python carbon_accounting.py measure --model materialize --cluster sol-htc --tile 23833 ...

Then:

  python carbon_accounting.py extrapolate --measurements .../carbon/*.json --n-tiles 311
  python carbon_accounting.py report      --measurements .../carbon/*.json --n-tiles 311 --format latex

--------------------------------------------------------------------------------
WHAT THIS DOES NOT DO
--------------------------------------------------------------------------------
* No FLOPs. Wall-clock energy and model FLOPs answer different questions and this script
  only does the former; count FLOPs separately with torch.utils.flop_counter.FlopCounterMode
  and put both in the table.
* No amortisation of AlphaEarth/Tessera pretraining -- that cost is the providers', not
  yours. Report your ingest (download + storage) for those rows instead.
* One tile is one sample. Tile cost here spans ~5.6 GB to ~24 GB of imagery, so a single
  measurement carries real variance; --n-tiles scaling assumes the measured tile is typical.
  Use --scale-by pixels (with --total-pixels) if you want to scale on tile size instead,
  and measure 2-3 tiles if you can afford it.
"""

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants you must check before these numbers go in a paper.
#
# Every entry marked VERIFY is a plausible default, NOT a measured fact about your site.
# PUE: ask PSC / ASU Research Computing -- they know theirs. 1.67 is the Green Algorithms
#      fallback for "unknown", and it is deliberately pessimistic; a modern academic
#      facility is usually 1.2-1.5, so leaving 1.67 in will overstate your footprint.
# carbon_intensity: kg CO2e per kWh for the grid the machine sits on. Look up the EPA
#      eGRID subregion (RFCE for Pittsburgh, AZNM for Tempe) or use an annual average from
#      app.electricitymaps.com. These move year to year -- cite the vintage you used.
# tdp_per_core_w: chip TDP / cores on the chip. Confirm the CPU with `lscpu` on a compute
#      node; Sol in particular has several node generations behind one partition name.
# ---------------------------------------------------------------------------
VERIFY_BEFORE_PUBLISHING = ("pue", "carbon_intensity_kg_per_kwh", "tdp_per_core_w",
                            "gpu_tdp_w")

CLUSTERS = {
    "bridges2-rm": {
        "site": "PSC Bridges-2, RM / RM-shared",
        "cpu_model": "AMD EPYC 7742 (64c, 225 W) x2 = 128 c/node",   # VERIFY with lscpu
        "tdp_per_core_w": 225.0 / 64,          # 3.52 VERIFY
        "gpu_model": None,
        "gpu_tdp_w": 0.0,
        "pue": 1.67,                            # VERIFY -- ask PSC
        "carbon_intensity_kg_per_kwh": 0.324,   # VERIFY -- eGRID RFCE, Pittsburgh PA
        "grid_region": "eGRID RFCE (Pittsburgh, PA)",
    },
    "bridges2-gpu": {
        "site": "PSC Bridges-2, GPU / GPU-shared (V100-32GB)",
        "cpu_model": "Intel Xeon Gold 6248 (20c, 150 W) x2 = 40 c/node",  # VERIFY
        "tdp_per_core_w": 150.0 / 20,          # 7.50 VERIFY
        "gpu_model": "NVIDIA V100-SXM2-32GB",
        "gpu_tdp_w": 300.0,                     # VERIFY -- SXM2 is 300 W, PCIe is 250 W
        "pue": 1.67,                            # VERIFY
        "carbon_intensity_kg_per_kwh": 0.324,   # VERIFY -- eGRID RFCE
        "grid_region": "eGRID RFCE (Pittsburgh, PA)",
    },
    "sol-htc": {
        "site": "ASU Sol, htc partition",
        "cpu_model": "AMD EPYC (225 W / 64c assumed)",   # VERIFY with lscpu -- Sol is mixed
        "tdp_per_core_w": 225.0 / 64,          # 3.52 VERIFY
        "gpu_model": None,
        "gpu_tdp_w": 0.0,
        "pue": 1.67,                            # VERIFY -- ask ASU RC
        "carbon_intensity_kg_per_kwh": 0.400,   # VERIFY -- eGRID AZNM, Tempe AZ
        "grid_region": "eGRID AZNM (Tempe, AZ)",
    },
    "sol-gpu": {
        "site": "ASU Sol, GPU partition",
        "cpu_model": "AMD EPYC (225 W / 64c assumed)",   # VERIFY
        "tdp_per_core_w": 225.0 / 64,
        "gpu_model": "NVIDIA A100",             # VERIFY which GPU you were given
        "gpu_tdp_w": 400.0,                     # VERIFY -- A100 SXM 400 W, PCIe 250 W
        "pue": 1.67,                            # VERIFY
        "carbon_intensity_kg_per_kwh": 0.400,   # VERIFY -- eGRID AZNM
        "grid_region": "eGRID AZNM (Tempe, AZ)",
    },
}

MEM_POWER_W_PER_GB = 0.3725      # Green Algorithms; this one is from the paper, not a guess
GPU_SAMPLE_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def maxrss_to_gb(maxrss: int) -> float:
    """getrusage ru_maxrss is KiB on Linux and bytes on macOS. Normalise to GB."""
    return maxrss / (1024 ** 2) if platform.system() == "Linux" else maxrss / (1024 ** 3)


def npz_member_shape(path: Path, key: str = "X"):
    """
    Shape of one array inside an .npz WITHOUT decompressing it.

    The dense Presto files are GB-scale; np.load(...)["X"].shape would inflate the whole
    array just to read a tuple. Reading the .npy header off the zip stream touches ~128 B.
    """
    with zipfile.ZipFile(path) as zf:
        names = {n.rsplit(".npy", 1)[0]: n for n in zf.namelist()}
        if key not in names:
            raise KeyError(f"{path.name} has no array {key!r} (has: {sorted(names)})")
        with zf.open(names[key]) as fh:
            major, _minor = np.lib.format.read_magic(fh)
            # numpy 2.x dropped the private _read_array_header; the versioned readers are
            # public API in both 1.x and 2.x.
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, _fortran, _dtype = reader(fh)
    return shape


def pixels_from_file(path: Path) -> dict:
    """
    Tokens actually pushed through the encoder, inferred from the output file.

    dense  healpix_<id>_<model>.npz   X=(H, W, C) -> H*W  (every pixel was embedded)
    xy     healpix_<id>_float32.npz   X=(N, C)    -> N    (GEDI-valid pixels only)

    Prefer the DENSE file when both exist: the xy file is a post-hoc subset, so using it
    would undercount the work by ~100x on a sparse GEDI tile.
    """
    shape = npz_member_shape(path, "X")
    if len(shape) == 3:
        return {"pixels": int(shape[0] * shape[1]), "kind": "dense",
                "grid": [int(shape[0]), int(shape[1])], "dim": int(shape[2])}
    if len(shape) == 2:
        return {"pixels": int(shape[0]), "kind": "xy", "grid": None, "dim": int(shape[1])}
    raise ValueError(f"unexpected X shape {shape} in {path}")


class GPUPowerSampler(threading.Thread):
    """
    Integrate real GPU power over the run instead of assuming TDP.

    nvidia-smi power.draw is unprivileged (NVML), so this works on a normal batch job --
    unlike RAPL-based CPU sampling, which needs root on most clusters. Silently degrades
    to the TDP model if nvidia-smi is absent or reports [N/A].
    """

    def __init__(self, interval=GPU_SAMPLE_INTERVAL_S):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []          # (t_monotonic, total_watts_across_gpus)
        self._stop = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None

    def run(self):
        if not self.available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                watts = [float(v) for v in out.stdout.split("\n")
                         if v.strip() and "N/A" not in v]
                if watts:
                    self.samples.append((time.monotonic(), sum(watts)))
            except (subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def energy_kwh(self):
        """Trapezoidal integral of the power samples. None if we never got two samples."""
        if len(self.samples) < 2:
            return None, None
        t = np.array([s[0] for s in self.samples])
        w = np.array([s[1] for s in self.samples])
        joules = np.trapezoid(w, t) if hasattr(np, "trapezoid") else np.trapz(w, t)
        return joules / 3.6e6, float(w.mean())


def slurm_alloc_defaults():
    """Allocation as Slurm sees it, so a measurement inside a batch job needs no flags."""
    cores = os.environ.get("SLURM_CPUS_PER_TASK")
    mem = os.environ.get("SLURM_MEM_PER_NODE")          # MB
    mem_cpu = os.environ.get("SLURM_MEM_PER_CPU")       # MB
    gpus = os.environ.get("SLURM_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE")
    mem_gb = None
    if mem:
        mem_gb = float(mem) / 1024
    elif mem_cpu and cores:
        mem_gb = float(mem_cpu) * int(cores) / 1024
    return {
        "cores": int(cores) if cores else None,
        "mem_gb": mem_gb,
        "gpus": int(gpus) if gpus and gpus.isdigit() else None,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
    }


def compute_energy(elapsed_s, cores, mem_gb, core_usage, gpus, gpu_energy_kwh, const):
    """Green Algorithms, split so the table can show where the joules went."""
    hours = elapsed_s / 3600.0
    cpu_kwh = hours * cores * const["tdp_per_core_w"] * core_usage / 1000.0
    mem_kwh = hours * mem_gb * MEM_POWER_W_PER_GB / 1000.0
    if gpu_energy_kwh is not None:
        gpu_kwh, gpu_src = gpu_energy_kwh, "measured (nvidia-smi)"
    elif gpus:
        gpu_kwh, gpu_src = hours * gpus * const["gpu_tdp_w"] / 1000.0, "modelled (TDP x time)"
    else:
        gpu_kwh, gpu_src = 0.0, "n/a"

    # PUE covers cooling/distribution for the whole facility, so it multiplies everything.
    total_kwh = (cpu_kwh + mem_kwh + gpu_kwh) * const["pue"]
    return {
        "cpu_kwh": cpu_kwh * const["pue"],
        "mem_kwh": mem_kwh * const["pue"],
        "gpu_kwh": gpu_kwh * const["pue"],
        "total_kwh": total_kwh,
        "gpu_energy_source": gpu_src,
        "co2e_kg": total_kwh * const["carbon_intensity_kg_per_kwh"],
    }


def warn_unverified(const, cluster):
    vals = ", ".join(f"{k}={const[k]:g}" for k in VERIFY_BEFORE_PUBLISHING if k in const)
    print("\n" + "!" * 78, file=sys.stderr)
    print(f"! UNVERIFIED CONSTANTS for {cluster}: {vals}", file=sys.stderr)
    print(f"! grid: {const['grid_region']}   cpu: {const['cpu_model']}", file=sys.stderr)
    print("! Confirm PUE with the facility, carbon intensity against eGRID/ElectricityMaps,",
          file=sys.stderr)
    print("! and the CPU with `lscpu`, before any of this goes in a manuscript.", file=sys.stderr)
    print("!" * 78 + "\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def cmd_measure(args):
    if not args.command:
        sys.exit("ERROR: no command given. Put the real command after `--`.")

    const = dict(CLUSTERS[args.cluster])
    for key in ("pue", "carbon_intensity_kg_per_kwh", "tdp_per_core_w", "gpu_tdp_w"):
        override = getattr(args, key, None)
        if override is not None:
            const[key] = override
            const.setdefault("_overridden", []).append(key)

    alloc = slurm_alloc_defaults()
    cores = args.cores or alloc["cores"] or os.cpu_count()
    mem_gb = args.mem_gb or alloc["mem_gb"] or 4.0
    gpus = args.gpus if args.gpus is not None else (alloc["gpus"] or 0)

    warn_unverified(const, args.cluster)
    print(f"[carbon] model={args.model} tile={args.tile} cluster={args.cluster}")
    print(f"[carbon] alloc: {cores} cores, {mem_gb:.1f} GB, {gpus} GPU(s)"
          f"{'  job=' + alloc['job_id'] if alloc['job_id'] else ''}")
    print(f"[carbon] running: {' '.join(args.command)}\n", flush=True)

    sampler = GPUPowerSampler(args.gpu_sample_interval)
    if gpus:
        if not sampler.available:
            print("[carbon] WARNING: nvidia-smi not found -- falling back to TDP x time.",
                  file=sys.stderr)
        else:
            sampler.start()

    # RUSAGE_CHILDREN accumulates over ALL reaped children, so snapshot before/after and
    # difference. In a fresh process the "before" is ~0, but subtracting makes this safe
    # to call more than once in one interpreter.
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    proc = subprocess.run(args.command)
    elapsed = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    sampler.stop()

    cpu_seconds = ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime))
    peak_rss_gb = maxrss_to_gb(after.ru_maxrss)
    # Fraction of the cores we RESERVED that were actually busy. Capped at 1.0: oversubscribed
    # BLAS can report >100%, and charging for cores you never held would overstate the cost.
    core_usage = min(cpu_seconds / (elapsed * cores), 1.0) if elapsed > 0 else 0.0

    gpu_kwh, gpu_mean_w = sampler.energy_kwh() if gpus else (None, None)
    energy = compute_energy(elapsed, cores, mem_gb, core_usage, gpus, gpu_kwh, const)

    pixels = None
    if args.pixels_from:
        p = Path(args.pixels_from)
        if p.exists():
            try:
                pixels = pixels_from_file(p)
            except Exception as exc:                      # noqa: BLE001 - never fail the run
                print(f"[carbon] could not read pixel count from {p}: {exc}", file=sys.stderr)
        else:
            print(f"[carbon] --pixels-from does not exist yet: {p}", file=sys.stderr)

    record = {
        "model": args.model,
        "tile": args.tile,
        "cluster": args.cluster,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exit_status": proc.returncode,
        "command": args.command,
        "alloc": {"cores": cores, "mem_gb": mem_gb, "gpus": gpus,
                  "job_id": alloc["job_id"], "node": alloc["node"]},
        "measured": {
            "elapsed_s": elapsed,
            "elapsed_h": elapsed / 3600.0,
            "cpu_seconds": cpu_seconds,
            "core_usage_fraction": core_usage,
            "peak_rss_gb": peak_rss_gb,
            "gpu_mean_w": gpu_mean_w,
            "gpu_samples": len(sampler.samples),
        },
        "constants": const,
        "energy": energy,
        "pixels": pixels,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_tile{args.tile}_{args.cluster}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"\n[carbon] ---- {args.model} / tile {args.tile} ----")
    print(f"[carbon] exit           : {proc.returncode}")
    print(f"[carbon] wall clock     : {elapsed / 60:.1f} min  ({elapsed / 3600:.3f} h)")
    print(f"[carbon] core-hours     : {cores * elapsed / 3600:.2f}  "
          f"(cores busy {core_usage * 100:.0f}% of reserved)")
    print(f"[carbon] peak RSS       : {peak_rss_gb:.1f} GB of {mem_gb:.0f} GB requested")
    if gpus:
        src = energy["gpu_energy_source"]
        mean = f"{gpu_mean_w:.0f} W mean" if gpu_mean_w else "no samples"
        print(f"[carbon] GPU            : {mean}, {src}")
    print(f"[carbon] energy         : {energy['total_kwh']:.4f} kWh  "
          f"(cpu {energy['cpu_kwh']:.4f} / mem {energy['mem_kwh']:.4f} / gpu {energy['gpu_kwh']:.4f})")
    print(f"[carbon] CO2e           : {energy['co2e_kg'] * 1000:.1f} g")
    if pixels:
        print(f"[carbon] pixels embedded: {pixels['pixels']:,} ({pixels['kind']})")
    print(f"[carbon] wrote          : {out_path}")

    if proc.returncode != 0:
        print("[carbon] NOTE: the wrapped command FAILED. This measurement reflects a failed "
              "run and should not be used for extrapolation.", file=sys.stderr)
    return proc.returncode


# ---------------------------------------------------------------------------
# extrapolate / report
# ---------------------------------------------------------------------------

def load_measurements(paths):
    out = []
    for p in paths:
        rec = json.loads(Path(p).read_text())
        if rec.get("exit_status") not in (0, None):
            print(f"[carbon] skipping failed measurement: {p} (exit {rec['exit_status']})",
                  file=sys.stderr)
            continue
        out.append(rec)
    if not out:
        sys.exit("ERROR: no usable measurements (all missing or failed).")
    return out


def scale(rec, n_tiles, total_pixels=None):
    """
    Scale one tile to the whole run.

    Default is linear in tile count. --total-pixels switches to scaling by embedded pixels,
    which is better when tile sizes vary a lot -- yours span ~5.6-24 GB of imagery, so the
    two can differ substantially. Both assume cost is linear in work, which holds for
    per-pixel encoders like Presto and roughly holds for the patch-wise ViTs.
    """
    if total_pixels and rec.get("pixels"):
        factor = total_pixels / rec["pixels"]["pixels"]
        basis = f"pixels ({total_pixels:,} / {rec['pixels']['pixels']:,})"
    else:
        factor = float(n_tiles)
        basis = f"tiles (x{n_tiles})"
    e = rec["energy"]
    m = rec["measured"]
    return {
        "model": rec["model"],
        "cluster": rec["cluster"],
        "basis": basis,
        "factor": factor,
        "node_hours": m["elapsed_h"] * factor,
        "core_hours": m["elapsed_h"] * rec["alloc"]["cores"] * factor,
        "gpu_hours": m["elapsed_h"] * (rec["alloc"]["gpus"] or 0) * factor,
        "kwh": e["total_kwh"] * factor,
        "co2e_kg": e["co2e_kg"] * factor,
        "per_tile_min": m["elapsed_s"] / 60.0,
        "peak_rss_gb": m["peak_rss_gb"],
    }


def check_pixel_bases(recs):
    """
    Pixel-count scaling is only meaningful within one counting convention.

    Presto/OLMo report DENSE pixels (H*W, every pixel embedded); Prithvi has no dense
    output so its count is the GEDI-valid subset (N). Dividing a dense total by an xy
    measurement -- or vice versa -- is off by ~100x on a sparse tile.
    """
    kinds = {r["pixels"]["kind"] for r in recs if r.get("pixels")}
    if len(kinds) > 1:
        print("[carbon] WARNING: measurements mix dense and xy pixel counts "
              f"({sorted(kinds)}). --total-pixels scaling across them is not comparable; "
              "scale by tile count instead, or supply a per-model --total-pixels.",
              file=sys.stderr)


def cmd_extrapolate(args):
    recs = load_measurements(args.measurements)
    if args.total_pixels:
        check_pixel_bases(recs)
    rows = [scale(r, args.n_tiles, args.total_pixels) for r in recs]

    print(f"\nExtrapolated to {args.n_tiles} tiles"
          + (f" / {args.total_pixels:,} pixels" if args.total_pixels else ""))
    print("-" * 96)
    print(f"{'model':<14}{'cluster':<15}{'min/tile':>9}{'core-h':>10}{'gpu-h':>8}"
          f"{'kWh':>10}{'kg CO2e':>10}{'peak GB':>9}")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: -x["co2e_kg"]):
        print(f"{r['model']:<14}{r['cluster']:<15}{r['per_tile_min']:>9.1f}"
              f"{r['core_hours']:>10.1f}{r['gpu_hours']:>8.1f}"
              f"{r['kwh']:>10.2f}{r['co2e_kg']:>10.2f}{r['peak_rss_gb']:>9.1f}")
    print("-" * 96)
    print(f"{'TOTAL':<14}{'':<15}{'':>9}{sum(r['core_hours'] for r in rows):>10.1f}"
          f"{sum(r['gpu_hours'] for r in rows):>8.1f}"
          f"{sum(r['kwh'] for r in rows):>10.2f}{sum(r['co2e_kg'] for r in rows):>10.2f}")
    print("\nScaling basis: " + "; ".join(sorted({r["basis"] for r in rows})))
    print("Reminder: 'materialize' is shared by Presto and Prithvi -- attribute it once, or "
          "split it, but do not add it into both model rows.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_tiles": args.n_tiles, "total_pixels": args.total_pixels, "rows": rows},
            indent=2) + "\n")
        print(f"\nWrote {args.out}")


def cmd_report(args):
    recs = load_measurements(args.measurements)
    rows = sorted((scale(r, args.n_tiles, args.total_pixels) for r in recs),
                  key=lambda x: -x["co2e_kg"])
    hdr = ["Encoder", "Platform", "min/tile", "Core-h", "GPU-h", "Energy (kWh)", "kg CO2e"]

    def cells(r):
        return [r["model"], r["cluster"], f"{r['per_tile_min']:.1f}",
                f"{r['core_hours']:.1f}", f"{r['gpu_hours']:.1f}",
                f"{r['kwh']:.2f}", f"{r['co2e_kg']:.2f}"]

    if args.format == "csv":
        print(",".join(hdr))
        for r in rows:
            print(",".join(cells(r)))
    elif args.format == "latex":
        print(r"\begin{tabular}{llrrrrr}")
        print(r"\hline")
        print(" & ".join(hdr) + r" \\")
        print(r"\hline")
        for r in rows:
            print(" & ".join(cells(r)) + r" \\")
        print(r"\hline")
        print(r"\end{tabular}")
        # One line per cluster: the two sites have different PUE/grid, so a single
        # footnote taken from the first measurement would misattribute the other one.
        for cluster in sorted({r["cluster"] for r in recs}):
            c = next(r["constants"] for r in recs if r["cluster"] == cluster)
            print(f"% {cluster}: PUE={c['pue']}, CI={c['carbon_intensity_kg_per_kwh']} "
                  f"kgCO2e/kWh ({c['grid_region']})")
        print(f"% Green Algorithms (Lannelongue et al. 2021). Extrapolated from "
              f"single-tile measurements x{args.n_tiles} tiles.")
    else:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for r in rows:
            print("| " + " | ".join(cells(r)) + " |")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="wrap a single-tile command and record its cost")
    m.add_argument("--model", required=True,
                   help="presto | prithvi | olmo | materialize (free text, used in the table)")
    m.add_argument("--tile", required=True, help="healpix id, for the filename")
    m.add_argument("--cluster", required=True, choices=sorted(CLUSTERS))
    m.add_argument("--out-dir", default="carbon", help="where the JSON goes")
    m.add_argument("--cores", type=int, default=None, help="default: $SLURM_CPUS_PER_TASK")
    m.add_argument("--mem-gb", type=float, default=None, help="default: $SLURM_MEM_PER_NODE")
    m.add_argument("--gpus", type=int, default=None, help="default: $SLURM_GPUS; 0 disables sampling")
    m.add_argument("--pixels-from", default=None,
                   help="output npz to read the embedded-pixel count from (prefer the DENSE file)")
    m.add_argument("--gpu-sample-interval", type=float, default=GPU_SAMPLE_INTERVAL_S)
    m.add_argument("--pue", type=float, default=None, help="override the cluster PUE")
    m.add_argument("--carbon-intensity-kg-per-kwh", type=float, default=None, dest="carbon_intensity_kg_per_kwh")
    m.add_argument("--tdp-per-core-w", type=float, default=None, dest="tdp_per_core_w")
    m.add_argument("--gpu-tdp-w", type=float, default=None, dest="gpu_tdp_w")
    m.add_argument("command", nargs=argparse.REMAINDER,
                   help="the real command, after `--`")
    m.set_defaults(func=cmd_measure)

    e = sub.add_parser("extrapolate", help="scale measurements to the full run")
    e.add_argument("--measurements", nargs="+", required=True)
    e.add_argument("--n-tiles", type=int, required=True)
    e.add_argument("--total-pixels", type=int, default=None,
                   help="scale by pixels instead of tile count")
    e.add_argument("--out", default=None, help="write the scaled rows as JSON")
    e.set_defaults(func=cmd_extrapolate)

    r = sub.add_parser("report", help="render a manuscript table")
    r.add_argument("--measurements", nargs="+", required=True)
    r.add_argument("--n-tiles", type=int, required=True)
    r.add_argument("--total-pixels", type=int, default=None)
    r.add_argument("--format", choices=["markdown", "csv", "latex"], default="markdown")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    if args.cmd == "measure" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
