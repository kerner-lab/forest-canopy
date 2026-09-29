# Canopy Height from Geospatial Foundation Model Embeddings

Benchmarking geospatial foundation model (GFM) embeddings against GEDI canopy height,
on a fixed HEALPix tile grid, with published canopy-height maps as baselines.

The question this repo answers: **given the same imagery, the same tiles and the same
GEDI pixels, which embedding is the best feature set for predicting canopy height?**
Every design decision below exists to keep that comparison fair — the encoders differ,
nothing else does.

Embeddings compared:

| Source | Dim | How it is obtained | Script |
|---|---|---|---|
| AlphaEarth (AEF) | 64 | precomputed archive, dequantized int8 → float32 | `data_processing/alphaearth_processing.py` |
| TESSERA | 128 | precomputed archive (GeoTessera blocks) | `data_processing/tessera_processing.py` |
| Major TOM | 2048 | precomputed archive (Core-S2L1C-SSL4EO) | `data_processing/majortom_processing.py` |
| OlmoEarth | 128 | model run via rslearn | `olmoearth_download.py`, `data_processing/unified_embedding_pipeline.py` |
| Presto | 128 | model run locally on monthly S2 (+S1) time series | `data_processing/presto_processing.py` |
| Prithvi-EO-2.0 | 1024 | model run via rslearn | `data_processing/unified_embedding_pipeline.py` |
| Clay v1.5 | 1024 | model run locally, reuses the Presto imagery | `data_processing/clay_processing.py` |

Baselines compared:

| Baseline | Source | Script |
|---|---|---|
| Potapov et al. 2019 | Earth Engine (`GEDI_V27`) | `data_processing/potapov_baseline.py` |
| ETH Global 2020 10 m | Earth Engine (single global Image) | `data_processing/eth_canopy_baseline.py` |
| ECHOSAT (Pauls et al.) | HTTP, windowed `/vsicurl` reads | `data_processing/echosat_baseline.py` |
| Meta/WRI CHM v2 | AWS Open Data (public, no creds) | `data_processing/meta_chm_baseline.py` |

---

## Conventions that hold everywhere

Read this before anything else — most of the pipeline's constraints follow from it.

**The tile grid.** All work is on a HEALPix tile shapefile with columns
`healpix_id`, `tile_idx`, `split`. 311 tiles: 217 `train`, 47 `val`, 47 `test`.
Tiles are EPSG:4326, ~0.00027° (~30 m), roughly 0.7° × 0.8° each.

**GEDI target rasters.** `healpix_<id>_gedi.tif`, one band, **centimetres**,
nodata `-9999`. A pixel is a valid label iff:

```python
gedi != -9999  and  gedi > 0  and  gedi < 12000     # 12000 cm = 120 m
```

That triple condition is hardcoded in five places and **must stay identical across
all of them** or the baselines are scored on a different pixel set than the models:

```
run_gedi_prediction_30m.py:283
data_processing/presto_processing.py          (GEDI_MAX_HEIGHT_CM)
data_processing/potapov_baseline.py
data_processing/eth_canopy_baseline.py
data_processing/echosat_baseline.py           (GEDI_MAX_HEIGHT_CM)
```

**Embedding cache format.** Every embedding script writes the same thing, so the RF
trainer does not care which encoder produced it:

```
healpix_<id>_float32.npz        X = (N, D) float32,  y = (N,) float32 heights in METRES
healpix_<id>_float32_withzeros.npz   same, but --keep-zeros kept gedi == 0
```

`N` is the number of valid GEDI pixels in that tile; rows are in `np.where()` order
over the GEDI raster. Some scripts also store `emb_rows`, `emb_cols`,
`grid_transform`, `grid_crs`, `grid_shape` so predictions can be re-rasterized later.

**Two levels of "aligned".** These are different and both matter:

- *Tile*-level — every source has a cache file for this tile (`build_master_tile_list.py`)
- *Pixel*-level — every source sampled the **same pixels** within that tile
  (`count_measurements.py`). Three sources apply different pixel-drop rules, so tile
  agreement does **not** imply pixel agreement.

**Which metric to quote.** `balanced_rmse` and `balanced_r2`, not the plain ones.
GEDI heights are heavily skewed toward short vegetation, so plain `r2` is dominated by
the low bins. See [Stage 5](#stage-5--metrics).

---

## Pipeline

```
 Stage 0   GEDI + HEALPix rasters  ──────────────►  healpix_<id>_gedi.tif
                                                     │
 Stage 1   run each encoder ───────────────────────► healpix_<id>_float32.npz  (per source)
                                                     │
 Stage 2   coverage: which tiles exist for this year? │
 Stage 3   alignment: master tile list + pixel check  │
                                                     ▼
 Stage 4   RandomForest ───────────────────────────► metrics.json, predictions_*.parquet
                                                     │
 Stage 5   stratified metrics, seed aggregation ─────► reportable numbers
```

---

## Stage 0 — Build the tile grid and GEDI targets

One-time setup. Everything downstream keys off these two artefacts.

| Script | Purpose |
|---|---|
| `data_processing/healpix_to_shapefile.py` | HEALPix cells → shapefile with `healpix_id` / `split` |
| `data_processing/tile_gedi_with_healpix.py` | split a large GEDI GeoTIFF into `healpix_<id>_gedi.tif` |
| `data_processing/process_GEDI.py` | upstream GEDI granule processing |

```bash
python data_processing/tile_gedi_with_healpix.py \
    --shapefile  /path/healpix_shapefile.shp \
    --gedi-tif   /path/gedi_<year>_mosaic.tif \
    --output-dir /path/gedi_tiles_<year>_healpix
```

---

## Stage 1 — Generate or query the embeddings

Two families, same output format.

### 1a. Precomputed archives (download and subset)

These read an existing embedding product and extract `(X, y)` at valid GEDI pixels.
They share a CLI: `--shapefile --gedi-dir --output-dir --year --splits --healpix-id
--check-only --overwrite --qa --keep-zeros`.

```bash
# AlphaEarth — 64-d, int8 archive dequantized via the dataset's signed-square formula
python data_processing/alphaearth_processing.py \
    --shapefile /path/healpix.shp --gedi-dir /path/gedi_2020_healpix \
    --output-dir /path/embeddings/2020/aef --year 2020 --splits train val test

# TESSERA — 128-d; --cache-dir is important, ~60% of blocks are shared between tiles
python data_processing/tessera_processing.py \
    --healpix-id 12345 --year 2020 \
    --shapefile ... --gedi-dir ... --output-dir ... --cache-dir /scratch/tessera_cache

# Major TOM — 2048-d; --list-years first to see what the archive holds
python data_processing/majortom_processing.py --list-years
```

### 1b. Models run over imagery (fetch S2, then encode)

These fetch Sentinel-2 themselves via [rslearn](https://github.com/allenai/rslearn)
and run an encoder. Slower, GPU for some, and network-bound.

```bash
# Presto — 12 monthly S2 composites per tile, optional S1, 128-d, CPU-friendly
python data_processing/presto_processing.py \
    --healpix-id 12345 --year 2020 \
    --shapefile ... --gedi-dir ... --output-dir ... \
    --rslearn-root /scratch/presto_rslearn --s2-source pc [--with-s1]

# OlmoEarth + Prithvi + Presto from ONE shared S2 download
python data_processing/unified_embedding_pipeline.py \
    --year 2020 --shapefile ... --gedi-dir ... \
    --dataset-path /scratch/unified/2020/dataset \
    --olmo-model-config /path/model-gpu.yaml \
    --output-dir /path/embeddings/2020 --emit-xy --do-prithvi

# Clay v1.5 — reuses the Presto rslearn plumbing so the two are apples-to-apples
python data_processing/clay_processing.py --healpix-id 12345 --year 2020 ...

# OlmoEarth predictions from the hosted API
python olmoearth_download.py --help
```

Useful flags on the model-based scripts:

- `--subset-fraction 0.1` — crop each tile to a centred 10% window for fast local tests
- `--subset-tile-fraction` / `--subset-tile-seed` — use only a fraction of a split's
  tiles, selecting *the same* tiles the RF trainer would
- `--check-only` — report cache coverage and exit
- `--save-tif` — also write a georeferenced embedding GeoTIFF for QGIS

**HPC wrappers.** The array jobs that drive these at scale, one tile per array index:

| Wrapper | Runs |
|---|---|
| `run_presto_embed_sol.sh` | Presto (ASU Sol) |
| `run_prithvi_embed_sol.sh`, `run_prithvi_embed_bridges.sh` | unified OLMo/Prithvi/Presto |
| `run_tessera_embed.sh` | TESSERA |
| `run_echosat_baseline.sh` | ECHOSAT baseline cache |
| `run_make_xy.sh` | rebuild `xy` npz from dense npz |

Array index → `healpix_id` is a 1-based line number in `tile_ids.txt`.

### 1c. Repair and rescue

Things that go wrong at scale, and the script for each:

| Script | Fixes |
|---|---|
| `audit_materialized.py` | Phase-1 S2 imagery missing or **truncated** (quota exhaustion). Run before spending GPU hours. |
| `make_xy_from_dense.py` | rebuild `xy` npz from an existing dense npz — seconds, no re-encode |
| `data_processing/fix_rslearn_invalid_geom.py` | invalid geometry that makes rslearn drop a whole layer |
| `find_reclaimable_tiles.py` | find materialized imagery that is safe to delete |
| `diagnose_embeddings.py` | inspect a cache file that looks wrong |

---

## Stage 2 — Check which tiles exist for a year

Before training, find out what you actually have.

**`check_missing_embeddings.py`** — per-directory coverage against the shapefile.

```bash
python check_missing_embeddings.py \
    --shapefile /path/healpix.shp \
    --dir       /path/embeddings/2020/presto \
    --splits train test \
    --validate \
    --out missing_2020.txt
```

**Always pass `--validate`.** `np.load` is lazy on `.npz`, so a run that hit a disk
quota leaves a **truncated but existing** file that a plain existence check accepts
forever. `--validate` opens each file and touches every array. Corrupt files are
reported separately and included in `--out`, which is a plain id-per-line list ready
to feed a re-run.

Add `--keep-zeros` if you generated the `_withzeros` cache — it looks for a different
filename, and without it you get a false "all missing".

Related: `check_olmo_outputs.py` (OlmoEarth-specific), `--check-only` on any Stage-1
script for the same answer without the deep validation.

---

## Stage 3 — Align the tiles, then verify the pixels

Two separate checks. Do them in this order.

### 3a. Master tile list — TILE-level intersection

**`build_master_tile_list.py`** — the tiles that *every* source actually has. Without
this, each encoder is scored on whatever it happens to have on disk, which confounds
encoder quality with coverage.

```bash
python build_master_tile_list.py \
    --source olmo=/path/embeddings/2020/olmo \
    --source aef=/path/embeddings/2020/aef \
    --source presto=/path/embeddings/2020/presto \
    --source tessera=/path/embeddings/2020/tessera \
    --source prithvi=/path/embeddings/2020/prithvi \
    --tile-list train=/path/train_ids.txt \
    --tile-list test=/path/test_ids.txt \
    --out-dir ./aligned_2020 \
    --validate
```

`--validate` applies the same deep read as Stage 2. `--require NAME ...` restricts the
intersection to a subset of sources. Output is the per-split id lists that every later
stage passes as `--tile-list`.

### 3b. Pixel alignment — the check that tile agreement does *not* give you

**`count_measurements.py`** — verifies the sources sampled the **same pixels**, by
comparing the `y` vectors (identical GEDI labels ⇒ identical pixels).

```bash
python count_measurements.py \
    --source presto=/path/embeddings/2020/presto \
    --source aef=/path/embeddings/2020/aef \
    --tile-list test=./aligned_2020/test_ids.txt \
    --reference aef \
    --out-md alignment_2020.md
```

This exists because three sources apply **different pixel-drop rules**, so two caches
can both cover a tile while disagreeing on which pixels inside it are usable. Key flags:

- `--reference NAME` — the source everything else is compared against
- `--atol` — tolerance on `y` (default `1e-4`)
- `--max-pixel-diff-frac F` — tolerate a small disagreement instead of failing
- `--out-aligned-dir DIR` — write caches restricted to the common pixels
- `--out-csv` / `--out-md` / `--out-json` — machine- and human-readable tables

The `n` you report must match the aligned test count from here, not each source's raw
count.

---

## Stage 4 — Train and evaluate the RandomForest

**`run_gedi_prediction_30m.py`** is the trainer. It reads the `.npz` caches directly,
so it never touches imagery.

```bash
python run_gedi_prediction_30m.py \
    --shapefile       /path/healpix.shp \
    --gedi-dir        /path/gedi_tiles_2020_healpix \
    --embeddings-dir  /path/embeddings \
    --preprocessed-dir /path/embeddings/2020/presto \
    --embedding-source presto \
    --embedding-format float32 \
    --model rf --n-estimators 100 \
    --year 2020 --random-state 42 \
    --tile-list ./aligned_2020/test_ids.txt \
    --output-dir /path/outputs/presto/seed_42
```

Key flags:

| Flag | Notes |
|---|---|
| `--preprocessed-dir` | the Stage-1 cache. Omit it and the path is built from `--embeddings-dir`/`--year`/`--embedding-source` |
| `--embedding-source` | `olmo`, `aef`, `tessera`, `presto`, `prithvi` — naming/metadata only |
| `--model` | `rf`, `xgb`, `gb`, `lr` (binned classifier) |
| `--tile-list` | **use the master list** so every source sees identical tiles |
| `--max-tiles N` | cap tiles per split — smoke tests |
| `--subset-fraction` / `--subset-seed` / `--subset-splits` | random tile subset, mirrored by the Stage-1 `--subset-tile-fraction` |
| `--stratified-sample-fraction` | two-pass height-stratified pixel sampling; keeps peak memory ~10% of the full set |
| `--include-zeros` | use the `_withzeros` caches; outputs gain a `_withzeros` suffix |
| `--metric-bin-size` / `--metric-max-height` / `--metric-min-bin-n` | must match the baselines or the numbers are not comparable |

Missing caches are **not fatal** — the trainer logs a warning and skips the tile
(`missing_tiles.json`). If you want the reported tile count to match what was actually
used, filter the tile list first rather than relying on the skip.

Outputs, all in `--output-dir`:

| File | Contents |
|---|---|
| `metrics.json` | train/val/test × plain + stratified metrics, plus the run's config |
| `predictions_{val,test}.parquet` | `y_true`, `y_pred` per pixel |
| `predictions_{val,test}.png` | scatter + residuals |
| `feature_importance.png` | top-30 |
| `missing_tiles.json`, `subset_tiles.json`, `stratified_sample_info.json` | provenance |
| `model_*.pkl` | the fitted model |

**Seed sweeps.** `run_seed_sweep.sh` loops sources × seeds; array index = seed.
`SOURCES` defaults to `"olmo aef presto tessera prithvi"`, and each run lands in
`${OUT_ROOT}/${SOURCE}/seed_${SEED}/`.

---

## Stage 5 — Metrics

### Why stratified

GEDI height distributions are dominated by short vegetation, so a plain `R²` mostly
measures performance on the low bins. `metrics_stratified.py` is the single
implementation, shared by the trainer, the backfill tool and every baseline, so the
numbers cannot drift apart.

Three views, computed per split and nested under `metrics.json → <split> → stratified`:

| Key | Meaning |
|---|---|
| `bins[]` | per 10 m bin: `lo, hi, n, rmse, mae, r2, bias` |
| `macro_rmse` / `macro_r2` | unweighted mean **across bins** — every bin counts equally |
| `balanced_rmse` / `balanced_r2` | resample **every bin to the smallest bin's n**, then compute ordinary metrics. Report these. |
| `balanced_n_per_bin`, `balanced_n_samples` | the `n` to quote alongside them |

`macro_*` and `balanced_*` answer different questions — macro averages per-bin metrics,
balanced pools an equal-size resample. Never mix them between runs.

> **Balanced metrics are sensitive to a handful of outliers.** The balanced resample
> weights the top height bin equally with bins 25× larger, so a few implausible GEDI
> pixels swing `balanced_r2` substantially — measured: raising the GEDI cap from
> 12000 to 20000 cm admitted 17 pixels (0.003%) and moved `balanced_r2` by ~12%
> relative. This is why the `< 12000` cap is not negotiable.

### Tools

| Script | Purpose |
|---|---|
| `metrics_stratified.py` | the metric implementation (`evaluate_and_report`). Import it; don't reimplement. |
| `stratify_from_predictions.py` | backfill stratified metrics into runs that predate them, from `predictions_*.parquet`. No retraining. |
| `aggregate_seed_sweep.py` | mean ± sample std (ddof=1) of RMSE/MAE/R² across seeds, per source |
| `results-plotting/evaluate.py` | scatter/density plot + summary from a parquet |
| `results-plotting/evaluate_bins.py` | binned version of the same |
| `results-plotting/plot_binned_rmse.py` | RMSE-vs-height-bin comparison across sources |
| `results-plotting/plot_prediction_scatter.py` | publication scatter |
| `results-plotting/plot_track_comparison.py` | canopy height along a single GEDI track, models overlaid |
| `data_processing/reconstruct_predictions_to_raster.py` | re-attach geo coords to a coord-less parquet → per-tile prediction rasters |

```bash
# aggregate a sweep
python aggregate_seed_sweep.py --root /path/outputs_seed_sweep_2020 \
    --split test --csv sweep_2020.csv

# backfill stratified metrics into older runs
python stratify_from_predictions.py --root /path/outputs_seed_sweep_2020 \
    --splits test --bin-size 10 --max-height 60

# plot from a parquet
python results-plotting/evaluate.py --parquet /path/outputs/presto/seed_42/predictions_test.parquet
python results-plotting/evaluate_bins.py --parquet /path/.../predictions_test.parquet
```

---

## Baselines

Each baseline writes `predictions_test.parquet` + `metrics.json` in the **same format**
as a model run, computed through the same `metrics_stratified.py`, so they drop straight
into the comparison. All support `--tile-list`, `--cache-only` and a per-tile raster
cache that can be re-sampled against any GEDI year without re-downloading.

```bash
# ETH 2020 (Earth Engine)
python data_processing/eth_canopy_baseline.py --split test \
    --gedi-dir /path/gedi_tiles_2020_healpix --out-dir /path/eth_baseline

# Potapov 2019 (Earth Engine); --ee-project only needed if something is uncached
python data_processing/potapov_baseline.py --split test --ee-project <project> ...

# ECHOSAT (plain HTTP, windowed reads — never download a whole tile, they are 51-195 GB)
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
python data_processing/echosat_baseline.py --split test --year 2020 \
    --gedi-dir /path/gedi_tiles_2020_healpix --out-dir /path/echosat --all-years

# validate the ECHOSAT cache before trusting it
python check_echosat_cache.py --tiles-dir /path/echosat/echosat_tiles \
    --gedi-dir /path/gedi_tiles_2020_healpix --shapefile /path/healpix.shp \
    --split test --deep --verify-source 8
```

`check_echosat_cache.py` is worth copying as a pattern: its `--verify-source` re-reads
the upstream server and correlates, which is the only check that can catch a wrong CRS
transform or a unit error. Everything else is self-consistency and passes happily on
confidently wrong data.

---

## Wall-to-wall prediction

Applying a fitted model to every pixel rather than only GEDI pixels:

| Script | Purpose |
|---|---|
| `run_aef_rf_wall_to_wall.py` | AlphaEarth wall-to-wall RF |
| `data_processing/alphaearth_wall_to_wall.py` | dense AEF embeddings for a region |
| `hybrid_rf_wall_to_wall.py` | hybrid feature-set variant |
| `reconstruct_to_map.py` | stitch per-tile predictions into a map |
| `compare_maps_to_als.py` | validate a map against airborne lidar (ALS) |
| `carbon_accounting.py` | height → biomass/carbon |

---

## Environments

Two conda environments are used, because the GFM dependencies conflict:

- **`jpl`** — RF training, metrics, plotting (sklearn, geopandas, rasterio, pyarrow)
- **`jpl-geo`** — embedding generation (adds rslearn, torch, presto, GDAL ≥ 3.10)

Notes that will bite otherwise:

- **Presto** must be installed from source with `pip install -e . --no-deps` plus
  openmapflow guards; the PyPI package is not equivalent.
- **rslearn durations** are parsed by `pytimeparse`, which reads `"M"` as **minutes**.
  Use `"30d"`, never `"1M"`.
- On HPC, `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...` is needed for the conda
  GDAL/PROJ stack but **breaks system binaries** linked against system libs
  (`/usr/bin/curl` in particular). Probe network reachability with Python/GDAL, not curl.
- `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` and
  `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif` are **required**, not optional, for any
  `/vsicurl` work.

---

## Repository layout

```
.
├── data_processing/          embedding generation, baselines, GEDI/tile prep
├── results-plotting/         evaluation plots and figures
├── run_gedi_prediction_30m.py    the RF trainer
├── metrics_stratified.py         the metric implementation
├── build_master_tile_list.py     tile-level alignment
├── count_measurements.py         pixel-level alignment
├── check_missing_embeddings.py   per-year coverage
├── aggregate_seed_sweep.py       multi-seed summary
└── run_*.sh                      SLURM wrappers
```

---

## Citation and licensing

This repo wraps several third-party products, each with its own terms. If you publish
numbers from it, cite the upstream sources:

- **ECHOSAT** — Pauls et al., arXiv:2602.21421. Data CC-BY 4.0.
- **ETH Global Canopy Height 2020** — Lang et al. 2023, doi:10.1038/s41477-023-01454-5
- **Potapov et al. 2019** — GEDI-derived forest height
- **Meta/WRI CHM v2** — Brandt et al. 2026
- **Presto** — Tseng et al. (nasaharvest/presto)
- **Clay**, **Prithvi-EO-2.0**, **OlmoEarth**, **TESSERA**, **AlphaEarth**, **Major TOM** — see each project
- **GEDI L2A/L2B** — NASA GEDI mission

---

## TODO before open-sourcing

Known gaps, in rough priority order:

- [ ] **Add a `LICENSE`** — there is none, so the repo is currently "all rights
      reserved" and nobody can legally reuse it.
- [ ] **Add `environment.yml` / `requirements.txt`** — the two environments are
      described above but not reproducible from the repo.
- [ ] **Add a `.gitignore`** and untrack the committed `__pycache__/`, `*.pyc` and
      `.DS_Store` files.
- [ ] **Remove hardcoded absolute paths.** Several scripts default to
      `/Users/kdoerkse/...`, `/scratch/kdoerkse/...` or `/ocean/projects/...`, and
      `data_processing/eth_canopy_baseline.py` defaults `--ee-project` to a personal
      GCP project.
      Make them required arguments or read them from a config file.
- [ ] **Remove or clearly mark scratch files** — `scratch.py`, `benchmark_acquisition.py`,
      and the `qa_*` / `dequant_check_*` one-offs.
- [ ] **Hoist the GEDI validity rule** (`> 0`, `< 12000`) into one shared module; it is
      currently duplicated in five files where it must not diverge.
- [ ] Add a small end-to-end smoke test on one tile so contributors can verify a setup.
- [ ] Document how to obtain the HEALPix shapefile and GEDI mosaics, or ship a tiny
      example tile — right now Stage 0 cannot be reproduced from the repo alone.
