#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 04:00:00
#SBATCH -p RM-shared
#SBATCH -A cis260017p
#SBATCH -J echosat
#SBATCH --array=0-7%4
#SBATCH -o /ocean/projects/agr260002p/kdoerkse/bash-scripts/logs/log_echosat_%A_%a.out
#SBATCH -e /ocean/projects/agr260002p/kdoerkse/bash-scripts/logs/log_echosat_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kdoerkse@asu.edu
# =============================================================================
# ECHOSAT canopy-height baseline over the GEDI HEALPix tiles.
#
# PHASE 1 (this array job): populate the per-tile cache in parallel. Each task
# takes every Nth tile of the split and windowed-reads ECHOSAT over it. Tasks
# share one --tiles-dir; they write disjoint files, so there is no race.
# --no-eval stops each task after caching, so no task emits a partial parquet.
#
# PHASE 2 (one short job, after the array finishes): re-run over the whole split
# with --cache-only to build predictions_test.parquet + metrics.json from the
# shared cache. No network. See the command at the bottom of this file.
#
# WHY AN ARRAY AND WHY %4
# The work is NETWORK-bound, not CPU-bound: ~100-150 s per tile against a
# university web server at ~1 MB/s. %4 caps concurrency at four tasks so we
# stay a polite client -- do NOT raise this to one task per tile. With 47 test
# tiles and 8 tasks that is ~6 tiles each, ~15 min per task.
#
# IMPORTANT: this reads whole-file-sized rasters through HTTP range requests.
# The GDAL_* exports below are not optional -- without
# GDAL_DISABLE_READDIR_ON_OPEN, GDAL issues a directory listing per open against
# a server hosting ~560 multi-GB files.
#
# ECHOSAT data is CC-BY 4.0; cite Pauls et al., arXiv:2602.21421.
#
# Usage:
#   sbatch run_echosat_baseline.sh                      # test split, 2020
#   SPLIT=train YEAR=2019 GEDI_YEAR=2019 sbatch run_echosat_baseline.sh
# =============================================================================
set -o pipefail

module load anaconda3
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /jet/home/kdoerkse/.conda/envs/jpl
set -u

cd /jet/home/kdoerkse/code/jpl-bio || { echo "ERROR: cannot cd to repo"; exit 1; }

echo "python: $(which python)"
python -c "import rasterio, geopandas, pandas; print('imports OK')" || {
    echo "ERROR: env not active or missing deps"; exit 1; }

# --- GDAL /vsicurl tuning: required, not cosmetic -----------------------------
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR   # no dir listing per open
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif    # skip sidecar probes (.aux.xml etc)
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=3
export GDAL_CACHEMAX=512                        # MB of block cache
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=67108864                  # 64 MB per-file read cache
export CPL_VSIL_CURL_CHUNK_SIZE=1048576         # 1 MB range requests

SHAPEFILE=/ocean/projects/agr260002p/kdoerkse/data/shapefiles/healpix_shapefile.shp
SPLIT=${SPLIT:-test}
YEAR=${YEAR:-2020}                              # ECHOSAT year (band) to evaluate
GEDI_YEAR=${GEDI_YEAR:-${YEAR}}                 # GEDI epoch to sample against
GEDI_DIR=/ocean/projects/agr260002p/kdoerkse/data/gedi/gedi_tiles_${GEDI_YEAR}_healpix

OUT_ROOT=/ocean/projects/agr260002p/kdoerkse/echosat_baseline
OUT_DIR=${OUT_ROOT}/${SPLIT}_${YEAR}
# One cache shared by every year and split: the cached tifs hold ALL 7 years, so
# a 2019 run and a 2020 run reuse the same downloads. Never make this per-year.
TILES_DIR=${OUT_ROOT}/echosat_tiles

[ -f "${SHAPEFILE}" ] || { echo "ERROR: missing ${SHAPEFILE}"; exit 1; }
[ -d "${GEDI_DIR}" ]  || { echo "ERROR: missing GEDI_DIR ${GEDI_DIR}"; exit 1; }
mkdir -p "${OUT_DIR}" "${TILES_DIR}"

NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASKID=${SLURM_ARRAY_TASK_ID:-0}

# Slice the split's tile ids: task k takes ids where (index %% NTASKS) == k.
# Derived from the shapefile so it needs no pre-built list and stays consistent
# with what the python script would select for the same --split.
SLICE=${OUT_DIR}/tiles_task_${TASKID}.txt
python - "$SHAPEFILE" "$SPLIT" "$NTASKS" "$TASKID" "$SLICE" <<'PY'
import sys
import geopandas as gpd
shp, split, n, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ids = sorted(int(t) for t in gpd.read_file(shp).query("split == @split")['healpix_id'])
mine = ids[k::n]
open(out, 'w').write("\n".join(map(str, mine)) + "\n")
print(f"task {k}/{n}: {len(mine)}/{len(ids)} '{split}' tiles -> {out}")
PY
[ -s "${SLICE}" ] || { echo "task ${TASKID}: empty slice, nothing to do"; exit 0; }

echo "=============================================================="
echo "ECHOSAT cache pass  split=${SPLIT} year=${YEAR} gedi=${GEDI_YEAR}"
echo "  task ${TASKID}/${NTASKS}   cache=${TILES_DIR}"
echo "=============================================================="
echo "Start: $(date)"

python -u data_processing/echosat_baseline.py \
    --shapefile "${SHAPEFILE}" \
    --gedi-dir  "${GEDI_DIR}" \
    --out-dir   "${OUT_DIR}" \
    --tiles-dir "${TILES_DIR}" \
    --split     "${SPLIT}" \
    --year      "${YEAR}" \
    --tile-list "${SLICE}" \
    --no-eval
STATUS=$?

echo "Completed at $(date)  (exit ${STATUS})"

# -----------------------------------------------------------------------------
# PHASE 2 -- run ONCE after the whole array finishes (seconds, no network):
#
#   python data_processing/echosat_baseline.py \
#       --shapefile /ocean/projects/agr260002p/kdoerkse/data/shapefiles/healpix_shapefile.shp \
#       --gedi-dir  /ocean/projects/agr260002p/kdoerkse/data/gedi/gedi_tiles_2020_healpix \
#       --out-dir   /ocean/projects/agr260002p/kdoerkse/echosat_baseline/test_2020 \
#       --tiles-dir /ocean/projects/agr260002p/kdoerkse/echosat_baseline/echosat_tiles \
#       --split test --year 2020 --cache-only --all-years \
#       --tile-list /jet/home/kdoerkse/code/jpl-bio/aligned_2020/aligned_tile_ids.txt
#
# Add --tile-list so the baseline is scored on exactly the tiles the models were.
# --all-years reports every band against this GEDI, which is the check on the
# band->year assumption (see the script docstring).
#
# Or chain it automatically:
#   AID=$(sbatch --parsable run_echosat_baseline.sh)
#   sbatch --dependency=afterok:$AID run_echosat_eval.sh
# -----------------------------------------------------------------------------
exit ${STATUS}
