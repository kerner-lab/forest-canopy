#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00-08:00:00
#SBATCH -q public
#SBATCH -p htc
#SBATCH --mem=100G
#SBATCH --array=1-311%6
#SBATCH -o logs/presto_gen_2021_%j_%A_%a.out
#SBATCH -e logs/presto_gen_2021_%j_%A_%a.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kdoerkse@asu.edu

# ---------------------------------------------------------------------------
# Presto embedding generation (Sentinel-2 fetched free via rslearn + Planetary
# Computer). Mirrors the AEF generation job. Outputs land in:
#     /scratch/kdoerkse/presto/<YEAR>/healpix_<id>_float32.npz
# ---------------------------------------------------------------------------

# Year of the 12-month Sentinel-2 window to embed. CHANGE THIS to re-run a
# different year (and update the 2021 in the #SBATCH -o/-e log names above,
# since SBATCH directives can't read shell variables).
YEAR=2021

TILE_IDS=/scratch/kdoerkse/olmo/tile_ids.txt
HEALPIX_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TILE_IDS}")

export SHAPEFILE_PATH=/scratch/kdoerkse/olmo/healpix_shapefile.shp
export WORKING_DIR=/home/kdoerkse/code/jpl-bio
export GEDI_DIR=/scratch/kdoerkse/olmo/${YEAR}/gedi
export OUTPUT_DIR=/scratch/kdoerkse/presto/${YEAR}

# Per-tile rslearn working dir: array tasks run concurrently and rslearn
# prepare/materialize operate on the whole dataset group, so each tile needs
# its own isolated dataset root.
export RSLEARN_ROOT=/scratch/kdoerkse/presto/${YEAR}/rslearn/healpix_${HEALPIX_ID}

mkdir -p "${OUTPUT_DIR}" "${RSLEARN_ROOT}" logs

module purge
module load mamba/latest
source activate jpl-geo

unset PROJ_DATA PROJ_LIB PROJ_NETWORK
export LD_LIBRARY_PATH=$HOME/.conda/envs/jpl-geo/lib:$LD_LIBRARY_PATH
export PROJ_DATA=$HOME/.conda/envs/jpl-geo/share/proj

# Let torch use the cores we requested (Presto runs on CPU).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Starting Presto embedding extraction"
echo "  Year:             ${YEAR}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  rslearn root:     ${RSLEARN_ROOT}"
echo "  Job ID:           ${SLURM_JOB_ID}"
echo "  Start time:       $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  healpix_id: ${HEALPIX_ID}"

cd "${WORKING_DIR}"
python -u data_processing/presto_processing.py \
    --healpix-id "${HEALPIX_ID}" \
    --year "${YEAR}" \
    --shapefile   "${SHAPEFILE_PATH}" \
    --gedi-dir    "${GEDI_DIR}" \
    --output-dir  "${OUTPUT_DIR}" \
    --rslearn-root "${RSLEARN_ROOT}" \
    --s2-source pc \
    --rslearn-workers "${SLURM_CPUS_PER_TASK}" \
    --qa
STATUS=$?
# To also write a QGIS-viewable embedding GeoTIFF per tile, add: --save-tif

# Free the fetched Sentinel-2 imagery once the .npz is written (the npz is the
# durable artifact). Comment this out if you want to keep/reuse the imagery.
if [ ${STATUS} -eq 0 ]; then
    rm -rf "${RSLEARN_ROOT}"
fi

echo "Completed at $(date)  (exit ${STATUS})"
echo "Outputs: ${OUTPUT_DIR}"
exit ${STATUS}
