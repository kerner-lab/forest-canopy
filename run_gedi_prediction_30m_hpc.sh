#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -t 02:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=100G

#SBATCH -o logs/gedi_rf_30m_%j.out
#SBATCH -e logs/gedi_rf_30m_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kdoerkse@asu.edu

# =============================================================================
# GEDI 30m RF prediction using OLMo embeddings
# =============================================================================

module purge
module load mamba/latest
source activate jpl-geo

unset PROJ_DATA PROJ_LIB PROJ_NETWORK
export LD_LIBRARY_PATH=$HOME/.conda/envs/jpl-geo/lib:$LD_LIBRARY_PATH
export PROJ_DATA=$HOME/.conda/envs/jpl-geo/share/proj

export PYTHONUNBUFFERED=1

cd /home/kdoerkse/code/jpl-bio

# ---------------------------------------------------------------------------
# Config — edit these before submitting
# ---------------------------------------------------------------------------
SHAPEFILE=/scratch/kdoerkse/olmo/healpix_shapefile.shp
GEDI_DIR=/scratch/kdoerkse/olmo/gedi
EMBEDDINGS_DIR=/scratch/kdoerkse/olmo/california_embeddings_30m_reproj
EMBEDDING_FORMAT=float32   # float32 or int8
N_ESTIMATORS=100
MAX_DEPTH=               # leave blank for unlimited
OUTPUT_DIR=/scratch/kdoerkse/olmo/outputs_gedi_rf_30m_${EMBEDDING_FORMAT}_$(date +%Y%m%d_%H%M%S)
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "Starting GEDI 30m RF prediction"
echo "  Embedding format: ${EMBEDDING_FORMAT}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  Job ID:           ${SLURM_JOB_ID}"
echo "  Start time:       $(date)"

DEPTH_ARG=""
if [ -n "${MAX_DEPTH}" ]; then
    DEPTH_ARG="--max-depth ${MAX_DEPTH}"
fi

python -u run_gedi_prediction_30m.py \
    --shapefile        "${SHAPEFILE}" \
    --gedi-dir         "${GEDI_DIR}" \
    --embeddings-dir   "${EMBEDDINGS_DIR}" \
    --output-dir       "${OUTPUT_DIR}" \
    --embedding-format "${EMBEDDING_FORMAT}" \
    --n-estimators     "${N_ESTIMATORS}" \
    --min-samples-split 2 \
    ${DEPTH_ARG}

echo "Completed at $(date)"
echo "Outputs: ${OUTPUT_DIR}"
