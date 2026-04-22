#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=10
#SBATCH -t 00:30:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=50G

#SBATCH -o logs/gedi_rf_30m_test_%j.out
#SBATCH -e logs/gedi_rf_30m_test_%j.err

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kdoerkse@asu.edu

# =============================================================================
# Pipeline smoke test — 5 tiles per split, 10 trees
# =============================================================================

module purge
module load mamba/latest
source activate jpl-geo

unset PROJ_DATA PROJ_LIB PROJ_NETWORK
export LD_LIBRARY_PATH=$HOME/.conda/envs/jpl-geo/lib:$LD_LIBRARY_PATH
export PROJ_DATA=$HOME/.conda/envs/jpl-geo/share/proj

export PYTHONUNBUFFERED=1

cd /home/kdoerkse/code/jpl-bio

SHAPEFILE=/scratch/kdoerkse/olmo/healpix_shapefile.shp
GEDI_DIR=/scratch/kdoerkse/olmo/gedi
EMBEDDINGS_DIR=/scratch/kdoerkse/olmo/california_embeddings_30m_reproj
EMBEDDING_FORMAT=float32
OUTPUT_DIR=/scratch/kdoerkse/olmo/outputs_gedi_rf_30m_test_$(date +%Y%m%d_%H%M%S)

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=== PIPELINE SMOKE TEST (5 tiles per split) ==="
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Start time: $(date)"

python -u run_gedi_prediction_30m.py \
    --shapefile        "${SHAPEFILE}" \
    --gedi-dir         "${GEDI_DIR}" \
    --embeddings-dir   "${EMBEDDINGS_DIR}" \
    --output-dir       "${OUTPUT_DIR}" \
    --embedding-format "${EMBEDDING_FORMAT}" \
    --n-estimators     10 \
    --max-tiles        5

echo "=== TEST COMPLETE at $(date) ==="
echo "Outputs: ${OUTPUT_DIR}"
