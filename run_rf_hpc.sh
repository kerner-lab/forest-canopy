#!/bin/bash

#SBATCH -N 1                            # 1 node
#SBATCH -n 10                           # 10 CPUs
#SBATCH -t 02-00:00:00                  # wall time (2 days)
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=100G                      # Request 100GB RAM (IMPORTANT!)
# Remove GPU allocation if not needed for Random Forest
# #SBATCH -G a100:1

#SBATCH -o logs/log_temp_jpl_rf_norcal.out
#SBATCH -e logs/log_temp_jpl_rf_norcal.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kdoerkse@asu.edu

module load mamba/latest
cd /home/kdoerkse/code/jpl-bio

# Run memory-efficient version with smaller batch size
mamba run -n jpl python -u simple_random_forest_hpc.py \
    --root_dir /data/hkerner/kdoerkse/jpl-bio/tiles \
    --save_dir /home/kdoerkse/scratch-results-jpl

echo "Job completed!"
