#!/bin/bash
#SBATCH --job-name=precompute_patch_interval_stats
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python tools/precompute_patch_interval_stats.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8/index.csv \
  --out-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8/index_with_interval_stats_h.csv \
  --out-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8/patch_interval_summary_h.json \
  --target-var h \
  --h-slight 0.1 \
  --h-severe 0.5 \
  --h-extreme 1.0 \
  --split-stats-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8/split_stats_h_asinh_wet.json \
  --summary-split train