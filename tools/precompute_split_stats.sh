#!/bin/bash
#SBATCH --job-name=precompute_stats_h
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
conda activate /nesi/project/uoa04425/zluo784/envs/pft39

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python tools/precompute_split_stats.py \
  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset/index.csv \
  --root      /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset \
  --out_json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset/split_stats_h.json \
  --target_var h \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192
