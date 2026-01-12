#!/bin/bash
#SBATCH --job-name=precompute_stats_h
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

# python tools/precompute_split_stats.py \
#  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/index.csv \
#  --root      /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup \
#  --out_json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_log1p.json \
#  --target_var h \
#  --h_transform log1p \
#  --by scenario --val_ratio 0.2 --seed 61 \
#  --bins 8192

# python tools/precompute_split_stats.py \
#  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/index.csv \
#  --root      /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup \
#  --out_json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json \
#  --target_var h \
#  --h_transform asinh \
#  --h_tau 0.05 \
#  --h_q_list 50,75,90,95,99 \
#  --by scenario --val_ratio 0.2 --seed 61 \
#  --bins 8192

python tools/precompute_split_stats.py \
  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/index.csv \
  --root      /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup \
  --out_json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_u.json \
  --target_var u \
  --uv_tau 0.1 \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192
