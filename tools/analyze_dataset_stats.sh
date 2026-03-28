#!/bin/bash
#SBATCH --job-name=analyze_dataset_stats_dataset_maxclip_coarse
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=5:00:00
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

python tools/analyze_dataset_stats.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_maxclip/index.csv \
  --out-dir   /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_maxclip \
  --var h --grid coarse
