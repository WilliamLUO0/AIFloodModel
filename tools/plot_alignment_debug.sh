#!/bin/bash
#SBATCH --job-name=plot_alignment_debug
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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

python tools/plot_alignment_debug.py \
  --debug-root /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_debug \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_debug/alignment_figures \
  --threshold 0.1 \
  --max-shift 3
