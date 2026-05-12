#!/bin/bash
#SBATCH --job-name=plot_assembled_floodmaps
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

python plot_assembled_floodmaps.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --vis-root /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/visualization \
  --scenario 100y_42h_0c \
  --var h \
  --scale 16 \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/9_4_present_plot/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/assembled_maps \
  --time-start 45 \
  --time-end 47 \
  --flood-thr 0.1 \
  --flood-q 99 \
  --abs-tol 0.02 \
  --err-q 99