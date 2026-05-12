#!/bin/bash
#SBATCH --job-name=cal_flood_metrics_global
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
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

python cal_flood_metrics_global.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --vis-root /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/visualization \
  --out-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/9_4_present_plot/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/dataset_metrics.json \
  --out-csv-patch /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/9_4_present_plot/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/patch_metrics.csv \
  --out-csv-time /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/9_4_present_plot/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/time_metrics.csv \
  --out-csv-scenario /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/9_4_present_plot/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/scenario_metrics.csv \
  --var h

