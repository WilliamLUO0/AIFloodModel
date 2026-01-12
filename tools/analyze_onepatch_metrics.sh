#!/bin/bash
#SBATCH --job-name=analyze_onepatch_metrics_testdataset_100y42h0c_t47_0_6
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

python tools/analyze_onepatch_metrics.py \
  --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/visualization \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --key h_100y_42h_0c_t0047_r000_c006_s16 \
  --delta 0.05 \
  --out-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/key_debug/h_100y_42h_0c_t0047_r000_c006_s16_report.json
