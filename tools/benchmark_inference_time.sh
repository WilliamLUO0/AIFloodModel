#!/bin/bash
#SBATCH --job-name=benchmark_infer_time
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# GPU-ONLY forward-latency benchmark of the 4 flood-SR models (factor 8). Unlike
# compute_flops.sh this NEEDS a GPU (real forward passes) AND the pft39 env (V8's
# forward calls the smm_cuda kernels). Pin ONE named A100 so the number is reproducible.
# Report per-patch median+IQR, and (with --test-index) the full-map / full-scenario time.

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

ROOT=/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel

# ===================== EDIT HERE: which test set to size the full-scenario time on =====================
TEST_INDEX="$ROOT/testdataset_100y42h0c/index.csv"
# TEST_INDEX="$ROOT/testdataset_gabrielle/index.csv"
# ====================================================================================================

nvidia-smi

python tools/benchmark_inference_time.py \
  --batch 1 --warmup 20 --iters 100 \
  --exp-root "$ROOT/experiments" \
  --test-index "$TEST_INDEX" \
  --csv tools/model_infer_time.csv
