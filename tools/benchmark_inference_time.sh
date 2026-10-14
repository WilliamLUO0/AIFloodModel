#!/bin/bash
#SBATCH --job-name=benchmark_infer_time
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
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

# ===================== the test sets the full-scenario time is sized on =====================
# Every model is extrapolated over ITS OWN test set, so the script only needs
# the folder that holds them:
#   testdataset_100y42h0c        s = 8 (all four architectures)
#   testdataset_100y42h0c_ds2    s = 2
#   testdataset_100y42h0c_ds4    s = 4
#   testdataset_100y42h0c_ds16   s = 16
# No checkpoint is loaded - latency depends on layer shapes, not on weights -
# so --exp-root is not needed either.
# ====================================================================================================

nvidia-smi

python tools/benchmark_inference_time.py \
  --batch 1 --warmup 20 --iters 100 \
  --test-root "$ROOT" \
  --csv tools/model_infer_time.csv
