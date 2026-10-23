#!/bin/bash
# Submit the 01 FMPFTV8 evaluation job on the held-out validation split (20% of the 18 train scenarios) (SLURM / NeSI).
#SBATCH --job-name=eval_08_FloodDM_SRx8_Filter_EpsL1_eval_valsub256_ts1000
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs1/%x_%j.out
#SBATCH --error=logs1/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nvidia-smi

srun torchrun --nproc_per_node=4 --standalone \
  basicsr/test_flood_map.py -opt options/test/08_FloodDM_SRx8_Filter_EpsL1_eval_valsub256_ts1000.yml \
  --launcher pytorch

