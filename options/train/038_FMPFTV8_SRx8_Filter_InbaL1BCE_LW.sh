#!/bin/bash
#SBATCH --job-name=fmpft_sr8_gpu4_038_FMPFTV8_SRx8_Filter_InbaL1BCE_LW
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
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
  basicsr/train_flood_map_v2.py -opt options/train/038_FMPFTV8_SRx8_Filter_InbaL1BCE_LW.yml \
  --launcher pytorch --auto_resume
