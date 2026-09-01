#!/bin/bash
# Submit the 06 FMPFTV8 SRx4 downscaling-factor eval (SLURM / NeSI).
#SBATCH --job-name=eval_06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW_eval_gabrielle
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

srun torchrun --nproc_per_node=1 --standalone \
  basicsr/test_flood_map.py -opt options/test/06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW_eval_gabrielle.yml \
  --launcher pytorch
