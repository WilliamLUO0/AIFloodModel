#!/bin/bash
# Submit the 03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW val-set evaluation job (SLURM / NeSI).
#SBATCH --job-name=eval_03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW_eval_val
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
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

srun torchrun --nproc_per_node=1 --standalone \
  basicsr/test_flood_map.py -opt options/test/03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW_eval_val.yml \
  --launcher pytorch

