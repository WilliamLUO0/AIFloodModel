#!/bin/bash
#SBATCH --job-name=merge_interp_dataset_ds8_engine5
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
conda activate /nesi/project/uoa04425/zluo784/envs/python310

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python ./merge_multiscale.py \
  --fine-dir "/nesi/nobackup/uoa04425/zluo784/Exp2/Gisborne_basin/results/1000y_48h_0c/dx8" \
  --input-nc "BGout.nc" \
  --coarse-nc "/nesi/nobackup/uoa04425/zluo784/Exp2/Gisborne_basin/results/1000y_48h_0c/dx64/BGout.nc" \
  --out-perdir "per_timestep_merged" \
  --vars zs u v h

python ./merge_multiscale.py \
  --fine-dir "/nesi/nobackup/uoa04425/zluo784/Exp2/Gisborne_basin/results/100y_48h_0c/dx8" \
  --input-nc "BGout.nc" \
  --coarse-nc "/nesi/nobackup/uoa04425/zluo784/Exp2/Gisborne_basin/results/100y_48h_0c/dx64/BGout.nc" \
  --out-perdir "per_timestep_merged" \
  --vars zs u v h