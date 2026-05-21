#!/bin/bash
#SBATCH --job-name=plot_selected_patch_alignment
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/python310
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export GDAL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export RASTERIO_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATASET_ROOT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8"
OUT_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_debug/selected_alignment_figures"

SCENARIOS=(
  "100y_48h_0c"
  "1000y_48h_0c"
)

TIMESTEPS=(
  "t0024"
  "t0029"
  "t0034"
)

PATCHES=(
  "4 15"
  "5 15"
  "6 15"
  "7 15"
  "8 15"
  "9 15"
  "10 15"
  "11 15"
  "10 14"
  "11 14"
  "12 14"
  "11 13"
  "12 13"
  "6 16"
  "7 16"
  "8 16"
  "6 17"
  "7 17"
)

mkdir -p "${OUT_DIR}"

#python tools/plot_selected_patch_alignment.py \
#  --dataset-root /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8 \
#  --scenario 100y_48h_0c \
#  --timestep t0000 \
#  --row 3 \
#  --col 5 \
#  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_debug/selected_alignment_figures \
#  --threshold 0.1 \
#  --max-shift 3

for SCENARIO in "${SCENARIOS[@]}"; do
  for TIMESTEP in "${TIMESTEPS[@]}"; do
    for PATCH in "${PATCHES[@]}"; do
      ROW=$(echo "${PATCH}" | awk '{print $1}')
      COL=$(echo "${PATCH}" | awk '{print $2}')

      echo "Plotting scenario=${SCENARIO}, timestep=${TIMESTEP}, row=${ROW}, col=${COL}"

      python tools/plot_selected_patch_alignment.py \
        --dataset-root "${DATASET_ROOT}" \
        --scenario "${SCENARIO}" \
        --timestep "${TIMESTEP}" \
        --row "${ROW}" \
        --col "${COL}" \
        --out-dir "${OUT_DIR}" \
        --threshold 0.1 \
        --max-shift 3
    done
  done
done
