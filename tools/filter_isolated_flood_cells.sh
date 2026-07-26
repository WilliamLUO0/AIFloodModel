#!/bin/bash
#SBATCH --job-name=filter_isolated_flood_cells_filtered_thr0p1_gabrielle
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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

BASE_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results"

THRESHOLD=0.1
CONNECTIVITY=8
MIN_PATCH_DX64=5
MIN_PATCH_DX8=100

SCENARIOS=(
  # "1000y_48h_0c"
  # "100y_48h_0c"
  # "1000y_6h_0c"
  # "100y_6h_0c"
  # "500y_48h_0c"
  # "500y_6h_0c"
  # "200y_48h_0c"
  # "200y_6h_0c"
  # "50y_48h_0c"
  # "50y_6h_0c"
  # "20y_48h_0c"
  # "20y_6h_0c"
  # "10y_48h_0c"
  # "10y_6h_0c"
  # "5y_48h_0c"
  # "5y_6h_0c"
  # "2y_48h_0c"
  # "2y_6h_0c"
  # "100y_42h_0c"
  # "2y_42h_0c"
  # "1000y_42h_0c"
  "gabrielle"
)

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "============================================================"
  echo "[info] Processing scenario: ${SCENARIO}"
  echo "============================================================"

  # ------------------------------------------------------------
  # dx64: filter BGout.nc directly
  # ------------------------------------------------------------
  DX64_INPUT="${BASE_DIR}/${SCENARIO}/dx64/BGout.nc"
  DX64_OUTPUT="${BASE_DIR}/${SCENARIO}/dx64/BGout_filtered_thr0p1_min${MIN_PATCH_DX64}.nc"

  if [[ -f "${DX64_INPUT}" ]]; then
    echo "[info] Filtering dx64:"
    echo "       input : ${DX64_INPUT}"
    echo "       output: ${DX64_OUTPUT}"

    python tools/filter_isolated_flood_cells.py \
      --input-nc "${DX64_INPUT}" \
      --output-nc "${DX64_OUTPUT}" \
      --threshold "${THRESHOLD}" \
      --min-patch-size "${MIN_PATCH_DX64}" \
      --connectivity "${CONNECTIVITY}" \
      --overwrite
  else
    echo "[warn] dx64 input not found, skipping:"
    echo "       ${DX64_INPUT}"
  fi

  # ------------------------------------------------------------
  # dx8: filter per-timestep merged files
  # ------------------------------------------------------------
  DX8_INPUT_DIR="${BASE_DIR}/${SCENARIO}/dx8/per_timestep_merged"
  DX8_OUTPUT_DIR="${BASE_DIR}/${SCENARIO}/dx8/per_timestep_merged_filtered_thr0p1_min${MIN_PATCH_DX8}"

  if [[ -d "${DX8_INPUT_DIR}" ]]; then
    echo "[info] Filtering dx8:"
    echo "       input_dir : ${DX8_INPUT_DIR}"
    echo "       output_dir: ${DX8_OUTPUT_DIR}"

    python tools/filter_isolated_flood_cells.py \
      --input-dir "${DX8_INPUT_DIR}" \
      --output-dir "${DX8_OUTPUT_DIR}" \
      --threshold "${THRESHOLD}" \
      --min-patch-size "${MIN_PATCH_DX8}" \
      --connectivity "${CONNECTIVITY}" \
      --overwrite
  else
    echo "[warn] dx8 input directory not found, skipping:"
    echo "       ${DX8_INPUT_DIR}"
  fi

done

echo "[done] All scenarios processed."
