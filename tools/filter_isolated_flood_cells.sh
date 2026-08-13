#!/bin/bash
#SBATCH --job-name=filter_isolated_flood_cells_dx16_32_128
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
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

# min_patch_size per resolution.
# Rationale: follow the BG-Flood developers' per-resolution guidance (conservative
# upper end of each band). This is ~area-preserving (~6400 m^2) at the fine end
# (dx8=100, dx16=25 both equal 6400 m^2) and deliberately flattens toward a
# single-cell-noise floor at the coarse end, so filtering never collapses to "do
# nothing" where one cell already covers a large area. Monotone ladder:
#   dx8=100  dx16=25  dx32=10  dx64=5  dx128=2
#
# ACTIVE this run: dx16 / dx32 / dx128 (new downscaling factors 2 / 4 / 16).
MIN_PATCH_DX16=25     # factor 2  : dx16  -> dx8
MIN_PATCH_DX32=10     # factor 4  : dx32  -> dx8
MIN_PATCH_DX128=2     # factor 16 : dx128 -> dx8
# Already filtered previously (factor 8 + shared target). Kept for provenance /
# reproducibility; the exact commands are preserved as commented branches below.
MIN_PATCH_DX8=100     # target, shared by all factors; per_timestep_merged dir
MIN_PATCH_DX64=5      # factor 8 input; single BGout.nc

SCENARIOS=(
  "1000y_48h_0c"
  "100y_48h_0c"
  "1000y_6h_0c"
  "100y_6h_0c"
  "500y_48h_0c"
  "500y_6h_0c"
  "200y_48h_0c"
  "200y_6h_0c"
  "50y_48h_0c"
  "50y_6h_0c"
  "20y_48h_0c"
  "20y_6h_0c"
  "10y_48h_0c"
  "10y_6h_0c"
  "5y_48h_0c"
  "5y_6h_0c"
  "2y_48h_0c"
  "2y_6h_0c"
  # --- test sets: enable later when filtering test data for the factor experiment ---
  # "100y_42h_0c"
  # "2y_42h_0c"
  # "1000y_42h_0c"
  # "gabrielle"
)

for SCENARIO in "${SCENARIOS[@]}"; do
  echo "============================================================"
  echo "[info] Processing scenario: ${SCENARIO}"
  echo "============================================================"

  # ------------------------------------------------------------
  # dx16 (factor 2): filter BGout.nc directly
  # ------------------------------------------------------------
  DX16_INPUT="${BASE_DIR}/${SCENARIO}/dx16/BGout.nc"
  DX16_OUTPUT="${BASE_DIR}/${SCENARIO}/dx16/BGout_filtered_thr0p1_min${MIN_PATCH_DX16}.nc"

  if [[ -f "${DX16_INPUT}" ]]; then
    echo "[info] Filtering dx16:"
    echo "       input : ${DX16_INPUT}"
    echo "       output: ${DX16_OUTPUT}"

    python tools/filter_isolated_flood_cells.py \
      --input-nc "${DX16_INPUT}" \
      --output-nc "${DX16_OUTPUT}" \
      --threshold "${THRESHOLD}" \
      --min-patch-size "${MIN_PATCH_DX16}" \
      --connectivity "${CONNECTIVITY}" \
      --overwrite
  else
    echo "[warn] dx16 input not found, skipping:"
    echo "       ${DX16_INPUT}"
  fi

  # ------------------------------------------------------------
  # dx32 (factor 4): filter BGout.nc directly
  # ------------------------------------------------------------
  DX32_INPUT="${BASE_DIR}/${SCENARIO}/dx32/BGout.nc"
  DX32_OUTPUT="${BASE_DIR}/${SCENARIO}/dx32/BGout_filtered_thr0p1_min${MIN_PATCH_DX32}.nc"

  if [[ -f "${DX32_INPUT}" ]]; then
    echo "[info] Filtering dx32:"
    echo "       input : ${DX32_INPUT}"
    echo "       output: ${DX32_OUTPUT}"

    python tools/filter_isolated_flood_cells.py \
      --input-nc "${DX32_INPUT}" \
      --output-nc "${DX32_OUTPUT}" \
      --threshold "${THRESHOLD}" \
      --min-patch-size "${MIN_PATCH_DX32}" \
      --connectivity "${CONNECTIVITY}" \
      --overwrite
  else
    echo "[warn] dx32 input not found, skipping:"
    echo "       ${DX32_INPUT}"
  fi

  # ------------------------------------------------------------
  # dx128 (factor 16): filter BGout.nc directly
  # ------------------------------------------------------------
  DX128_INPUT="${BASE_DIR}/${SCENARIO}/dx128/BGout.nc"
  DX128_OUTPUT="${BASE_DIR}/${SCENARIO}/dx128/BGout_filtered_thr0p1_min${MIN_PATCH_DX128}.nc"

  if [[ -f "${DX128_INPUT}" ]]; then
    echo "[info] Filtering dx128:"
    echo "       input : ${DX128_INPUT}"
    echo "       output: ${DX128_OUTPUT}"

    python tools/filter_isolated_flood_cells.py \
      --input-nc "${DX128_INPUT}" \
      --output-nc "${DX128_OUTPUT}" \
      --threshold "${THRESHOLD}" \
      --min-patch-size "${MIN_PATCH_DX128}" \
      --connectivity "${CONNECTIVITY}" \
      --overwrite
  else
    echo "[warn] dx128 input not found, skipping:"
    echo "       ${DX128_INPUT}"
  fi

  # ============================================================
  # PROVENANCE — already filtered in previous runs (factor 8 + shared target).
  # Kept verbatim for traceability. Do NOT re-enable without intent: dx64 would be
  # re-filtered (harmless, same params) and dx8 is a slow per-timestep pass.
  # ============================================================
  #
  # # ------------------------------------------------------------
  # # dx64 (factor 8): filter BGout.nc directly
  # # ------------------------------------------------------------
  # DX64_INPUT="${BASE_DIR}/${SCENARIO}/dx64/BGout.nc"
  # DX64_OUTPUT="${BASE_DIR}/${SCENARIO}/dx64/BGout_filtered_thr0p1_min${MIN_PATCH_DX64}.nc"
  #
  # if [[ -f "${DX64_INPUT}" ]]; then
  #   echo "[info] Filtering dx64:"
  #   echo "       input : ${DX64_INPUT}"
  #   echo "       output: ${DX64_OUTPUT}"
  #
  #   python tools/filter_isolated_flood_cells.py \
  #     --input-nc "${DX64_INPUT}" \
  #     --output-nc "${DX64_OUTPUT}" \
  #     --threshold "${THRESHOLD}" \
  #     --min-patch-size "${MIN_PATCH_DX64}" \
  #     --connectivity "${CONNECTIVITY}" \
  #     --overwrite
  # else
  #   echo "[warn] dx64 input not found, skipping:"
  #   echo "       ${DX64_INPUT}"
  # fi
  #
  # # ------------------------------------------------------------
  # # dx8 (target): filter per-timestep merged files (directory mode)
  # # ------------------------------------------------------------
  # DX8_INPUT_DIR="${BASE_DIR}/${SCENARIO}/dx8/per_timestep_merged"
  # DX8_OUTPUT_DIR="${BASE_DIR}/${SCENARIO}/dx8/per_timestep_merged_filtered_thr0p1_min${MIN_PATCH_DX8}"
  #
  # if [[ -d "${DX8_INPUT_DIR}" ]]; then
  #   echo "[info] Filtering dx8:"
  #   echo "       input_dir : ${DX8_INPUT_DIR}"
  #   echo "       output_dir: ${DX8_OUTPUT_DIR}"
  #
  #   python tools/filter_isolated_flood_cells.py \
  #     --input-dir "${DX8_INPUT_DIR}" \
  #     --output-dir "${DX8_OUTPUT_DIR}" \
  #     --threshold "${THRESHOLD}" \
  #     --min-patch-size "${MIN_PATCH_DX8}" \
  #     --connectivity "${CONNECTIVITY}" \
  #     --overwrite
  # else
  #   echo "[warn] dx8 input directory not found, skipping:"
  #   echo "       ${DX8_INPUT_DIR}"
  # fi

done

echo "[done] All scenarios processed."
