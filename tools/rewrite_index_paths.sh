#!/bin/bash
#SBATCH --job-name=rewrite_index_paths_all
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export PYTHONUNBUFFERED=1

# Rewrite index CSV path prefixes NeSI -> NIWA for EVERY dataset that will be used
# on the NIWA HPC (training + test). rewrite_index_paths.py replaces by parent-dir
# prefix, so it covers dataset_ds2/4/8/16 and testdataset_* alike. It writes a
# *_niwa.csv beside each source CSV (never in-place), which is what the NIWA
# *_pbs.yml dataroot / eval configs load, and what pack_dataset.sh ships.
#
# Run this on NeSI AFTER precompute_* (so the training datasets already have their
# index_with_interval_stats_{h,u,v}.csv) and BEFORE pack_dataset.sh.
# Per dataset it rewrites every index*.csv it finds (skipping *_niwa.csv):
#   training  -> index.csv + index_with_interval_stats_h.csv (+ _u/_v for ds8)
#   test      -> index.csv

BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

# ===================== EDIT HERE: which datasets to rewrite =====================
DATASETS=(
  # --- training (4) ---
  "dataset_ds8_filtered_thr0p1_min5100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds2_filtered_thr0p1_min25100_full"
  # --- test: factor-8 main-model suite ---
  "testdataset_2y42h0c"
  "testdataset_5y42h0c"
  "testdataset_10y42h0c"
  "testdataset_50y42h0c"
  "testdataset_100y42h0c"
  "testdataset_1000y42h0c"
  # --- test: downscaling-factor ablation (100y only) ---
  "testdataset_100y42h0c_ds2"
  "testdataset_100y42h0c_ds4"
  "testdataset_100y42h0c_ds16"
  # --- deferred: gabrielle (uncomment once its sim + patches exist) ---
  # "testdataset_gabrielle"
  # "testdataset_gabrielle_ds2"
  # "testdataset_gabrielle_ds4"
  # "testdataset_gabrielle_ds16"
)
# ================================================================================

rewrite_one() {
  local csv="$1"
  local out="${csv%.csv}_niwa.csv"
  echo "[rewrite] $(basename "${csv}") -> $(basename "${out}")"
  python tools/rewrite_index_paths.py --in_csv "${csv}" --out_csv "${out}"
}

for DS in "${DATASETS[@]}"; do
  DSDIR="${BASE_OUT}/${DS}"
  echo "============================================================"
  echo "[info] rewrite index paths for ${DS}"
  echo "============================================================"
  if [[ ! -d "${DSDIR}" ]]; then
    echo "[warn] dataset dir not found, skipping: ${DSDIR}"
    continue
  fi

  shopt -s nullglob
  found=0
  for csv in "${DSDIR}"/index*.csv; do
    case "${csv}" in
      *_niwa.csv) continue ;;   # skip already-rewritten outputs
    esac
    rewrite_one "${csv}"
    found=1
  done
  shopt -u nullglob

  if [[ "${found}" -eq 0 ]]; then
    echo "[warn] no index*.csv found in ${DSDIR}"
  fi
done

echo "[done] rewrite_index_paths finished for ${#DATASETS[@]} datasets."
