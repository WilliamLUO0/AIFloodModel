#!/bin/bash
#SBATCH --job-name=precompute_patch_interval_stats_ds2_ds4_ds16
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Precompute per-patch interval stats for the new datasets. Run AFTER
# precompute_split_stats (needs the split_stats_*.json for the train split).
# --split-stats-json is used ONLY to select the train split for the JSON summary
# (h uses split_stats_h_asinh.json, u/v use their own; all share the same split).
# The h output index_with_interval_stats_h.csv is the index_csv the yml loads.

BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

DATASETS=(
  "dataset_ds2_filtered_thr0p1_min25100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
)

for DS in "${DATASETS[@]}"; do
  DSDIR="${BASE_OUT}/${DS}"
  echo "============================================================"
  echo "[info] patch interval stats for ${DS}"
  echo "============================================================"

  # h (water-depth thresholds 0.1/0.5/1.0 m)
  python tools/precompute_patch_interval_stats.py \
    --index-csv "${DSDIR}/index.csv" \
    --root "${DSDIR}" \
    --out-csv "${DSDIR}/index_with_interval_stats_h.csv" \
    --out-json "${DSDIR}/patch_interval_summary_h.json" \
    --target-var h \
    --h-slight 0.1 \
    --h-severe 0.5 \
    --h-extreme 1.0 \
    --split-stats-json "${DSDIR}/split_stats_h_asinh.json" \
    --summary-split train

  # u/v interval stats SKIPPED (h-only; see the note in precompute_split_stats.sh).
done

echo "[done] patch interval stats finished for: ${DATASETS[*]}"

# ============================================================
# PROVENANCE — the ds8 (factor 8) recipe. h used split_stats_h_asinh_wet.json
# back then, but that only supplies the split (identical to the _asinh.json used
# above), so results match. See git history for the full original file.
#   h: --target-var h --h-slight 0.1 --h-severe 0.5 --h-extreme 1.0
#      --split-stats-json .../split_stats_h_asinh_wet.json --summary-split train
#      --out-csv .../index_with_interval_stats_h.csv
#      --out-json .../patch_interval_summary_h.json
#   u: --target-var u --split-stats-json .../split_stats_u.json --summary-split train
#      -> index_with_interval_stats_u.csv / patch_interval_summary_u.json
#   v: same as u -> ..._v.csv / ..._v.json
#   (dataset_ds8_filtered_thr0p1_min5100_full)
# ============================================================
