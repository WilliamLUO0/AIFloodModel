#!/bin/bash
#SBATCH --job-name=precompute_split_stats_ds2_ds4_ds16
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

# Precompute split_stats for the new downscaling-factor datasets. Run AFTER
# make_patches, and BEFORE precompute_patch_interval_stats.
# Per dataset: h (scope=all) MUST run first (creates the split); u and v then
# --reuse_split_from it. The wet-scope h variant is intentionally NOT computed
# (unused by the yml; interval-stats only needs the split, which is identical).

BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

DATASETS=(
  "dataset_ds2_filtered_thr0p1_min25100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
)

for DS in "${DATASETS[@]}"; do
  DSDIR="${BASE_OUT}/${DS}"
  echo "============================================================"
  echo "[info] split_stats for ${DS}"
  echo "============================================================"

  # 1) h (asinh, scope=all) -> split_stats_h_asinh.json  (used by the yml, and
  #    provides the train/val split that u/v reuse)
  python tools/precompute_split_stats.py \
    --index_csv "${DSDIR}/index.csv" \
    --root "${DSDIR}" \
    --out_json "${DSDIR}/split_stats_h_asinh.json" \
    --target_var h \
    --aux_vars zs \
    --h_transform asinh \
    --h_tau 0.1 \
    --h_q_list 50,75,90,95,99 \
    --h_asinh_scale_scope all \
    --compute_h_flood_intervals \
    --h_flood_interval_thresholds 0.1,0.5,1.0 \
    --by scenario --val_ratio 0.2 --seed 61 \
    --bins 8192

  # 2/3) u and v split_stats are SKIPPED (h-only factor experiment; u/v models
  # won't be trained, and make_patches now generates h patches only). If u/v is
  # ever needed for a factor, first make its u/v patches, then run the u/v steps
  # (--target_var u/v --reuse_split_from split_stats_h_asinh.json --uv_tau 0.1
  # --compute_uv_intervals ...); see the PROVENANCE block below / git history.
done

echo "[done] split_stats finished for: ${DATASETS[*]}"

# ============================================================
# PROVENANCE — the ds8 (factor 8) recipe actually used (h-all + u + v).
# The wet-scope variant (split_stats_h_asinh_wet.json) was also produced back
# then but is unused; see git history for the full original file.
#   h:  --target_var h --aux_vars zs --h_transform asinh --h_tau 0.1
#       --h_q_list 50,75,90,95,99 --h_asinh_scale_scope all
#       --compute_h_flood_intervals --h_flood_interval_thresholds 0.1,0.5,1.0
#       --out_json .../dataset_ds8_filtered_thr0p1_min5100_full/split_stats_h_asinh.json
#   u:  --target_var u --reuse_split_from .../split_stats_h_asinh.json --uv_tau 0.1
#       --compute_uv_intervals --uv_interval_thresholds 0.1,0.5,1.0
#       --out_json .../split_stats_u.json
#   v:  same as u -> .../split_stats_v.json
#   (all with --by scenario --val_ratio 0.2 --seed 61 --bins 8192)
# ============================================================
