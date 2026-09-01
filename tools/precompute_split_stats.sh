#!/bin/bash
#SBATCH --job-name=precompute_split_stats_train_all
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

# Precompute split_stats for the FOUR training datasets. Run AFTER make_patches,
# and BEFORE precompute_patch_interval_stats. Test sets do NOT need split_stats
# (they are not split into train/val; eval uses the training dataset's stats).
# Per dataset: h (asinh, scope=all) runs first and CREATES the train/val split;
# ds8 additionally runs u and v with --reuse_split_from so they share that split.
# ds2/4/16 are h-only. (The wet-scope h variant is unused and not computed.)

BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

DATASETS=(
  "dataset_ds8_filtered_thr0p1_min5100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds2_filtered_thr0p1_min25100_full"
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

  # 2/3) ds8 only: u and v reuse h's train/val split. ds2/4/16 are h-only -> skip.
  if [[ "$DS" == *ds8* ]]; then
    for UV in u v; do
      echo "[info] split_stats ${UV} for ${DS}"
      python tools/precompute_split_stats.py \
        --index_csv "${DSDIR}/index.csv" \
        --root "${DSDIR}" \
        --out_json "${DSDIR}/split_stats_${UV}.json" \
        --target_var "${UV}" \
        --reuse_split_from "${DSDIR}/split_stats_h_asinh.json" \
        --uv_tau 0.1 \
        --compute_uv_intervals \
        --uv_interval_thresholds 0.1,0.5,1.0 \
        --by scenario --val_ratio 0.2 --seed 61 \
        --bins 8192
    done
  fi
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
