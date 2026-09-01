#!/bin/bash
# Batch eval_flood.py aggregation (NeSI / SLURM).
# Assembles the saved per-patch predictions into GLOBAL / patch_mean metrics for
# many experiments in one job. PREREQUISITE: the matching test_flood_map jobs
# (options/test/<exp>_eval_*.sh) must already have written predictions under
# results/<evalname>/visualization/. Experiments with no vis-root are skipped.
# Submit from the repo root:  sbatch tools/eval_flood_batch.sh
#SBATCH --job-name=eval_flood_batch
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
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

# ===================== cluster paths (NeSI) =====================
ROOT=/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel
SUF=""                                       # NeSI eval-result dirs carry no _pbs suffix
RESULTS=$ROOT/results
BOUT="$RESULTS/_baselines"

# Variable to aggregate: h (water depth), u or v (x/y water velocity). For u/v the
# eval classifies wet/dry & bands by |v| (--abs) and the experiment / val-index names
# carry the _u / _v suffix. VAR=h reproduces the original behaviour exactly.
# Experiment names in the job lists below are BASE (h) names; $EXPSUF is appended
# automatically, so the SAME lists work for h/u/v -- BUT the factor (06_) models are
# h-only, so comment their lines out for VAR=u/v (they have no _u/_v checkpoints).
VAR=h                                        # <-- set to h / u / v
if [[ "$VAR" == "h" ]]; then EXPSUF=""; ABS=""; else EXPSUF="_${VAR}"; ABS="--abs"; fi

TRAIN_INDEX=index_with_interval_stats_${VAR}.csv  # val index (per training dataset)
TEST_INDEX=index.csv                              # test-dataset index (all vars)

# ===================== EDIT HERE: explicit eval jobs =====================
# Fully explicit -- each line pins WHICH model reads WHICH dataset; no factor is
# inferred from the name. Comment out anything already aggregated.
#
# VAL jobs  -> "exp_base:train_dataset_dir"
#   val patches live in the training dataset; read as *_patch_mean, pinned to the
#   seed-61 val 20% subset via --vis-root.
VAL_JOBS=(
  # --- 01 main model (h; u/v via VAR knob) ---
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  # --- 02 baselines (matched-loss + native-loss variants) ---
  "02_FMPFTV7_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_HeUNet_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_HeUNet_SRx8_Filter_InbaL2:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_SwinFlood_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_SwinFlood_SRx8_Filter_InbaL2:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "02_RSwinUNet_SRx8_Filter_InbaL1:dataset_ds8_filtered_thr0p1_min5100_full"
  # --- 03 input ablations ---
  "03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_coarse_zsOnly_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_demOnly_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_noAspect_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_noDEM_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_noRough_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_noSlope_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "03_FMPFTV8_Abl_static_noTWI_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  # --- 04 loss / mechanism ablations (noFilter trains on the UNFILTERED ds8) ---
  "04_FMPFTV8_Abl_noAoiGate_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "04_FMPFTV8_Abl_noBCE_SRx8_Filter_InbaL1_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "04_FMPFTV8_Abl_noFilter_SRx8_InbaL1BCE_LW:dataset_ds8"
  "04_FMPFTV8_Abl_noIBS_SRx8_Filter_L1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "04_FMPFTV8_Abl_noResidual_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  # --- 05 architecture ablations ---
  "05_FMPFTV8Abl_convBottleneck_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "05_FMPFTV8Abl_noUshape_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "05_FMPFTV8Abl_noUshapeDFuse_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "05_FMPFTV9_Abl_deepSkip_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "05_FMPFTV9_Abl_deepSkipFuse_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  # --- 06 downscaling-factor models (h-only; comment out for VAR=u/v) ---
  "06_FMPFTV8_SRx2_Filter_InbaL1BCE_LW:dataset_ds2_filtered_thr0p1_min25100_full"
  "06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW:dataset_ds4_filtered_thr0p1_min10100_full"
  "06_FMPFTV8_SRx16_Filter_InbaL1BCE_LW:dataset_ds16_filtered_thr0p1_min2100_full"
)

# TEST jobs -> "exp_base:eval_tag:test_dataset_dir"
#   complete maps; read as *_global. Each factor points at ITS OWN test dataset
#   (testdataset_<tag>_ds<N>), so nothing is inferred.
TEST_JOBS=(
  # --- 01 main model -> ALL 7 test sets (factor-8) ---
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test2y42h0c:testdataset_2y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test5y42h0c:testdataset_5y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test10y42h0c:testdataset_10y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test50y42h0c:testdataset_50y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:test1000y42h0c:testdataset_1000y42h0c"
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  # --- 02 baselines -> 100y + gabrielle ---
  "02_FMPFTV7_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "02_FMPFTV7_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "02_HeUNet_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "02_HeUNet_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "02_HeUNet_SRx8_Filter_InbaL2:test100y42h0c:testdataset_100y42h0c"
  "02_HeUNet_SRx8_Filter_InbaL2:gabrielle:testdataset_gabrielle"
  "02_SwinFlood_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "02_SwinFlood_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "02_SwinFlood_SRx8_Filter_InbaL2:test100y42h0c:testdataset_100y42h0c"
  "02_SwinFlood_SRx8_Filter_InbaL2:gabrielle:testdataset_gabrielle"
  "02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "02_RSwinUNet_SRx8_Filter_InbaL1:test100y42h0c:testdataset_100y42h0c"
  "02_RSwinUNet_SRx8_Filter_InbaL1:gabrielle:testdataset_gabrielle"
  # --- 03 input ablations -> 100y + gabrielle ---
  "03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_coarse_demOnly_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_coarse_zsOnly_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_coarse_zsOnly_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_demOnly_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_demOnly_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_noAspect_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_noAspect_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_noDEM_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_noDEM_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_noRough_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_noRough_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_noSlope_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_noSlope_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "03_FMPFTV8_Abl_static_noTWI_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "03_FMPFTV8_Abl_static_noTWI_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  # --- 04 loss / mechanism ablations -> 100y + gabrielle ---
  "04_FMPFTV8_Abl_noAoiGate_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "04_FMPFTV8_Abl_noAoiGate_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "04_FMPFTV8_Abl_noBCE_SRx8_Filter_InbaL1_LW:test100y42h0c:testdataset_100y42h0c"
  "04_FMPFTV8_Abl_noBCE_SRx8_Filter_InbaL1_LW:gabrielle:testdataset_gabrielle"
  "04_FMPFTV8_Abl_noFilter_SRx8_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "04_FMPFTV8_Abl_noFilter_SRx8_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "04_FMPFTV8_Abl_noIBS_SRx8_Filter_L1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "04_FMPFTV8_Abl_noIBS_SRx8_Filter_L1BCE_LW:gabrielle:testdataset_gabrielle"
  "04_FMPFTV8_Abl_noResidual_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "04_FMPFTV8_Abl_noResidual_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  # --- 05 architecture ablations -> 100y + gabrielle ---
  "05_FMPFTV8Abl_convBottleneck_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "05_FMPFTV8Abl_convBottleneck_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "05_FMPFTV8Abl_noUshape_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "05_FMPFTV8Abl_noUshape_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "05_FMPFTV8Abl_noUshapeDFuse_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "05_FMPFTV8Abl_noUshapeDFuse_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "05_FMPFTV9_Abl_deepSkip_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "05_FMPFTV9_Abl_deepSkip_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  "05_FMPFTV9_Abl_deepSkipFuse_SRx8_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c"
  "05_FMPFTV9_Abl_deepSkipFuse_SRx8_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle"
  # --- 06 downscaling-factor models -> 100y + gabrielle on their OWN factor test sets ---
  "06_FMPFTV8_SRx2_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c_ds2"
  "06_FMPFTV8_SRx2_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle_ds2"
  "06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c_ds4"
  "06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle_ds4"
  "06_FMPFTV8_SRx16_Filter_InbaL1BCE_LW:test100y42h0c:testdataset_100y42h0c_ds16"
  "06_FMPFTV8_SRx16_Filter_InbaL1BCE_LW:gabrielle:testdataset_gabrielle_ds16"
)

# coarse-upsample baselines (bicubic upsample of the coarse map). MODEL-INDEPENDENT ->
# ONE per dataset. These are EXPLICIT lists (NOT derived from the eval jobs above): list
# ONLY datasets whose baseline is not already in results/_baselines/. Output naming:
# eval_val_baseline_<dsN>{EXPSUF}.json  and  eval_<tag>_baseline_<dsN>{EXPSUF}.json.
# NOTE: pre-existing ds8 baselines on NeSI use OLD names (no _dsN / no h-suffix, e.g.
# eval_val_baseline.json, eval_test100y42h0c_baseline.json, gabrielle_baseline.json) --
# leave those out here; anything you (re)run through these lists gets the new _dsN name.
RUN_BASELINES=1
# VAL baseline -> "selector_exp_base:train_dataset_dir" (selector's val vis pins the subset)
VAL_BASELINE_JOBS=(
  "01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW:dataset_ds8_filtered_thr0p1_min5100_full"
  "04_FMPFTV8_Abl_noFilter_SRx8_InbaL1BCE_LW:dataset_ds8"
  "06_FMPFTV8_SRx2_Filter_InbaL1BCE_LW:dataset_ds2_filtered_thr0p1_min25100_full"
  "06_FMPFTV8_SRx4_Filter_InbaL1BCE_LW:dataset_ds4_filtered_thr0p1_min10100_full"
  "06_FMPFTV8_SRx16_Filter_InbaL1BCE_LW:dataset_ds16_filtered_thr0p1_min2100_full"
)
# TEST baseline -> "eval_tag:test_dataset_dir"
TEST_BASELINE_JOBS=(
  "test2y42h0c:testdataset_2y42h0c"
  "test5y42h0c:testdataset_5y42h0c"
  "test10y42h0c:testdataset_10y42h0c"
  "test50y42h0c:testdataset_50y42h0c"
  "test100y42h0c:testdataset_100y42h0c"
  "test1000y42h0c:testdataset_1000y42h0c"
  "gabrielle:testdataset_gabrielle"
  "test100y42h0c:testdataset_100y42h0c_ds2"
  "gabrielle:testdataset_gabrielle_ds2"
  "test100y42h0c:testdataset_100y42h0c_ds4"
  "gabrielle:testdataset_gabrielle_ds4"
  "test100y42h0c:testdataset_100y42h0c_ds16"
  "gabrielle:testdataset_gabrielle_ds16"
)
# =======================================================================

# dataset dir (dataset_ds8_..._full or testdataset_..._ds16) -> factor tag ds2/4/8/16.
factor_tag () {
  case "$1" in
    *ds2_*|*_ds2)   echo ds2  ;;
    *ds4_*|*_ds4)   echo ds4  ;;
    *ds16_*|*_ds16) echo ds16 ;;
    *)              echo ds8  ;;   # factor-8 test dirs have no suffix; ds8 training dir
  esac
}

# ---- model eval on VAL (read *_patch_mean; --vis-root pins the val 20% subset) ----
eval_val () {
  local exp="$1" ds="$2"
  local idx="$ROOT/$ds/$TRAIN_INDEX"
  local vis="$RESULTS/${exp}_eval_val${SUF}/visualization"
  local out="$RESULTS/${exp}_eval_val${SUF}"
  if [[ ! -d "$vis" ]]; then echo "[skip][val] no vis-root: $vis"; return; fi
  echo "[eval][val] $exp  (ds=$ds)"
  python tools/eval_flood.py \
    --index-csv "$idx" \
    --vis-root  "$vis" \
    --var "$VAR" $ABS \
    --out-json  "$out/eval_val_summary.json" \
    --out-csv-patch    "$out/eval_val_patch.csv" \
    --out-csv-time     "$out/eval_val_time.csv" \
    --out-csv-scenario "$out/eval_val_scenario.csv"
}

# ---- model eval on a TEST set (read *_global; complete maps) ----
eval_one_test () {
  local exp="$1" tag="$2" dsdir="$3"
  local idx="$ROOT/$dsdir/$TEST_INDEX"
  local vis="$RESULTS/${exp}_eval_${tag}${SUF}/visualization"
  local out="$RESULTS/${exp}_eval_${tag}${SUF}"
  if [[ ! -d "$vis" ]]; then echo "[skip][$tag] no vis-root: $vis"; return; fi
  echo "[eval][$tag] $exp  (ds=$dsdir)"
  python tools/eval_flood.py \
    --index-csv "$idx" \
    --vis-root  "$vis" \
    --var "$VAR" $ABS \
    --out-json  "$out/eval_${tag}_summary.json" \
    --out-csv-time     "$out/eval_${tag}_time.csv" \
    --out-csv-scenario "$out/eval_${tag}_scenario.csv"
}

echo "==================== MODEL EVALS ===================="
for job in "${VAL_JOBS[@]}"; do
  eval_val "${job%%:*}${EXPSUF}" "${job#*:}"
done
for job in "${TEST_JOBS[@]}"; do
  IFS=":" read -r ebase tag dsdir <<< "$job"
  eval_one_test "${ebase}${EXPSUF}" "$tag" "$dsdir"
done

# ==================== BASELINES (model-independent; explicit lists) ====================
if [[ "$RUN_BASELINES" == "1" ]]; then
  mkdir -p "$BOUT"
  echo "==================== VAL BASELINES ===================="
  declare -A _VBASE_DONE=()
  for job in "${VAL_BASELINE_JOBS[@]}"; do
    ebase="${job%%:*}"; ds="${job#*:}"
    if [[ -n "${_VBASE_DONE[$ds]:-}" ]]; then continue; fi
    selvis="$RESULTS/${ebase}${EXPSUF}_eval_val${SUF}/visualization"
    if [[ ! -d "$selvis" ]]; then
      echo "[skip][baseline][val] no val vis-root for selector ${ebase}${EXPSUF} ($ds)"; continue
    fi
    _VBASE_DONE[$ds]=1
    dstag=$(factor_tag "$ds")
    echo "[baseline][val] dataset=$ds ($dstag) selector=${ebase}${EXPSUF}"
    python tools/eval_flood.py \
      --index-csv "$ROOT/$ds/$TRAIN_INDEX" \
      --vis-root  "$selvis" \
      --var "$VAR" $ABS \
      --pred-source coarse_upsample \
      --out-json "$BOUT/eval_val_baseline_${dstag}${EXPSUF}.json"
  done

  echo "==================== TEST BASELINES ===================="
  declare -A _TBASE_DONE=()
  for job in "${TEST_BASELINE_JOBS[@]}"; do
    IFS=":" read -r tag dsdir <<< "$job"
    if [[ -n "${_TBASE_DONE[$dsdir]:-}" ]]; then continue; fi
    idx="$ROOT/$dsdir/$TEST_INDEX"
    if [[ ! -f "$idx" ]]; then echo "[skip][baseline][$tag] no index: $idx"; continue; fi
    _TBASE_DONE[$dsdir]=1
    dstag=$(factor_tag "$dsdir")
    echo "[baseline][$tag] dataset=$dsdir ($dstag)"
    python tools/eval_flood.py \
      --index-csv "$idx" \
      --var "$VAR" $ABS \
      --pred-source coarse_upsample \
      --out-json "$BOUT/eval_${tag}_baseline_${dstag}${EXPSUF}.json" \
      --out-csv-time "$BOUT/eval_${tag}_baseline_${dstag}${EXPSUF}_time.csv"
  done
fi

echo "==================== DONE ===================="
