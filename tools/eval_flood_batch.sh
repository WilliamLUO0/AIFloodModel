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

# Variable to aggregate: h (water depth), u or v (x/y water velocity). For u/v the
# eval classifies wet/dry & bands by |v| (--abs) and the experiment / val-index
# names carry the _u / _v suffix. VAR=h reproduces the original behaviour exactly.
VAR=u                                        # <-- set to h / u / v
if [[ "$VAR" == "h" ]]; then EXPSUF=""; ABS=""; else EXPSUF="_${VAR}"; ABS="--abs"; fi

TRAIN_INDEX=index_with_interval_stats_${VAR}.csv  # val index (per training dataset)
TEST_INDEX=index.csv                              # test-dataset index (all vars)

# ===================== EDIT HERE: which experiments =====================
# Main model -> evaluated on val + the three designed 42h test scenarios + Gabrielle.
MAIN_EXPERIMENTS=(
  01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW${EXPSUF}
)

# Ablations / baselines -> evaluated on the VAL set only. Add or remove lines freely.
# NOTE: these are literal experiment names (NOT auto-suffixed by $VAR). They are
# h-only ablations, so when VAR=u/v leave this list EMPTY (or list _u/_v models).
VAL_ONLY_EXPERIMENTS=(

)

# Compute the coarse-upsample baseline too? It is MODEL-INDEPENDENT (just bicubic
# upsampling of the coarse map), so it is computed ONCE per dataset, not per model.
RUN_BASELINE=1
# =======================================================================

FILTERED_DS=dataset_ds8_filtered_thr0p1_min5100_full
RESULTS=$ROOT/results

# The VAL patches live in the training dataset; noFilter trains on the unfiltered one.
val_dataset_dir () {
  local exp="$1"
  # noFilter ablation trains on the UNFILTERED ds8.
  if [[ "$exp" == *noFilter* ]]; then echo "dataset_ds8"; return; fi
  # Otherwise the SRx{N} tag in the name fixes the downscaling factor -> its dataset
  # (SRx8<->ds8, SRx2<->ds2, SRx4<->ds4, SRx16<->ds16). To evaluate a factor model
  # just add it to MAIN_EXPERIMENTS / VAL_ONLY_EXPERIMENTS; the dataset comes from here.
  case "$exp" in
    *_SRx2_*)  echo "dataset_ds2_filtered_thr0p1_min25100_full" ;;
    *_SRx4_*)  echo "dataset_ds4_filtered_thr0p1_min10100_full" ;;
    *_SRx16_*) echo "dataset_ds16_filtered_thr0p1_min2100_full" ;;
    *_SRx8_*)  echo "dataset_ds8_filtered_thr0p1_min5100_full" ;;
    *)         echo "$FILTERED_DS" ;;
  esac
}

# ---- model eval on VAL (read *_patch_mean; --vis-root pins the val 20% subset) ----
eval_val () {
  local exp="$1"
  local ds; ds=$(val_dataset_dir "$exp")
  local idx="$ROOT/$ds/$TRAIN_INDEX"
  local vis="$RESULTS/${exp}_eval_val${SUF}/visualization"
  local out="$RESULTS/${exp}_eval_val${SUF}"
  if [[ ! -d "$vis" ]]; then echo "[skip][val] no vis-root: $vis"; return; fi
  echo "[eval][val] $exp"
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
# args: exp, eval_tag (test2y42h0c / test100y42h0c / test1000y42h0c / gabrielle), dataset_dir
eval_one_test () {
  local exp="$1" tag="$2" dsdir="$3"
  local idx="$ROOT/$dsdir/$TEST_INDEX"
  local vis="$RESULTS/${exp}_eval_${tag}${SUF}/visualization"
  local out="$RESULTS/${exp}_eval_${tag}${SUF}"
  if [[ ! -d "$vis" ]]; then echo "[skip][$tag] no vis-root: $vis"; return; fi
  echo "[eval][$tag] $exp"
  python tools/eval_flood.py \
    --index-csv "$idx" \
    --vis-root  "$vis" \
    --var "$VAR" $ABS \
    --out-json  "$out/eval_${tag}_summary.json" \
    --out-csv-time     "$out/eval_${tag}_time.csv" \
    --out-csv-scenario "$out/eval_${tag}_scenario.csv"
}

echo "==================== MODEL EVALS ===================="
if (( ${#MAIN_EXPERIMENTS[@]} )); then
  for exp in "${MAIN_EXPERIMENTS[@]}"; do
    eval_val "$exp"
    eval_one_test "$exp" test2y42h0c    testdataset_2y42h0c
    eval_one_test "$exp" test100y42h0c  testdataset_100y42h0c
    eval_one_test "$exp" test1000y42h0c testdataset_1000y42h0c
    eval_one_test "$exp" gabrielle      testdataset_gabrielle
  done
fi
if (( ${#VAL_ONLY_EXPERIMENTS[@]} )); then
  for exp in "${VAL_ONLY_EXPERIMENTS[@]}"; do
    eval_val "$exp"
  done
fi

# ==================== BASELINES (model-independent; once per dataset) ====================
# coarse_upsample = bicubic upsample of the coarse map -> identical for every model on a
# given dataset, so we run it once here rather than per experiment.
if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "==================== BASELINES ===================="
  BOUT="$RESULTS/_baselines"; mkdir -p "$BOUT"

  # VAL baselines: coarse_upsample is model-independent but DATASET-dependent, so run
  # it once per UNIQUE val dataset among all evaluated experiments (main + val-only),
  # using one of that dataset's val vis-roots to pin the same val subset. noFilter's
  # unfiltered dataset_ds8 is handled automatically (it is a distinct dataset).
  # Output is named per dataset: eval_val_baseline_<ds2|ds4|ds8|ds16><EXPSUF>.json.
  ALL_EVAL_EXPS=()
  if (( ${#MAIN_EXPERIMENTS[@]} )); then ALL_EVAL_EXPS+=("${MAIN_EXPERIMENTS[@]}"); fi
  if (( ${#VAL_ONLY_EXPERIMENTS[@]} )); then ALL_EVAL_EXPS+=("${VAL_ONLY_EXPERIMENTS[@]}"); fi
  declare -A _BASE_DONE=()
  if (( ${#ALL_EVAL_EXPS[@]} )); then
    for exp in "${ALL_EVAL_EXPS[@]}"; do
      ds=$(val_dataset_dir "$exp")
      if [[ -n "${_BASE_DONE[$ds]:-}" ]]; then continue; fi
      selvis="$RESULTS/${exp}_eval_val${SUF}/visualization"
      if [[ ! -d "$selvis" ]]; then
        echo "[skip][baseline][val] no val vis-root for selector $exp ($ds)"
        continue
      fi
      _BASE_DONE[$ds]=1
      dstag="${ds#dataset_}"; dstag="${dstag%%_*}"      # -> ds2 / ds4 / ds8 / ds16
      echo "[baseline][val] dataset=$ds selector=$exp"
      python tools/eval_flood.py \
        --index-csv "$ROOT/$ds/$TRAIN_INDEX" \
        --vis-root  "$selvis" \
        --var "$VAR" $ABS \
        --pred-source coarse_upsample \
        --out-json "$BOUT/eval_val_baseline_${dstag}${EXPSUF}.json"
    done
  else
    echo "[skip][baseline][val] no experiments listed -> no val-subset selector"
  fi

  # TEST baselines: no --vis-root (the test index IS the whole set).
  for pair in test2y42h0c:testdataset_2y42h0c \
              test100y42h0c:testdataset_100y42h0c \
              test1000y42h0c:testdataset_1000y42h0c \
              gabrielle:testdataset_gabrielle; do
    tag="${pair%%:*}"; dsdir="${pair##*:}"
    idx="$ROOT/$dsdir/$TEST_INDEX"
    if [[ -f "$idx" ]]; then
      echo "[baseline][$tag]"
      python tools/eval_flood.py \
        --index-csv "$idx" \
        --var "$VAR" $ABS \
        --pred-source coarse_upsample \
        --out-json "$BOUT/eval_${tag}_baseline${EXPSUF}.json" \
        --out-csv-time "$BOUT/eval_${tag}_baseline${EXPSUF}_time.csv"
    else
      echo "[skip][baseline][$tag] no index: $idx"
    fi
  done
fi

echo "==================== DONE ===================="
