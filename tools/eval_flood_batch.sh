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
#SBATCH --output=logs1/%x_%j.out
#SBATCH --error=logs1/%x_%j.err

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
TRAIN_INDEX=index_with_interval_stats_h.csv  # val index (per training dataset)
TEST_INDEX=index.csv                         # test-dataset index

# ===================== EDIT HERE: which experiments =====================
# Main model -> evaluated on val + the three designed 42h test scenarios + Gabrielle.
MAIN_EXPERIMENTS=(
  01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW
)

# Ablations / baselines -> evaluated on the VAL set only. Add or remove lines freely.
VAL_ONLY_EXPERIMENTS=(
  02_HeUNet_SRx8_Filter_InbaL1BCE_LW
  02_SwinFlood_SRx8_Filter_InbaL1BCE_LW
  02_RSwinUNet_SRx8_Filter_InbaL1BCE_LW
  03_FMPFTV8_Abl_static_noDEM_SRx8_Filter_InbaL1BCE_LW
  04_FMPFTV8_Abl_noAoiGate_SRx8_Filter_InbaL1BCE_LW
  05_FMPFTV8Abl_noUshapeDFuse_SRx8_Filter_InbaL1BCE_LW
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
  if [[ "$exp" == *noFilter* ]]; then echo "dataset_ds8"; else echo "$FILTERED_DS"; fi
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

  # VAL baseline: needs a val vis-root to pin the same 20% subset the models saw.
  # Any evaluated experiment's val vis-root works (same split); we use the first main one.
  if (( ${#MAIN_EXPERIMENTS[@]} )); then
    SEL="${MAIN_EXPERIMENTS[0]}"
    SELVIS="$RESULTS/${SEL}_eval_val${SUF}/visualization"
    if [[ -d "$SELVIS" ]]; then
      echo "[baseline][val] selector=$SEL"
      python tools/eval_flood.py \
        --index-csv "$ROOT/$FILTERED_DS/$TRAIN_INDEX" \
        --vis-root  "$SELVIS" \
        --pred-source coarse_upsample \
        --out-json "$BOUT/eval_val_baseline.json"
    else
      echo "[skip][baseline][val] no selector vis-root: $SELVIS"
    fi
  else
    echo "[skip][baseline][val] MAIN_EXPERIMENTS empty -> no val-subset selector"
  fi
  # NOTE: noFilter trains on dataset_ds8, so its val baseline differs; add a separate
  # run against $ROOT/dataset_ds8/$TRAIN_INDEX if you report a noFilter val baseline.

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
        --pred-source coarse_upsample \
        --out-json "$BOUT/eval_${tag}_baseline.json" \
        --out-csv-time "$BOUT/eval_${tag}_baseline_time.csv"
    else
      echo "[skip][baseline][$tag] no index: $idx"
    fi
  done
fi

echo "==================== DONE ===================="
