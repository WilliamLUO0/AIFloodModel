#!/bin/bash
# Assemble per-patch predictions into full flood-map NetCDFs (NeSI / SLURM).
# PREREQUISITE: the matching test_flood_map eval jobs have written per-patch
# predictions under results/<evalname>/visualization/. Assembly only makes sense
# for the COMPLETE test sets (every patch present); the val split is a scattered
# 20% subset and cannot be stitched into full maps, so it is not listed here.
# Submit from the repo root:  sbatch tools/assemble_flood_batch.sh
#SBATCH --job-name=assemble_flood_batch
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/python310
set -u

export PYTHONUNBUFFERED=1

# ===================== cluster paths (NeSI) =====================
ROOT=/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel
SUF=""                 # NeSI eval-result dirs carry no _pbs suffix
TEST_INDEX=index.csv   # test-dataset index (all vars)

# ===================== EDIT HERE =====================
VAR=h                  # h (water depth) / u / v (x/y velocity)
SOURCE=pred            # pred = model output (needs vis-root) | gt = simulated fine (for comparison)
PER_TS_NPY=1           # 1 to also dump one assembled .npy per timestep

# Which test sets to assemble -- comment out any line you don't want.
# format:  eval_tag : test-dataset-dir : scenario-token (as in the patch core names)
TESTSETS=(
  # test2y42h0c:testdataset_2y42h0c:2y_42h_0c
  # test100y42h0c:testdataset_100y42h0c:100y_42h_0c
  # test1000y42h0c:testdataset_1000y42h0c:1000y_42h_0c
  gabrielle:testdataset_gabrielle:gabrielle
)
# =====================================================
if [[ "$VAR" == "h" ]]; then EXPSUF=""; else EXPSUF="_${VAR}"; fi
if [[ "$PER_TS_NPY" == "1" ]]; then NPYFLAG="--per-timestep-npy"; else NPYFLAG=""; fi

EXP=01_FMPFTV8_SRx8_Filter_InbaL1BCE_LW${EXPSUF}
RESULTS=$ROOT/results

for triple in "${TESTSETS[@]}"; do
  tag="${triple%%:*}"; rest="${triple#*:}"; dsdir="${rest%%:*}"; scen="${rest##*:}"
  idx="$ROOT/$dsdir/$TEST_INDEX"
  vis="$RESULTS/${EXP}_eval_${tag}${SUF}/visualization"
  out="$RESULTS/${EXP}_eval_${tag}${SUF}/assembled"
  if [[ ! -f "$idx" ]]; then echo "[skip][$tag] no index: $idx"; continue; fi
  if [[ "$SOURCE" == "pred" && ! -d "$vis" ]]; then echo "[skip][$tag] no vis-root: $vis"; continue; fi
  echo "[assemble][$tag] var=$VAR source=$SOURCE scenario=$scen"
  python tools/assemble_flood_map.py \
    --index-csv "$idx" \
    --vis-root  "$vis" \
    --var "$VAR" --scenario "$scen" \
    --source "$SOURCE" \
    --out-dir "$out" $NPYFLAG
done

echo "==================== DONE ===================="
