#!/bin/bash
#SBATCH --job-name=filter_isolated_flood_cells
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
# NOTE: --mem=128G covers the WORST case (dx16 single-file load; see the OOM history).
# dx8 (per-timestep dir mode) and the coarser single files (dx32/dx64/dx128) need far
# less; if RESOLUTIONS excludes dx16 you can drop --mem to ~32-64G for a faster queue.

BASE_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results"

THRESHOLD=0.1
CONNECTIVITY=8

# min_patch_size per resolution. Rationale: BG-Flood devs' per-resolution guidance
# (conservative upper end), ~area-preserving (~6400 m^2) at the fine end, flattening
# to a single-cell-noise floor at the coarse end. Monotone ladder below.
# dx8 is the FINE target (adaptive mesh) -> the tool first MERGES dx8/BGout.nc onto
#   the finest grid (per_timestep_merged), then filters (dir mode). No separate
#   merge step is needed -- merge_multiscale is folded into filter (--merge-from).
# dx16/dx32/dx64/dx128 are COARSE inputs -> single BGout.nc.
declare -A MINPATCH=( [dx8]=100 [dx16]=25 [dx32]=10 [dx64]=5 [dx128]=2 )

# ===================== EDIT HERE =====================
# Explicit per-scenario resolution list (NO cartesian product, because different
# scenarios need different resolutions). Each entry is "scenario:res1 res2 ...".
#   - factor-8 test sets (5y/10y/50y_42h): fine target dx8 + its coarse dx64.
#   - factor 2/4/16 ablation on 100y_42h / gabrielle: coarse inputs dx16/dx32/dx128
#     (factor 2<-dx16, factor 4<-dx32, factor 16<-dx128). Their dx8 fine target was
#     already filtered with the earlier 2y/100y/1000y/gabrielle run, so it is NOT
#     repeated here.
JOBS=(
  "5y_42h_0c:dx8 dx64"
  "10y_42h_0c:dx8 dx64"
  "50y_42h_0c:dx8 dx64"
  "100y_42h_0c:dx16 dx32 dx128"
  # "gabrielle:dx16 dx32 dx128"
)
# ====================================================
# Already filtered previously (do NOT re-add):
#   - training scenarios (18): dx8/dx16/dx32/dx64/dx128
#   - test scenarios 2y_42h_0c / 100y_42h_0c / 1000y_42h_0c / gabrielle: dx8 + dx64

filter_one () {
  # $1 = scenario, $2 = resolution tag (dx8/dx16/dx32/dx64/dx128)
  local scen="$1" res="$2"
  local minp="${MINPATCH[$res]}"

  if [[ "$res" == "dx8" ]]; then
    # fine target: dx8 is an ADAPTIVE-MESH run. Merge BGout.nc onto the finest
    # grid (-> per_timestep_merged) and filter, in ONE step -- the merge is folded
    # into the filter tool (--merge-from), so only dx8/BGout.nc must exist on disk
    # (no separate merge_multiscale run needed anymore).
    local finedir="${BASE_DIR}/${scen}/dx8"
    local bgout="${finedir}/BGout.nc"
    local indir="${finedir}/per_timestep_merged"
    local outdir="${finedir}/per_timestep_merged_filtered_thr0p1_min${minp}"
    if [[ -f "${bgout}" ]]; then
      echo "[info] ${scen} ${res} (min${minp}, merge+filter)"
      echo "       ${bgout} -> ${indir} -> ${outdir}"
      python tools/filter_isolated_flood_cells.py \
        --merge-from "${bgout}" \
        --input-dir "${indir}" --output-dir "${outdir}" \
        --threshold "${THRESHOLD}" --min-patch-size "${minp}" \
        --connectivity "${CONNECTIVITY}" --overwrite
    else
      echo "[warn] ${scen} ${res}: dx8 BGout.nc not found: ${bgout}"
    fi
  else
    # coarse input: single BGout.nc
    local infile="${BASE_DIR}/${scen}/${res}/BGout.nc"
    local outfile="${BASE_DIR}/${scen}/${res}/BGout_filtered_thr0p1_min${minp}.nc"
    if [[ -f "${infile}" ]]; then
      echo "[info] ${scen} ${res} (min${minp}, single file)"
      echo "       ${infile} -> ${outfile}"
      python tools/filter_isolated_flood_cells.py \
        --input-nc "${infile}" --output-nc "${outfile}" \
        --threshold "${THRESHOLD}" --min-patch-size "${minp}" \
        --connectivity "${CONNECTIVITY}" --overwrite
    else
      echo "[warn] ${scen} ${res}: input file not found: ${infile}"
    fi
  fi
}

for job in "${JOBS[@]}"; do
  SCENARIO="${job%%:*}"
  read -r -a RESLIST <<< "${job#*:}"
  echo "============================================================"
  echo "[info] scenario: ${SCENARIO} | resolutions: ${RESLIST[*]}"
  echo "============================================================"
  for RES in "${RESLIST[@]}"; do
    filter_one "${SCENARIO}" "${RES}"
  done
done

echo "[done] All scenarios/resolutions processed."
