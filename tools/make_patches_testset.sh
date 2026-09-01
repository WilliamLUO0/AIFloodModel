#!/bin/bash
#SBATCH --job-name=make_patches_testset
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00
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
export GDAL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export RASTERIO_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Test-set patcher. Builds per-scenario test datasets for BOTH the factor-8 main
# model AND the factor 2/4/16 downscaling-factor ablation, in one run. Every build
# is explicit (no hidden factor-8 assumption). Each JOBS entry is:
#   "scenario:scale:coarse_dx:coarse_minpatch:out_suffix"
#     scale       -> downscaling factor; patch_fine = 64 * scale (128/256/512/1024)
#     coarse_dx   -> coarse-input resolution dir (dx16/dx32/dx64/dx128)
#     coarse_min  -> min_patch used when that coarse dx was isolated-cell filtered
#                    (dx16=25, dx32=10, dx64=5, dx128=2). Fine target is ALWAYS dx8 min100.
#     out_suffix  -> "" for factor 8 (testdataset_<tag>); "_ds2"/"_ds4"/"_ds16" otherwise,
#                    matching the dataset_ds<N> training-set naming.
# Output dir: testdataset_<tag><out_suffix>   (e.g. testdataset_100y42h0c_ds2).
# --var: factor 8 -> h u v (the main model has u/v variants); factor 2/4/16 -> h only
#        (there are no u/v factor models), to save runtime/space.
# --filter-thresh 0.000001 keeps ~ALL patches (incl. basin-edge) so the full map can be
# reassembled at test time (AOI-coverage filtering would hole the mosaic). This is the
# TEST setting and must stay 1e-6 for EVERY factor (NOT the 0.2 used for training).
# PREREQUISITE: run filter_isolated_flood_cells.sh for each (scenario, coarse_dx) first.

BASE_RESULTS="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results"
BASE_INPUT="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files"
BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

# fine (target) is the SAME dx8 filtered series for every factor.
FINE_TMPL="${BASE_RESULTS}/{scenario}/dx8/per_timestep_merged_filtered_thr0p1_min100"

# ===================== EDIT HERE: which (scenario, factor) builds this run =====================
JOBS=(
  # factor-8 main-model test suite, dx64 min5 -> dx8, no suffix.
  # 2y/100y/1000y were built earlier with the OLD broken topo -> rebuilt here;
  # 5y/10y/50y are new. gabrielle deferred (sim not finished).
  "2y_42h_0c:8:dx64:5:"
  "5y_42h_0c:8:dx64:5:"
  "10y_42h_0c:8:dx64:5:"
  "50y_42h_0c:8:dx64:5:"
  "100y_42h_0c:8:dx64:5:"
  "1000y_42h_0c:8:dx64:5:"
  # "gabrielle:8:dx64:5:"        # ds8 gabrielle test set — deferred, sim not finished
  # factor 2/4/16 downscaling-factor ablation on 100y_42h
  "100y_42h_0c:2:dx16:25:_ds2"
  "100y_42h_0c:4:dx32:10:_ds4"
  "100y_42h_0c:16:dx128:2:_ds16"
  # gabrielle ds2/4/16: COMMENTED OUT — gabrielle BG-Flood simulation not finished
  # yet. Uncomment once its dx8/dx16/dx32/dx128 sims + isolated-cell filters exist.
  # "gabrielle:2:dx16:25:_ds2"
  # "gabrielle:4:dx32:10:_ds4"
  # "gabrielle:16:dx128:2:_ds16"
)
# NOTE: topo repointed to Topo_Attrs_fixed (Slope/Aspect/TWI bug fix). All active
# jobs above are rebuilt/built with the fixed topo. The only deferred sets are the
# gabrielle ones (ds8 + ds2/4/16): uncomment them once gabrielle's BG-Flood sim +
# isolated-cell filters exist.
# =============================================================================================

for spec in "${JOBS[@]}"; do
  IFS=":" read -r SCEN SCALE CDX CMIN SUFFIX <<< "${spec}"
  TAG="${SCEN//_/}"                          # 5y_42h_0c -> 5y42h0c ; gabrielle -> gabrielle
  OUT_DIR="${BASE_OUT}/testdataset_${TAG}${SUFFIX}"
  COARSE_TMPL="${BASE_RESULTS}/{scenario}/${CDX}/BGout_filtered_thr0p1_min${CMIN}.nc"
  if [[ "${SCALE}" == "8" ]]; then VARS=(h u v); else VARS=(h); fi

  echo "============================================================"
  echo "[info] factor ${SCALE} | coarse=${CDX} (min${CMIN}) -> fine dx8 | patch_fine=$((64 * SCALE))"
  echo "[info] scenario ${SCEN} | vars: ${VARS[*]} | out_dir: ${OUT_DIR}"
  echo "============================================================"

  python tools/make_patches.py \
    --var "${VARS[@]}" \
    --scenarios "${SCEN}" \
    --fine-template  "${FINE_TMPL}" \
    --coarse-template "${COARSE_TMPL}" \
    --file-elev  "${BASE_INPUT}/Elevation.nc" \
    --file-rough  "${BASE_INPUT}/Roughness.nc" \
    --file-slope  "${BASE_INPUT}/Topo_Attrs_fixed/Slope_Deg.nc" \
    --file-twi  "${BASE_INPUT}/Topo_Attrs_fixed/TWI.nc" \
    --file-aspect-sin  "${BASE_INPUT}/Topo_Attrs_fixed/Aspect_SIN.nc" \
    --file-aspect-cos  "${BASE_INPUT}/Topo_Attrs_fixed/Aspect_COS.nc" \
    --aoi  "${BASE_INPUT}/Gisborne_basin.shp" \
    --scale "${SCALE}" --patch-coarse 64 \
    --filter-enable --filter-thresh 0.000001 \
    --out-dir "${OUT_DIR}" \
    --depth-eps 5e-5 --vel-eps 1e-5
done

echo "[done] testset patches finished for ${#JOBS[@]} (scenario,factor) builds."

# ============================================================
# PROVENANCE — earlier factor-8 test-set builds (already done), same command shape:
#   2y_42h_0c -> testdataset_2y42h0c ; 100y_42h_0c -> testdataset_100y42h0c ;
#   1000y_42h_0c -> testdataset_1000y42h0c ; gabrielle -> testdataset_gabrielle.
# ============================================================
