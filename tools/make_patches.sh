#!/bin/bash
#SBATCH --job-name=make_patches_train_ds8_ds16_ds4_ds2
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
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

# Submit from the repo root (tools/ and logs/ must be reachable from CWD).
# NOTE: uses `python tools/make_patches.py` (the previous train run used
# `python make_patches.py`, i.e. it was launched from inside tools/).

BASE_RESULTS="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results"
BASE_INPUT="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files"
BASE_OUT="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

# fine (target) is the SAME dx8 filtered series for every factor.
FINE_TMPL="${BASE_RESULTS}/{scenario}/dx8/per_timestep_merged_filtered_thr0p1_min100"

# 18 design-rain training scenarios (8:2 scenario split happens later in the yml).
SCENARIOS=(
  2y_6h_0c   2y_48h_0c
  5y_6h_0c   5y_48h_0c
  10y_6h_0c  10y_48h_0c
  20y_6h_0c  20y_48h_0c
  50y_6h_0c  50y_48h_0c
  100y_6h_0c 100y_48h_0c
  200y_6h_0c 200y_48h_0c
  500y_6h_0c 500y_48h_0c
  1000y_6h_0c 1000y_48h_0c
)

# One entry per downscaling factor: "scale:coarse_dx:coarse_minpatch:ds_tag".
# patch_fine = 64 * scale  ->  factor 2=128, 4=256, 8=512, 16=1024.
# out-dir name: dataset_<ds_tag>_filtered_thr0p1_min<coarse_minpatch>100_full
#   ds8 -> dataset_ds8_filtered_thr0p1_min5100_full (the main-model dataset,
#   matches the dataroot in options/train/01_*.yml).
# VAR: ds8 = h u v (main model has u/v variants); ds2/4/16 = h only.
#
# RUNTIME: patch count ~ 1/patch_fine^2. h-only estimates: factor 16 ~1.5h,
# factor 4 ~6h, factor 2 ~23h; ds8 is h u v so ~3x its h cost (~9h). All four in
# one job is ~40h and can brush the 48h walltime -> ds8 (main dataset) runs FIRST
# so a timeout loses only a factor-ablation set, never ds8. To split across jobs,
# comment factors out and submit separately.
FACTORS=(
  "8:dx64:5:ds8"
  "16:dx128:2:ds16"
  "4:dx32:10:ds4"
  "2:dx16:25:ds2"
)

for spec in "${FACTORS[@]}"; do
  IFS=":" read -r SCALE CDX CMIN ODS <<< "${spec}"

  COARSE_TMPL="${BASE_RESULTS}/{scenario}/${CDX}/BGout_filtered_thr0p1_min${CMIN}.nc"
  OUT_DIR="${BASE_OUT}/dataset_${ODS}_filtered_thr0p1_min${CMIN}100_full"

  echo "============================================================"
  echo "[info] factor ${SCALE} | coarse=${CDX} (min${CMIN}) | patch_fine=$((64 * SCALE))"
  echo "[info] out_dir: ${OUT_DIR}"
  echo "============================================================"

  # ds8 (main model) trains h + u + v. The downscaling-factor ablation (ds2/4/16)
  # is h-only (no u/v factor models) to save runtime/space.
  if [[ "${SCALE}" == "8" ]]; then VARS=(h u v); else VARS=(h); fi

  python tools/make_patches.py \
    --var "${VARS[@]}" \
    --scenarios "${SCENARIOS[@]}" \
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
    --filter-enable --filter-thresh 0.2 \
    --out-dir "${OUT_DIR}" \
    --depth-eps 5e-5 --vel-eps 1e-5
done

echo "[done] make_patches finished for factors: ${FACTORS[*]}"

# ============================================================
# REFERENCE — standalone ds8 build command. ds8 is now the FIRST FACTORS entry
# above (so this script already builds it, h u v, with the fixed topo); this
# block is kept only as the explicit single-dataset invocation for reference.
# Coarse = dx64 min5, fine = dx8 min100, scale 8, patch_fine 512
#   -> dataset_ds8_filtered_thr0p1_min5100_full.
# ============================================================
#
# python make_patches.py \
#   --var h u v \
#   --scenarios 2y_6h_0c 2y_48h_0c 5y_6h_0c 5y_48h_0c 10y_6h_0c 10y_48h_0c 20y_6h_0c 20y_48h_0c 50y_6h_0c 50y_48h_0c 100y_6h_0c 100y_48h_0c 200y_6h_0c 200y_48h_0c 500y_6h_0c 500y_48h_0c 1000y_6h_0c 1000y_48h_0c \
#   --fine-template  "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx8/per_timestep_merged_filtered_thr0p1_min100" \
#   --coarse-template "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx64/BGout_filtered_thr0p1_min5.nc" \
#   --file-elev  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
#   --file-rough  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
#   --file-slope  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs_fixed/Slope_Deg.nc \
#   --file-twi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs_fixed/TWI.nc \
#   --file-aspect-sin  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs_fixed/Aspect_SIN.nc \
#   --file-aspect-cos  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs_fixed/Aspect_COS.nc \
#   --aoi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
#   --scale 8 --patch-coarse 64 \
#   --filter-enable --filter-thresh 0.2 \
#   --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_filtered_thr0p1_min5100_full \
#   --depth-eps 5e-5 --vel-eps 1e-5
