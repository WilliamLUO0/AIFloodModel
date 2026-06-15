#!/bin/bash
#SBATCH --job-name=plot_static_features
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
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

INPUT=/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files
TOPO=${INPUT}/Topo_Attrs
CPT=/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt
AOI=${INPUT}/Gisborne_basin.shp

# run from repo root so `from plot_utils import ...` resolves (tools/ is added to sys.path)

# (1) all six static features as one combined panel + AOI
python tools/plot_static_features.py \
  --file-elev "${INPUT}/Elevation.nc" --cpt "${CPT}" \
  --file-rough "${INPUT}/Roughness.nc" \
  --file-slope "${TOPO}/Slope_Deg.nc" \
  --file-twi "${TOPO}/TWI.nc" \
  --file-aspect-sin "${TOPO}/Aspect_SIN.nc" \
  --file-aspect-cos "${TOPO}/Aspect_COS.nc" \
  --aoi "${AOI}" --dem-vmin -500 --dem-vmax 500 \
  --features all --mode panel \
  --out tools/static_features_panel.png

# (2) individual feature pngs (same args, --mode individual)
# python tools/plot_static_features.py \
#   --file-elev "${INPUT}/Elevation.nc" --cpt "${CPT}" \
#   --file-rough "${INPUT}/Roughness.nc" --file-slope "${TOPO}/Slope_Deg.nc" \
#   --file-twi "${TOPO}/TWI.nc" --file-aspect-sin "${TOPO}/Aspect_SIN.nc" \
#   --file-aspect-cos "${TOPO}/Aspect_COS.nc" --aoi "${AOI}" \
#   --features all --mode individual --out-dir tools/static_features

# (3) DEM only, with the 8x training patch grid overlaid
# python tools/plot_static_features.py --file-elev "${INPUT}/Elevation.nc" --cpt "${CPT}" \
#   --aoi "${AOI}" --features dem --mode individual --out-dir tools/static_features \
#   --patch-grid --scale 8 --patch-coarse 64 --dem-vmin -500 --dem-vmax 500
