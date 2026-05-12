#!/bin/bash
#SBATCH --job-name=make_patches_dataset_ds8_h_debug
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
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

python make_patches.py \
  --var h \
  --scenarios 100y_48h_0c \
  --dir-fine-template  "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx8/per_timestep_merged" \
  --file-coarse-template "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx64/BGout.nc" \
  --file-elev  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --file-rough  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
  --file-slope  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Slope_Deg.nc \
  --file-twi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/TWI.nc \
  --file-aspect-sin  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_SIN.nc \
  --file-aspect-cos  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_COS.nc \
  --aoi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --scale 8 --patch-coarse 64 \
  --filter-enable --filter-thresh 0.2 \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_debug \
  --depth-eps 5e-5 --vel-eps 1e-5 \
  --clip-max-depth -1 --clip-max-vel-u -1 --clip-max-vel-v -1 \
  --debug-align \
  --debug-align-max-times 50 \
  --debug-align-max-patches 6

