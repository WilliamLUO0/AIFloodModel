#!/bin/bash
#SBATCH --job-name=make_patches_dataset_ds8_filtered_thr0p1_min5100_full
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
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
export GDAL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export RASTERIO_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python make_patches.py \
  --var h u v \
  --scenarios 2y_6h_0c 2y_48h_0c 5y_6h_0c 5y_48h_0c 10y_6h_0c 10y_48h_0c 20y_6h_0c 20y_48h_0c 50y_6h_0c 50y_48h_0c 100y_6h_0c 100y_48h_0c 200y_6h_0c 200y_48h_0c 500y_6h_0c 500y_48h_0c 1000y_6h_0c 1000y_48h_0c \
  --dir-fine-template  "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx8/per_timestep_merged_filtered_thr0p1_min100" \
  --file-coarse-template "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx64/BGout_filtered_thr0p1_min5.nc" \
  --file-elev  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --file-rough  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
  --file-slope  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Slope_Deg.nc \
  --file-twi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/TWI.nc \
  --file-aspect-sin  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_SIN.nc \
  --file-aspect-cos  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_COS.nc \
  --aoi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --scale 8 --patch-coarse 64 \
  --filter-enable --filter-thresh 0.2 \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_filtered_thr0p1_min5100_full \
  --depth-eps 5e-5 --vel-eps 1e-5 \
  --clip-max-depth -1 --clip-max-vel-u -1 --clip-max-vel-v -1
