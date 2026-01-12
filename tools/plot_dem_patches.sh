#!/bin/bash
#SBATCH --job-name=plot_dem_patches
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
conda activate /nesi/project/uoa04425/zluo784/envs/pft39

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python tools/plot_dem_patches.py \
  --file-elev /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --cpt /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt \
  --aoi /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --scale 16 --patch-coarse 64 \
  --label-every 1 --label-font 7 \
  --vmin -500 --vmax 500 \
  --out /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/dem_patches.png

python tools/plot_dem_patches.py --file-elev /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc --cpt /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt --aoi /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp --scale 16 --patch-coarse 64 --label-every 1 --label-font 7 --vmin -500 --vmax 500 --out /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/tools/dem_patches.png
