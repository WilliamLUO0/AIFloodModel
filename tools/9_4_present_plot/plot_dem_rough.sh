#!/bin/bash
#SBATCH --job-name=plot_dem_rough
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

python plot_dem_rough.py \
  --file-elev /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --elev-var Band1 \
  --file-rough /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
  --rough-var Band1 \
  --cpt /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt \
  --aoi /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --out-dem ./dem_aoi.png \
  --out-rough ./roughness_aoi.png \
  --dem-vmin -500 --dem-vmax 500 \
  --rough-qmax 99 \
  --rough-cmap viridis