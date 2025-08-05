#!/bin/bash
#SBATCH --job-name      Gabrielle_flood_basin
#SBATCH --time          20:00:00
#SBATCH --account       uoa04425
#SBATCH --partition     genoa
#SBATCH --gpus-per-node A100:1
#SBATCH --ntasks        1
#SBATCH --cpus-per-task 8
#SBATCH --mem           20GB

#SBATCH --mail-user     williamlzh00@gmail.com
#SBATCH --mail-type     ALL,TIME_LIMIT_80

#SBATCH --output        logs/%x_%j_%A_%a.out
#SBATCH --error         logs/%x_%j_%A_%a.err


#When running on Mahuika
module load CUDA/11.2.0
module load netCDF/4.7.3-gimpi-2020a

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

srun ./BG_Flood BG_param.txt

