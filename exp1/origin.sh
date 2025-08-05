#!/bin/bash
#SBATCH --job-name=Gabrielle_flood
#SBATCH --time=72:00:00
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB

#When running on Mahuika
module load CUDA/11.2.0
module load netCDF/4.7.3-gimpi-2020a

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

srun ./BG_Flood BG_param.txt

