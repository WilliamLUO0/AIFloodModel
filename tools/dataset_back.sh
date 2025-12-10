#!/bin/bash
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --job-name=dataset_backup
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/copy_dataset_%j.out
#SBATCH --error=logs/copy_dataset_%j.err

BASE_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"

cd "$BASE_DIR"
echo "[$(date)] Start rsync"
echo "Working directory: $(pwd)"

rsync -a --partial --info=progress2 --bwlimit=50M \
  "$BASE_DIR/dataset/" "$BASE_DIR/dataset_backup/"

echo "[$(date)] rsync done"
