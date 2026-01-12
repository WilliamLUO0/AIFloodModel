#!/bin/bash
#SBATCH --job-name=pack_dataset
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /home/zluo784/00_nesi_projects/uoa04425_nobackup/zluo784/Exp1/AIFloodModel

module load zstd 2>/dev/null || true

tar -I "zstd -T${SLURM_CPUS_PER_TASK:-16} -10" -cf dataset.tar.zst dataset
sha256sum dataset.tar.zst > dataset.tar.zst.sha256
