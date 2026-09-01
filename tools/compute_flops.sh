#!/bin/bash
#SBATCH --job-name=compute_flops
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-ONLY: computing #Params + FLOPs is a single forward pass on CPU -> NO GPU
# (no --gres), so this queues fast. It's fast enough (~minutes) that you can also
# just run `python tools/compute_flops.py` directly on a LOGIN node.
#
# PREREQUISITE: fvcore must be importable in the env. NeSI compute nodes have no
# internet, so install it ONCE on a login node first:
#     conda activate /nesi/project/uoa04425/zluo784/envs/pft39
#     pip install fvcore
# Without fvcore the script still prints #Params and just skips the MACs column.

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"
set +u
conda activate /nesi/project/uoa04425/zluo784/envs/pft39
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python tools/compute_flops.py --csv tools/model_flops.csv
