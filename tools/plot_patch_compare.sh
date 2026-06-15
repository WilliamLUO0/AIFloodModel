#!/bin/bash
#SBATCH --job-name=plot_patch_compare
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
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

ROOT=/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel
CPT=/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt

# --- test set (held-out 100y_42h), 039 eval visualization ---
INDEX=${ROOT}/testdataset_100y42h0c/index.csv
VIS=${ROOT}/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/visualization

# run from repo root so `import plot_utils` resolves (tools/ added to sys.path)

# (1) single patch, full 12-panel (static features + flood + diff)
python tools/plot_patch_compare.py --panels full \
  --index-csv "${INDEX}" --vis-root "${VIS}" --elev-cpt "${CPT}" \
  --scale 8 --flood-thr 0.1 --flood-q 99 --abs-tol 0.02 --err-q 99 \
  --key h_100y_42h_0c_t0047_r012_c012_s8 \
  --out tools/patch_full_t0047_r012_c012.png

# (2) batch, compact 2x3 over all patches in the scenario (limit a few)
# python tools/plot_patch_compare.py --panels compact \
#   --index-csv "${INDEX}" --vis-root "${VIS}" --elev-cpt "${CPT}" \
#   --scale 8 --scenario 100y_42h_0c --limit 20 \
#   --out-dir tools/patch_compare_compact
