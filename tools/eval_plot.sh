#!/bin/bash
#SBATCH --job-name=eval_visualize_plot_007_FMPFT_SRx16_scratch_eval_test_100y42h0c
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
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

python tools/eval_plot.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/visualization \
  --out-dir   /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/plot_results \
  --elev-cpt  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt \
  --elev-vmin -500 --elev-vmax 500 \
  --flood-thr 0.05 \
  --flood-vmax-mode q95 --flood-q 95 \
  --abs-tol 0.01 \
  --diff-vabs-mode q95abs --diff-q 95

# python tools/eval_plot.py --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/visualization --elev-cpt  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/wiki-france.cpt --key h_100y_42h_0c_t0047_r000_c006_s16 --out /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/007_FMPFT_SRx16_scratch_eval_test_100y42h0c/key_debug/h_100y_42h_0c_t0047_r000_c006_s16_QC.png --elev-vmin -500 --elev-vmax 500 --flood-thr 0.05 --flood-vmax-mode q95 --flood-q 95 --abs-tol 0.01 --diff-vabs-mode q95abs --diff-q 95
