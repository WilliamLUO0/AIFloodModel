#!/bin/bash
#SBATCH --job-name=eval_flood_testdataset_039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs1/%x_%j.out
#SBATCH --error=logs1/%x_%j.err

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

#--job-name=eval_flood_valset_039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val
#python tools/eval_flood.py \
#  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_filtered_thr0p1_min5100_full/index_with_interval_stats_h.csv \
#  --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/visualization \
#  --out-json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/eval_val_summary.json \
#  --out-csv-patch /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/eval_val_patch.csv \
#  --out-csv-time /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/eval_val_time.csv \
#  --out-csv-scenario /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/eval_val_scenario.csv
#
#python tools/eval_flood.py \
#  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_filtered_thr0p1_min5100_full/index_with_interval_stats_h.csv \
#  --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/visualization \
#  --pred-source coarse_upsample \
#  --out-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/eval_val_baseline.json

# --job-name=eval_flood_testdataset_039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c
python tools/eval_flood.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --vis-root  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/visualization \
  --out-json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/eval_test_summary.json \
  --out-csv-time /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/eval_test_time.csv \
  --out-csv-scenario /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/eval_test_scenario.csv

python tools/eval_flood.py \
  --index-csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/testdataset_100y42h0c/index.csv \
  --pred-source coarse_upsample \
  --out-json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/eval_test_baseline.json \
  --out-csv-time /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_test100y42h0c/eval_test_baseline_time.csv