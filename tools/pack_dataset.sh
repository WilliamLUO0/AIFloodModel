#!/bin/bash
#SBATCH --job-name=pack_dataset_ds2_ds4_ds16
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs1/%x_%j.out
#SBATCH --error=logs1/%x_%j.err

set -euo pipefail

# Pack the h-only factor-experiment datasets (ds2/ds4/ds16) to ship NeSI -> NIWA.
# Run AFTER rewrite_index_paths.py so each dataset already contains its
# index_with_interval_stats_h_niwa.csv (NIWA-path index used by the *_pbs.yml).
# ds8 was packed earlier into archive2 (see PROVENANCE below) -- do NOT repack it.

BASE="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"
OUT_DIR="${BASE}/archive3"

DATASETS=(
  "dataset_ds2_filtered_thr0p1_min25100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
)

mkdir -p "$OUT_DIR"

for DS in "${DATASETS[@]}"; do
  SRC_DIR="${BASE}/${DS}"
  ARCHIVE_NAME="${DS}.tar.gz"

  echo "============================================================"
  echo "Start ${DS}: $(date)"
  echo "Source: $SRC_DIR"
  echo "Output: $OUT_DIR/$ARCHIVE_NAME"

  tar -C "$BASE" -cf - "$DS" \
    | pigz -p "${SLURM_CPUS_PER_TASK:-8}" -1 \
    > "$OUT_DIR/$ARCHIVE_NAME"

  echo "Compression finished: $(date)"

  ( cd "$OUT_DIR" && sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256" )
  echo "SHA256:"
  cat "$OUT_DIR/$ARCHIVE_NAME.sha256"
  echo "Done ${DS}: $(date)"
done

echo "All done: $(date)"

# ============================================================
# PROVENANCE -- the original ds8 pack (already done, in archive2). Do NOT repack.
#   SRC_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8_filtered_thr0p1_min5100_full"
#   OUT_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/archive2"
#   ARCHIVE_NAME="dataset_ds8_filtered_thr0p1_min5100_full.tar.gz"
#   mkdir -p "$OUT_DIR" logs
#   tar -C "$(dirname "$SRC_DIR")" -cf - "$(basename "$SRC_DIR")" \
#     | pigz -p "${SLURM_CPUS_PER_TASK:-8}" -1 \
#     > "$OUT_DIR/$ARCHIVE_NAME"
#   cd "$OUT_DIR"
#   sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
# ============================================================
