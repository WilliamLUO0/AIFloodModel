#!/bin/bash
#SBATCH --job-name=pack_ds8
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SRC_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8"
OUT_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/archive"
ARCHIVE_NAME="dataset_ds8.tar.gz"

mkdir -p "$OUT_DIR" logs

echo "Start: $(date)"
echo "Source: $SRC_DIR"
echo "Output: $OUT_DIR/$ARCHIVE_NAME"

tar -C "$(dirname "$SRC_DIR")" -cf - "$(basename "$SRC_DIR")" \
  | pigz -p "${SLURM_CPUS_PER_TASK:-8}" -1 \
  > "$OUT_DIR/$ARCHIVE_NAME"

echo "Compression finished: $(date)"

cd "$OUT_DIR"
sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"

echo "SHA256:"
cat "$ARCHIVE_NAME.sha256"

echo "Done: $(date)"