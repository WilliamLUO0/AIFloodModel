#!/bin/bash
#SBATCH --job-name=backup_file
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/nesi/nobackup/uoa04425/zluo784/backups/logs/%x_%j.out
#SBATCH --error=/nesi/nobackup/uoa04425/zluo784/backups/logs/%x_%j.err

set -euo pipefail

SRC_DIR=/nesi/nobackup/uoa04425/zluo784/Exp1
BACKUP_DIR=/nesi/nobackup/uoa04425/zluo784/backups
mkdir -p "$BACKUP_DIR/logs"

STAMP=$(date +%Y%m%d)

module load zstd 2>/dev/null || true
ZSTD_THREADS="${SLURM_CPUS_PER_TASK:-16}"

for d in AIFloodModel Gisborne_basin Merge_Grid; do
  if [[ ! -d "$SRC_DIR/$d" ]]; then
    echo "[WARN] skip missing dir: $SRC_DIR/$d"
    continue
  fi

  ARCHIVE="$BACKUP_DIR/${d}_${STAMP}.tar.zst"
  echo "[INFO] packing $d -> $ARCHIVE"

  tar -C "$SRC_DIR" -I "zstd -T${ZSTD_THREADS} -10" -cf "$ARCHIVE" "$d"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

  echo "[OK] $(ls -lh "$ARCHIVE")"
done
