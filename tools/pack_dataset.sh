#!/bin/bash
#SBATCH --job-name=pack_dataset_ds8
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ==============================
# User settings
# ==============================

SRC_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_ds8"
OUT_DIR="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/archive"

ARCHIVE_NAME="dataset_ds8.tar.gz"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
SHA_PATH="${ARCHIVE_PATH}.sha256"

N_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ==============================
# Preparation
# ==============================

mkdir -p "${OUT_DIR}"

echo "Source directory: ${SRC_DIR}"
echo "Output archive:   ${ARCHIVE_PATH}"
echo "SHA256 file:      ${SHA_PATH}"
echo "Threads:          ${N_THREADS}"
echo "Start time:       $(date)"
echo

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: Source directory does not exist: ${SRC_DIR}"
    exit 1
fi

# Avoid overwriting existing archive by accident
if [ -f "${ARCHIVE_PATH}" ]; then
    echo "ERROR: Archive already exists: ${ARCHIVE_PATH}"
    echo "Please remove it manually or change ARCHIVE_NAME."
    exit 1
fi

# ==============================
# Compress
# ==============================

echo "Compressing dataset..."

tar \
    -C "$(dirname "${SRC_DIR}")" \
    -cf - "$(basename "${SRC_DIR}")" \
    | pigz -p "${N_THREADS}" -9 > "${ARCHIVE_PATH}"

echo "Compression finished."
echo

# ==============================
# Generate checksum
# ==============================

echo "Generating SHA256 checksum..."

cd "${OUT_DIR}"
sha256sum "${ARCHIVE_NAME}" > "${SHA_PATH}"

echo "Checksum generated:"
cat "${SHA_PATH}"
echo

# ==============================
# Basic archive test
# ==============================

echo "Testing archive integrity..."

tar -tzf "${ARCHIVE_PATH}" > /dev/null

echo "Archive test passed."
echo "End time: $(date)"
echo "Done."
