#!/bin/bash
#SBATCH --job-name=pack_dataset
#SBATCH --account=uoa04425
#SBATCH --partition=milan,genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Pack datasets ONE-BY-ONE for the NeSI -> NIWA transfer. Each dataset becomes
#   archive_dataset/<DS>.tar.gz  (+ <DS>.tar.gz.sha256)
# For every dataset this prints, to the job log:
#   * a one-line [metrics] record of the SOURCE dir (apparent bytes + file/entry
#     counts) -- diff this against the NIWA unpack log to prove nothing was lost;
#   * a sha256 self-verify of the tarball (catches a bad write before transfer).
#
# Run AFTER rewrite_index_paths.sh (so *_niwa.csv are inside each dataset dir).
# Edit DATASETS to pack a subset (e.g. ship a few first, the rest later) -- the
# SAME list convention is used by unpack_dataset.pbs.

BASE="/nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel"
OUT_DIR="${BASE}/archive_dataset"

# ===================== EDIT HERE: which datasets to pack =====================
DATASETS=(
  # --- training (4) ---
  "dataset_ds8_filtered_thr0p1_min5100_full"
  "dataset_ds16_filtered_thr0p1_min2100_full"
  "dataset_ds4_filtered_thr0p1_min10100_full"
  "dataset_ds2_filtered_thr0p1_min25100_full"
  # --- test: factor-8 main-model suite ---
  "testdataset_2y42h0c"
  "testdataset_5y42h0c"
  "testdataset_10y42h0c"
  "testdataset_50y42h0c"
  "testdataset_100y42h0c"
  "testdataset_1000y42h0c"
  # --- test: downscaling-factor ablation (100y only) ---
  "testdataset_100y42h0c_ds2"
  "testdataset_100y42h0c_ds4"
  "testdataset_100y42h0c_ds16"
  # --- deferred: gabrielle (uncomment once built) ---
  # "testdataset_gabrielle"
  # "testdataset_gabrielle_ds2"
  # "testdataset_gabrielle_ds4"
  # "testdataset_gabrielle_ds16"
)
# ============================================================================

NCPU="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$OUT_DIR" logs

# One-line, diff-friendly record of a dataset DIRECTORY. Single find pass gives
# apparent bytes (sum of file sizes), file count and total entry count. The SAME
# function runs in unpack_dataset.pbs on the extracted dir, so the two logs match
# byte-for-byte when the transfer is intact.
report_metrics() {
  local dir="$1"
  local size files entries
  size=$(du -sh --apparent-size "$dir" | awk '{print $1}')   # apparent size (human)
  files=$(find "$dir" -type f | wc -l)                       # number of files
  entries=$(find "$dir" | wc -l)                             # files + dirs
  echo "[metrics] DS=$(basename "$dir") apparent_size=${size} files=${files} entries=${entries}"
}

for DS in "${DATASETS[@]}"; do
  SRC_DIR="${BASE}/${DS}"
  ARCHIVE="${OUT_DIR}/${DS}.tar.gz"

  echo "============================================================"
  echo "[$(date)] PACK ${DS}"
  echo "  source : ${SRC_DIR}"
  echo "  archive: ${ARCHIVE}"

  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[error] source dir not found: ${SRC_DIR}" >&2
    exit 1
  fi

  # source-dir metrics (compare against the NIWA unpack log)
  report_metrics "${SRC_DIR}"

  # pack: top-level dir inside the tar is exactly ${DS}/ (tar -C "$BASE")
  tar -C "$BASE" -cf - "$DS" | pigz -p "${NCPU}" -1 > "${ARCHIVE}"
  echo "[$(date)] compressed"

  # checksum + self-verify (bad write on NeSI is caught here, before transfer)
  ( cd "$OUT_DIR" && sha256sum "${DS}.tar.gz" > "${DS}.tar.gz.sha256" )
  ( cd "$OUT_DIR" && sha256sum -c "${DS}.tar.gz.sha256" )
  echo "[sha256] $(cat "${ARCHIVE}.sha256")"
  echo "[archive-size] $(du -sh --apparent-size "${ARCHIVE}" | awk '{print $1}')"
  echo "[$(date)] done ${DS}"
done

echo "============================================================"
echo "[all done] $(date)  packed ${#DATASETS[@]} datasets -> ${OUT_DIR}"
