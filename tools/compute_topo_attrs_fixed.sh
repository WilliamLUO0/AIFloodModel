#!/bin/bash
#SBATCH --job-name=topo_attrs_fixed
#SBATCH --account=uoa04425
#SBATCH --partition=milan
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1
eval "$(conda shell.bash hook)"

ENV_DIR=/nesi/project/uoa04425/zluo784/envs/python310
conda activate "$ENV_DIR"

INPUT="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc"
# Write to a NEW dir first so you can diff against the old (buggy) Topo_Attrs
# before swapping. Once verified, point make_patches at this dir (or replace
# the four files in Topo_Attrs/).
OUTDIR="/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs_fixed"
VAR="Band1"

mkdir -p "$OUTDIR" logs

python compute_topo_attrs_fixed.py --input "$INPUT" --var "$VAR" --outdir "$OUTDIR" --dx 8

# Quick sanity checks. Expect slope zero-fraction << 1% (not ~50%) AND
# TWI==ln(8e6) spike << 1% (the earlier plain-fill run left it at ~5.5%).
python - "$OUTDIR/Slope_Deg.nc" "$OUTDIR/TWI.nc" <<'PY'
import sys, math, numpy as np, xarray as xr
sp, tp = sys.argv[1], sys.argv[2]
da = xr.open_dataset(sp)
v = da[[k for k in da.data_vars if 'slope' in k.lower()][0]].values.astype('float64')
v = v[np.isfinite(v)]
print(f"[check] slope zero-fraction = {100*np.mean(v==0):.3f}%  median={np.median(v):.2f} deg  max={v.max():.1f}")
dt = xr.open_dataset(tp)
w = dt[[k for k in dt.data_vars if 'twi' in k.lower()][0]].values.astype('float64')
w = w[np.isfinite(w)]
const = math.log(8.0 / 1e-6)  # ln(cs/eps) = ln(8e6) = 15.895
print(f"[check] TWI==ln(8e6) spike = {100*np.mean(np.abs(w-const)<1e-3):.3f}%   "
      f"pct[10,50,90]={np.percentile(w,[10,50,90]).round(2)}  (spike should be <<1%)")
PY
