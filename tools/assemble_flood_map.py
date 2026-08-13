#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reassemble per-patch fine-grid predictions into full flood maps and write NetCDF.

The model eval (test_flood_map.py) saves one predicted fine patch (patch_fine x
patch_fine, e.g. 512x512) per coarse patch, under
    results/<eval>/visualization/<core>_coarse/<core>_<exp>.npy
where core = <var>_<scenario>_t<NNNN>_r<RRR>_c<CCC>_s<S>. Patches tile the fine
grid with stride == patch_fine (no overlap), so patch (r,c) occupies fine rows
[r*P : r*P+P) and cols [c*P : c*P+P). This tool stitches them back per timestep.

Outputs (per --var, --scenario):
  <out-dir>/<scenario>_<var>_<source>.nc       # (time, y, x)  — all timesteps, BGout-style
  <out-dir>/<scenario>_<var>_<source>_max.nc   # (y, x)        — max inundation
      h   : max over time of the value
      u/v : max over time of |value| (peak speed magnitude)
Optional:
  --per-timestep-npy   also dump <out-dir>/npy/<core-map>_t<NNNN>.npy per timestep

Grid geometry & coordinates are reconstructed from index.csv (x0_fine/y0_fine/
x1_fine/y1_fine per patch), so the original huge fine .nc files are NOT needed.
AOI-outside cells are set to NaN using the per-patch fine mask.

Examples
--------
# model prediction, gabrielle
python tools/assemble_flood_map.py \
  --index-csv /.../testdataset_gabrielle/index.csv \
  --vis-root  /.../results/01_..._eval_gabrielle/visualization \
  --var h --scenario gabrielle \
  --out-dir   /.../results/01_..._eval_gabrielle/assembled

# also assemble the simulated ground truth for side-by-side comparison
python tools/assemble_flood_map.py --index-csv ... --var h --scenario gabrielle \
  --source gt --out-dir ...

# velocity (max uses |v|)
python tools/assemble_flood_map.py --index-csv ... --vis-root ... \
  --var u --scenario gabrielle --out-dir ...
"""

import os
import re
import glob
import csv
import argparse
from collections import defaultdict

import numpy as np

CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>.+?)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)

VAR_META = {
    "h": ("h", "water depth", "m"),
    "zs": ("zs", "water surface elevation", "m"),
    "u": ("u", "x-direction depth-averaged velocity", "m s-1"),
    "v": ("v", "y-direction depth-averaged velocity", "m s-1"),
}


def t_to_int(t_tag: str) -> int:
    return int(t_tag[1:])  # 't0047' -> 47


def pick_pred_file(folder: str, core: str) -> str:
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npys:
        raise RuntimeError(f"No .npy in {folder}")
    cand = [p for p in npys
            if os.path.basename(p).startswith(core + "_")
            and not p.endswith("_coarse.npy") and not p.endswith("_fine.npy")]
    if cand:
        return cand[0]
    if len(npys) == 1:
        return npys[0]
    raise RuntimeError(f"Cannot identify predicted .npy in {folder} for core={core}")


def load_hw(path: str) -> np.ndarray:
    a = np.asarray(np.load(path))
    if a.ndim == 3:
        a = a[0]
    elif a.ndim == 4:
        a = a[0, 0]
    return a


def build_index(index_csv, var, scenario):
    """core -> row dict, filtered to var+scenario and filtered_out==0."""
    rows = {}
    with open(index_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("var") != var or row.get("scenario") != scenario:
                continue
            if str(row.get("filtered_out", "0")) == "1":
                continue
            core = (f"{row['var']}_{row['scenario']}_{row['t']}"
                    f"_r{int(row['patch_row']):03d}_c{int(row['patch_col']):03d}_s{int(row['downscale'])}")
            rows[core] = row
    if not rows:
        raise RuntimeError(f"No index rows for var={var} scenario={scenario} (filtered_out==0)")
    return rows


def reconstruct_grid(rows, dx_fine_default=8.0):
    """
    Reconstruct (x, y) coordinate vectors and (Ny, Nx) of the full fine grid from
    the per-patch corner coords in index.csv. Geometry is time-invariant, so we
    dedupe by (patch_row, patch_col).
    """
    any_row = next(iter(rows.values()))
    P = int(any_row["patch_size_fine"])

    by_rc = {}
    for r in rows.values():
        rc = (int(r["patch_row"]), int(r["patch_col"]))
        by_rc.setdefault(rc, r)

    cols = sorted({c for (_, c) in by_rc})
    rws = sorted({rr for (rr, _) in by_rc})

    x0_col, x1_col, y0_row, y1_row = {}, {}, {}, {}
    for (rr, cc), r in by_rc.items():
        if cc not in x0_col:
            x0_col[cc] = float(r["x0_fine"]); x1_col[cc] = float(r["x1_fine"])
        if rr not in y0_row:
            y0_row[rr] = float(r["y0_fine"]); y1_row[rr] = float(r["y1_fine"])

    def _step(origin_by_idx, span_hi_by_idx, idxs):
        lo, hi = idxs[0], idxs[-1]
        if hi > lo:  # exact: origin difference across patches / (index gap * P)
            return (origin_by_idx[hi] - origin_by_idx[lo]) / ((hi - lo) * P)
        # Only ONE patch along this axis: can't infer the step from patch origins.
        # Fall back to the known fine cell size (--dx-fine), taking the sign from
        # the within-patch span (a partial edge patch has an unknown real width, so
        # span/(P-1) would be wrong -- use the magnitude of dx_fine_default).
        span = span_hi_by_idx[lo] - origin_by_idx[lo]
        return float(np.copysign(dx_fine_default, span if span != 0 else 1.0))

    dx = _step(x0_col, x1_col, cols)
    dy = _step(y0_row, y1_row, rws)

    cmax, rmax = cols[-1], rws[-1]
    w_last = int(round((x1_col[cmax] - x0_col[cmax]) / dx)) + 1
    h_last = int(round((y1_row[rmax] - y0_row[rmax]) / dy)) + 1
    Nx = cmax * P + w_last
    Ny = rmax * P + h_last

    # Anchor coordinates to the TRUE grid origin (fine column/row 0). The canvas
    # places patches at ABSOLUTE indices c*P / r*P, but the westmost/northmost
    # patches can be MISSING from the index when they fall entirely outside the
    # AOI (aoi_ratio==0 -> filtered even at thresh 1e-6), so cols[0]/rws[0] may be
    # > 0. Using the first PRESENT patch's coord as the origin would then shift
    # every coordinate by cols[0]*P cells (the "x-axis shifted eastward" artefact).
    # Extrapolate back to column/row 0 (x0_col[c] == x_f[0] + c*P*dx for all c).
    x_f0 = x0_col[cols[0]] - cols[0] * P * dx
    y_f0 = y0_row[rws[0]] - rws[0] * P * dy
    x = x_f0 + np.arange(Nx) * dx
    y = y_f0 + np.arange(Ny) * dy
    return x.astype(np.float64), y.astype(np.float64), Ny, Nx, P


def assemble_canvas(rows_t, Ny, Nx, P, vis_root, source):
    """Assemble one timestep's patches into a (Ny, Nx) float32 canvas (NaN gaps)."""
    canvas = np.full((Ny, Nx), np.nan, dtype=np.float32)
    for core, row in rows_t:
        r = int(row["patch_row"]); c = int(row["patch_col"])
        if source == "gt":
            patch = load_hw(row["fine_path"])
        else:
            folder = os.path.join(vis_root, core + "_coarse")
            patch = load_hw(pick_pred_file(folder, core))
        y0, x0 = r * P, c * P
        h = min(P, Ny - y0); w = min(P, Nx - x0)
        canvas[y0:y0 + h, x0:x0 + w] = patch[:h, :w].astype(np.float32)
    return canvas


def load_mask_canvas(rows, Ny, Nx, P):
    """Full AOI mask (True inside AOI), from per-patch mask_fine (time-invariant)."""
    mask = np.zeros((Ny, Nx), dtype=bool)
    by_rc = {}
    for r in rows.values():
        by_rc.setdefault((int(r["patch_row"]), int(r["patch_col"])), r)
    for (r, c), row in by_rc.items():
        mp = row.get("mask_fine_path", "")
        if not mp or not os.path.exists(mp):
            continue
        m = load_hw(mp).astype(bool)
        y0, x0 = r * P, c * P
        h = min(P, Ny - y0); w = min(P, Nx - x0)
        mask[y0:y0 + h, x0:x0 + w] = m[:h, :w]
    return mask


def write_nc_series(path, varname, long_name, units, x, y, times, canvases_iter,
                    crs="EPSG:2193"):
    """Incrementally write a (time,y,x) NetCDF from an iterator of (t_int, canvas)."""
    from netCDF4 import Dataset
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nc = Dataset(path, "w", format="NETCDF4")
    nc.createDimension("time", None)
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))
    yv = nc.createVariable("y", "f8", ("y",)); yv[:] = y; yv.units = "m"; yv.standard_name = "projection_y_coordinate"
    xv = nc.createVariable("x", "f8", ("x",)); xv[:] = x; xv.units = "m"; xv.standard_name = "projection_x_coordinate"
    tv = nc.createVariable("time", "i4", ("time",)); tv.long_name = "timestep index"
    dv = nc.createVariable(varname, "f4", ("time", "y", "x"),
                           zlib=True, complevel=4, fill_value=np.float32(np.nan))
    dv.long_name = long_name; dv.units = units; dv.grid_mapping = "crs"
    cv = nc.createVariable("crs", "i4"); cv.crs_wkt = crs; cv.spatial_ref = crs
    nc.Conventions = "CF-1.8"; nc.crs = crs
    for i, (t_int, canvas) in enumerate(canvases_iter):
        dv[i, :, :] = canvas
        tv[i] = int(t_int)
    nc.close()


def write_nc_2d(path, varname, long_name, units, x, y, arr2d, crs="EPSG:2193"):
    from netCDF4 import Dataset
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nc = Dataset(path, "w", format="NETCDF4")
    nc.createDimension("y", len(y)); nc.createDimension("x", len(x))
    yv = nc.createVariable("y", "f8", ("y",)); yv[:] = y; yv.units = "m"; yv.standard_name = "projection_y_coordinate"
    xv = nc.createVariable("x", "f8", ("x",)); xv[:] = x; xv.units = "m"; xv.standard_name = "projection_x_coordinate"
    dv = nc.createVariable(varname, "f4", ("y", "x"),
                           zlib=True, complevel=4, fill_value=np.float32(np.nan))
    dv.long_name = long_name; dv.units = units; dv.grid_mapping = "crs"
    cv = nc.createVariable("crs", "i4"); cv.crs_wkt = crs; cv.spatial_ref = crs
    nc.Conventions = "CF-1.8"; nc.crs = crs
    dv[:, :] = arr2d.astype(np.float32)
    nc.close()


def main():
    ap = argparse.ArgumentParser(description="Assemble per-patch predictions into full flood-map NetCDFs.")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--var", required=True, choices=["h", "zs", "u", "v"])
    ap.add_argument("--scenario", required=True, help="e.g. gabrielle, 100y_42h_0c")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="pred", choices=["pred", "gt"],
                    help="'pred' = model prediction from --vis-root; 'gt' = simulated fine patch (index fine_path)")
    ap.add_argument("--vis-root", default="", help="results/<eval>/visualization (required for --source pred)")
    ap.add_argument("--max-mode", default="auto", choices=["auto", "value", "abs"],
                    help="max-inundation: 'value'=max(x), 'abs'=max(|x|), 'auto'= abs for u/v else value")
    ap.add_argument("--clip-nonneg", action="store_true", help="clip the assembled value to >=0 (depth only)")
    ap.add_argument("--no-mask", action="store_true", help="do NOT set AOI-outside cells to NaN")
    ap.add_argument("--per-timestep-npy", action="store_true", help="also dump one assembled .npy per timestep")
    ap.add_argument("--crs", default="EPSG:2193")
    ap.add_argument("--dx-fine", type=float, default=8.0, help="fallback fine cell size if grid step can't be inferred")
    args = ap.parse_args()

    if args.source == "pred" and not args.vis_root:
        raise SystemExit("[ERROR] --vis-root is required for --source pred")

    rows = build_index(args.index_csv, args.var, args.scenario)
    x, y, Ny, Nx, P = reconstruct_grid(rows, dx_fine_default=args.dx_fine)
    print(f"[grid] {args.scenario}/{args.var}: Ny={Ny} Nx={Nx} P={P} "
          f"dx={(x[1]-x[0]):.3f} dy={(y[1]-y[0]):.3f} patches={len(rows)}")

    mask = None if args.no_mask else load_mask_canvas(rows, Ny, Nx, P)

    # group by timestep
    by_t = defaultdict(list)
    for core, row in rows.items():
        by_t[row["t"]].append((core, row))
    t_tags = sorted(by_t.keys(), key=t_to_int)
    print(f"[time] {len(t_tags)} timesteps: {t_tags[0]}..{t_tags[-1]}")

    varname, long_name, units = VAR_META[args.var]
    use_abs = (args.max_mode == "abs") or (args.max_mode == "auto" and args.var in ("u", "v"))
    max_map = np.full((Ny, Nx), np.nan, dtype=np.float32)

    npy_dir = os.path.join(args.out_dir, "npy")
    if args.per_timestep_npy:
        os.makedirs(npy_dir, exist_ok=True)

    def canvases():
        for t in t_tags:
            canvas = assemble_canvas(by_t[t], Ny, Nx, P, args.vis_root, args.source)
            if args.clip_nonneg:
                np.maximum(canvas, 0.0, out=canvas, where=np.isfinite(canvas))
            if mask is not None:
                canvas[~mask] = np.nan
            # running max (value or |value|), NaN-aware
            m_upd = np.abs(canvas) if use_abs else canvas
            np.fmax(max_map, m_upd, out=max_map)
            if args.per_timestep_npy:
                np.save(os.path.join(npy_dir, f"{args.scenario}_{args.var}_{args.source}_{t}.npy"), canvas)
            print(f"  assembled {t}")
            yield t_to_int(t), canvas

    series_path = os.path.join(args.out_dir, f"{args.scenario}_{args.var}_{args.source}.nc")
    write_nc_series(series_path, varname, long_name, units, x, y, t_tags, canvases(), crs=args.crs)
    print("[OK] saved series:", series_path)

    if mask is not None:
        max_map[~mask] = np.nan
    max_name = f"{varname}_absmax" if use_abs else f"{varname}_max"
    max_long = (f"peak |{varname}| over time" if use_abs else f"maximum {long_name} over time")
    max_path = os.path.join(args.out_dir, f"{args.scenario}_{args.var}_{args.source}_max.nc")
    write_nc_2d(max_path, max_name, max_long, units, x, y, max_map, crs=args.crs)
    print("[OK] saved max:  ", max_path)
    print("[done]")


if __name__ == "__main__":
    main()
