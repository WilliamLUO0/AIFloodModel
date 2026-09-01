#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_topo_attrs_fixed.py

Rewritten terrain-attribute exporter. Fixes three bugs in the original
tools/compute_topo_attrs.py that produced Slope_Deg==0 / TWI==ln(8e6) over
~half of the valid grid (~79% of the Gisborne basin polygon):

  BUG A (no-data not recognised):
      xr.open_dataset(..) with the default mask_and_scale=True turns the
      DEM's _FillValue (9.96921e+36) into NaN and MOVES the _FillValue
      attribute into .encoding. The original loop `if k in z.attrs` then
      never finds it, so `nodata` stays None and no masking happens.
      richdem detects no-data by equality (cell == no_data); NaN == NaN is
      always False, so the NaN cells are treated as real terrain, poisoning
      the Priority-Flood queue and flattening huge regions of the filled DEM.
    -> FIX: open with mask_and_scale=False, read _FillValue explicitly, build
       an explicit valid mask, and hand richdem a REAL sentinel no_data value
       (not NaN) that its equality test can actually match.

  BUG B (slope/aspect computed on the FILLED DEM):
      The original took slope/aspect on dem_filled. Depression-filled regions
      are perfectly horizontal by construction -> slope == 0 there. Standard
      practice: compute slope/aspect on the ORIGINAL DEM; use the filled DEM
      only for flow accumulation.
    -> FIX: slope_deg / aspect are computed on the RAW (masked) DEM.
       Only TWI uses the filled DEM (for flow accumulation + a well-defined
       tan(beta) in flat pits).

  BUG C (cell size / geotransform never set):
      rd.rdarray(...) had no geotransform ("Warning! No geotransform defined.
      ... cells are 1x1"), so richdem's Horn slope used a 1 m run instead of
      8 m -> even the non-zero slopes were inflated (true 20 deg -> ~71 deg),
      and --dx 8 was only used inside TWI, never for the slope itself.
    -> FIX: slope/aspect use the true 8 m cell size (Horn finite differences);
       TWI's specific catchment area uses the true cell size too.

Slope and aspect are computed with a self-contained numpy Horn kernel (the
same one used to verify the bug), so they do not depend on richdem's
geotransform handling at all. richdem is used only for FillDepressions +
FlowAccumulation (its hydrology strength). If richdem is unavailable the TWI
step is skipped with a warning; slope/aspect are always produced.

Usage (matches the original CLI):
  python compute_topo_attrs_fixed.py \
    --input  /.../input_files/Elevation.nc \
    --var    Band1 \
    --outdir /.../input_files/Topo_Attrs \
    --dx 8
"""

import os
import argparse
import numpy as np
import xarray as xr

try:
    import rioxarray as rxr  # noqa: F401  (registers .rio accessor)
    _HAS_RIO = True
except Exception:
    _HAS_RIO = False

try:
    import richdem as rd
    _HAS_RICHDEM = True
except Exception:
    _HAS_RICHDEM = False


FILL_SENTINEL = np.float32(9.96921e36)  # keep the same _FillValue as Elevation.nc


# --------------------------------------------------------------------------- #
# Horn (1981) 3x3 finite-difference terrain attributes on the RAW DEM.
# `z` is float64 with np.nan at no-data. `cs` is the cell size in metres.
# Returns slope in degrees, plus sin/cos of aspect (azimuth of steepest
# descent, measured clockwise from north). Any output cell whose 3x3 window
# touches a no-data cell (or the grid edge) is set to NaN.
# --------------------------------------------------------------------------- #
def horn_attrs(z, cs):
    ny, nx = z.shape
    slope_deg = np.full((ny, nx), np.nan, dtype=np.float64)
    asin = np.full((ny, nx), np.nan, dtype=np.float64)
    acos = np.full((ny, nx), np.nan, dtype=np.float64)

    # 3x3 neighbourhood of the interior (row 0 = north because dy < 0).
    #   a b c      NW N NE
    #   d e f  =    W . E
    #   g h i      SW S SE
    a = z[:-2, :-2]; b = z[:-2, 1:-1]; c = z[:-2, 2:]
    d = z[1:-1, :-2];                  f = z[1:-1, 2:]
    g = z[2:,  :-2]; h = z[2:,  1:-1]; i = z[2:,  2:]

    # east-west gradient (+x = east) and north-south gradient (+y = north)
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cs)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * cs)

    slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))

    # azimuth of steepest DESCENT = direction of -grad, clockwise from north.
    azimuth = np.arctan2(-dz_dx, -dz_dy)  # 0 = north, +pi/2 = east
    s = np.sin(azimuth)
    co = np.cos(azimuth)

    # flat cells have no defined aspect -> encode as (0, 0)
    flat = slope == 0
    s = np.where(flat, 0.0, s)
    co = np.where(flat, 0.0, co)

    slope_deg[1:-1, 1:-1] = slope
    asin[1:-1, 1:-1] = s
    acos[1:-1, 1:-1] = co
    return slope_deg, asin, acos


def slope_radians_from(z, cs):
    """Horn slope in radians on `z` (used for TWI's tan(beta))."""
    ny, nx = z.shape
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    a = z[:-2, :-2]; b = z[:-2, 1:-1]; c = z[:-2, 2:]
    d = z[1:-1, :-2];                  f = z[1:-1, 2:]
    g = z[2:,  :-2]; h = z[2:,  1:-1]; i = z[2:,  2:]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cs)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * cs)
    out[1:-1, 1:-1] = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="DEM NetCDF (Elevation.nc)")
    ap.add_argument("--var", default="Band1", help="DEM variable (default Band1)")
    ap.add_argument("--outdir", required=True, help="output dir")
    ap.add_argument("--dx", type=float, default=8.0, help="grid size in metres")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cs = float(args.dx)

    # ---- read DEM WITHOUT auto-masking so _FillValue stays in .attrs ------- #
    ds = xr.open_dataset(args.input, mask_and_scale=False)
    var = args.var if args.var in ds.data_vars else list(ds.data_vars)[0]
    z_da = ds[var]

    # explicit no-data value (attrs first, then encoding, then a huge-value
    # fallback that also traps GDAL's 9.96921e+36)
    nodata = None
    for src in (z_da.attrs, z_da.encoding):
        for k in ("_FillValue", "missing_value", "nodata"):
            if k in src:
                try:
                    nodata = float(src[k]); break
                except Exception:
                    pass
        if nodata is not None:
            break

    z = np.array(z_da.values, dtype=np.float64)
    valid = np.isfinite(z) & (np.abs(z) < 1e30)
    if nodata is not None and np.isfinite(nodata):
        valid &= (z != nodata)
    z[~valid] = np.nan
    n_valid = int(valid.sum())
    print(f"[info] DEM {z.shape}  valid={n_valid:,} ({100.0*n_valid/z.size:.1f}%)  "
          f"nodata={nodata}  cell={cs} m")

    # ---- slope / aspect on the RAW DEM (fix B + fix C) --------------------- #
    slope_deg, aspect_sin, aspect_cos = horn_attrs(z, cs)

    zero_pct = 100.0 * np.mean(slope_deg[valid & np.isfinite(slope_deg)] == 0) \
        if n_valid else float("nan")
    print(f"[info] slope zero-fraction over valid = {zero_pct:.3f}%  "
          f"(median={np.nanmedian(slope_deg[valid]):.2f} deg)")

    # ---- TWI: FillDepressions + FlowAccumulation on the FILLED DEM --------- #
    twi = None
    if _HAS_RICHDEM:
        # hand richdem a REAL sentinel no_data (fix A) + the true geotransform.
        z_rd = np.where(valid, z, float(FILL_SENTINEL)).astype(np.float64)
        dem = rd.rdarray(z_rd, no_data=float(FILL_SENTINEL))
        # GeoTransform of Elevation.nc: [x0, dx, 0, y0, 0, dy]
        gt = None
        for src in (ds[var].attrs, ds.attrs):
            if "GeoTransform" in src:
                try:
                    gt = [float(t) for t in str(src["GeoTransform"]).split()]
                except Exception:
                    gt = None
            for vn in ds.variables:
                a = ds[vn].attrs
                if "GeoTransform" in a and gt is None:
                    try:
                        gt = [float(t) for t in str(a["GeoTransform"]).split()]
                    except Exception:
                        gt = None
        if gt and len(gt) == 6:
            dem.geotransform = gt
        else:
            dem.geotransform = [0.0, cs, 0.0, 0.0, 0.0, -cs]

        # EPSILON fill: adds a tiny monotone drainage gradient so FlowAccumulation
        # can route across would-be flats. Plain FillDepressions can leave an
        # enclosed sub-basin (e.g. a high massif whose real outlet is cut off by
        # the nodata boundary) perfectly flat -> D8 has no direction -> acc stalls
        # at 1 -> TWI pinned to ln(cs/eps)=ln(8e6) over the whole region. Verified
        # this happened to a 300 km^2 high massif with plain fill.
        try:
            dem_filled = rd.FillDepressions(dem, epsilon=True, in_place=False)
        except TypeError:
            dem_filled = rd.FillDepressions(dem, in_place=False)
        acc_cells = rd.FlowAccumulation(dem_filled, method='D8')

        # tan(beta) = LOCAL surface slope from the RAW DEM (standard TWI
        # definition, e.g. Beven & Kirkby / SAGA / GRASS r.topidx). NOT the
        # filled DEM: filled/flattened cells have beta=0 -> tan(beta) clamped to
        # eps -> a spurious constant-high TWI on real terrain. z already carries
        # NaN at nodata, so edge cells fall out as NaN (masked below).
        slope_rad = slope_radians_from(z, cs)

        eps = 1e-6
        a_spec = np.asarray(acc_cells, dtype=np.float64) * cs   # specific catchment area
        tanb = np.tan(slope_rad)
        twi = np.log(np.maximum(a_spec, eps) / np.maximum(tanb, eps))
        twi[~valid] = np.nan
        const = float(np.mean(np.abs(twi[valid & np.isfinite(twi)] - np.log(cs / eps)) < 1e-3)) \
            if n_valid else float("nan")
        print(f"[info] TWI computed (acc max={np.nanmax(a_spec)/cs:.0f} cells; "
              f"TWI==ln(cs/eps) sanity spike = {100*const:.3f}% of valid, want <<1%)")
    else:
        print("[warn] richdem not available -> TWI NOT written "
              "(run this step on NeSI where richdem is installed).")

    # ---- write outputs ----------------------------------------------------- #
    coords = {d: ds[d] for d in z_da.dims if d in ds.coords}
    dims = z_da.dims

    def write(arr, name, fname):
        da = xr.DataArray(np.asarray(arr, dtype=np.float32), coords=coords, dims=dims, name=name)
        if _HAS_RIO:
            try:
                da.rio.write_crs("EPSG:2193", inplace=True)
            except Exception:
                pass
        da.encoding["_FillValue"] = float(FILL_SENTINEL)
        da.to_netcdf(os.path.join(args.outdir, fname))
        v = np.asarray(arr, dtype=np.float64)
        v = v[np.isfinite(v)]
        print(f"[write] {fname:16s} min/max = {v.min():.4f} / {v.max():.4f}")

    write(slope_deg, "slope_deg", "Slope_Deg.nc")
    write(aspect_sin, "aspect_sin", "Aspect_SIN.nc")
    write(aspect_cos, "aspect_cos", "Aspect_COS.nc")
    if twi is not None:
        write(twi, "TWI", "TWI.nc")

    ds.close()
    print("[done] outputs in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
