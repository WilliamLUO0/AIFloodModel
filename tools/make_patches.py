#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python make_patches.py --var h --dir-fine ./100y_42h_0c_dx8/per_timestep_merged --file-coarse ./100y_42h_0c_dx128/BGout.nc --file-elev ./Elevation.nc --file-rough ./Roughness.nc --file-slope ./Slope_Deg.nc --file-twi ./TWI.nc --file-aspect-sin ./Aspect_SIN.nc --file-aspect-cos ./Aspect_COS.nc --aoi ./Gisborne_basin.shp --scale 16 --patch-coarse 64 --filter-enable --filter-thresh 0.2 --out-dir ./dataset_patches

python make_patches.py
  --var h u v
  --scenarios 100y_42h_0c 2y_6h_0c:3
  --dir-fine-template "./{scenario}/dx8/per_timestep_merged"
  --file-coarse-template "./{scenario}/dx128/BGout.nc"
  --file-elev ./Elevation.nc
  --file-rough ./Roughness.nc
  --file-slope ./Slope_Deg.nc
  --file-twi ./TWI.nc
  --file-aspect-sin ./Aspect_SIN.nc
  --file-aspect-cos ./Aspect_COS.nc
  --aoi ./Gisborne_basin.shp
  --scale 16 --patch-coarse 64
  --filter-enable --filter-thresh 0.2
  --out-dir ./dataset_patches

python make_patches.py \
  --var h u v \
  --scenarios 2y_6h_0c 2y_48h_0c 5y_6h_0c 5y_48h_0c 10y_6h_0c 10y_48h_0c 20y_6h_0c 20y_48h_0c 50y_6h_0c 50y_48h_0c 100y_6h_0c 100y_48h_0c 200y_6h_0c 200y_48h_0c 500y_6h_0c 500y_48h_0c 1000y_6h_0c 1000y_48h_0c \
  --dir-fine-template  "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx8/per_timestep_merged" \
  --file-coarse-template "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/{scenario}/dx128/BGout.nc" \
  --file-elev  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --file-rough  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
  --file-slope  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Slope_Deg.nc \
  --file-twi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/TWI.nc \
  --file-aspect-sin  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_SIN.nc \
  --file-aspect-cos  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_COS.nc \
  --aoi  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --scale 16 --patch-coarse 64 \
  --filter-enable --filter-thresh 0.2 \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset

python make_patches.py \
  --var h \
  --dir-fine /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/100y_42h_0c/dx8/per_timestep_merged \
  --file-coarse /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/100y_42h_0c/dx128/BGout.nc \
  --file-elev   /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Elevation.nc \
  --file-rough  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Roughness.nc \
  --file-slope       /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Slope_Deg.nc \
  --file-twi         /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/TWI.nc \
  --file-aspect-sin  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_SIN.nc \
  --file-aspect-cos  /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Topo_Attrs/Aspect_COS.nc \
  --aoi /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/Gisborne_basin.shp \
  --scale 16 --patch-coarse 64 \
  --filter-enable --filter-thresh 0.2 \
  --out-dir /nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/input_files/dataset_patches
"""

import os
import re
import glob
import csv
import argparse
import numpy as np
import xarray as xr
from affine import Affine


def _natural_key(path):
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return tuple(int(x) for x in nums) if nums else (base,)


def _mask_fill(da):
    fv = None
    for k in ("_FillValue", "missing_value", "nodata"):
        if k in da.attrs:
            fv = float(da.attrs[k])
            break
        if k in da.encoding:
            fv = float(da.encoding[k])
            break
    if fv is not None and np.isfinite(fv):
        da = da.where(~xr.apply_ufunc(np.isclose, da, fv))
        da = da.where(~xr.apply_ufunc(np.isclose, da, -fv))
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < 1e30)
    return da


def _infer_scenario(path):
    m = re.search(r'(\d+y_\d+h_\d+c)', path)
    return m.group(1) if m else "unknown"


def _extract_t_from_filename(path):
    m = re.search(r'_t(\d{4})', os.path.basename(path))
    return f"t{m.group(1)}" if m else None


def _load_fine(path, var):
    ds = xr.open_dataset(path)
    x = ds["xx"] if "xx" in ds else (ds["x"] if "x" in ds else None)
    y = ds["yy"] if "yy" in ds else (ds["y"] if "y" in ds else None)
    if x is None or y is None:
        ds.close()
        raise RuntimeError(f"{path} lacks xx/yy or x/y coordinates")
    if var not in ds:
        ds.close()
        raise KeyError(f"{path} cannot find variable: {var}")
    da = _mask_fill(ds[var]).assign_coords(xx=x, yy=y).load()
    out = (da, x.values.astype(np.float64), y.values.astype(np.float64))
    ds.close()
    return out


def _load_coarse_at_time(ds_coarse, var, ti):
    key = f"{var}_P0"
    if key not in ds_coarse:
        raise KeyError(f"coarse file cannot find {key}")
    da = _mask_fill(ds_coarse[key].isel(time=ti)).load()
    da = da.rename({"yy_P0": "yy", "xx_P0": "xx"})
    x_vals = ds_coarse["xx_P0"].values.astype(np.float64)
    y_vals = ds_coarse["yy_P0"].values.astype(np.float64)
    da = da.assign_coords(xx=("xx", x_vals), yy=("yy", y_vals))
    return da


def _load_static_gdal(path):
    ds = xr.open_dataset(path)
    if "Band1" not in ds:
        ds.close()
        raise RuntimeError(f"{path} lacks var Band1")
    da = _mask_fill(ds["Band1"]).load()
    if "x" in ds and "y" in ds:
        da = da.rename({"y": "yy", "x": "xx"})
        x = ds["x"].values.astype(np.float64)
        y = ds["y"].values.astype(np.float64)
    else:
        ds.close()
        raise RuntimeError(f"{path} lacks x/y coordinates")
    ds.close()
    return da, x, y


def _load_static_nc(path, prefer_vars=None):
    ds = xr.open_dataset(path)
    var = None
    if prefer_vars:
        for v in prefer_vars:
            if v in ds.data_vars:
                var = v
                break
    if var is None:
        var = list(ds.data_vars)[0]
    da = _mask_fill(ds[var]).load()
    if "x" in ds and "y" in ds:
        x = ds["x"].values.astype(np.float64)
        y = ds["y"].values.astype(np.float64)
        da = da.rename({"y": "yy", "x": "xx"})
    elif "xx" in ds and "yy" in ds:
        x = ds["xx"].values.astype(np.float64)
        y = ds["yy"].values.astype(np.float64)
    else:
        ds.close()
        raise RuntimeError(f"{path} 缺少 x/y 或 xx/yy 坐标")
    ds.close()
    return da, x, y


def _pixel_offset_align(x_f, y_f, x_s, y_s):
    yf = y_f[::-1] if y_f[0] > y_f[-1] else y_f
    ys = y_s[::-1] if y_s[0] > y_s[-1] else y_s
    dx = np.mean(np.diff(x_f))
    dy = np.mean(np.diff(yf))
    ox = (x_s[0] - x_f[0]) / dx
    oy = (ys[0] - yf[0]) / dy
    ok = (
        np.isclose(ox, round(ox), atol=1e-9)
        and np.isclose(oy, round(oy), atol=1e-9)
        and np.isclose(np.mean(np.diff(x_s)) / dx, 1.0, atol=1e-9)
        and np.isclose(np.mean(np.diff(ys)) / dy, 1.0, atol=1e-9)
    )
    return int(round(oy)), int(round(ox)), ok


def _center_coords_to_affine(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("x and y must be 1D arrays with length >= 2")

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    x0 = float(x[0] - dx / 2.0)
    y0 = float(y[0] - dy / 2.0)

    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def _place_on_fine_canvas(static_da, x_f, y_f, x_s, y_s):
    Ny_f, Nx_f = len(y_f), len(x_f)
    Ny_s, Nx_s = len(y_s), len(x_s)
    oy, ox, ok = _pixel_offset_align(x_f, y_f, x_s, y_s)
    static_da = static_da.astype(np.float32)
    if ok:
        canvas = np.full((Ny_f, Nx_f), np.nan, dtype=np.float32)
        y0 = max(0, oy)
        x0 = max(0, ox)
        y1 = min(Ny_f, y0 + Ny_s)
        x1 = min(Nx_f, x0 + Nx_s)
        sy0 = 0 if oy >= 0 else -oy
        sx0 = 0 if ox >= 0 else -ox
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)
        canvas[y0:y1, x0:x1] = static_da.values[sy0:sy1, sx0:sx1]
        return xr.DataArray(canvas, coords={"yy": y_f, "xx": x_f}, dims=("yy", "xx"))
    else:
        static_da = static_da.assign_coords(xx=("xx", x_s), yy=("yy", y_s))
        return static_da.interp(xx=x_f, yy=y_f, method="nearest").astype(np.float32)


def _rasterize_mask_to_fine(shp_path, x_f, y_f):
    base, _ = os.path.splitext(shp_path)
    needed = [base + s for s in (".shp", ".shx", ".dbf")]
    if not all(os.path.exists(p) for p in needed):
        print(f"[warn] Shapefile lacks ({needed}), mask will be 1 for all.")
        return None
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=2193)
        Ny, Nx = len(y_f), len(x_f)
        transform = _center_coords_to_affine(x_f, y_f)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        mask = rasterize(
            shapes,
            out_shape=(Ny, Nx),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )
        return mask.astype(np.uint8)
    except Exception as e:
        print(f"[warn] AOI fails, mask will be 1 for all. ({e})")
        return None


def _rasterize_mask_to_coarse(shp_path, x_c, y_c):
    base, _ = os.path.splitext(shp_path)
    needed = [base + s for s in (".shp", ".shx", ".dbf")]
    if not all(os.path.exists(p) for p in needed):
        print(f"[warn] Shapefile lacks ({needed}), coarse mask will be 1 for all.")
        return None
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=2193)
        Ny, Nx = len(y_c), len(x_c)
        transform = _center_coords_to_affine(x_c, y_c)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        mask = rasterize(
            shapes,
            out_shape=(Ny, Nx),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False
        )
        return mask.astype(np.uint8)
    except Exception as e:
        print(f"[warn] AOI rasterize to coarse failed, coarse mask will be all 1. ({e})")
        return None


def _clamp_nonneg_inplace(arr: np.ndarray, eps: float = 1e-9):
    np.copyto(arr, 0.0, where=~np.isfinite(arr))
    np.maximum(arr, 0.0, out=arr)
    if eps > 0:
        tiny = (arr > 0) & (arr < eps)
        arr[tiny] = 0.0


def _parse_scenarios_with_limits(scen_list):
    names, limits = [], {}
    if not scen_list:
        return names, limits
    for tok in scen_list:
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, n = tok.split(":", 1)
            name = name.strip()
            try:
                limits[name] = int(n.strip())
            except Exception:
                raise ValueError(f"Invalid steps limit in --scenarios token: {tok}")
            names.append(name)
        else:
            names.append(tok)
    return names, limits


def _coarse_patch_start_from_fine_patch_center(
    x_f, y_f, x_c, y_c,
    x0f, y0f,
    patch_fine, patch_coarse
):
    x_center = 0.5 * (x_f[x0f] + x_f[min(x0f + patch_fine - 1, len(x_f) - 1)])
    y_center = 0.5 * (y_f[y0f] + y_f[min(y0f + patch_fine - 1, len(y_f) - 1)])

    i_center = int(np.argmin(np.abs(x_c - x_center)))
    j_center = int(np.argmin(np.abs(y_c - y_center)))

    i0 = int(np.round(i_center - (patch_coarse / 2.0 - 0.5)))
    j0 = int(np.round(j_center - (patch_coarse / 2.0 - 0.5)))

    i0 = max(0, min(i0, len(x_c) - patch_coarse))
    j0 = max(0, min(j0, len(y_c) - patch_coarse))

    return j0, i0


def _infer_template(base_path: str, scenarios: list[str]):
    if not base_path:
        return None
    for s in scenarios:
        if s in base_path:
            return base_path.replace(s, "{scenario}")
    m = re.search(r"\d+y_\d+h_\d+c", base_path)
    if m:
        return base_path.replace(m.group(0), "{scenario}")
    return None


def _downsample_fine_to_coarse_mean(fine_da, x_f, y_f, x_c, y_c, scale):
    fine_vals = fine_da.values.astype(np.float32)

    Ny_c = len(y_c)
    Nx_c = len(x_c)

    expected_h = Ny_c * scale
    expected_w = Nx_c * scale

    if fine_vals.shape[0] < expected_h or fine_vals.shape[1] < expected_w:
        raise RuntimeError(
            f"fine DEM size {fine_vals.shape} is smaller than expected coarse*scale "
            f"({expected_h}, {expected_w})"
        )

    fine_crop = fine_vals[:expected_h, :expected_w]

    coarse_vals = np.nanmean(
        fine_crop.reshape(Ny_c, scale, Nx_c, scale),
        axis=(1, 3)
    )

    return xr.DataArray(
        coarse_vals.astype(np.float32),
        coords={"yy": y_c, "xx": x_c},
        dims=("yy", "xx")
    )


def _pad_to_patch(arr, patch_fine, fill_value):
    padH = patch_fine - arr.shape[0]
    padW = patch_fine - arr.shape[1]
    return np.pad(arr, ((0, padH), (0, padW)), mode="constant", constant_values=fill_value)


def _downsample_patch_mean_np(fine_block: np.ndarray, scale: int, patch_coarse: int):
    """
    Downsample a fine patch [patch_fine, patch_fine] to coarse resolution
    [patch_coarse, patch_coarse] by block mean.

    This is only for alignment diagnostics.
    """
    patch_fine = patch_coarse * scale

    if fine_block.shape[0] != patch_fine or fine_block.shape[1] != patch_fine:
        raise RuntimeError(
            f"[debug-align] fine_block shape {fine_block.shape} does not match "
            f"expected {(patch_fine, patch_fine)}"
        )

    x = fine_block.astype(np.float32)
    x = x.reshape(patch_coarse, scale, patch_coarse, scale)
    return np.nanmean(x, axis=(1, 3)).astype(np.float32)


def run(
    var, dir_fine, pat_fine, file_coarse, file_elev, file_rough, shp_aoi,
    scale=16, patch_coarse=64, out_dir=None, scenario=None, snap_mode="center",
    dx_fine=8.0, file_slope="", file_twi="", file_aspect_sin="", file_aspect_cos="",
    filter_enable=False, filter_thresh=0.2, write_header=True, limit_steps=None,
    depth_eps=5e-5, vel_eps=1e-5, clip_max_depth=5.0, clip_max_vel_u=5.0, clip_max_vel_v=5.0,
    debug_align=False, debug_align_max_times=3, debug_align_max_patches=4, debug_align_min_wet_ratio=0.01
):
    patch_fine = patch_coarse * scale
    stride_fine = patch_fine

    if scenario is None:
        scenario = _infer_scenario(file_coarse)

    if out_dir is None:
        out_dir = "./dataset_patches"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    d_coarse = os.path.join(out_dir, f"{var}/{scenario}/coarse")
    d_fine = os.path.join(out_dir, f"{var}/{scenario}/fine")
    os.makedirs(d_coarse, exist_ok=True)
    os.makedirs(d_fine, exist_ok=True)

    d_zs_fine = os.path.join(out_dir, "zs", "fine")
    d_zs_coarse = os.path.join(out_dir, "zs", "coarse")
    os.makedirs(d_zs_fine, exist_ok=True)
    os.makedirs(d_zs_coarse, exist_ok=True)

    d_elev_fine = os.path.join(out_dir, "elevation", "fine")
    d_elev_coarse = os.path.join(out_dir, "elevation", "coarse")
    os.makedirs(d_elev_fine, exist_ok=True)
    os.makedirs(d_elev_coarse, exist_ok=True)

    d_rough = os.path.join(out_dir, "roughness")
    os.makedirs(d_rough, exist_ok=True)

    d_mask_fine = os.path.join(out_dir, "mask", "fine")
    d_mask_coarse = os.path.join(out_dir, "mask", "coarse")
    os.makedirs(d_mask_fine, exist_ok=True)
    os.makedirs(d_mask_coarse, exist_ok=True)

    d_slope = os.path.join(out_dir, "slope")
    d_twi = os.path.join(out_dir, "twi")
    d_as_sin = os.path.join(out_dir, "aspect_sin")
    d_as_cos = os.path.join(out_dir, "aspect_cos")
    os.makedirs(d_slope, exist_ok=True)
    os.makedirs(d_twi, exist_ok=True)
    os.makedirs(d_as_sin, exist_ok=True)
    os.makedirs(d_as_cos, exist_ok=True)

    index_csv = os.path.join(out_dir, "index.csv")

    fine_files = sorted(glob.glob(os.path.join(dir_fine, pat_fine)), key=_natural_key)
    print(f"[check] {scenario}/{var}: fine_files={len(fine_files)} at {dir_fine} pattern={pat_fine}")
    if not fine_files:
        raise RuntimeError(f"Cannot find fine-grid file under {dir_fine} with pattern {pat_fine}")

    ds_c = xr.open_dataset(file_coarse)

    try:
        n_coarse_time = ds_c.sizes.get("time", None)
        if n_coarse_time is None:
            n_coarse_time = ds_c["time"].sizes["time"]
    except Exception:
        n_coarse_time = None

    if isinstance(limit_steps, int) and limit_steps > 0:
        fine_files = fine_files[:limit_steps]

    if (n_coarse_time is not None) and (len(fine_files) > n_coarse_time):
        print(f"[warn] coarse time steps ({n_coarse_time}) < fine files ({len(fine_files)}), truncating fine to coarse.")
        fine_files = fine_files[:n_coarse_time]

    x_c = ds_c["xx_P0"].values.astype(np.float64)
    y_c = ds_c["yy_P0"].values.astype(np.float64)

    _, x_f, y_f = _load_fine(fine_files[0], var)
    Ny_f, Nx_f = len(y_f), len(x_f)

    elev_raw, x_e, y_e = _load_static_gdal(file_elev)
    rough_raw, x_r, y_r = _load_static_gdal(file_rough)
    elev_on_f = _place_on_fine_canvas(elev_raw, x_f, y_f, x_e, y_e)
    rough_on_f = _place_on_fine_canvas(rough_raw, x_f, y_f, x_r, y_r)
    elev_on_c = _downsample_fine_to_coarse_mean(elev_on_f, x_f, y_f, x_c, y_c, scale)

    slope_on_f = twi_on_f = as_sin_on_f = as_cos_on_f = None
    if file_slope:
        slope_raw, x_slope, y_slope = _load_static_nc(file_slope, prefer_vars=["slope_deg", "slope_rad", "slope"])
        slope_on_f = _place_on_fine_canvas(slope_raw, x_f, y_f, x_slope, y_slope)
    if file_twi:
        twi_raw, x_twi, y_twi = _load_static_nc(file_twi, prefer_vars=["TWI", "twi"])
        twi_on_f = _place_on_fine_canvas(twi_raw, x_f, y_f, x_twi, y_twi)
    if file_aspect_sin:
        as_sin_raw, x_asin, y_asin = _load_static_nc(file_aspect_sin, prefer_vars=["aspect_sin"])
        as_sin_on_f = _place_on_fine_canvas(as_sin_raw, x_f, y_f, x_asin, y_asin)
    if file_aspect_cos:
        as_cos_raw, x_acos, y_acos = _load_static_nc(file_aspect_cos, prefer_vars=["aspect_cos"])
        as_cos_on_f = _place_on_fine_canvas(as_cos_raw, x_f, y_f, x_acos, y_acos)

    aoi_mask_full_fine = None
    if shp_aoi and os.path.exists(shp_aoi):
        aoi_mask_full_fine = _rasterize_mask_to_fine(shp_aoi, x_f, y_f)
    if aoi_mask_full_fine is None:
        aoi_mask_full_fine = np.ones((Ny_f, Nx_f), dtype=np.uint8)

    aoi_mask_full_coarse = None
    if shp_aoi and os.path.exists(shp_aoi):
        aoi_mask_full_coarse = _rasterize_mask_to_coarse(shp_aoi, x_c, y_c)
    if aoi_mask_full_coarse is None:
        aoi_mask_full_coarse = np.ones((len(y_c), len(x_c)), dtype=np.uint8)

    n_rows = (Ny_f + stride_fine - 1) // stride_fine
    n_cols = (Nx_f + stride_fine - 1) // stride_fine

    if write_header or (not os.path.exists(index_csv)):
        with open(index_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow([
                "scenario", "var", "t", "time_index", "patch_row", "patch_col",
                "downscale", "patch_size_coarse", "patch_size_fine",
                "x0_fine", "y0_fine", "x1_fine", "y1_fine",
                "src_fine_file",
                "coarse_path", "fine_path",
                "zs_coarse_path", "zs_fine_path",
                "elev_fine_path", "elev_coarse_path", "rough_path",
                "mask_fine_path", "mask_coarse_path",
                "slope_path", "twi_path", "aspect_sin_path", "aspect_cos_path",
                "aoi_ratio", "filtered_out"
            ])

    for ti, ffile in enumerate(fine_files):
        t_tag = _extract_t_from_filename(ffile) or f"t{ti:04d}"
        debug_saved_patches_this_time = 0

        if debug_align:
            if "time" in ds_c:
                coarse_time_val = ds_c["time"].values[ti]
            else:
                coarse_time_val = "NA"

            print(
                f"[time-check] scenario={scenario}, var={var}, "
                f"ti={ti}, fine_file={os.path.basename(ffile)}, "
                f"fine_t_tag={t_tag}, coarse_time={coarse_time_val}"
            )

        da_f, x_f2, y_f2 = _load_fine(ffile, var)
        if not (np.array_equal(x_f, x_f2) and np.array_equal(y_f, y_f2)):
            raise RuntimeError("different fine-grid file coordinates")

        da_c = _load_coarse_at_time(ds_c, var, ti)

        da_zs_f = None
        da_zs_c = None
        if var == "h":
            da_zs_f, x_zs_f, y_zs_f = _load_fine(ffile, "zs")
            if not (np.array_equal(x_f, x_zs_f) and np.array_equal(y_f, y_zs_f)):
                raise RuntimeError(f"fine-grid zs coordinates do not match h coordinates in {ffile}")
            da_zs_c = _load_coarse_at_time(ds_c, "zs", ti)

        for r in range(n_rows):
            y0f = r * stride_fine
            y1f = min(y0f + patch_fine, Ny_f)

            for c in range(n_cols):
                x0f = c * stride_fine
                x1f = min(x0f + patch_fine, Nx_f)

                fine_slice = da_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values
                elev_slice = elev_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values
                rough_slice = rough_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values
                aoi_slice = aoi_mask_full_fine[y0f:y1f, x0f:x1f].astype(np.uint8)

                zs_fine_slice = None
                if var == "h":
                    zs_fine_slice = da_zs_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values

                slope_slice = slope_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values if slope_on_f is not None else None
                twi_slice = twi_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values if twi_on_f is not None else None
                asin_slice = as_sin_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values if as_sin_on_f is not None else None
                acos_slice = as_cos_on_f.isel(yy=slice(y0f, y1f), xx=slice(x0f, x1f)).values if as_cos_on_f is not None else None

                H, W = aoi_slice.shape
                aoi_ratio = float(np.count_nonzero(aoi_slice == 1)) / float(H * W) if H > 0 and W > 0 else 0.0
                filtered = filter_enable and (aoi_ratio < filter_thresh)

                if (H != patch_fine) or (W != patch_fine):
                    fine_block = _pad_to_patch(fine_slice, patch_fine, np.nan)
                    elev_block = _pad_to_patch(elev_slice, patch_fine, np.nan)
                    rough_block = _pad_to_patch(rough_slice, patch_fine, np.nan)
                    aoi_block = _pad_to_patch(aoi_slice, patch_fine, 0).astype(np.uint8)

                    zs_fine_block = None
                    if var == "h":
                        zs_fine_block = _pad_to_patch(zs_fine_slice, patch_fine, np.nan)

                    slope_block = _pad_to_patch(slope_slice, patch_fine, np.nan) if slope_slice is not None else None
                    twi_block = _pad_to_patch(twi_slice, patch_fine, np.nan) if twi_slice is not None else None
                    asin_block = _pad_to_patch(asin_slice, patch_fine, np.nan) if asin_slice is not None else None
                    acos_block = _pad_to_patch(acos_slice, patch_fine, np.nan) if acos_slice is not None else None
                else:
                    fine_block = fine_slice
                    elev_block = elev_slice
                    rough_block = rough_slice
                    aoi_block = aoi_slice

                    zs_fine_block = zs_fine_slice if var == "h" else None

                    slope_block = slope_slice if slope_slice is not None else None
                    twi_block = twi_slice if twi_slice is not None else None
                    asin_block = asin_slice if asin_slice is not None else None
                    acos_block = acos_slice if acos_slice is not None else None

                if snap_mode == "center":
                    j0, i0 = _coarse_patch_start_from_fine_patch_center(
                        x_f=x_f, y_f=y_f,
                        x_c=x_c, y_c=y_c,
                        x0f=x0f, y0f=y0f,
                        patch_fine=patch_fine,
                        patch_coarse=patch_coarse
                    )
                else:
                    x_ref = x_f[x0f]
                    y_ref = y_f[y0f]

                    i0 = int(np.argmin(np.abs(x_c - x_ref)))
                    j0 = int(np.argmin(np.abs(y_c - y_ref)))

                    i0 = max(0, min(i0, x_c.size - patch_coarse))
                    j0 = max(0, min(j0, y_c.size - patch_coarse))

                coarse_block = da_c.isel(
                    yy=slice(j0, j0 + patch_coarse),
                    xx=slice(i0, i0 + patch_coarse)
                ).values.astype(np.float32)

                coarse_mask_block = aoi_mask_full_coarse[j0:j0 + patch_coarse, i0:i0 + patch_coarse]

                elev_coarse_block = elev_on_c.isel(
                    yy=slice(j0, j0 + patch_coarse),
                    xx=slice(i0, i0 + patch_coarse)
                ).values.astype(np.float32)

                if var == "h":
                    zs_coarse_block = da_zs_c.isel(
                        yy=slice(j0, j0 + patch_coarse),
                        xx=slice(i0, i0 + patch_coarse)
                    ).values.astype(np.float32)
                else:
                    zs_coarse_block = None

                elev_coarse_block = np.where(
                    (coarse_mask_block == 1) & np.isfinite(elev_coarse_block),
                    elev_coarse_block,
                    0.0
                ).astype(np.float32)

                coarse_block = np.nan_to_num(coarse_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                coarse_block = np.where(
                    (coarse_mask_block == 1) & np.isfinite(coarse_block),
                    coarse_block,
                    0.0
                ).astype(np.float32)

                if var == "h":
                    zs_coarse_block = np.nan_to_num(zs_coarse_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    zs_coarse_block = np.where(
                        (coarse_mask_block == 1) & np.isfinite(zs_coarse_block),
                        zs_coarse_block,
                        0.0
                    ).astype(np.float32)

                joint_valid = (
                    np.isfinite(fine_block) &
                    np.isfinite(elev_block) &
                    np.isfinite(rough_block) &
                    (aoi_block == 1)
                )

                fine_block[~joint_valid] = 0.0
                elev_block[~joint_valid] = 0.0
                rough_block[~joint_valid] = 0.0

                if var == "h" and zs_fine_block is not None:
                    zs_fine_block[~joint_valid] = 0.0

                if slope_block is not None:
                    slope_block[~joint_valid] = 0.0
                if twi_block is not None:
                    twi_block[~joint_valid] = 0.0
                if asin_block is not None:
                    asin_block[~joint_valid] = 0.0
                if acos_block is not None:
                    acos_block[~joint_valid] = 0.0

                fine_block = np.nan_to_num(fine_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                elev_block = np.nan_to_num(elev_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                rough_block = np.nan_to_num(rough_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                if var == "h" and zs_fine_block is not None:
                    zs_fine_block = np.nan_to_num(zs_fine_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                if var == "h":
                    _clamp_nonneg_inplace(fine_block, eps=depth_eps)
                    _clamp_nonneg_inplace(coarse_block, eps=depth_eps)
                    if clip_max_depth is not None and clip_max_depth > 0:
                        np.minimum(fine_block, clip_max_depth, out=fine_block)
                        np.minimum(coarse_block, clip_max_depth, out=coarse_block)

                    if debug_align and var == "h" and ti < debug_align_max_times:
                        if debug_saved_patches_this_time < debug_align_max_patches:
                            fine_down = _downsample_patch_mean_np(
                                fine_block,
                                scale=scale,
                                patch_coarse=patch_coarse
                            )

                            valid_c = coarse_mask_block.astype(bool)
                            fine_wet_ratio = float(np.mean((fine_down >= 0.1)[valid_c])) if np.any(valid_c) else 0.0
                            coarse_wet_ratio = float(np.mean((coarse_block >= 0.1)[valid_c])) if np.any(
                                valid_c) else 0.0

                            # only save patches with visible flooding
                            if max(fine_wet_ratio, coarse_wet_ratio) >= debug_align_min_wet_ratio:
                                debug_saved_patches_this_time += 1

                                debug_dir = os.path.join(out_dir, "_debug_alignment", scenario, t_tag)
                                os.makedirs(debug_dir, exist_ok=True)

                                diff = fine_down - coarse_block
                                mae = float(np.mean(np.abs(diff[valid_c]))) if np.any(valid_c) else float("nan")
                                rmse = float(np.sqrt(np.mean((diff[valid_c]) ** 2))) if np.any(valid_c) else float(
                                    "nan")

                                fine_wet = (fine_down >= 0.1) & valid_c
                                coarse_wet = (coarse_block >= 0.1) & valid_c
                                tp = int(np.sum(fine_wet & coarse_wet))
                                fp = int(np.sum(coarse_wet & (~fine_wet)))
                                fn = int(np.sum((~coarse_wet) & fine_wet))
                                csi = float(tp / max(tp + fp + fn, 1))

                                print(
                                    f"[space-check] scenario={scenario}, t={t_tag}, "
                                    f"patch r={r}, c={c}, coarse_start=(j0={j0}, i0={i0}), "
                                    f"fine_start=(y0f={y0f}, x0f={x0f}), "
                                    f"fine_wet_ratio={fine_wet_ratio:.6f}, "
                                    f"coarse_wet_ratio={coarse_wet_ratio:.6f}, "
                                    f"fine_down_vs_coarse: MAE={mae:.6f}, RMSE={rmse:.6f}, CSI@0.1={csi:.6f}"
                                )

                                np.savez_compressed(
                                    os.path.join(debug_dir, f"align_r{r:03d}_c{c:03d}.npz"),
                                    coarse=coarse_block.astype(np.float32),
                                    fine=fine_block.astype(np.float32),
                                    fine_down=fine_down.astype(np.float32),
                                    diff=diff.astype(np.float32),
                                    mask_coarse=coarse_mask_block.astype(np.uint8),
                                    mask_fine=aoi_block.astype(np.uint8),
                                    fine_wet_ratio=float(fine_wet_ratio),
                                    coarse_wet_ratio=float(coarse_wet_ratio),
                                    scenario=str(scenario),
                                    t_tag=str(t_tag),
                                    ti=int(ti),
                                    patch_row=int(r),
                                    patch_col=int(c),
                                    j0=int(j0),
                                    i0=int(i0),
                                    y0f=int(y0f),
                                    x0f=int(x0f),
                                    scale=int(scale),
                                )

                elif var in ("u", "v"):
                    if vel_eps > 0.0:
                        np.copyto(fine_block, 0.0, where=np.abs(fine_block) < vel_eps)
                        np.copyto(coarse_block, 0.0, where=np.abs(coarse_block) < vel_eps)
                    if var == "u" and clip_max_vel_u is not None and clip_max_vel_u > 0:
                        np.clip(fine_block, -clip_max_vel_u, clip_max_vel_u, out=fine_block)
                        np.clip(coarse_block, -clip_max_vel_u, clip_max_vel_u, out=coarse_block)
                    if var == "v" and clip_max_vel_v is not None and clip_max_vel_v > 0:
                        np.clip(fine_block, -clip_max_vel_v, clip_max_vel_v, out=fine_block)
                        np.clip(coarse_block, -clip_max_vel_v, clip_max_vel_v, out=coarse_block)

                if slope_block is not None:
                    slope_block = np.nan_to_num(slope_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if twi_block is not None:
                    twi_block = np.nan_to_num(twi_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if asin_block is not None:
                    asin_block = np.nan_to_num(asin_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if acos_block is not None:
                    acos_block = np.nan_to_num(acos_block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                core_tag = f"{var}_{scenario}_{t_tag}_r{r:03d}_c{c:03d}_s{scale}"
                p_coarse = os.path.join(d_coarse, f"{core_tag}_coarse.npy")
                p_fine = os.path.join(d_fine, f"{core_tag}_fine.npy")

                if var == "h":
                    zs_core_tag = f"zs_{scenario}_{t_tag}_r{r:03d}_c{c:03d}_s{scale}"
                    p_zs_coarse = os.path.join(d_zs_coarse, f"{zs_core_tag}_coarse.npy")
                    p_zs_fine = os.path.join(d_zs_fine, f"{zs_core_tag}_fine.npy")
                else:
                    p_zs_coarse = ""
                    p_zs_fine = ""

                p_elev_fine = os.path.join(d_elev_fine, f"elevation_r{r:03d}_c{c:03d}.npy")
                p_elev_coarse = os.path.join(d_elev_coarse, f"elevation_r{r:03d}_c{c:03d}.npy")
                p_rough = os.path.join(d_rough, f"roughness_r{r:03d}_c{c:03d}.npy")
                p_mask_fine = os.path.join(d_mask_fine, f"mask_fine_r{r:03d}_c{c:03d}.npy")
                p_mask_coarse = os.path.join(d_mask_coarse, f"mask_coarse_r{r:03d}_c{c:03d}.npy")
                p_slope = os.path.join(d_slope, f"slope_r{r:03d}_c{c:03d}.npy") if slope_on_f is not None else ""
                p_twi = os.path.join(d_twi, f"twi_r{r:03d}_c{c:03d}.npy") if twi_on_f is not None else ""
                p_asin = os.path.join(d_as_sin, f"aspect_sin_r{r:03d}_c{c:03d}.npy") if as_sin_on_f is not None else ""
                p_acos = os.path.join(d_as_cos, f"aspect_cos_r{r:03d}_c{c:03d}.npy") if as_cos_on_f is not None else ""

                if not filtered:
                    np.save(p_coarse, coarse_block)
                    np.save(p_fine, fine_block)

                    if var == "h":
                        np.save(p_zs_coarse, zs_coarse_block)
                        np.save(p_zs_fine, zs_fine_block)

                    np.save(p_elev_fine, elev_block)
                    np.save(p_elev_coarse, elev_coarse_block)
                    np.save(p_rough, rough_block)
                    np.save(p_mask_fine, aoi_block.astype(np.uint8))
                    np.save(p_mask_coarse, coarse_mask_block.astype(np.uint8))

                    if slope_on_f is not None:
                        np.save(p_slope, slope_block)
                    if twi_on_f is not None:
                        np.save(p_twi, twi_block)
                    if as_sin_on_f is not None:
                        np.save(p_asin, asin_block)
                    if as_cos_on_f is not None:
                        np.save(p_acos, acos_block)
                else:
                    p_coarse = ""
                    p_fine = ""
                    p_zs_coarse = ""
                    p_zs_fine = ""
                    p_elev_fine = ""
                    p_elev_coarse = ""
                    p_rough = ""
                    p_mask_fine = ""
                    p_mask_coarse = ""
                    p_slope = ""
                    p_twi = ""
                    p_asin = ""
                    p_acos = ""

                with open(index_csv, "a", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow([
                        scenario, var, t_tag, ti, r, c,
                        scale, patch_coarse, patch_fine,
                        float(x_f[x0f]), float(y_f[y0f]),
                        float(x_f[min(x1f, x_f.size) - 1]), float(y_f[min(y1f, y_f.size) - 1]),
                        os.path.abspath(ffile),
                        p_coarse, p_fine,
                        p_zs_coarse, p_zs_fine,
                        p_elev_fine, p_elev_coarse, p_rough,
                        p_mask_fine, p_mask_coarse,
                        p_slope, p_twi, p_asin, p_acos,
                        f"{aoi_ratio:.6f}", int(filtered)
                    ])
        print(f"[{scenario}_{var}_{t_tag}] done")

    ds_c.close()
    print("✔ 输出目录：", out_dir)
    print("✔ 索引 CSV：", index_csv)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Cut patches aligned to fine grid; coarse snaps by nearest coarse window to fine-block center."
    )
    ap.add_argument("--var", nargs="+", choices=["h", "u", "v"], required=True, help="vars (h/u/v)")
    ap.add_argument("--scenarios", nargs="+", help="list of rainfall scenarios, e.g. 100y_42h_0c 100y_48h_0c")
    ap.add_argument("--dir-fine-template", default="", help="fine dir template with {scenario} placeholder")
    ap.add_argument("--file-coarse-template", default="", help="coarse file template with {scenario} placeholder")
    ap.add_argument("--dir-fine", default="", help="per_timestep_merged path")
    ap.add_argument("--pat-fine", default="merged_series_t*.nc", help="fine re (default: merged_series_t*.nc)")
    ap.add_argument("--file-coarse", default="", help="coarse BGout.nc path")
    ap.add_argument("--file-elev", required=True, help="Elevation.nc path")
    ap.add_argument("--file-rough", required=True, help="Roughness.nc path")
    ap.add_argument("--aoi", default="", help="AOI shapefile (.shp) path (可空)")
    ap.add_argument("--scale", type=int, default=16, help="downscaling factor (fine/coarse ratio, default: 16)")
    ap.add_argument("--patch-coarse", type=int, default=64, help="coarse patch size(default: 64)")
    ap.add_argument("--out-dir", default="", help="output dir")
    ap.add_argument("--scenario", default="", help="rainfall scenario (comes from file-coarse, e.g. 100y_42h_0c)")
    ap.add_argument("--snap-mode", choices=["center", "ll"], default="center",
                    help="coarse alignment：center=use fine patch center to align (recommend), ll=use lower left corner to align")
    ap.add_argument("--dx-fine", type=float, default=8.0, help="fine grid size, default: 8.0 m")
    ap.add_argument("--file-slope", default="", help="slope NetCDF path")
    ap.add_argument("--file-twi", default="", help="TWI NetCDF path")
    ap.add_argument("--file-aspect-sin", default="", help="aspect_sin NetCDF path")
    ap.add_argument("--file-aspect-cos", default="", help="aspect_cos NetCDF path")
    ap.add_argument("--filter-enable", action="store_true", help="enable AOI coverage ratio to filter (dont save filtering file .npy)")
    ap.add_argument("--filter-thresh", type=float, default=0.2, help="AOI coverage threshold (default: 0.2)")
    ap.add_argument("--depth-eps", type=float, default=5e-5, help="for h only: tiny positive depth -> 0 threshold (m). threshold (0.05 m) x 0.1% = 5e-5")
    ap.add_argument("--vel-eps", type=float, default=1e-5, help="|velocity|<vel_eps -> 0 (m/s). threshold (0.01 m/s) x 0.1% = 1e-5")
    ap.add_argument("--clip-max-depth", type=float, default=5.0, help="clip water depth to this max value (<=0 disables)")
    ap.add_argument("--clip-max-vel-u", type=float, default=5.0, help="clip |u| to this max value (<=0 disables)")
    ap.add_argument("--clip-max-vel-v", type=float, default=5.0, help="clip |v| to this max value (<=0 disables)")
    ap.add_argument("--debug-align", action="store_true", help="print time alignment info and save coarse/fine alignment diagnostics")
    ap.add_argument("--debug-align-max-times", type=int, default=3, help="maximum number of timesteps for alignment debug")
    ap.add_argument("--debug-align-max-patches", type=int, default=4, help="maximum number of patches per timestep for alignment debug")
    ap.add_argument("--debug-align-min-wet-ratio", type=float, default=0.01, help="minimum wet ratio for saving alignment debug patches")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scenario_names, scenario_step_limits = _parse_scenarios_with_limits(args.scenarios or [])
    print(f"scenarios: [{scenario_names}] + limits: [{scenario_step_limits}]")

    if scenario_names:
        dir_fine_tmpl = args.dir_fine_template or _infer_template(args.dir_fine, scenario_names)
        coarse_file_tmpl = args.file_coarse_template or _infer_template(args.file_coarse, scenario_names)

        if not dir_fine_tmpl or not coarse_file_tmpl:
            raise RuntimeError(
                "When using --scenarios, please provide --dir-fine-template and --file-coarse-template, "
                "or make sure --dir-fine/--file-coarse contain a recognizable scenario token."
            )

        header_written = False
        for s in scenario_names:
            dir_fine = dir_fine_tmpl.format(scenario=s)
            file_coarse = coarse_file_tmpl.format(scenario=s)
            s_limit = scenario_step_limits.get(s, None)
            print(f"[info] scenario={s} limit_steps={s_limit if s_limit is not None else 'ALL'}")
            print(f"[info] scenario={s} dir_fine={dir_fine} file_coarse={file_coarse}")

            for v in args.var:
                run(
                    var=v,
                    dir_fine=dir_fine,
                    pat_fine=args.pat_fine,
                    file_coarse=file_coarse,
                    file_elev=args.file_elev,
                    file_rough=args.file_rough,
                    shp_aoi=(args.aoi if args.aoi else None),
                    scale=args.scale,
                    patch_coarse=args.patch_coarse,
                    out_dir=(args.out_dir if args.out_dir else None),
                    scenario=s,
                    snap_mode=args.snap_mode,
                    dx_fine=args.dx_fine,
                    file_slope=args.file_slope,
                    file_twi=args.file_twi,
                    file_aspect_sin=args.file_aspect_sin,
                    file_aspect_cos=args.file_aspect_cos,
                    filter_enable=args.filter_enable,
                    filter_thresh=args.filter_thresh,
                    write_header=(not header_written),
                    limit_steps=s_limit,
                    depth_eps=args.depth_eps,
                    vel_eps=args.vel_eps,
                    clip_max_depth=args.clip_max_depth,
                    clip_max_vel_u=args.clip_max_vel_u,
                    clip_max_vel_v=args.clip_max_vel_v,
                    debug_align=args.debug_align,
                    debug_align_max_times=args.debug_align_max_times,
                    debug_align_max_patches=args.debug_align_max_patches,
                    debug_align_min_wet_ratio=args.debug_align_min_wet_ratio
                )
                header_written = True
    else:
        for i, v in enumerate(args.var):
            run(
                var=v,
                dir_fine=args.dir_fine,
                pat_fine=args.pat_fine,
                file_coarse=args.file_coarse,
                file_elev=args.file_elev,
                file_rough=args.file_rough,
                shp_aoi=(args.aoi if args.aoi else None),
                scale=args.scale,
                patch_coarse=args.patch_coarse,
                out_dir=(args.out_dir if args.out_dir else None),
                scenario=(args.scenario if args.scenario else None),
                snap_mode=args.snap_mode,
                dx_fine=args.dx_fine,
                file_slope=args.file_slope,
                file_twi=args.file_twi,
                file_aspect_sin=args.file_aspect_sin,
                file_aspect_cos=args.file_aspect_cos,
                filter_enable=args.filter_enable,
                filter_thresh=args.filter_thresh,
                write_header=(i == 0),
                depth_eps=args.depth_eps,
                vel_eps=args.vel_eps,
                clip_max_depth=args.clip_max_depth,
                clip_max_vel_u=args.clip_max_vel_u,
                clip_max_vel_v=args.clip_max_vel_v,
                debug_align=args.debug_align,
                debug_align_max_times=args.debug_align_max_times,
                debug_align_max_patches=args.debug_align_max_patches,
                debug_align_min_wet_ratio=args.debug_align_min_wet_ratio
            )