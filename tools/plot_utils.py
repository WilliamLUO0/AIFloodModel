#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting helpers for the tools/ figure scripts.

Consolidated (2026-06-15) from the duplicated boilerplate previously copy-pasted
across plot_dem_rough.py, plot_dem_patches.py, eval_plot.py,
plot_assembled_floodmaps.py and plot_patch_panels.py.

Groups:
  - colormaps:   load_gmt_cpt, make_blues_zero_white
  - raster IO:   load_raster_nc, auto_color_limits, auto_roughness_limits
  - drawing:     extent_origin, overlay_aoi, imshow_raster, add_colorbar, imshow_with_cb
  - patch utils: CORE_RE/FOLDER_RE, parse_core_key, parse_folder_name, build_index_map,
                 pick_pred_file_in_folder, ensure_hw, kron_upsample, apply_aoi_nan
  - flood/diff:  flood_apply_threshold_zero, diff_apply_abs_tol_zero,
                 percentile_positive, percentile_abs_nonzero, percentile_range,
                 get_static_vmin_vmax, make_elev_ticks
"""

import os
import re
import csv
import glob
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ---------------------------------------------------------------------------
# patch-name parsing (e.g. h_100y_42h_0c_t0047_r000_c006_s8)
# ---------------------------------------------------------------------------
CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>\d+y_\d+h_\d+c)_(?P<t>t\d{4})_'
    r'r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)
FOLDER_RE = re.compile(r'^(?P<core>.+)_coarse$')


def parse_core_key(core: str) -> str:
    if not CORE_RE.match(core):
        raise ValueError(f"Bad core key: {core}")
    return core


def parse_folder_name(folder_basename: str):
    m = FOLDER_RE.match(folder_basename)
    if not m:
        return None
    core = m.group("core")
    return core if CORE_RE.match(core) else None


def build_index_map(index_csv: str) -> dict:
    """core key -> index.csv row."""
    mp = {}
    with open(index_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                core = (f'{row["var"]}_{row["scenario"]}_{row["t"]}'
                        f'_r{int(row["patch_row"]):03d}_c{int(row["patch_col"]):03d}'
                        f'_s{int(row["downscale"])}')
            except Exception:
                continue
            mp[core] = row
    return mp


def pick_pred_file_in_folder(folder: str, core: str) -> str:
    """Find the predicted .npy in a results/<core>_coarse/ folder (not *_coarse/_fine)."""
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npys:
        raise RuntimeError(f"No .npy found in {folder}")

    def _ok(p):
        bn = os.path.basename(p)
        return bn.endswith(".npy") and not bn.endswith("_coarse.npy") and not bn.endswith("_fine.npy")

    # prefer names containing pred/sr/output
    cand1 = [p for p in npys if _ok(p) and os.path.basename(p).lower().startswith((core + "_").lower())
             and any(k in os.path.basename(p).lower() for k in ["pred", "sr", "output"])]
    if cand1:
        cand1.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        return cand1[0]
    cand2 = [p for p in npys if _ok(p) and os.path.basename(p).startswith(core + "_")]
    if cand2:
        cand2.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        return cand2[0]
    if len(npys) == 1:
        return npys[0]
    raise RuntimeError(f"Cannot identify predicted .npy in {folder} for core={core}")


def ensure_hw(arr, name="arr"):
    """Squeeze an array of shape (H,W) / (1,H,W) / (1,1,H,W) down to (H, W)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0, ...]
    if a.ndim == 4:
        return a[0, 0, ...]
    raise RuntimeError(f"{name} has unsupported shape: {a.shape}")


def kron_upsample(coarse, scale):
    return np.kron(coarse, np.ones((scale, scale), dtype=coarse.dtype))


def apply_aoi_nan(arr, mask):
    return np.where(mask == 1, arr, np.nan)


# ---------------------------------------------------------------------------
# colormaps
# ---------------------------------------------------------------------------
def load_gmt_cpt(cpt_path, name="gmt_cpt"):
    """Parse a GMT .cpt (RGB) file into a matplotlib LinearSegmentedColormap.

    Handles lines `z1 r1 g1 b1  z2 r2 g2 b2`; skips B/F/N rows, comments and
    `COLOR_MODEL = RGB`. NaN -> white.
    """
    if not cpt_path or not os.path.exists(cpt_path):
        raise RuntimeError(f"CPT not found: {cpt_path}")

    xs, cols = [], []
    with open(cpt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "COLOR_MODEL" in line.upper():
                continue
            if line[0] in ("B", "F", "N"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                z1, r1, g1, b1, z2, r2, g2, b2 = parts[:8]
                z1 = float(z1); z2 = float(z2)
                c1 = (float(r1) / 255.0, float(g1) / 255.0, float(b1) / 255.0)
                c2 = (float(r2) / 255.0, float(g2) / 255.0, float(b2) / 255.0)
            except Exception:
                continue
            xs.extend([z1, z2])
            cols.extend([c1, c2])

    if not xs:
        raise RuntimeError(f"Failed to parse CPT: {cpt_path}")
    zmin, zmax = min(xs), max(xs)
    if np.isclose(zmax, zmin):
        raise RuntimeError(f"Invalid CPT range: zmin==zmax=={zmin}")
    t = [(z - zmin) / (zmax - zmin) for z in xs]
    pts = {ti: ci for ti, ci in zip(t, cols)}
    tt = sorted(pts.keys())
    cc = [pts[k] for k in tt]
    cm = LinearSegmentedColormap.from_list(name, list(zip(tt, cc)))
    try:
        cm.set_bad("white")
    except Exception:
        pass
    return cm


def make_blues_zero_white():
    """Blues colormap with value 0 and NaN forced to white (flood-depth maps)."""
    base = plt.get_cmap("Blues", 256)
    cols = base(np.linspace(0, 1, 256))
    cols[0] = np.array([1, 1, 1, 1], dtype=np.float64)  # 0 -> white
    cm = ListedColormap(cols, name="BluesZeroWhite")
    cm.set_bad(color="white")
    return cm


# ---------------------------------------------------------------------------
# raster IO + color limits
# ---------------------------------------------------------------------------
def load_raster_nc(file_path, var="Band1"):
    """Open a NetCDF raster; return (DataArray[load], x, y). Coords unified to xx/yy."""
    import xarray as xr
    ds = xr.open_dataset(file_path)
    try:
        if var not in ds:
            raise RuntimeError(f"{file_path} lacks var '{var}'. Available: {list(ds.data_vars)}")
        da = ds[var]
        if ("x" in ds) and ("y" in ds):
            da = da.rename({"x": "xx", "y": "yy"})
            x = ds["x"].values.astype(np.float64); y = ds["y"].values.astype(np.float64)
        elif ("x" in ds.coords) and ("y" in ds.coords):
            da = da.rename({"x": "xx", "y": "yy"})
            x = ds.coords["x"].values.astype(np.float64); y = ds.coords["y"].values.astype(np.float64)
        elif ("xx" in ds) and ("yy" in ds):
            x = ds["xx"].values.astype(np.float64); y = ds["yy"].values.astype(np.float64)
        elif ("xx" in ds.coords) and ("yy" in ds.coords):
            x = ds.coords["xx"].values.astype(np.float64); y = ds.coords["yy"].values.astype(np.float64)
        else:
            raise RuntimeError(f"{file_path} lacks x/y or xx/yy coordinates")
        return da.load(), x, y
    finally:
        ds.close()


def auto_color_limits(Z, qmin=1.0, qmax=99.0, nonnegative=False):
    vals = Z[np.isfinite(Z)]
    if vals.size == 0:
        raise RuntimeError("No finite values found for automatic color scaling.")
    vmin = float(np.percentile(vals, qmin))
    vmax = float(np.percentile(vals, qmax))
    if nonnegative:
        vmin = max(0.0, vmin)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(vals)); vmax = float(np.max(vals))
        if nonnegative:
            vmin = max(0.0, vmin)
    return vmin, vmax


def auto_roughness_limits(Z, qmax=99.0):
    vals = Z[np.isfinite(Z)]
    if vals.size == 0:
        raise RuntimeError("No finite values found for roughness scaling.")
    vals = vals[vals >= 0]
    if vals.size == 0:
        raise RuntimeError("No nonnegative finite values found for roughness scaling.")
    vmin = 0.0
    vmax = float(np.percentile(vals, qmax))
    if (not np.isfinite(vmax)) or (vmax <= vmin):
        vmax = float(np.max(vals))
    return vmin, vmax


def percentile_range(arr, qlo, qhi):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [qlo, qhi])
    if np.isclose(lo, hi):
        lo2, hi2 = float(v.min()), float(v.max())
        return (lo2, lo2 + 1.0) if np.isclose(lo2, hi2) else (lo2, hi2)
    return float(lo), float(hi)


def get_static_vmin_vmax(arr, mode, qlo, qhi):
    if mode == "minmax":
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = float(v.min()), float(v.max())
        return (lo, lo + 1.0) if np.isclose(lo, hi) else (lo, hi)
    return percentile_range(arr, qlo, qhi)


def percentile_positive(arr, q):
    v = arr[np.isfinite(arr)]
    v = v[v > 0]
    if v.size == 0:
        return 1.0
    out = float(np.percentile(v, q))
    if (not np.isfinite(out)) or (out <= 0):
        out = float(np.max(v))
    return max(out, 1e-8)


def percentile_abs_nonzero(arr, q, fallback):
    v = np.abs(arr[np.isfinite(arr)])
    v = v[v > 0]
    if v.size == 0:
        return max(fallback, 1e-6)
    out = float(np.percentile(v, q))
    if (not np.isfinite(out)) or (out <= 0):
        out = float(np.max(v))
    return max(out, fallback, 1e-6)


def make_elev_ticks(vmin, vmax, step=100):
    if vmin is None or vmax is None or (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return None
    t0 = int(np.ceil(vmin / step) * step)
    t1 = int(np.floor(vmax / step) * step)
    return list(range(t0, t1 + step, step)) if t1 >= t0 else None


# ---------------------------------------------------------------------------
# flood / diff thresholding
# ---------------------------------------------------------------------------
def flood_apply_threshold_zero(a, thr):
    """AOI-masked array: values < thr -> 0 (outside AOI stays NaN)."""
    if thr is None or thr <= 0:
        return a
    return np.where(np.isfinite(a) & (a < thr), 0.0, a)


def diff_apply_abs_tol_zero(d, abs_tol):
    """AOI-masked array: |d| < abs_tol -> 0 (outside AOI stays NaN)."""
    if abs_tol is None or abs_tol <= 0:
        return d
    return np.where(np.isfinite(d) & (np.abs(d) < abs_tol), 0.0, d)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def extent_origin(x, y):
    """imshow extent + origin from 1D coordinate arrays."""
    origin = "lower" if (y[0] < y[-1]) else "upper"
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    return extent, origin


def overlay_aoi(ax, aoi_path, dem_epsg=2193, edgecolor="black", linewidth=2.0):
    """Draw an AOI shapefile outline on `ax`. No-op (with warning) on failure."""
    if not aoi_path:
        return
    try:
        import geopandas as gpd
        gdf = gpd.read_file(aoi_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=dem_epsg)
        else:
            try:
                gdf = gdf.to_crs(epsg=dem_epsg)
            except Exception:
                pass
        gdf.plot(ax=ax, facecolor="none", edgecolor=edgecolor, linewidth=linewidth)
    except Exception as e:
        print(f"[warn] AOI plot failed: {e}")


def imshow_raster(ax, Z, x, y, cmap="viridis", vmin=None, vmax=None):
    extent, origin = extent_origin(x, y)
    return ax.imshow(Z, extent=extent, origin=origin, cmap=cmap,
                     vmin=vmin, vmax=vmax, interpolation="nearest")


def add_colorbar(fig, im, ax, scientific=False, ticks=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if scientific:
        fmt = mticker.ScalarFormatter(useMathText=False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        cb.formatter = fmt
        cb.update_ticks()
    else:
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        cb.formatter = fmt
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.update_ticks()
    return cb


def imshow_with_cb(fig, ax, data, title, cmap, vmin=None, vmax=None,
                   contour_mask=None, scientific=False, cb_ticks=None, origin="lower"):
    im = ax.imshow(data, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if contour_mask is not None:
        ax.contour(contour_mask, levels=[0.5], colors="k", linewidths=0.8)
    add_colorbar(fig, im, ax, scientific=scientific, ticks=cb_ticks)
    return im
