#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the fine-grid STATIC features used by the model, each with the AOI outline,
either as individual figures or as one combined panel; optionally overlay the
training patch grid (r/c boxes) on the DEM.

Merges the old tools/9_4_present_plot/plot_dem_rough.py (DEM + roughness rasters
+ AOI) and tools/plot_dem_patches.py (DEM + AOI + patch-grid overlay).

Static features (full-domain NetCDF rasters; same sources as make_patches.py):
  dem          --file-elev        Elevation.nc          (GMT .cpt, --cpt)
  roughness    --file-rough       Roughness.nc
  slope        --file-slope       Topo_Attrs/Slope_Deg.nc
  twi          --file-twi         Topo_Attrs/TWI.nc
  aspect_sin   --file-aspect-sin  Topo_Attrs/Aspect_SIN.nc
  aspect_cos   --file-aspect-cos  Topo_Attrs/Aspect_COS.nc
(mask is the AOI itself, drawn as the outline overlaid on every feature.)

Examples
--------
# all features as one panel + AOI
python tools/plot_static_features.py \
  --file-elev .../Elevation.nc --cpt .../wiki-france.cpt \
  --file-rough .../Roughness.nc --file-slope .../Slope_Deg.nc --file-twi .../TWI.nc \
  --file-aspect-sin .../Aspect_SIN.nc --file-aspect-cos .../Aspect_COS.nc \
  --aoi .../Gisborne_basin.shp --mode panel --out tools/static_features_panel.png

# just the DEM, individually, with the 8x patch grid overlaid
python tools/plot_static_features.py --file-elev .../Elevation.nc --cpt .../wiki-france.cpt \
  --aoi .../Gisborne_basin.shp --features dem --mode individual --out-dir tools/static \
  --patch-grid --scale 8 --patch-coarse 64
"""

import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_utils import (
    load_gmt_cpt, load_raster_nc, auto_color_limits, auto_roughness_limits,
    overlay_aoi, extent_origin,
)

ALL_FEATURES = ["dem", "roughness", "slope", "twi", "aspect_sin", "aspect_cos"]

# per-feature display defaults: (matplotlib cmap, colorbar label, color-limit rule)
#   rule: "dem"=GMT cpt + fixed ticks; "rough"=0..q99; "pos"=q1..q99 nonneg; "pm1"=[-1,1]
FEATURE_CFG = {
    "dem":        dict(cmap=None,       label="Elevation [m]", rule="dem",   aoi_edge="black"),
    "roughness":  dict(cmap="viridis",  label="Roughness",     rule="rough", aoi_edge="white"),
    "slope":      dict(cmap="cividis",  label="Slope [deg]",   rule="pos",   aoi_edge="white"),
    "twi":        dict(cmap="YlGnBu",   label="TWI",           rule="pos",   aoi_edge="black"),
    "aspect_sin": dict(cmap="twilight", label="Aspect sin",    rule="pm1",   aoi_edge="black"),
    "aspect_cos": dict(cmap="twilight", label="Aspect cos",    rule="pm1",   aoi_edge="black"),
}

DEM_TICKS = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]


def feature_file(args, feat):
    return {
        "dem":        args.file_elev,
        "roughness":  args.file_rough,
        "slope":      args.file_slope,
        "twi":        args.file_twi,
        "aspect_sin": args.file_aspect_sin,
        "aspect_cos": args.file_aspect_cos,
    }[feat]


def resolve_limits(feat, Z, args, dem_cmap):
    """Return (cmap, vmin, vmax) for a feature."""
    cfg = FEATURE_CFG[feat]
    rule = cfg["rule"]
    if rule == "dem":
        return dem_cmap, args.dem_vmin, args.dem_vmax
    if rule == "rough":
        vmin, vmax = auto_roughness_limits(Z, qmax=args.qmax)
        return cfg["cmap"], vmin, vmax
    if rule == "pm1":
        return cfg["cmap"], -1.0, 1.0
    # "pos"
    vmin, vmax = auto_color_limits(Z, qmin=args.qmin, qmax=args.qmax, nonnegative=True)
    return cfg["cmap"], vmin, vmax


def draw_patch_grid(ax, x, y, Z_shape, scale, patch_coarse, lw=1.0, alpha=0.9,
                    label_every=1, label_font=7.0, edgecolor="red"):
    """Overlay the r/c training-patch boxes (stride = patch_coarse*scale) with labels."""
    patch_fine = patch_coarse * scale
    stride = patch_fine
    Ny, Nx = Z_shape
    n_rows = (Ny + stride - 1) // stride
    n_cols = (Nx + stride - 1) // stride
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    label_every = max(1, int(label_every))
    for r in range(n_rows):
        y0 = r * stride; y1 = min(y0 + patch_fine, Ny)
        for c in range(n_cols):
            x0 = c * stride; x1 = min(x0 + patch_fine, Nx)
            x_left = x[x0] - dx / 2.0; x_right = x[x1 - 1] + dx / 2.0
            y_a = y[y0] - dy / 2.0;    y_b = y[y1 - 1] + dy / 2.0
            xmin, xmax = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
            ymin, ymax = (y_a, y_b) if y_a <= y_b else (y_b, y_a)
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, linewidth=lw, edgecolor=edgecolor, alpha=alpha))
            if ((r * n_cols + c) % label_every) == 0:
                ax.text(xmin + 0.02 * (xmax - xmin), ymax - 0.02 * (ymax - ymin),
                        f"r{r:02d}_c{c:02d}", fontsize=label_font, color=edgecolor,
                        ha="left", va="top", alpha=0.95)
    return n_rows, n_cols


def render_feature(ax, feat, args, dem_cmap):
    """Load a feature raster and draw it (imshow + AOI + optional patch grid) on ax."""
    path = feature_file(args, feat)
    da, x, y = load_raster_nc(path, var=args.var_name)
    Z = da.values.astype(np.float32)
    cmap, vmin, vmax = resolve_limits(feat, Z, args, dem_cmap)
    extent, origin = extent_origin(x, y)
    im = ax.imshow(Z, extent=extent, origin=origin, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(FEATURE_CFG[feat]["label"])
    if feat == "dem" and (vmin is not None) and (vmax is not None):
        ticks = [t for t in DEM_TICKS if vmin <= t <= vmax] or None
        if ticks:
            cb.set_ticks(ticks)
            cb.ax.set_yticklabels([f"{t:.0f}" for t in ticks])
    ax.set_xlabel("Easting [m]"); ax.set_ylabel("Northing [m]")
    ax.set_title(feat)
    overlay_aoi(ax, args.aoi, dem_epsg=args.dem_epsg, edgecolor=FEATURE_CFG[feat]["aoi_edge"])
    if args.patch_grid and feat == "dem":
        draw_patch_grid(ax, x, y, Z.shape, args.scale, args.patch_coarse,
                        lw=args.lw, label_every=args.label_every, label_font=args.label_font)
    ax.set_aspect("equal", adjustable="box")


def main():
    ap = argparse.ArgumentParser("Plot fine-grid static features (+AOI), individual or panel.")
    ap.add_argument("--file-elev", default="", help="Elevation.nc (dem)")
    ap.add_argument("--file-rough", default="", help="Roughness.nc")
    ap.add_argument("--file-slope", default="", help="Slope_Deg.nc")
    ap.add_argument("--file-twi", default="", help="TWI.nc")
    ap.add_argument("--file-aspect-sin", default="", help="Aspect_SIN.nc")
    ap.add_argument("--file-aspect-cos", default="", help="Aspect_COS.nc")
    ap.add_argument("--var-name", default="Band1", help="NetCDF variable name (default Band1)")
    ap.add_argument("--cpt", default="", help="GMT .cpt for the DEM (required if dem is plotted)")

    ap.add_argument("--features", default="all",
                    help="comma list from {dem,roughness,slope,twi,aspect_sin,aspect_cos} or 'all'")
    ap.add_argument("--mode", choices=["individual", "panel"], default="panel")
    ap.add_argument("--out", default="", help="output png (panel mode)")
    ap.add_argument("--out-dir", default="", help="output dir (individual mode)")

    ap.add_argument("--aoi", default="", help="AOI shapefile (.shp), optional")
    ap.add_argument("--dem-epsg", type=int, default=2193)
    ap.add_argument("--dem-vmin", type=float, default=None)
    ap.add_argument("--dem-vmax", type=float, default=None)
    ap.add_argument("--qmin", type=float, default=1.0, help="auto color min percentile")
    ap.add_argument("--qmax", type=float, default=99.0, help="auto color max percentile")

    # patch-grid overlay (on DEM)
    ap.add_argument("--patch-grid", action="store_true", help="overlay r/c patch boxes on the DEM")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--patch-coarse", type=int, default=64)
    ap.add_argument("--lw", type=float, default=1.0)
    ap.add_argument("--label-every", type=int, default=1)
    ap.add_argument("--label-font", type=float, default=7.0)

    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    feats = ALL_FEATURES if args.features.strip() == "all" else \
        [f.strip() for f in args.features.split(",") if f.strip()]
    # keep only features whose source file was provided
    feats = [f for f in feats if feature_file(args, f)]
    if not feats:
        raise SystemExit("[plot_static_features] no features to plot (provide --file-* paths).")
    if "dem" in feats and not args.cpt:
        raise SystemExit("[plot_static_features] --cpt is required when plotting the dem feature.")

    dem_cmap = load_gmt_cpt(args.cpt) if ("dem" in feats and args.cpt) else None

    if args.mode == "individual":
        out_dir = args.out_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        for feat in feats:
            fig, ax = plt.subplots(figsize=(12, 10), dpi=args.dpi)
            render_feature(ax, feat, args, dem_cmap)
            plt.tight_layout()
            out = os.path.join(out_dir, f"static_{feat}.png")
            fig.savefig(out, bbox_inches="tight"); plt.close(fig)
            print(f"[OK] saved: {out}")
    else:
        if not args.out:
            raise SystemExit("[plot_static_features] --out is required in panel mode.")
        n = len(feats)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), dpi=args.dpi)
        axes = np.atleast_1d(axes).ravel()
        for ax, feat in zip(axes, feats):
            render_feature(ax, feat, args, dem_cmap)
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] saved panel ({nrows}x{ncols}, {n} features): {args.out}")


if __name__ == "__main__":
    main()
