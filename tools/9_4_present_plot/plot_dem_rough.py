import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_grid(file_path: str, var="Band1"):
    ds = xr.open_dataset(file_path)
    try:
        if var not in ds:
            raise RuntimeError(f"{file_path} lacks var '{var}'. Available: {list(ds.data_vars)}")

        da = ds[var]

        # 统一坐标名到 xx/yy
        if ("x" in ds) and ("y" in ds):
            da = da.rename({"x": "xx", "y": "yy"})
            x = ds["x"].values.astype(np.float64)
            y = ds["y"].values.astype(np.float64)
        elif ("x" in ds.coords) and ("y" in ds.coords):
            da = da.rename({"x": "xx", "y": "yy"})
            x = ds.coords["x"].values.astype(np.float64)
            y = ds.coords["y"].values.astype(np.float64)
        elif ("xx" in ds) and ("yy" in ds):
            x = ds["xx"].values.astype(np.float64)
            y = ds["yy"].values.astype(np.float64)
        elif ("xx" in ds.coords) and ("yy" in ds.coords):
            x = ds.coords["xx"].values.astype(np.float64)
            y = ds.coords["yy"].values.astype(np.float64)
        else:
            raise RuntimeError(f"{file_path} lacks x/y or xx/yy coordinates")

        return da.load(), x, y
    finally:
        ds.close()


def load_gmt_cpt(cpt_path: str, name="gmt_cpt"):
    """
    读取常见 GMT CPT (RGB)：
      z1 r1 g1 b1  z2 r2 g2 b2
    忽略 B/F/N 行、注释行；支持 'COLOR_MODEL = RGB'
    """
    xs = []
    cols = []

    with open(cpt_path, "r") as f:
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
                z1 = float(z1)
                z2 = float(z2)
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

    pts = {}
    for ti, ci in zip(t, cols):
        pts[ti] = ci

    tt = sorted(pts.keys())
    cc = [pts[k] for k in tt]

    return LinearSegmentedColormap.from_list(name, list(zip(tt, cc)))


def auto_color_limits(Z, qmin=1.0, qmax=99.0, nonnegative=False):
    vals = Z[np.isfinite(Z)]
    if vals.size == 0:
        raise RuntimeError("No finite values found for automatic color scaling.")

    vmin = float(np.percentile(vals, qmin))
    vmax = float(np.percentile(vals, qmax))

    if nonnegative:
        vmin = max(0.0, vmin)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("Invalid automatic color limits.")

    if vmax <= vmin:
        # fallback，避免色条范围出问题
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
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

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.max(vals))

    return vmin, vmax


def plot_raster_with_aoi(
    data_da,
    x,
    y,
    out_path,
    title,
    cbar_label,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aoi_path="",
    dem_epsg=2193,
    figsize=(12, 10),
    dpi=200,
    cbar_ticks=None,
    aoi_edgecolor="black",
):
    Z = data_da.values.astype(np.float32)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    origin = "lower" if (y[0] < y[-1]) else "upper"
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

    im = ax.imshow(
        Z,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label)

    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
        cb.ax.set_yticklabels([f"{t:.0f}" for t in cbar_ticks])

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(title)

    if aoi_path:
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

            gdf.plot(ax=ax, facecolor="none", edgecolor=aoi_edgecolor, linewidth=2.0)
        except Exception as e:
            print(f"[warn] AOI plot failed for {out_path}: {e}")

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out_path}")

def main():
    ap = argparse.ArgumentParser("Plot DEM + AOI and roughness + AOI")

    # DEM
    ap.add_argument("--file-elev", required=True, help="Elevation nc path")
    ap.add_argument("--elev-var", default="Band1", help="DEM variable name")

    # Roughness
    ap.add_argument("--file-rough", required=True, help="Roughness nc path")
    ap.add_argument("--rough-var", default="Band1", help="Roughness variable name")

    # DEM colormap
    ap.add_argument("--cpt", required=True, help="GMT .cpt file for DEM")

    # AOI
    ap.add_argument("--aoi", default="", help="AOI shapefile (.shp), optional")
    ap.add_argument("--dem-epsg", type=int, default=2193, help="assumed DEM CRS EPSG (default 2193)")

    # Output
    ap.add_argument("--out-dem", required=True, help="output DEM png path")
    ap.add_argument("--out-rough", required=True, help="output roughness png path")

    # DEM display range
    ap.add_argument("--dem-vmin", type=float, default=None, help="DEM color min")
    ap.add_argument("--dem-vmax", type=float, default=None, help="DEM color max")

    # Roughness display range
    ap.add_argument("--rough-vmin", type=float, default=None, help="roughness color min")
    ap.add_argument("--rough-vmax", type=float, default=None, help="roughness color max")
    ap.add_argument("--rough-qmin", type=float, default=1.0, help="roughness auto color min percentile")
    ap.add_argument("--rough-qmax", type=float, default=99.0, help="roughness auto color max percentile")

    # Roughness cmap
    ap.add_argument("--rough-cmap", default="viridis", help="matplotlib cmap for roughness")

    args = ap.parse_args()

    # DEM
    dem_da, dem_x, dem_y = load_grid(args.file_elev, var=args.elev_var)
    dem_cmap = load_gmt_cpt(args.cpt)
    dem_ticks = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]

    plot_raster_with_aoi(
        data_da=dem_da,
        x=dem_x,
        y=dem_y,
        out_path=args.out_dem,
        title=f"DEM",
        cbar_label="Elevation [m]",
        cmap=dem_cmap,
        vmin=args.dem_vmin,
        vmax=args.dem_vmax,
        aoi_path=args.aoi,
        dem_epsg=args.dem_epsg,
        cbar_ticks=dem_ticks,
        aoi_edgecolor="black",
    )

    # Roughness
    rough_da, rough_x, rough_y = load_grid(args.file_rough, var=args.rough_var)
    rough_Z = rough_da.values.astype(np.float32)

    if args.rough_vmin is None or args.rough_vmax is None:
        auto_vmin, auto_vmax = auto_roughness_limits(rough_Z, qmax=args.rough_qmax)
        rough_vmin = args.rough_vmin if args.rough_vmin is not None else auto_vmin
        rough_vmax = args.rough_vmax if args.rough_vmax is not None else auto_vmax
    else:
        rough_vmin = args.rough_vmin
        rough_vmax = args.rough_vmax

    print(f"[INFO] roughness color limits: vmin={rough_vmin:.6f}, vmax={rough_vmax:.6f}")

    plot_raster_with_aoi(
        data_da=rough_da,
        x=rough_x,
        y=rough_y,
        out_path=args.out_rough,
        title=f"Roughness",
        cbar_label="Roughness",
        cmap=args.rough_cmap,
        vmin=rough_vmin,
        vmax=rough_vmax,
        aoi_path=args.aoi,
        dem_epsg=args.dem_epsg,
        aoi_edgecolor="white",
    )

if __name__ == "__main__":
    main()