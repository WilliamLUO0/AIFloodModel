import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def load_dem(file_elev: str, var="Band1"):
    ds = xr.open_dataset(file_elev)
    try:
        if var not in ds:
            raise RuntimeError(f"{file_elev} lacks var '{var}'. Available: {list(ds.data_vars)}")

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
            raise RuntimeError(f"{file_elev} lacks x/y or xx/yy coordinates")

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
            # 跳过色彩模型声明
            if "COLOR_MODEL" in line.upper():
                continue
            # 跳过 B/F/N
            if line[0] in ("B", "F", "N"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                z1, r1, g1, b1, z2, r2, g2, b2 = parts[:8]
                z1 = float(z1)
                z2 = float(z2)
                c1 = (float(r1)/255.0, float(g1)/255.0, float(b1)/255.0)
                c2 = (float(r2)/255.0, float(g2)/255.0, float(b2)/255.0)
            except Exception:
                continue

            xs.extend([z1, z2])
            cols.extend([c1, c2])

    if not xs:
        raise RuntimeError(f"Failed to parse CPT: {cpt_path}")

    zmin, zmax = min(xs), max(xs)
    if np.isclose(zmax, zmin):
        raise RuntimeError(f"Invalid CPT range: zmin==zmax=={zmin}")

    # 归一化到 [0,1]
    t = [(z - zmin) / (zmax - zmin) for z in xs]

    # 去重（保留后出现的点也没问题）
    pts = {}
    for ti, ci in zip(t, cols):
        pts[ti] = ci

    tt = sorted(pts.keys())
    cc = [pts[k] for k in tt]

    return LinearSegmentedColormap.from_list(name, list(zip(tt, cc)))


def main():
    ap = argparse.ArgumentParser("Plot DEM with GMT CPT + overlay AOI + patch boxes with r/c labels.")
    ap.add_argument("--file-elev", required=True, help="Elevation.nc path")
    ap.add_argument("--elev-var", default="Band1", help="DEM var name (default Band1)")
    ap.add_argument("--cpt", required=True, help="GMT .cpt file path (e.g., wiki-france.cpt)")
    ap.add_argument("--aoi", default="", help="AOI shapefile (.shp), optional")
    ap.add_argument("--out", required=True, help="output png path")

    ap.add_argument("--scale", type=int, default=16)
    ap.add_argument("--patch-coarse", type=int, default=64)

    ap.add_argument("--lw", type=float, default=1.0, help="box line width")
    ap.add_argument("--alpha", type=float, default=0.9, help="box alpha")
    ap.add_argument("--label-every", type=int, default=1, help="label every Nth patch (default 1=all)")
    ap.add_argument("--label-font", type=float, default=7.0, help="label font size")

    ap.add_argument("--vmin", type=float, default=None, help="color min (e.g., -500)")
    ap.add_argument("--vmax", type=float, default=None, help="color max (e.g., 500)")

    # 如果 AOI CRS 和 DEM 不一致，你可以显式告诉脚本 DEM 的 EPSG（NZTM 常用 2193）
    ap.add_argument("--dem-epsg", type=int, default=2193, help="assumed DEM CRS EPSG (default 2193)")

    args = ap.parse_args()

    patch_fine = args.patch_coarse * args.scale
    stride = patch_fine

    dem_da, x, y = load_dem(args.file_elev, var=args.elev_var)
    Z = dem_da.values.astype(np.float32)

    Ny, Nx = Z.shape
    n_rows = (Ny + stride - 1) // stride
    n_cols = (Nx + stride - 1) // stride

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))

    cmap = load_gmt_cpt(args.cpt)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    origin = "lower" if (y[0] < y[-1]) else "upper"
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

    im = ax.imshow(
        Z,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Elevation [m]")

    if args.vmin is not None and args.vmax is not None:
        ticks = np.linspace(args.vmin, args.vmax, 11)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{t:.0f}" for t in ticks])

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"DEM (CPT={os.path.basename(args.cpt)}) + AOI + patches ({n_rows}x{n_cols}={n_rows*n_cols})")

    # 画 AOI（可选）
    if args.aoi:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(args.aoi)

            # 如果 AOI 没 CRS，就按 2193 处理；如果有 CRS，但不是 DEM CRS，就重投影
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=args.dem_epsg)
            else:
                try:
                    gdf = gdf.to_crs(epsg=args.dem_epsg)
                except Exception:
                    # 如果投影失败，就直接画（至少能看到大概位置）
                    pass

            gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0)
        except Exception as e:
            print(f"[warn] AOI plot failed: {e}")

    # 画 patch 框 + r/c 标签
    label_every = max(1, int(args.label_every))
    drawn = 0
    labeled = 0

    for r in range(n_rows):
        y0 = r * stride
        y1 = min(y0 + patch_fine, Ny)

        for c in range(n_cols):
            x0 = c * stride
            x1 = min(x0 + patch_fine, Nx)

            # 像元边界：中心 +/- 半像元
            x_left = x[x0] - dx / 2.0
            x_right = x[x1 - 1] + dx / 2.0
            y_a = y[y0] - dy / 2.0
            y_b = y[y1 - 1] + dy / 2.0

            xmin, xmax = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
            ymin, ymax = (y_a, y_b) if y_a <= y_b else (y_b, y_a)

            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linewidth=args.lw,
                edgecolor="red",
                alpha=args.alpha,
            )
            ax.add_patch(rect)
            drawn += 1

            idx = r * n_cols + c
            if (idx % label_every) == 0:
                tx = xmin + 0.02 * (xmax - xmin)
                ty = ymax - 0.02 * (ymax - ymin)
                ax.text(tx, ty, f"r{r:02d}_c{c:02d}",
                        fontsize=args.label_font, color="red",
                        ha="left", va="top", alpha=0.95)
                labeled += 1

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] out={args.out}")
    print(f"[OK] boxes={drawn}, labels={labeled}, grid={n_rows}x{n_cols}={n_rows*n_cols}")


if __name__ == "__main__":
    main()
