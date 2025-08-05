import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import rioxarray

# ==== 1. 加载 DEM ====
dem_path = "./data/porangahau_basin/merged_porangahau_basin.nc"
# dem_path = "./data/Porangahau/8m_geofabric.nc"
ds = xr.open_dataset(dem_path)
# z = ds["z"]  # 变量名是 z（地面高程）
z = ds["Band1"]
# print(ds)
print("====0====")
print(z)
print("====1====")

# 设置空间维度 + 坐标系（NZTM2000）+ 处理FillValue为NaN
z.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
z.rio.write_crs("EPSG:2193", inplace=True)

if "_FillValue" in z.attrs:
    z = z.where(z != z.attrs["_FillValue"])

print(z)
print("CRS:", z.rio.crs)
print("x_dim:", z.rio.x_dim)
print("y_dim:", z.rio.y_dim)
print("====2====")

# ==== 2. 显示原始空间范围 ====
print("Original DEM extent:")
print(f"xmin = {float(z.x.min())}, xmax = {float(z.x.max())}")
print(f"ymin = {float(z.y[0])}, ymax = {float(z.y[-1])}")
print("====3====")

# # ==== 3. 按 BG-Flood 给定区域裁剪 ====
# xmin_clip = 1898150.0
# xmax_clip = 1922940.0
# ymin_clip = 5523060.0
# ymax_clip = 5545740.0

# # 注意：Y 是从上到下 → 需要从 ymax 到 ymin
# print("=== 检查Y的方向 ===")
# print(z.y.values[:10])  # 打印前10个y
# print(z.y.values[-10:])  # 打印最后10个y
# # 从大到小 ymax - ymin
# # z_clipped = z.sel(x=slice(xmin_clip, xmax_clip), y=slice(ymax_clip, ymin_clip))
# # 从小到大 ymin - ymax
# z_clipped = z.sel(x=slice(xmin_clip, xmax_clip), y=slice(ymin_clip, ymax_clip))
# print("=== 裁剪结果 ===")
# print(z_clipped)
z_clipped = z

# ==== 4. 上采样到 128m（block mean）====
original_dx = float(z.x[1] - z.x[0])  # 原始分辨率 = 8m
print(z.x[1])
print(z.x[0])
target_dx = 128.0
scale_factor = int(target_dx / original_dx)

z_upsampled = z_clipped.coarsen(x=scale_factor, y=scale_factor, boundary="trim").mean()
print("=== 上采样结果 ===")
print(z_upsampled)
print(z_upsampled.values.shape)
print("=== x coord ===")
print(z_upsampled.x)
print("====4====")

# ==== 5. 读取 AOI 并构建 Polygon ====
aoi_path = "./data/porangahau_basin/aoi_BGFLOOD.txt"
aoi_points = pd.read_csv(aoi_path, sep="\s+", header=None, names=["x", "y"])
aoi_polygon = Polygon(aoi_points.values)

# ==== 6. 构建 AOI 掩膜 ====
transform = from_origin(
    float(z_upsampled.x.min()), float(z_upsampled.y.max()), target_dx, target_dx
)
mask = geometry_mask(
    [aoi_polygon],
    out_shape=(z_upsampled.sizes["y"], z_upsampled.sizes["x"]),
    transform=transform,
    invert=True
)

# ==== 7. 应用掩膜 ====
z_masked = z_upsampled.where(mask)
print(z_masked)

# ==== 8. 可视化结果 ====
plt.figure(figsize=(10, 8))
z.plot(cmap="terrain")
plt.title("DEM (128m)")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
z_upsampled.plot(cmap="terrain")
plt.title("DEM (128m)")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
z_masked.plot(cmap="terrain")
plt.title("Downsampled DEM (128m) with AOI mask applied")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()
