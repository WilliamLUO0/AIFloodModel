import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt

# ==== 1. 加载 Manning 粗糙度数据 ====
manning_path = "./data/porangahau_basin/merged_porangahau_basin_zo.nc"
ds = xr.open_dataset(manning_path)
n = ds["Band1"]
print("====0====")
# print(ds)
print(n)
print("====1====")

# === 2. 设置空间维度和投影信息 ===
n.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
n.rio.write_crs("EPSG:2193", inplace=True)

if "_FillValue" in n.attrs:
    n = n.where(n != n.attrs["_FillValue"])

print("\nCRS:", n.rio.crs)
print("Original DEM extent:")
print("x range:", float(n.x.min()), "to", float(n.x.max()))
print("y range:", float(n.y[0]), "to", float(n.y[-1]))
print("====2====")

# # === 3. 裁剪到 BG-FLOOD 区域 ===
# xmin, xmax = 1898150.0, 1922940.0
# ymin, ymax = 5523060.0, 5545740.0
# n_clipped = n.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))  # y 是从上到下
# print("\n")
# print(n_clipped)
# print(n_clipped.values.shape)
# print("====3====")
n_clipped = n

# === 4. 读取 AOI 并构建 Polygon ===
aoi_path = "./data/porangahau_basin/aoi_BGFLOOD.txt"
aoi_points = pd.read_csv(aoi_path, sep="\s+", header=None, names=["x", "y"])
aoi_polygon = Polygon(aoi_points.values)

# ==== 5. 构建 AOI 掩膜 ====
transform = from_origin(
    float(n_clipped.x.min()), float(n_clipped.y.max()), 8, 8
)
mask = geometry_mask(
    [aoi_polygon],
    out_shape=(n_clipped.sizes["y"], n_clipped.sizes["x"]),
    transform=transform,
    invert=True
)

# ==== 6. 应用掩膜 ====
n_masked = n_clipped.where(mask)

# === 7. 可视化 ===
plt.figure(figsize=(10, 8))
n_clipped.plot(cmap="viridis")
plt.title("Manning Roughness Coefficient (clipped)")
plt.xlabel("X (EPSG:2193)")
plt.ylabel("Y (EPSG:2193)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
n_masked.plot(cmap="viridis")
plt.title("Manning Roughness Coefficient (clipped) with AOI mask applied")
plt.xlabel("X (EPSG:2193)")
plt.ylabel("Y (EPSG:2193)")
plt.tight_layout()
plt.show()

# print("非 NaN 值的最大值：", float(n_clipped.max()))
# print("非 NaN 值的最小值：", float(n_clipped.min()))
# print("非 NaN 值的均值：", float(n_clipped.mean()))
# print("一共有多少 NaN：", int(n_clipped.isnull().sum()))
# print("总像素数：", int(n_clipped.size))

print("\n")
print("非 NaN 值的最大值：", float(n_masked.max()))
print("非 NaN 值的最小值：", float(n_masked.min()))
print("非 NaN 值的均值：", float(n_masked.mean()))
print("一共有多少 NaN：", int(n_masked.isnull().sum()))
print("总像素数：", int(n_masked.size))

vals = n_masked.values.flatten()
vals = vals[np.isfinite(vals)]  # 去掉 NaN

plt.hist(vals, bins=20, range=(0.0, 0.12), edgecolor='black')
plt.title("Histogram of Manning Values (including 0.0)")
plt.xlabel("Manning Coefficient")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

vc = pd.Series(vals).value_counts().sort_index()
print(vc)

val = n_masked.sel(x=1920000.00, y=5535000.00, method="nearest")
print("对应位置的 Manning 值是：", float(val))

val = n_masked.sel(x=1905000.00, y=5535000.00, method="nearest")
print("对应位置的 Manning 值是：", float(val))


# 找出值为0.0的位置（排除nan）
zero_mask = (n_masked == 0.0).values
zero_count = np.sum(zero_mask)
print("值为0.0的像素数量：", zero_count)

zero_masked_array = xr.DataArray(zero_mask, coords=n_masked.coords, dims=n_masked.dims)

# 可视化0.0值的位置
plt.figure(figsize=(10, 8))
zero_masked_array.plot(cmap="gray")
plt.title("Pixels with Manning = 0.0")
plt.xlabel("X (EPSG:2193)")
plt.ylabel("Y (EPSG:2193)")
plt.tight_layout()
plt.show()


