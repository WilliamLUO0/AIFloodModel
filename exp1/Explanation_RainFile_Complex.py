import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from rasterio.transform import from_origin
from rasterio.features import geometry_mask

# ==== 1. 加载降雨数据 ====
rain_path = "./data/porangahau_basin/sum_rain_amount_with_proj_clip.nc"
ds = xr.open_dataset(rain_path)
rain = ds["sum_rain_amount"]
time = ds["time2"]
x = ds["x"]
y = ds["y"]
print("====0====")
# print(ds)
print("\n")
print(rain)

print("====1====")
print("\nSpatial Range:")
print(f"x: {float(x.min())} ~ {float(x.max())}")
print(f"y: {float(y[0])} ~ {float(y[-1])}")

print("\nTime Range:")
print(f"time: {float(time.min())} ~ {float(time.max())}")
print("Time steps:", time.shape)

print("\nRainfall Stats:")
print(f"Max: {float(rain.max())} mm")
print(f"Min: {float(rain.min())} mm")
print(f"Mean: {float(rain.mean())} mm")
print(f"NaN count: {int(np.isnan(rain).sum())}")
print(f"Total cells: {rain.size}")

# ==== 2. 设置空间信息 ====
rain.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
rain.rio.write_crs("EPSG:2193", inplace=True)

# === 3. 读取 AOI 并构建 Polygon ===
aoi_path = "./data/porangahau_basin/aoi_BGFLOOD.txt"
aoi_points = pd.read_csv(aoi_path, sep="\s+", header=None, names=["x", "y"])
aoi_polygon = Polygon(aoi_points.values)
#
# print(aoi_points.head())
# print(rain.x.values[:5])
# print(rain.y.values[:5])

# # ==== 5. 构建 AOI 掩膜 ====
# dx = float(rain.x[1] - rain.x[0])
# dy = float(rain.y[1] - rain.y[0])
# transform = from_origin(
#     float(rain.x.min()), float(rain.y.max()), dx, dy
# )
# mask = geometry_mask(
#     [aoi_polygon],
#     out_shape=(rain.sizes["y"], rain.sizes["x"]),
#     transform=transform,
#     invert=True
# )
# rain = rain.where(mask)

# ==== 3. 可视化部分 ====
plt.figure(figsize=(10, 8))
rain.isel(time2=5).plot(cmap="Blues")
plt.title("Rainfall depth at time=5")
plt.xlabel("x (NZTM2000)")
plt.ylabel("y (NZTM2000)")
plt.show()

plt.figure(figsize=(10, 8))
rain.isel(time2=42).plot(cmap="Blues")
plt.title("Rainfall depth at time=42")
plt.xlabel("x (NZTM2000)")
plt.ylabel("y (NZTM2000)")
plt.show()

# ==== 4. 时间平均 ====
rain_mean = rain.mean(dim=("x", "y"))
plt.figure(figsize=(10, 8))
rain_mean.plot()
plt.title("Domain-Averaged Rainfall Time Series")
plt.xlabel("Time step")
plt.ylabel("Mean Rainfall (mm)")
plt.grid(True)
plt.show()

# ==== 5. 时间累加 ====
rain_cumsum = rain.sum(dim="time2")
plt.figure(figsize=(10, 8))
rain_cumsum.plot(cmap="Blues")
plt.title("Total Accumulated Rainfall Over Period")
plt.xlabel("x (NZTM2000)")
plt.ylabel("y (NZTM2000)")
plt.show()

# ==== 6. 极端降雨检测 ====
extreme_mask = rain > 25  # 25 mm 是暴雨阈值
extreme_times = extreme_mask.any(dim=("x", "y"))
print("Extreme rainfall occurred at time steps:", time.values[extreme_times.values])

# # ==== 7. 日累计降雨 (假设每时间步是1小时，48步 = 2天) ====
# hours_per_day = 24
# days = rain.sizes["time2"] // hours_per_day
#
# plt.figure(figsize=(16, 12))
#
# for i in range(days):
#     start = i * hours_per_day
#     end = start + hours_per_day + 1
#
#     daily_rain = rain.isel(time2=slice(start, end)).sum(dim="time2")
#     # slice(start, end)，最后end超过实际上并不会报错
#
#     ax = plt.subplot(2, 2, i + 1)
#     daily_rain.plot(ax=ax, cmap="Blues")
#     plt.title(f"Day {i + 1}: Accumulated Rainfall (mm)")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#
# plt.suptitle("Daily Accumulated Rainfall Maps")
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

from shapely.geometry import box
import matplotlib.pyplot as plt

aoi = aoi_polygon
rain_bbox = box(float(rain.x.min()), float(rain.y.min()), float(rain.x.max()), float(rain.y.max()))

fig, ax = plt.subplots(figsize=(10, 8))
gpd.GeoSeries(aoi).plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, label='AOI')
gpd.GeoSeries(rain_bbox).plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, label='Rain Domain')

ax.legend()
ax.set_title('AOI vs Rain Data Domain')
plt.show()
