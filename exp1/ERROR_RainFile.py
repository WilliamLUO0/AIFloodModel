import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import geopandas as gpd
import numpy as np

# ==== 1. 读取 DEM，提取原始空间范围 ====
dem = xr.open_dataset("./data/Porangahau/8m_geofabric.nc")
dem_xmin = float(dem["x"].min())
dem_xmax = float(dem["x"].max())
dem_ymin = float(dem["y"].min())
dem_ymax = float(dem["y"].max())

# ==== 2. BG-Flood 的裁剪范围 ====
clip_xmin = 1898150.0
clip_xmax = 1922940.0
clip_ymin = 5523060.0
clip_ymax = 5545740.0

# ==== 3. 读取 AOI ====
aoi_coords = []
with open("./data/Porangahau/aoi_BGFLOOD.txt", "r") as f:
    for line in f:
        x, y = map(float, line.strip().split())
        aoi_coords.append((x, y))
aoi_polygon = Polygon(aoi_coords)

# ==== 4. 读取 Rainfile 范围 ====
rain = xr.open_dataset("./data/Porangahau/outputrain_test.nc")
rain_xmin = float(rain["x"].min())
rain_xmax = float(rain["x"].max())
rain_ymin = float(rain["y"].min())
rain_ymax = float(rain["y"].max())

# ==== 5. 可视化全部框线 ====
fig, ax = plt.subplots(figsize=(10, 10))

# DEM 原始范围（黑）
dem_box = box(dem_xmin, dem_ymin, dem_xmax, dem_ymax)
gpd.GeoSeries([dem_box]).boundary.plot(ax=ax, color="black", label="DEM extent")

# BG-Flood 配置范围（红）
clip_box = box(clip_xmin, clip_ymin, clip_xmax, clip_ymax)
gpd.GeoSeries([clip_box]).boundary.plot(ax=ax, color="red", label="Clip extent")

# AOI 多边形（绿）
gpd.GeoSeries([aoi_polygon]).boundary.plot(ax=ax, color="green", label="AOI extent")

# Rainfile 范围（蓝）
rain_box = box(rain_xmin, rain_ymin, rain_xmax, rain_ymax)
gpd.GeoSeries([rain_box]).boundary.plot(ax=ax, color="blue", label="Rainfile extent")

ax.set_title("Spatial Coverage Comparison")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("spatial_coverage.png", dpi=300)
plt.show()

# ==== 6. 检查降雨强度 ====
print("==== Rainfile Depth (RainIntensity) Summary ====")
depth = rain["depth"]
depth_vals = depth.values.flatten()

has_nan = np.isnan(depth_vals).any()
print("Contains NaN:", has_nan)

depth_non_nan = depth_vals[~np.isnan(depth_vals)]
all_zero = np.all(depth_non_nan == 0)
print("All non-NaN values are 0:", all_zero)

if len(depth_non_nan) > 0:
    print(f"Min value: {np.min(depth_non_nan)}")
    print(f"Max value: {np.max(depth_non_nan)}")
else:
    print("All values are NaN")
