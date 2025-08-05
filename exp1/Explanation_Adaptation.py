import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon

target_level_path = "./data/TargetLevel.nc"

# ==== 读取 TargetLevel 数据 ====
ds = xr.open_dataset(target_level_path)
z = ds["z"]
z.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
z.rio.write_crs("EPSG:2193", inplace=True)
x = ds["x"]
y = ds["y"]

print("==== TargetLevel.nc ====")
print(z)
print("x range:", float(x.min()), "to", float(x.max()))
print("y range:", float(y.min()), "to", float(y.max()))

# ==== 可视化 ====
# plt.figure(figsize=(10, 8))
# ax = z.plot(cmap="viridis")
# cbar = ax.colorbar
# cbar.set_label("Adaptation Level (m)")
# plt.title("TargetLevel Adaptation Map")
# plt.xlabel("x (EPSG:2193)")
# plt.ylabel("y (EPSG:2193)")

plt.figure(figsize=(10, 8))
plt.imshow(
    z,
    origin="lower",
    cmap="viridis",
    extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())]
)
plt.colorbar(label="Target Level")
plt.title("TargetLevel Adaptation Map")
plt.xlabel("x (NZTM)")
plt.ylabel("y (NZTM)")

# ==== 原始裁剪范围 ====
xmin_clip = 1898150.0
xmax_clip = 1922940.0
ymin_clip = 5523060.0
ymax_clip = 5545740.0
plt.plot(
    [xmin_clip, xmax_clip, xmax_clip, xmin_clip, xmin_clip],
    [ymin_clip, ymin_clip, ymax_clip, ymax_clip, ymin_clip],
    color='red', linewidth=2, label="DEM Clipping Box"
)

# ==== 加载 AOI from txt ====
aoi_coords = []
with open("./data/aoi_BGFLOOD.txt", "r") as f:
    for line in f:
        x_val, y_val = map(float, line.strip().split())
        aoi_coords.append((x_val, y_val))

aoi_polygon = Polygon(aoi_coords)
aoi_gdf = gpd.GeoDataFrame(index=[0], geometry=[aoi_polygon], crs="EPSG:2193")
aoi_gdf.boundary.plot(ax=plt.gca(), color="orange", linewidth=1.5, label="AOI")

plt.legend()
plt.tight_layout()
plt.show()