import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

# ==== 1. 加载降雨数据 ====
rain_path = "./data/Porangahau/input_files/outputrain_test.nc"
ds = xr.open_dataset(rain_path)
depth = ds["depth"]
time = ds["time"]
x = ds["x"]
y = ds["y"]
print("====0====")
# print(ds)
print("\n")
print(depth)

print("====1====")
print("\n")
print("x range:", float(x.min()), "to", float(x.max()))
print("y range:", float(y.min()), "to", float(y.max()))
print("time range:", float(time.min()), "to", float(time.max()))
print(time.shape)
# print(time)

print("====2====")
print("\n")
# print("非 NaN 值的最大值：", float(depth.isel(time=32).max()))
# print("非 NaN 值的最小值：", float(depth.isel(time=32).min()))
# print("非 NaN 值的均值：", float(depth.isel(time=32).mean()))
# print("一共有多少 NaN：", int(depth.isel(time=32).isnull().sum()))
# print("总像素数：", int(depth.isel(time=32).size))
print("非 NaN 值的最大值：", float(depth.max()))
print("非 NaN 值的最小值：", float(depth.min()))
print("非 NaN 值的均值：", float(depth.mean()))
print("一共有多少 NaN：", int(depth.isnull().sum()))
print("总像素数：", int(depth.size))

# ==== 2. 设置空间信息 ====
depth.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
depth.rio.write_crs("EPSG:2193", inplace=True)

plt.figure(figsize=(10, 8))
depth.isel(time=5).plot(cmap="Blues")
plt.title("Rainfall depth at time=5")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(figsize=(10, 8))
depth.isel(time=42).plot(cmap="Blues")
plt.title("Rainfall depth at time=42")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

rain = ds["depth"]  # 假设变量叫 depth
rain_mean = rain.mean(dim=("x", "y"))

plt.figure(figsize=(10, 8))
rain_mean.plot()
plt.title("Average Rainfall Intensity Over Domain")
plt.ylabel("mm/h")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
rain_cumsum = rain.sum(dim="time")  # 在每个格点上累计 mm
rain_cumsum.plot(cmap="Blues")
plt.title("Total Accumulated Rainfall (mm) Over 4 Days")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

extreme_mask = rain > 25  # 25mm/h 是暴雨临界线
extreme_times = extreme_mask.any(dim=("x", "y"))
print("哪些时间步发生了暴雨级别降雨？", ds.time[extreme_times].values)

days = 4
hours_per_day = 24

plt.figure(figsize=(16, 12))

for i in range(days):
    start = i * hours_per_day
    end = start + hours_per_day + 1

    daily_rain = rain.isel(time=slice(start, end)).sum(dim="time")  # 累计 mm
    # slice(start, end)，最后end超过实际上并不会报错

    ax = plt.subplot(2, 2, i + 1)
    daily_rain.plot(ax=ax, cmap="Blues")
    plt.title(f"Day {i + 1}: Accumulated Rainfall (mm)")
    plt.xlabel("X")
    plt.ylabel("Y")

plt.suptitle("Daily Accumulated Rainfall Over 4 Days", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
