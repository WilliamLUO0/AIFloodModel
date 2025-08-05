import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==== 1. 加载 BG-FLOOD 输出 ====
ds = xr.open_dataset("./results/BGout_128_2.nc")
time = ds["time"]
x = ds["xx_P0"]
y = ds["yy_P0"]
print("====0====")
print(ds)
print("x range:", float(x.min()), "to", float(x.max()))
print("y range:", float(y.min()), "to", float(y.max()))
print(time.shape)
print(time[0])

zs_P0 = ds["zs_P0"]
u_P0 = ds["u_P0"]
v_P0 = ds["v_P0"]
h_P0 = ds["h_P0"]
hmax_P0 = ds["hmax_P0"]
zsmax_P0 = ds["zsmax_P0"]

# 设置空间维度 + 坐标系(NZTM2000)
output = [zs_P0, u_P0, v_P0, h_P0, hmax_P0, zsmax_P0]
for var in output:
    var.rio.set_spatial_dims(x_dim="xx_P0", y_dim="yy_P0", inplace=True)
    var.rio.write_crs("EPSG:2193", inplace=True)

# ==== 2. 指定时间步 ====
# t = 288000
# t = np.timedelta64(288000, "s")
# sel = ds.sel(time=t, method="nearest")
start_time = np.datetime64("2000-01-01T00:00:00")
target_time = start_time + np.timedelta64(288000, "s")

# ==== 3. 逐个变量绘图 ====

# zs
plt.figure(figsize=(10, 8))
zs_P0.sel(time=target_time, method="nearest").plot(cmap="viridis", cbar_kwargs={"label": "Water Surface Elevation [m]"})
# zs_P0.isel(time=42).plot(cmap="viridis", cbar_kwargs={"label": "Water Surface Elevation [m]"})
plt.title(f"zs at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

# u
plt.figure(figsize=(10, 8))
u_P0.sel(time=target_time).plot(cmap="plasma", cbar_kwargs={"label": "Velocity X [m/s]"})
plt.title(f"u at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

# v
plt.figure(figsize=(10, 8))
v_P0.sel(time=target_time).plot(cmap="plasma", cbar_kwargs={"label": "Velocity Y [m/s]"})
plt.title(f"v at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

# h
plt.figure(figsize=(10, 8))
h_P0.sel(time=target_time).plot(cmap="Blues", cbar_kwargs={"label": "Water Depth [m]"})
plt.title(f"h at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

# hmax
plt.figure(figsize=(10, 8))
hmax_P0.sel(time=target_time).plot(cmap="Blues", cbar_kwargs={"label": "Max Water Depth [m]"})
plt.title(f"hmax (Max Depth since simulation start) at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()

# zsmax
plt.figure(figsize=(10, 8))
zsmax_P0.sel(time=target_time).plot(cmap="viridis", cbar_kwargs={"label": "Max Water Elevation [m]"})
plt.title(f"zsmax (Max Water Surface Elevation) at t = {target_time}s")
plt.xlabel("x (EPSG:2193)")
plt.ylabel("y (EPSG:2193)")
plt.tight_layout()
plt.show()