import xarray as xr

# 读取 Rain
rain_ds = xr.open_dataset("./data/porangahau_basin/sum_rain_amount_with_proj_clip.nc")
rain = rain_ds["sum_rain_amount"]

# 读取 DEM
dem_ds = xr.open_dataset("./data/porangahau_basin/merged_porangahau_basin.nc")
dem = dem_ds["Band1"]

print("Rain Y (first 5):", rain.y.values[:5])
print("DEM Y (first 5):", dem.y.values[:5])

if rain.y[0] < rain.y[-1]:
    print("Rain Y is increasing")
    rain_clipped = rain.sel(
        x=slice(float(dem.x.min()), float(dem.x.max())),
        y=slice(float(dem.y.min()), float(dem.y.max()))
    )
else:
    print("Rain Y is decreasing")
    rain_clipped = rain.sel(
        x=slice(float(dem.x.min()), float(dem.x.max())),
        y=slice(float(dem.y.max()), float(dem.y.min()))
    )

print("Rain_clipped x size:", rain_clipped.sizes.get("x"))
print("Rain_clipped y size:", rain_clipped.sizes.get("y"))

if rain_clipped.sizes.get("y") == 0 or rain_clipped.sizes.get("x") == 0:
    raise ValueError("Clipped rain data is empty. Please check spatial overlap.")

print("Rain_clipped Y range:", float(rain_clipped.y.min()), "to", float(rain_clipped.y.max()))
print("Rain_clipped X range:", float(rain_clipped.x.min()), "to", float(rain_clipped.x.max()))

# 翻转 Y 轴
rain_clipped_flipped = rain_clipped.isel(y=slice(None, None, -1))
print("Flipped Y (first 5):", rain_clipped_flipped.y.values[:5])

# 保存
output_path = "./data/porangahau_basin/sum_rain_clipped_flipped.nc"
rain_clipped_flipped.to_netcdf(output_path)
print(f"Saved flipped file to: {output_path}\n")


# rain_clipped.to_netcdf("./data/porangahau_basin/sum_rain_clipped.nc")

# # 重采样 Rain 到 DEM 的分辨率 (8m)
# rain_resampled = rain_clipped.interp(
#     x=dem.x,
#     y=dem.y
# )
#
# # 保存为新文件
# rain_resampled.to_netcdf("./data/porangahau_basin/sum_rain_clipped_resampled.nc")

# import xarray as xr
#
# # 读取 Rain 数据
# rain_ds = xr.open_dataset("./data/porangahau_basin/sum_rain_amount_with_proj_clip.nc")
# rain = rain_ds["sum_rain_amount"]
#
# rain_ds2 = xr.open_dataset("./data/Porangahau/outputrain_test.nc")
# rain2 = rain_ds2["depth"]
#
# # 读取 DEM 数据
# dem_ds = xr.open_dataset("./data/porangahau_basin/merged_porangahau_basin.nc")
# dem = dem_ds["Band1"]
#
# dem_ds2 = xr.open_dataset("./data/Porangahau/8m_geofabric.nc")
# dem2 = dem_ds2["z"]
#
# # ==== 检查原始 Y 轴方向 ====
# print("Original Rain Y (first 5):", rain.y.values[:5])
# print("Original Rain2 Y (first 5):", rain2.y.values[:5])
# print("Original DEM Y (first 5):", dem.y.values[:5])
# print("Original DEM2 Y (first 5):", dem2.y.values[:5])
#
# # ==== 翻转 Rain Y 轴 ====
# rain_flipped = rain.isel(y=slice(None, None, -1))
#
# # ==== 检查翻转后的 Y 轴 ====
# print("Flipped Rain Y (first 5):", rain_flipped.y.values[:5])
#
# # ==== 保存为新文件 ====
# rain_flipped.to_netcdf("./data/porangahau_basin/sum_rain_flipped.nc")
#
# print("Flipped Rain saved successfully.")

# import xarray as xr
#
# def flip_and_save(input_path, var_name, output_path):
#     ds = xr.open_dataset(input_path)
#     data = ds[var_name]
#
#     print(f"--- {input_path} ---")
#     print("Original Y (first 5):", data.y.values[:5])
#
#     # 翻转 Y 轴
#     data_flipped = data.isel(y=slice(None, None, -1))
#     print("Flipped Y (first 5):", data_flipped.y.values[:5])
#
#     # 保存
#     data_flipped.to_netcdf(output_path)
#     print(f"Saved flipped file to: {output_path}\n")
#
#
# # ==== 1. DEM ====
# flip_and_save(
#     input_path="./data/porangahau_basin/merged_porangahau_basin.nc",
#     var_name="Band1",
#     output_path="./data/porangahau_basin/merged_porangahau_basin_flipped.nc"
# )
#
# # ==== 2. Rain ====
# flip_and_save(
#     input_path="./data/porangahau_basin/sum_rain_amount_with_proj_clip.nc",
#     var_name="sum_rain_amount",
#     output_path="./data/porangahau_basin/sum_rain_amount_with_proj_clip_flipped.nc"
# )
#
# # ==== 3. Manning 糙率 ====
# flip_and_save(
#     input_path="./data/porangahau_basin/merged_porangahau_basin_zo.nc",
#     var_name="Band1",
#     output_path="./data/porangahau_basin/merged_porangahau_basin_zo_flipped.nc"
# )