"""
python merge_multi-scale_rev1.py \
  --fine-dir "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/100y_42h_0c/dx8" \
  --input-nc "BGout.nc" \
  --coarse-nc "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/100y_42h_0c/dx128/BGout.nc" \
  --out-perdir "per_timestep_merged" \
  --vars zs u v h
"""

import os
import re
import argparse
import numpy as np
import xarray as xr
import dask


def merge_all_reso(input_nc_path, variables=None, num_time_steps=None, xmin=None, xmax=None, ymin=None, ymax=None,
                   name_output=None):
    # open as dataframe and extract basic info
    ds = xr.open_dataset(input_nc_path)
    maxlvl = int(ds.maxlevel)
    minlvl = int(ds.minlevel)

    # get the higher resolution
    suffix_max = '_' + ('N' if maxlvl < 0 else 'P') + str(abs(maxlvl))

    # create the list of variables
    if not variables:
        my_vars = list(ds.data_vars)
        my_var_type = []
        for var in my_vars:
            if suffix_max in var:
                my_var_type.append(var.split('_')[0])
    else:
        my_var_type = variables

    # Time steps reduction
    if num_time_steps is None:
        time2 = ds.time
    elif isinstance(num_time_steps, (np.integer, int)):
        time2 = ds.time[int(num_time_steps)]
    elif num_time_steps == -1:
        time2 = ds.time[-1]
    elif num_time_steps == 0:
        time2 = ds.time[0]
    else:
        time2 = xr.DataArray([num_time_steps]).values[0]

    # initialise the new dataframe using the higher resolution data
    i = maxlvl
    suffix = '_' + ('N' if i < 0 else 'P') + str(abs(i))
    if isinstance(num_time_steps, (np.integer, int)):
        do = ds.isel(time=int(num_time_steps))
    elif num_time_steps == -1:
        do = ds.isel(time=-1)
    elif num_time_steps == 0:
        do = ds.isel(time=0)
    else:
        do = ds.sel(time=time2)

    dict_name = {i + suffix: i for i in my_var_type}
    dict_name.update({i + suffix: i for i in ['xx', 'yy']})
    do = do.rename(dict_name)
    new = do[my_var_type]

    # project all resolutions on the higher resolution grid
    for i in range(minlvl, maxlvl):
        suffix = '_' + ('N' if i < 0 else 'P') + str(abs(i))
        if isinstance(num_time_steps, (np.integer, int)):
            do = ds.isel(time=int(num_time_steps))
        elif num_time_steps == -1:
            do = ds.isel(time=-1)
        elif num_time_steps == 0:
            do = ds.isel(time=0)
        else:
            do = ds.sel(time=time2)  # 不加 method

        dict_name = {i + suffix: i for i in my_var_type}
        dict_name.update({i + suffix: i for i in ['xx', 'yy']})
        do = do.rename(dict_name)
        for j in range(len(my_var_type)):
            dtmp = do[my_var_type[j]]
            dint = dtmp.interp(xx=new["xx"], yy=new["yy"], method="nearest")
            new = xr.merge([new, dint], compat="no_conflicts")

    if not name_output:
        name_output = os.path.splitext(os.path.basename(input_nc_path))[0] + '_' + variables[0] + '.nc'
    new.to_netcdf(name_output)
    ds.close()
    return name_output


def stack_per_timestep_to_bgout(per_dir: str, coarse_nc: str, out_nc_path: str) -> str:
    # Load coarse (decode_times=False to keep raw numeric values + units)
    ds128 = xr.open_dataset(coarse_nc, decode_times=False)
    time128 = ds128["time"].values
    time128_attrs = dict(ds128["time"].attrs)

    # Copy crs variable if exists
    crs_var = None
    if "crs" in ds128.variables:
        crs_var = xr.Variable((), 0, attrs=dict(ds128["crs"].attrs))

    Nt128 = time128.shape[0]

    # List per-timestep files
    all_files = sorted(
        [f for f in os.listdir(per_dir) if re.match(r"merged_series_t\d{4}\.nc$", f)]
    )
    if len(all_files) == 0:
        ds128.close()
        raise RuntimeError(f"No merged_series_tXXXX.nc found in {per_dir}")

    Nt = min(len(all_files), Nt128)
    if len(all_files) != Nt128:
        print(f"[warn] #files ({len(all_files)}) != #times in coarse ({Nt128}); using first {Nt} entries.")

    all_files = all_files[:Nt]
    used_time = time128[:Nt]

    pieces = []
    FILL = np.float32(9.9692e+36)

    for k, fname in enumerate(all_files):
        fpath = os.path.join(per_dir, fname)
        d = xr.open_dataset(fpath, decode_times=False, chunks={"yy": 1024, "xx": 1024})

        # Expect dims 'yy' and 'xx'
        if "yy" not in d.dims or "xx" not in d.dims:
            d.close()
            ds128.close()
            raise RuntimeError(f"{fname} is missing dims 'yy'/'xx'")

        # Rename dims & coords to *_P0
        d = d.rename_dims({"yy": "yy_P0", "xx": "xx_P0"})
        if "yy" in d.variables:
            d = d.rename_vars({"yy": "yy_P0"})
        if "xx" in d.variables:
            d = d.rename_vars({"xx": "xx_P0"})

        # Ensure coordinate attrs
        if "xx_P0" in d and "axis" not in d["xx_P0"].attrs:
            d["xx_P0"].attrs["axis"] = "X"
        if "yy_P0" in d and "axis" not in d["yy_P0"].attrs:
            d["yy_P0"].attrs["axis"] = "Y"

        # Rename variables to *_P0
        rename_map = {}
        for v in ["zs", "u", "v", "h"]:
            if v in d.variables:
                rename_map[v] = f"{v}_P0"
        d = d.rename_vars(rename_map)

        # Standardize encodings/attrs
        for v in ["zs_P0", "u_P0", "v_P0", "h_P0"]:
            if v in d.variables:
                d[v] = d[v].astype("float32")
                d[v].encoding["_FillValue"] = FILL
                d[v].attrs["missingvalue"] = FILL
                if "grid_mapping" not in d[v].attrs:
                    d[v].attrs["grid_mapping"] = "crs"

        # Add single time point with coarse time value
        d = d.expand_dims({"time": [used_time[k]]})
        d["time"].attrs.update(time128_attrs)

        # Attach crs scalar variable
        if crs_var is not None and "crs" not in d.variables:
            d = d.assign(crs=crs_var)

        # Default some global attrs, preserve existing
        gattrs = dict(d.attrs)
        if "maxlevel" not in gattrs:
            gattrs["maxlevel"] = 4
        if "minlevel" not in gattrs:
            gattrs["minlevel"] = 0
        d.attrs = gattrs

        pieces.append(d)

    # Concatenate along time
    ds_fine = xr.concat(pieces, dim="time", compat="no_conflicts")

    ds_fine["time"].attrs.update(time128_attrs)

    enc = {}
    for v in ["zs_P0", "u_P0", "v_P0", "h_P0"]:
        if v in ds_fine.variables:
            enc[v] = {
                "_FillValue": np.float32(9.9692e+36),
                "dtype": "float32",
                "chunksizes": (1, 1024, 1024),
            }

    tmp_out = out_nc_path + ".tmp"
    ds_fine.to_netcdf(tmp_out, encoding=enc, engine="netcdf4")
    os.replace(tmp_out, out_nc_path)
    ds128.close()
    return out_nc_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-resolution BG-Flood outputs per timestep onto the finest grid, "
                    "then stack along time using the coarse BGout.nc timeline."
    )
    parser.add_argument(
        "--fine-dir", required=True,
        help="Directory of the fine grid run (e.g., dx8). Outputs are also written here."
    )
    parser.add_argument(
        "--input-nc", default="BGout.nc",
        help="Input multi-resolution BG-Flood file under fine-dir (default: BGout.nc)."
    )
    parser.add_argument(
        "--coarse-nc", required=True,
        help="Path to coarse-grid BGout.nc (e.g., dx128/BGout.nc) used for time & crs."
    )
    parser.add_argument(
        "--out-perdir", default="per_timestep_merged",
        help="Subdirectory (under fine-dir) to store per-timestep merged files (default: per_timestep_merged)."
    )
    parser.add_argument(
        "--vars", nargs="+", default=["zs", "u", "v", "h"],
        help="Variables to process (default: zs u v h)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing per-timestep files if present."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional: only process the first N timesteps (debugging/quick run)."
    )
    parser.add_argument(
        "--final-out", default="BGout_merged.nc",
        help="Final merged fine-grid output filename under --fine-dir (default: BGout_merged.nc)."
    )

    args = parser.parse_args()

    fine_dir = args.fine_dir
    input_nc = os.path.join(fine_dir, args.input_nc)
    per_dir = os.path.join(fine_dir, args.out_perdir)
    os.makedirs(per_dir, exist_ok=True)

    print(f"[info] Stage 1: per-timestep merge from {input_nc}")
    ds0 = xr.open_dataset(input_nc)
    # Nt_total = int(ds0.dims.get('time', ds0['time'].sizes['time']))
    Nt_total = int(ds0.sizes.get('time', ds0['time'].sizes['time']))
    ds0.close()

    if args.limit is not None:
        Nt = min(Nt_total, args.limit)
        print(f"[info] Limiting timesteps to first {Nt} (of {Nt_total})")
    else:
        Nt = Nt_total

    for ti in range(Nt):
        out_name = f"merged_series_t{ti:04d}.nc"
        out_path = os.path.join(per_dir, out_name)
        if (not args.overwrite) and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue

        print(f"[info] processing time index {ti}/{Nt - 1} -> {out_path}")
        merge_all_reso(
            input_nc_path=input_nc,
            variables=args.vars,
            num_time_steps=ti,
            name_output=out_path
        )

    print(f"[info] Per-timestep files written to: {per_dir}")

    print(f"[info] Stage 2: stacking per-timestep files using time from {args.coarse_nc}")
    final_out = os.path.join(fine_dir, args.final_out)
    final_path = stack_per_timestep_to_bgout(
        per_dir=per_dir,
        coarse_nc=args.coarse_nc,
        out_nc_path=final_out
    )
    print(f"[info] Final fine-grid BGout saved to: {final_path}")
    print("[done]")


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     mydir = "/nesi/nobackup/uoa04425/zluo784/Exp1/Gisborne_basin/results/100y_42h_0c/dx8"
#     fn    = "BGout.nc"
#     out_dir = os.path.join(mydir, "per_timestep_merged")
#     os.makedirs(out_dir, exist_ok=True)
#
#     variables = ['zs', 'u', 'v', 'h']
#
#     ds0 = xr.open_dataset(os.path.join(mydir, fn))
#     Nt = int(ds0.dims.get('time', ds0['time'].sizes['time']))
#     ds0.close()
#
#     for ti in range(Nt):
#         out_name = f"merged_series_t{ti:04d}.nc"
#         out_path = os.path.join(out_dir, out_name)
#         print(f"[info] processing time index {ti}/{Nt-1} -> {out_path}")
#         merge_all_reso(
#             mydir=mydir,
#             fn=fn,
#             variables=variables,
#             num_time_steps=ti,
#             name_output=out_path
#         )
#
#     print(f"Done. Files in: {out_dir}")

