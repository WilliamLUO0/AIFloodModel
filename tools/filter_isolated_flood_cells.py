import os
import re
import argparse
import numpy as np
import xarray as xr
from scipy import ndimage
from tqdm import tqdm


def find_var(ds, base_name):
    """
    Find variable name in a BG-Flood-style NetCDF file.

    It first tries base_name, then base_name_P0.
    This supports both:
      merged_series_tXXXX.nc: h, u, v, zs
      BGout.nc / BGout_merged.nc: h_P0, u_P0, v_P0, zs_P0
    """
    if base_name in ds.variables:
        return base_name
    if f"{base_name}_P0" in ds.variables:
        return f"{base_name}_P0"
    return None


def get_time_dim(da):
    """
    Return time dimension name if it exists.
    """
    for d in da.dims:
        if d.lower() == "time":
            return d
    return None


def build_keep_mask(h2d, threshold, min_patch_size, connectivity=8):
    """
    Build keep mask from one 2D water depth map.

    keep_mask = True  -> keep this flooded cell
    keep_mask = False -> remove this cell
    """
    h = np.asarray(h2d, dtype=np.float32)
    h = np.where(np.isfinite(h), h, 0.0)

    wet = h >= threshold

    if connectivity == 8:
        struct = np.ones((3, 3), dtype=np.int8)
    elif connectivity == 4:
        struct = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=np.int8,
        )
    else:
        raise ValueError("connectivity must be 4 or 8.")

    labels, num_features = ndimage.label(wet, structure=struct)

    if num_features == 0:
        return np.zeros_like(wet, dtype=bool)

    counts = np.bincount(labels.ravel())

    large_component = counts >= min_patch_size
    large_component[0] = False  # label 0 is background, not flooded area

    keep = large_component[labels]

    return keep


def filter_one_2d_map(
    h_old,
    u_old=None,
    v_old=None,
    zs_old=None,
    threshold=0.1,
    min_patch_size=2,
    remove_below_threshold=True,
    connectivity=8,
):
    """
    Filter one 2D timestep.

    Default logic:
      h_new is kept only where:
        h_old > threshold and connected_patch_size >= min_patch_size

    If remove_below_threshold=False:
      this mimics the original script's active line:
        var = var_raw * (1 - mask)
      In that mode, values below threshold are not forced to zero.
    """
    h_old = np.asarray(h_old)

    keep_large_wet_patch = build_keep_mask(
        h_old,
        threshold=threshold,
        min_patch_size=min_patch_size,
        connectivity=connectivity,
    )

    finite_h = np.isfinite(h_old)

    if remove_below_threshold:
        final_keep = keep_large_wet_patch
    else:
        # Mimic var = var_raw * (1 - mask):
        # keep background / below-threshold cells, remove only small above-threshold components.
        final_keep = keep_large_wet_patch | ((h_old < threshold) & finite_h)

    h_new = np.where(final_keep, h_old, 0.0).astype(h_old.dtype, copy=False)

    removed_h = h_old - h_new
    removed_h = np.where(np.isfinite(removed_h), removed_h, 0.0)

    outputs = {"h": h_new}

    if u_old is not None:
        u_new = np.where(final_keep, u_old, 0.0).astype(u_old.dtype, copy=False)
        outputs["u"] = u_new

    if v_old is not None:
        v_new = np.where(final_keep, v_old, 0.0).astype(v_old.dtype, copy=False)
        outputs["v"] = v_new

    if zs_old is not None:
        # Important:
        # Do not set zs to zero.
        # If zs = dem + h, then removing water depth should give:
        #   zs_new = zs_old - removed_h
        zs_new = zs_old - removed_h
        zs_new = zs_new.astype(zs_old.dtype, copy=False)
        outputs["zs"] = zs_new

    return outputs


def filter_single_nc(
    input_nc,
    output_nc,
    threshold=0.1,
    min_patch_size=2,
    remove_below_threshold=True,
    connectivity=8,
    overwrite=False,
):
    if os.path.exists(output_nc) and not overwrite:
        print(f"[skip] output exists: {output_nc}")
        return

    print(f"[info] input : {input_nc}")
    print(f"[info] output: {output_nc}")

    ds = xr.open_dataset(input_nc, decode_times=False)
    ds_out = ds.copy(deep=True)

    h_name = find_var(ds, "h")
    u_name = find_var(ds, "u")
    v_name = find_var(ds, "v")
    zs_name = find_var(ds, "zs")

    if h_name is None:
        ds.close()
        raise RuntimeError(f"Cannot find h or h_P0 in {input_nc}")

    print(f"[info] h variable : {h_name}")
    print(f"[info] u variable : {u_name}")
    print(f"[info] v variable : {v_name}")
    print(f"[info] zs variable: {zs_name}")
    print(f"[info] threshold={threshold}, min_patch_size={min_patch_size}")
    print(f"[info] remove_below_threshold={remove_below_threshold}, connectivity={connectivity}")

    h_da = ds[h_name]
    time_dim = get_time_dim(h_da)

    if time_dim is None:
        h_old = ds[h_name].values
        u_old = ds[u_name].values if u_name is not None else None
        v_old = ds[v_name].values if v_name is not None else None
        zs_old = ds[zs_name].values if zs_name is not None else None

        outputs = filter_one_2d_map(
            h_old=h_old,
            u_old=u_old,
            v_old=v_old,
            zs_old=zs_old,
            threshold=threshold,
            min_patch_size=min_patch_size,
            remove_below_threshold=remove_below_threshold,
            connectivity=connectivity,
        )

        ds_out[h_name].values[:] = outputs["h"]
        if u_name is not None and "u" in outputs:
            ds_out[u_name].values[:] = outputs["u"]
        if v_name is not None and "v" in outputs:
            ds_out[v_name].values[:] = outputs["v"]
        if zs_name is not None and "zs" in outputs:
            ds_out[zs_name].values[:] = outputs["zs"]

    else:
        nt = ds[h_name].sizes[time_dim]

        for ti in tqdm(range(nt), desc="Filtering timesteps"):
            h_old = ds[h_name].isel({time_dim: ti}).values
            u_old = ds[u_name].isel({time_dim: ti}).values if u_name is not None else None
            v_old = ds[v_name].isel({time_dim: ti}).values if v_name is not None else None
            zs_old = ds[zs_name].isel({time_dim: ti}).values if zs_name is not None else None

            outputs = filter_one_2d_map(
                h_old=h_old,
                u_old=u_old,
                v_old=v_old,
                zs_old=zs_old,
                threshold=threshold,
                min_patch_size=min_patch_size,
                remove_below_threshold=remove_below_threshold,
                connectivity=connectivity,
            )

            idx = {time_dim: ti}
            ds_out[h_name][idx] = outputs["h"]
            if u_name is not None and "u" in outputs:
                ds_out[u_name][idx] = outputs["u"]
            if v_name is not None and "v" in outputs:
                ds_out[v_name][idx] = outputs["v"]
            if zs_name is not None and "zs" in outputs:
                ds_out[zs_name][idx] = outputs["zs"]

    tmp_out = output_nc + ".tmp"
    ds_out.to_netcdf(tmp_out, engine="netcdf4")
    os.replace(tmp_out, output_nc)

    ds.close()
    ds_out.close()

    print(f"[done] wrote: {output_nc}")


def filter_directory(
    input_dir,
    output_dir,
    pattern=r"merged_series_t\d{4}\.nc$",
    threshold=0.1,
    min_patch_size=100,
    remove_below_threshold=True,
    connectivity=8,
    overwrite=False,
):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if re.match(pattern, f)
    )

    if len(files) == 0:
        raise RuntimeError(f"No files matching pattern {pattern} found in {input_dir}")

    print(f"[info] input_dir : {input_dir}")
    print(f"[info] output_dir: {output_dir}")
    print(f"[info] number of files: {len(files)}")

    for i, fname in enumerate(files, start=1):
        print(f"[info] filtering file {i}/{len(files)}: {fname}", flush=True)
        input_nc = os.path.join(input_dir, fname)
        output_nc = os.path.join(output_dir, fname)

        filter_single_nc(
            input_nc=input_nc,
            output_nc=output_nc,
            threshold=threshold,
            min_patch_size=min_patch_size,
            remove_below_threshold=remove_below_threshold,
            connectivity=connectivity,
            overwrite=overwrite,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Filter isolated flooded cells in BG-Flood NetCDF outputs using h-based connected components."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-nc", help="Input NetCDF file.")
    mode.add_argument("--input-dir", help="Input directory containing per-timestep NetCDF files.")

    parser.add_argument("--output-nc", help="Output NetCDF file. Required when using --input-nc.")
    parser.add_argument("--output-dir", help="Output directory. Required when using --input-dir.")

    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--min-patch-size", type=int, required=True)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--pattern", default=r"merged_series_t\d{4}\.nc$")

    parser.add_argument(
        "--keep-below-threshold",
        action="store_true",
        help=(
            "Mimic the original active line var = var_raw * (1 - mask). "
            "By default, values below threshold are also removed from h/u/v, "
            "and zs is adjusted by subtracting removed h."
        ),
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    remove_below_threshold = not args.keep_below_threshold

    if args.input_nc is not None:
        if args.output_nc is None:
            raise ValueError("--output-nc is required when using --input-nc")

        filter_single_nc(
            input_nc=args.input_nc,
            output_nc=args.output_nc,
            threshold=args.threshold,
            min_patch_size=args.min_patch_size,
            remove_below_threshold=remove_below_threshold,
            connectivity=args.connectivity,
            overwrite=args.overwrite,
        )

    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --input-dir")

        filter_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            threshold=args.threshold,
            min_patch_size=args.min_patch_size,
            remove_below_threshold=remove_below_threshold,
            connectivity=args.connectivity,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()