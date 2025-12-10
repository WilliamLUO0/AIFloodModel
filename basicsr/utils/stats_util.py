import csv
import torch
import time
import errno
import numpy as np


def cal_percentile(x, q, mask=None):
    """ Calculate percentiles at positions where the value is not NaN and mask==1 """
    if mask is not None:
        x = x[mask.astype(bool)]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    return float(np.percentile(x, q))


def cal_mean_std(x, mask=None):
    """ Calculate mean and std at positions where the value is not NaN and mask==1 """
    if mask is not None:
        x = x[mask.astype(bool)]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(x)), float(np.std(x) + 1e-12)


def cal_log1p(x):
    return np.log1p(np.clip(x, 0, None))


def cal_asinh_p90(x, asinh_scale):
    if (asinh_scale is None) or (not np.isfinite(asinh_scale)) or (asinh_scale <= 0):
        return x.astype(np.float32)
    return np.arcsinh(x / float(asinh_scale)).astype(np.float32)


def cal_zscore(x, mean, std):
    return (x - mean) / (std + 1e-12)


def percentile_clip(x, lo, hi):
    if x.size == 0 or np.isnan(lo) or np.isnan(hi):
        return x
    return np.clip(x, lo, hi)


def load_npy_shape(path, expect_shape=None, dtype=np.float32, max_retries=7, base_delay=0.4):
    arr = None
    for k in range(max_retries):
        try:
            arr = np.load(path)
            break
        except OSError as e:
            code = getattr(e, "errno", None)
            msg = str(e).lower()
            transient = (code in (121, errno.EIO)) or ("remote i/o" in msg) or ("remote io" in msg) or (
                        "i/o error" in msg)
            if transient and k < max_retries - 1:
                time.sleep(base_delay * (2 ** k))
                continue
            raise
    if expect_shape is not None:
        h, w = expect_shape
        if arr.shape != expect_shape:
            hh = min(h, arr.shape[0])
            ww = min(w, arr.shape[1])
            canvas = np.zeros((h, w), dtype=arr.dtype)
            canvas[:hh, :ww] = arr[:hh, :ww]
            arr = canvas
    return arr.astype(dtype)


def group_by_scenario(rows):
    buckets = {}
    for r in rows:
        buckets.setdefault(r['scenario'], []).append(r)
    return buckets


def load_index_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            r = dict(r)
            r['_row_id'] = i
            rows.append(r)
    if not rows:
        raise RuntimeError(f'[ERROR] index.csv ({path}) has no rows.')
    return rows


def check_required_fields(rows):
    needed = [
        "coarse_path", "fine_path", "elev_path", "rough_path",
        "mask_fine_path", "mask_coarse_path", "slope_path", "twi_path",
        "aspect_sin_path", "aspect_cos_path", "downscale"
    ]
    missing_any = False
    example_missing = []
    for r in rows:
        for k in needed:
            if k not in r:
                missing_any = True
                example_missing.append(k)
    if missing_any:
        ks = ','.join(sorted(set(example_missing)))
        raise RuntimeError(f'[ERROR] index.csv missing field: {ks}')


def gather_vals(paths, mask_paths, postprocess=None):
    """Gather values from multiple patches (mask==1)"""
    vals = []
    for p, pm in zip(paths, mask_paths):
        arr = load_npy_shape(p)
        m = load_npy_shape(pm, expect_shape=arr.shape, dtype=np.uint8)
        if postprocess is not None:
            arr = postprocess(arr)
        sel = np.isfinite(arr) & (m == 1)
        if np.any(sel):
            vals.append(arr[sel])
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals, axis=0).astype(np.float64)


def cal_stats_on_train(rows, train_ids_set, var='static'):
    """
    h (water depth): log1p + clip[p1, p99] + z-score
    u (water velocity in x-direction): asinh(x/|x|_p90) + clip[p1, p99] + z-score
    v (water velocity in y-direction): asinh(x/|x|_p90) + clip[p1, p99] + z-score
    Elevation  : clip[p1, p99] + z-score
    Roughness  : clip[p1, p99] + z-score
    TWI        : clip[p1, p99] + z-score
    Slope_deg  : / 90
    Aspect_Sin : skip
    Aspect_COS : skip
    Mask       : skip
    """
    train_rows = [r for r in rows if int(r['_row_id']) in train_ids_set]

    if var in ('h', 'u', 'v'):
        train_rows = [r for r in train_rows if str(r.get('var', '')).lower() == var]
    else:
        train_rows = [r for r in train_rows if str(r.get('var', '')).lower() == 'h']

    if var == 'static':
        # mask_fine -> elevation + roughness + twi
        p_mask_fine = [r['mask_fine_path'] for r in train_rows]

        # elevation
        p_elev = [r['elev_path'] for r in train_rows]
        elev_vals = gather_vals(p_elev, p_mask_fine, postprocess=None)
        e_p1 = cal_percentile(elev_vals, 1.0)
        e_p99 = cal_percentile(elev_vals, 99.0)
        elev_vals_clip = percentile_clip(elev_vals, e_p1, e_p99)
        e_mean, e_std = cal_mean_std(elev_vals_clip)

        # roughness
        p_rough = [r['rough_path'] for r in train_rows]
        rough_vals = gather_vals(p_rough, p_mask_fine, postprocess=None)
        r_p1 = cal_percentile(rough_vals, 1.0)
        r_p99 = cal_percentile(rough_vals, 99.0)
        rough_vals_clip = percentile_clip(rough_vals, r_p1, r_p99)
        r_mean, r_std = cal_mean_std(rough_vals_clip)

        # twi
        p_twi = [r['twi_path'] for r in train_rows]
        twi_vals = gather_vals(p_twi, p_mask_fine, postprocess=None)
        t_p1 = cal_percentile(twi_vals, 1.0)
        t_p99 = cal_percentile(twi_vals, 99.0)
        twi_vals_clip = percentile_clip(twi_vals, t_p1, t_p99)
        t_mean, t_std = cal_mean_std(twi_vals_clip)

        stats = {
            "elevation": {"p1": e_p1, "p99": e_p99, "mean": e_mean, "std": e_std},
            "twi": {"p1": t_p1, "p99": t_p99, "mean": t_mean, "std": t_std},
            "roughness": {"p1": r_p1, "p99": r_p99, "mean": r_mean, "std": r_std},
        }
        return stats

    elif var == 'h':
        # mask_coarse -> coarse-grid flood map
        p_coarse = [r['coarse_path'] for r in train_rows]
        p_mask_coarse = [r['mask_coarse_path'] for r in train_rows]

        # h -> log1p + clip + zscore
        coarse_log_vals = gather_vals(p_coarse, p_mask_coarse, postprocess=cal_log1p)
        c_p1 = cal_percentile(coarse_log_vals, 1.0)
        c_p99 = cal_percentile(coarse_log_vals, 99.0)
        coarse_log_vals_clip = percentile_clip(coarse_log_vals, c_p1, c_p99)
        c_mean, c_std = cal_mean_std(coarse_log_vals_clip)

        stats = {
            "coarse": {
                "after": "log1p_clip_zscore",
                "p1": c_p1, "p99": c_p99,
                "mean": c_mean, "std": c_std
            }
        }
        return stats

    elif var in ('u', 'v'):
        # mask_coarse -> coarse-grid flood map
        p_coarse = [r['coarse_path'] for r in train_rows]
        p_mask_coarse = [r['mask_coarse_path'] for r in train_rows]

        # u, v -> asinh + zscore
        coarse_vals = gather_vals(p_coarse, p_mask_coarse, postprocess=None)
        if coarse_vals.size == 0:
            asinh_scale = float('nan')
            coarse_asinh_vals = coarse_vals
        else:
            coarse_abs_vals = np.abs(coarse_vals)
            asinh_scale = cal_percentile(coarse_abs_vals, 90.0)
            # use median or 1.0
            if (asinh_scale is None) or (not np.isfinite(asinh_scale)) or (asinh_scale <= 0):
                coarse_abs_med = float(np.median(coarse_abs_vals)) if coarse_abs_vals.size > 0 else float('nan')
                asinh_scale = coarse_abs_med if (np.isfinite(coarse_abs_med) and coarse_abs_med > 0) else 1.0
            asinh_scale = max(float(asinh_scale), 1e-6)
            coarse_asinh_vals = cal_asinh_p90(coarse_vals, asinh_scale)

        c_p1 = cal_percentile(coarse_asinh_vals, 1.0)
        c_p99 = cal_percentile(coarse_asinh_vals, 99.0)
        coarse_asinh_vals_clip = percentile_clip(coarse_asinh_vals, c_p1, c_p99)
        c_mean, c_std = cal_mean_std(coarse_asinh_vals_clip)

        stats = {
            "coarse": {
                "after": "asinh_clip_zscore",
                "asinh_scale": asinh_scale,
                "p1": c_p1, "p99": c_p99,
                "mean": c_mean, "std": c_std
            }
        }
        return stats

    else:
        raise ValueError(f'[ERROR] Unknown var: {var}. Expected "static", "h", "u", or "v".')


def destand_to_physical(z, var, stats_var):
    # z: [N, 1, Hf, Wf] model output with standardization
    mean = torch.tensor(stats_var['coarse']['mean'], device=z.device, dtype=z.dtype)
    std = torch.tensor(stats_var['coarse']['std'], device=z.device, dtype=z.dtype)
    x = z * std + mean
    if var == 'h':
        return torch.expm1(x)  # exp(x) - 1
    else:
        s = torch.tensor(stats_var['coarse']['asinh_scale'], device=z.device, dtype=z.dtype)
        return s * torch.sinh(x)

