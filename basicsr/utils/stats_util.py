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


def cal_minmaxnorm(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-12)


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
        "coarse_path", "fine_path", "elev_fine_path", "elev_coarse_path", "rough_path",
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
    DEPRECATED. Do not use.

    The legacy in-code stats computation produced a JSON schema that is
    incompatible with the dataset / loss readers (no ``shared`` block, no
    ``asinh_by_q`` block, no ``aux_stats``, no ``pos_weight_fine_raw_tau``).
    Calling it would silently write a broken split_stats JSON.

    Pre-compute the split_stats JSON via the scripts under ``tools/``
    (e.g. ``tools/precompute_split_stats*.py``) BEFORE training, point
    ``split_cfg.split_stats_json`` in your YAML at that file, and keep
    ``stats.calculate_if_missing`` set to ``false`` (or omit it).
    """
    raise NotImplementedError(
        "cal_stats_on_train() has been deprecated. Pre-compute the split_stats "
        "JSON with tools/precompute_split_stats*.py and reference it from the "
        "YAML (split_cfg.split_stats_json). The in-code fallback was removed "
        "because its output schema did not match the dataset/loss readers."
    )


def destand_to_physical(z, var, stats_var, transform, h_asinh_q):
    # z: [N, 1, Hf, Wf] model output with standardization
    if var == 'h' and transform == 'asinh':
        qk = str(int(h_asinh_q))
        S = stats_var['asinh_by_q'][qk]['shared']
        s_val = stats_var['asinh_by_q'][qk]['s']
    else:
        S = stats_var['shared']
        s_val = None
    mean = torch.tensor(S['mean'], device=z.device, dtype=z.dtype)
    std = torch.tensor(S['std'], device=z.device, dtype=z.dtype)
    x = z * std + mean

    if var == 'h':
        if transform == 'asinh':
            s = torch.tensor(float(s_val), device=z.device, dtype=z.dtype)
            return s * torch.sinh(x)
        else:
            return torch.expm1(x)
    else:
        s_key = 'asinh_scale_shared' if ('asinh_scale_shared' in S) else 'asinh_scale'
        s_val = S.get(s_key, 1.0)
        if (s_val is None) or (not np.isfinite(float(s_val))) or (float(s_val) <= 0):
            s_val = 1.0
        s = torch.tensor(float(s_val), device=z.device, dtype=z.dtype)
        return s * torch.sinh(x)


def denorm_to_physical(z, var, stats_var, transform, h_asinh_q):
    if var == 'h' and transform == 'asinh':
        qk = str(int(h_asinh_q))
        S = stats_var['asinh_by_q'][qk]['shared']
        s_val = stats_var['asinh_by_q'][qk]['s']
    else:
        S = stats_var['shared']
        s_val = None

    vmin = torch.tensor(S['min'], device=z.device, dtype=z.dtype)
    vmax = torch.tensor(S['max'], device=z.device, dtype=z.dtype)
    x = z * (vmax - vmin) + vmin

    if var == 'h':
        if transform == 'asinh':
            s = torch.tensor(float(s_val), device=z.device, dtype=z.dtype)
            return s * torch.sinh(x)
        else:
            return torch.expm1(x)
    else:
        s_key = 'asinh_scale_shared' if ('asinh_scale_shared' in S) else 'asinh_scale'
        s_val = S.get(s_key, 1.0)
        if (s_val is None) or (not np.isfinite(float(s_val))) or (float(s_val) <= 0):
            s_val = 1.0
        s = torch.tensor(float(s_val), device=z.device, dtype=z.dtype)
        return s * torch.sinh(x)

