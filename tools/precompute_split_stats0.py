"""
python tools/precompute_split_stats.py \
  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset/index.csv \
  --root /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset \
  --out_json /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset/split_stats_h.json \
  --target_var h \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192
"""

import os
import csv
import json
import math
import argparse
import random
import numpy as np
from basicsr.utils import load_index_csv, group_by_scenario, cal_log1p, cal_asinh_p90


def _load_npy_shape(path, expect_shape=None, dtype=np.float32):
    arr = np.load(path, mmap_mode='r')
    if expect_shape is not None:
        h, w = expect_shape
        if arr.shape != expect_shape:
            hh = min(h, arr.shape[0])
            ww = min(w, arr.shape[1])
            canvas = np.zeros((h, w), dtype=arr.dtype)
            canvas[:hh, :ww] = arr[:hh, :ww]
            arr = canvas
    return arr.astype(dtype)


def masked_iter_values(paths, mask_paths, postprocess=None, expect_shape=None):
    for p, pm in zip(paths, mask_paths):
        arr = _load_npy_shape(p)
        m = _load_npy_shape(pm, expect_shape=arr.shape, dtype=np.uint8)
        if postprocess is not None:
            arr = postprocess(arr)
        sel = np.isfinite(arr) & (m == 1)
        if np.any(sel):
            yield arr[sel].astype(np.float64, copy=False)


class StreamingHistogram:
    def __init__(self, bins=8192):
        self.bins = int(bins)
        self.min_ = math.inf
        self.max_ = -math.inf
        self.hist = None
        self.eps = 1e-12

    def update_minmax(self, x1d):
        if x1d.size == 0: return
        xm = np.nanmin(x1d)
        xM = np.nanmax(x1d)
        if np.isfinite(xm): self.min_ = min(self.min_, float(xm))
        if np.isfinite(xM): self.max_ = max(self.max_, float(xM))

    def allocate(self):
        if not np.isfinite(self.min_) or not np.isfinite(self.max_):
            self.min_, self.max_ = 0.0, 1.0
        if self.max_ <= self.min_:
            self.max_ = self.min_ + 1.0
        self.hist = np.zeros(self.bins, dtype=np.int64)

    def update_hist(self, x1d):
        if x1d.size == 0: return
        scale = (self.bins - 1) / (self.max_ - self.min_ + self.eps)
        idx = ((x1d - self.min_) * scale).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)
        bc = np.bincount(idx, minlength=self.bins)
        self.hist += bc

    def percentile(self, q):
        if self.hist is None:
            return float('nan')
        total = int(self.hist.sum())
        if total == 0:
            return float('nan')
        rank = (q / 100.0) * (total - 1)
        cdf = np.cumsum(self.hist)
        k = int(np.searchsorted(cdf, rank))
        lo = self.min_ + (self.max_ - self.min_) * (k / max(self.bins - 1, 1))
        return float(lo)


class Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x1d):
        x = x1d.astype(np.float64, copy=False)
        for v in x:
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    def result(self, sample=False):
        if self.n == 0:
            return float('nan'), float('nan')
        if sample and self.n > 1:
            var = self.M2 / (self.n - 1)
        else:
            var = self.M2 / max(self.n, 1)
        return float(self.mean), float(np.sqrt(var) + 1e-12)


def make_split(rows, by, val_ratio, seed):
    rng = random.Random(seed)
    buckets = group_by_scenario(rows) if by == 'scenario' else {'__all__': rows}
    train_ids, val_ids = [], []
    for _, items in buckets.items():
        ids = [int(r['_row_id']) for r in items]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    return sorted(train_ids), sorted(val_ids)


def compute_static_stats(train_rows, bins=8192):
    feats = {
        'elevation': ('elev_path',),
        'roughness': ('rough_path',),
        'twi': ('twi_path',),
    }
    out = {}
    for name, (key,) in feats.items():
        p_vals = [r[key] for r in train_rows]
        p_mask = [r['mask_fine_path'] for r in train_rows]

        # pass1: 直方图界定 p1/p99
        H = StreamingHistogram(bins=bins)
        for x in masked_iter_values(p_vals, p_mask):
            H.update_minmax(x)
        H.allocate()
        for x in masked_iter_values(p_vals, p_mask):
            H.update_hist(x)
        p1 = H.percentile(1.0)
        p99 = H.percentile(99.0)

        # pass2: 在 [p1,p99] 内做 Welford
        wf = Welford()
        for x in masked_iter_values(p_vals, p_mask):
            if np.isfinite(p1) and np.isfinite(p99):
                x = np.clip(x, p1, p99)
            wf.update(x)
        mean, std = wf.result(sample=False)

        out[name] = {"p1": p1, "p99": p99, "mean": mean, "std": std}
    return out


def compute_h_stats(train_rows, bins=8192):
    # coarse + mask_coarse 对 log1p(coarse)
    p_coarse = [r['coarse_path'] for r in train_rows]
    p_mask = [r['mask_coarse_path'] for r in train_rows]

    def _log1p_gen():
        for x in masked_iter_values(p_coarse, p_mask):
            yield cal_log1p(x)

    # pass1: 直方图得 p1/p99
    H = StreamingHistogram(bins=bins)
    for x in _log1p_gen():
        H.update_minmax(x)
    H.allocate()
    for x in _log1p_gen():
        H.update_hist(x)
    p1 = H.percentile(1.0)
    p99 = H.percentile(99.0)

    # pass2: 阈内 Welford
    wf = Welford()
    for x in _log1p_gen():
        if np.isfinite(p1) and np.isfinite(p99):
            x = np.clip(x, p1, p99)
        wf.update(x)
    mean, std = wf.result(sample=False)

    return {
        "coarse": {
            "after": "log1p_clip_zscore",
            "p1": p1, "p99": p99,
            "mean": mean, "std": std
        }
    }


def compute_uv_stats(train_rows, bins=8192):
    # coarse + mask_coarse 三趟
    p_coarse = [r['coarse_path'] for r in train_rows]
    p_mask = [r['mask_coarse_path'] for r in train_rows]

    # pass1: |x| 的 p90 -> asinh_scale
    H_abs = StreamingHistogram(bins=bins)
    for x in masked_iter_values(p_coarse, p_mask):
        xabs = np.abs(x)
        H_abs.update_minmax(xabs)
    H_abs.allocate()
    for x in masked_iter_values(p_coarse, p_mask):
        H_abs.update_hist(np.abs(x))
    asinh_scale = H_abs.percentile(90.0)
    if (not np.isfinite(asinh_scale)) or (asinh_scale <= 0):
        # 粗糙兜底
        asinh_scale = 1.0

    # pass2: asinh 变换后的 p1/p99
    def _asinh_gen():
        for x in masked_iter_values(p_coarse, p_mask):
            yield np.arcsinh(x / asinh_scale)

    H = StreamingHistogram(bins=bins)
    for x in _asinh_gen():
        H.update_minmax(x)
    H.allocate()
    for x in _asinh_gen():
        H.update_hist(x)
    p1 = H.percentile(1.0)
    p99 = H.percentile(99.0)

    # pass3: 阈内 Welford
    wf = Welford()
    for x in _asinh_gen():
        if np.isfinite(p1) and np.isfinite(p99):
            x = np.clip(x, p1, p99)
        wf.update(x)
    mean, std = wf.result(sample=False)

    return {
        "coarse": {
            "after": "asinh_clip_zscore",
            "asinh_scale": float(asinh_scale),
            "p1": p1, "p99": p99,
            "mean": mean, "std": std
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_csv', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_json', required=True, help='split_stats_*.json 输出路径')
    ap.add_argument('--target_var', required=True, choices=['h', 'u', 'v', 'static'])
    ap.add_argument('--by', default='scenario', choices=['scenario', 'all'])
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=61)
    ap.add_argument('--bins', type=int, default=8192)
    args = ap.parse_args()

    rows_all = load_index_csv(args.index_csv)
    if args.target_var in ('h', 'u', 'v'):
        rows = [r for r in rows_all if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == args.target_var]
    else:
        rows = [r for r in rows_all if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == 'h']

    def _maybe_join(p):
        return p if os.path.isabs(p) else os.path.join(args.root, p)

    for r in rows:
        need = ["coarse_path", "fine_path", "elev_path", "rough_path",
                "mask_fine_path", "mask_coarse_path", "slope_path", "twi_path",
                "aspect_sin_path", "aspect_cos_path"]
        for k in need:
            if k not in r:
                raise RuntimeError(f'[ERROR] index.csv missing field: {k}')
            r[k] = _maybe_join(r[k])

    if os.path.isfile(args.out_json):
        with open(args.out_json, 'r') as f:
            meta = json.load(f)
        if 'split' not in meta:
            raise RuntimeError(f'[ERROR] Existing out_json missing "split".')
        train_ids = set(meta['split']['train'])
        train_rows = [r for r in rows if int(r['_row_id']) in train_ids]
    else:
        train_ids, val_ids = make_split(rows, args.by, args.val_ratio, args.seed)
        train_rows = [r for r in rows if int(r['_row_id']) in set(train_ids)]
        meta = {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "split": {"train": train_ids, "val": val_ids},
            "note": "Split by scenario with fixed seed; stats calculated on train only with masks."
        }

    # 计算 static + var
    print(f'Computing stats on {len(train_rows)} train patches, target_var={args.target_var} ...')

    static_rows = [r for r in rows_all if int(r['filtered_out']) == 0 and str(r.get('var','')).lower() == 'h']
    static_train_ids = set(meta['split']['train'])
    static_train_rows = [r for r in static_rows if int(r['_row_id']) in static_train_ids]
    stats_static = compute_static_stats(static_train_rows, bins=args.bins)
    meta['stats'] = stats_static

    if args.target_var == 'h':
        stats_var = compute_h_stats(train_rows, bins=args.bins)
    elif args.target_var in ('u', 'v'):
        stats_var = compute_uv_stats(train_rows, bins=args.bins)
    else:
        stats_var = {"coarse": {}}

    meta['stats_var'] = stats_var
    meta['stats_var_for'] = args.target_var

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {args.out_json}')


if __name__ == '__main__':
    main()
