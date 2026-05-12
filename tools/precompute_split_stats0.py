"""
Example:
python tools/precompute_split_stats2.py \
  --index_csv /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/index.csv \
  --root      /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup \
  --out_json  /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h.json \
  --target_var h \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192
"""

import os
import json
import math
import argparse
import random
import numpy as np

from basicsr.utils import load_index_csv, group_by_scenario


# ----------------------------- IO helpers ----------------------------- #

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
    return np.asarray(arr, dtype=dtype)


def iter_masked_values(val_path, mask_path, postprocess=None):
    """
    Yield a 1D float64 array for each file (masked & finite).
    """
    arr = _load_npy_shape(val_path, dtype=np.float32)
    m = _load_npy_shape(mask_path, expect_shape=arr.shape, dtype=np.uint8)
    if postprocess is not None:
        arr = postprocess(arr)
    sel = np.isfinite(arr) & (m == 1)
    if np.any(sel):
        yield arr[sel].astype(np.float64, copy=False)


# ----------------------------- Running stats (exact) ----------------------------- #

class RunningStats:
    """
    Exact streaming mean/std/min/max using batch-combine Welford.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_ = math.inf
        self.max_ = -math.inf

    def update_batch(self, x1d: np.ndarray):
        if x1d.size == 0:
            return
        # batch min/max
        xm = float(np.min(x1d))
        xM = float(np.max(x1d))
        if xm < self.min_:
            self.min_ = xm
        if xM > self.max_:
            self.max_ = xM

        # batch mean/M2
        nb = int(x1d.size)
        mb = float(np.mean(x1d))
        # M2_b = sum((x - mb)^2)
        M2b = float(np.sum((x1d - mb) ** 2))

        if self.n == 0:
            self.n = nb
            self.mean = mb
            self.M2 = M2b
            return

        n1 = self.n
        n2 = nb
        delta = mb - self.mean
        n = n1 + n2
        self.mean = self.mean + delta * (n2 / n)
        self.M2 = self.M2 + M2b + (delta * delta) * (n1 * n2 / n)
        self.n = n

    def to_dict(self):
        if self.n <= 0:
            return {"n": 0, "min": float("nan"), "max": float("nan"),
                    "mean": float("nan"), "std": float("nan")}
        var = self.M2 / max(self.n, 1)
        std = float(np.sqrt(var) + 1e-12)
        return {"n": int(self.n), "min": float(self.min_), "max": float(self.max_),
                "mean": float(self.mean), "std": std}


# ----------------------------- Histogram percentile (approx) ----------------------------- #

class StreamingHistogram:
    """
    Approx histogram for percentile (for p90 of abs(u/v)).
    Two-pass: min/max then hist.
    """
    def __init__(self, bins=8192):
        self.bins = int(bins)
        self.min_ = math.inf
        self.max_ = -math.inf
        self.hist = None
        self.eps = 1e-12

    def update_minmax(self, x1d):
        if x1d.size == 0:
            return
        xm = float(np.min(x1d))
        xM = float(np.max(x1d))
        if np.isfinite(xm):
            self.min_ = min(self.min_, xm)
        if np.isfinite(xM):
            self.max_ = max(self.max_, xM)

    def allocate(self):
        if not np.isfinite(self.min_) or not np.isfinite(self.max_):
            self.min_, self.max_ = 0.0, 1.0
        if self.max_ <= self.min_:
            self.max_ = self.min_ + 1.0
        self.hist = np.zeros(self.bins, dtype=np.int64)

    def update_hist(self, x1d):
        if x1d.size == 0:
            return
        scale = (self.bins - 1) / (self.max_ - self.min_ + self.eps)
        idx = ((x1d - self.min_) * scale).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)
        self.hist += np.bincount(idx, minlength=self.bins)

    def percentile(self, q):
        if self.hist is None:
            return float("nan")
        total = int(self.hist.sum())
        if total <= 0:
            return float("nan")
        rank = (q / 100.0) * (total - 1)
        cdf = np.cumsum(self.hist)
        k = int(np.searchsorted(cdf, rank))
        val = self.min_ + (self.max_ - self.min_) * (k / max(self.bins - 1, 1))
        return float(val)


# ----------------------------- Split ----------------------------- #

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


# ----------------------------- Transform helpers ----------------------------- #

def log1p_nonneg(x: np.ndarray) -> np.ndarray:
    # avoid NaN if some file has negative numerical noise
    return np.log1p(np.clip(x, 0.0, None))


def asinh_scale(x: np.ndarray, s: float) -> np.ndarray:
    return np.arcsinh(x / s)


# ----------------------------- Compute stats ----------------------------- #

def compute_feature_stats(train_rows, *, feat_key, mask_key):
    rs = RunningStats()
    for r in train_rows:
        for x in iter_masked_values(r[feat_key], r[mask_key], postprocess=None):
            rs.update_batch(x)
    return rs.to_dict()


def compute_h_stats(train_rows, bins=8192):
    """
    h: compute stats on log1p(coarse) and log1p(fine) separately using corresponding masks.
    No clipping.
    """
    rs_c = RunningStats()
    rs_f = RunningStats()

    for r in train_rows:
        # coarse flood map uses mask_coarse
        for x in iter_masked_values(r['coarse_path'], r['mask_coarse_path'], postprocess=log1p_nonneg):
            rs_c.update_batch(x)
        # fine flood map uses mask_fine
        for x in iter_masked_values(r['fine_path'], r['mask_fine_path'], postprocess=log1p_nonneg):
            rs_f.update_batch(x)

    out = {
        "coarse": {"after": "log1p_zscore", **rs_c.to_dict()},
        "fine": {"after": "log1p_zscore", **rs_f.to_dict()},
    }
    out["shared"] = dict(out["fine"])  # use fine stats as shared
    return out


def compute_uv_stats(train_rows, bins=8192):
    """
    u/v: compute p90(|x|) with histogram, then asinh(x/p90), then mean/std/min/max.
    We compute p90 on fine (mask_fine), and use it as shared scale for both coarse/fine.
    (Also record coarse/fine p90 for reference.)
    """
    # pass A: p90(abs(coarse)) and p90(abs(fine)) (both approximate)
    Hc = StreamingHistogram(bins=bins)
    Hf = StreamingHistogram(bins=bins)

    for r in train_rows:
        for x in iter_masked_values(r['coarse_path'], r['mask_coarse_path'], postprocess=None):
            Hc.update_minmax(np.abs(x))
        for x in iter_masked_values(r['fine_path'], r['mask_fine_path'], postprocess=None):
            Hf.update_minmax(np.abs(x))

    Hc.allocate()
    Hf.allocate()

    for r in train_rows:
        for x in iter_masked_values(r['coarse_path'], r['mask_coarse_path'], postprocess=None):
            Hc.update_hist(np.abs(x))
        for x in iter_masked_values(r['fine_path'], r['mask_fine_path'], postprocess=None):
            Hf.update_hist(np.abs(x))

    p90_abs_coarse = Hc.percentile(90.0)
    p90_abs_fine = Hf.percentile(90.0)

    # shared scale = fine p90 (per your preference)
    s_shared = p90_abs_fine
    if (not np.isfinite(s_shared)) or (s_shared <= 0):
        s_shared = 1.0

    # pass B: exact stats after asinh(x/s_shared)
    rs_c = RunningStats()
    rs_f = RunningStats()

    for r in train_rows:
        for x in iter_masked_values(r['coarse_path'], r['mask_coarse_path'],
                                    postprocess=lambda a: asinh_scale(a, s_shared)):
            rs_c.update_batch(x)
        for x in iter_masked_values(r['fine_path'], r['mask_fine_path'],
                                    postprocess=lambda a: asinh_scale(a, s_shared)):
            rs_f.update_batch(x)

    out = {
        "coarse": {
            "after": "asinh_p90_zscore",
            "asinh_scale_shared": float(s_shared),
            "p90_abs_coarse_ref": float(p90_abs_coarse),
            "p90_abs_fine_ref": float(p90_abs_fine),
            **rs_c.to_dict()
        },
        "fine": {
            "after": "asinh_p90_zscore",
            "asinh_scale_shared": float(s_shared),
            "p90_abs_coarse_ref": float(p90_abs_coarse),
            "p90_abs_fine_ref": float(p90_abs_fine),
            **rs_f.to_dict()
        },
    }
    out["shared"] = dict(out["fine"])  # use fine stats as shared
    return out


def compute_static_stats(train_rows):
    """
    static: compute exact mean/std/min/max on train only (masked by mask_fine).
    No clip, no p1/p99.
    """
    feats = {
        "elevation": "elev_path",
        "roughness": "rough_path",
        "twi": "twi_path",
        "slope": "slope_path",
        "aspect_sin": "aspect_sin_path",
        "aspect_cos": "aspect_cos_path",
        "mask_fine": "mask_fine_path",
    }
    out = {}
    for name, key in feats.items():
        rs = RunningStats()
        for r in train_rows:
            # mask_fine itself: still apply mask_fine (so stats reflect AOI region);
            # for mask feature it's basically {1}, but it's okay.
            for x in iter_masked_values(r[key], r['mask_fine_path'], postprocess=None):
                rs.update_batch(x)
        out[name] = rs.to_dict()
    return out


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_csv', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_json', required=True, help='split_stats_*.json 输出路径')
    ap.add_argument('--target_var', required=True, choices=['h', 'u', 'v'])
    ap.add_argument('--by', default='scenario', choices=['scenario', 'all'])
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=61)
    ap.add_argument('--bins', type=int, default=8192)
    args = ap.parse_args()

    rows_all = load_index_csv(args.index_csv)
    rows = [r for r in rows_all if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == args.target_var]

    def _maybe_join(p):
        return p if os.path.isabs(p) else os.path.join(args.root, p)

    # finalize required fields + absolutize paths
    need = ["coarse_path", "fine_path", "elev_path", "rough_path",
            "mask_fine_path", "mask_coarse_path", "slope_path", "twi_path",
            "aspect_sin_path", "aspect_cos_path"]
    for r in rows:
        for k in need:
            if k not in r:
                raise RuntimeError(f'[ERROR] index.csv missing field: {k}')
            r[k] = _maybe_join(r[k])

    # split (if existing json has split, reuse it)
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
        }

    meta["note"] = (
        "Split by scenario with fixed seed; stats computed on TRAIN only with masks. "
        "No percentile clipping. h uses log1p; u/v uses asinh(x/p90) with histogram p90."
    )

    print(f'Computing stats on {len(train_rows)} train patches, target_var={args.target_var} ...')

    # static rows come from var=h (your convention: static stored alongside h)
    static_rows_all = [r for r in rows_all if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == 'h']
    # make paths absolute for static rows too
    for r in static_rows_all:
        for k in need:
            if k not in r:
                raise RuntimeError(f'[ERROR] index.csv missing field: {k}')
            r[k] = _maybe_join(r[k])

    static_train_ids = set(meta['split']['train'])
    static_train_rows = [r for r in static_rows_all if int(r['_row_id']) in static_train_ids]
    meta['stats'] = compute_static_stats(static_train_rows)

    # stats_var
    if args.target_var == 'h':
        meta['stats_var'] = compute_h_stats(train_rows, bins=args.bins)
    else:
        meta['stats_var'] = compute_uv_stats(train_rows, bins=args.bins)

    meta['stats_var_for'] = args.target_var

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {args.out_json}')


if __name__ == '__main__':
    main()
