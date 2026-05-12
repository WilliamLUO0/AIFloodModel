#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example (h, log1p):
python tools/precompute_split_stats.py \
  --index_csv /.../index.csv \
  --root      /.../dataset \
  --out_json  /.../split_stats_h_log1p.json \
  --target_var h \
  --h_transform log1p \
  --aux_vars zs \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192

Example (h, asinh, scale from wet pixels):
python tools/precompute_split_stats.py \
  --index_csv /.../index.csv \
  --root      /.../dataset \
  --out_json  /.../split_stats_h_asinh_wet.json \
  --target_var h \
  --aux_vars zs \
  --h_transform asinh \
  --h_tau 0.05 \
  --h_q_list 50,75,90,95,99 \
  --h_asinh_scale_scope wet \
  --compute_h_flood_intervals \
  --h_flood_interval_thresholds 0.05,0.5,1.0 \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192

Example (u):
python tools/precompute_split_stats.py \
  --index_csv /.../index.csv \
  --root      /.../dataset \
  --out_json  /.../split_stats_u.json \
  --target_var u \
  --uv_tau 0.1 \
  --compute_uv_intervals \
  --uv_interval_thresholds 0.1,0.5,1.0 \
  --by scenario --val_ratio 0.2 --seed 61 \
  --bins 8192
"""

import os
import json
import math
import argparse
import random
from typing import Dict, List, Tuple

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


# ----------------------------- Running stats ----------------------------- #

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

        xm = float(np.min(x1d))
        xM = float(np.max(x1d))
        if xm < self.min_:
            self.min_ = xm
        if xM > self.max_:
            self.max_ = xM

        nb = int(x1d.size)
        mb = float(np.mean(x1d))
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
            return {
                "n": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
        var = self.M2 / max(self.n, 1)
        std = float(np.sqrt(var) + 1e-12)
        return {
            "n": int(self.n),
            "min": float(self.min_),
            "max": float(self.max_),
            "mean": float(self.mean),
            "std": std,
        }


# ----------------------------- Histogram percentile ----------------------------- #

class StreamingHistogram:
    """
    Approx histogram for percentile.
    Two-pass: min/max then hist.
    """
    def __init__(self, bins=8192):
        self.bins = int(bins)
        self.min_ = math.inf
        self.max_ = -math.inf
        self.hist = None
        self.eps = 1e-12

    def update_minmax(self, x1d: np.ndarray):
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

    def update_hist(self, x1d: np.ndarray):
        if x1d.size == 0:
            return
        scale = (self.bins - 1) / (self.max_ - self.min_ + self.eps)
        idx = ((x1d - self.min_) * scale).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)
        self.hist += np.bincount(idx, minlength=self.bins)

    def percentile(self, q: float) -> float:
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
    return np.log1p(np.clip(x, 0.0, None))


def asinh_scale_signed(x: np.ndarray, s: float) -> np.ndarray:
    return np.arcsinh(x / (s + 1e-12))


# ----------------------------- Static stats ----------------------------- #

def compute_static_stats(train_rows):
    """
    Static stats on TRAIN only.
    Elevation is computed from fine-grid elevation only, and shared by fine/coarse elevation.
    """
    out = {}

    # elevation: fine reference only
    if 'elev_fine_path' in train_rows[0]:
        rs_elev = RunningStats()
        for r in train_rows:
            for x in iter_masked_values(r['elev_fine_path'], r['mask_fine_path'], postprocess=None):
                rs_elev.update_batch(x)
        out["elevation"] = {
            "reference_grid": "fine",
            "applies_to": ["elevation_fine", "elevation_coarse"],
            **rs_elev.to_dict()
        }
    elif 'elev_path' in train_rows[0]:
        rs_elev = RunningStats()
        for r in train_rows:
            for x in iter_masked_values(r['elev_path'], r['mask_fine_path'], postprocess=None):
                rs_elev.update_batch(x)
        out["elevation"] = {
            "reference_grid": "fine",
            "applies_to": ["elevation_fine", "elevation_coarse"],
            **rs_elev.to_dict()
        }

    static_feats = [
        ("roughness", "rough_path"),
        ("twi", "twi_path"),
        ("slope", "slope_path"),
        ("aspect_sin", "aspect_sin_path"),
        ("aspect_cos", "aspect_cos_path"),
        ("mask_fine", "mask_fine_path"),
    ]

    for name, key in static_feats:
        if key not in train_rows[0]:
            continue
        rs = RunningStats()
        for r in train_rows:
            for x in iter_masked_values(r[key], r['mask_fine_path'], postprocess=None):
                rs.update_batch(x)
        out[name] = rs.to_dict()

    return out


# ----------------------------- Aux stats: zs ----------------------------- #

def compute_aux_zs_stats(train_rows, zs_coarse_key='zs_coarse_path', zs_fine_key='zs_fine_path'):
    """
    zs is treated as an auxiliary variable for h task.
    Keep coarse/fine/shared stats, but no separate mode field.
    shared = fine-grid raw zs stats.
    """
    rs_c = RunningStats()
    rs_f = RunningStats()

    for r in train_rows:
        for x in iter_masked_values(r[zs_coarse_key], r['mask_coarse_path'], postprocess=None):
            rs_c.update_batch(x)
        for x in iter_masked_values(r[zs_fine_key], r['mask_fine_path'], postprocess=None):
            rs_f.update_batch(x)

    return {
        "reference_grid": "fine",
        "coarse": rs_c.to_dict(),
        "fine": rs_f.to_dict(),
        "shared": rs_f.to_dict(),
        "note": (
            "zs is treated as an auxiliary variable. "
            "Use shared.mean/shared.std (from fine-grid raw zs) for z-score normalization "
            "for both coarse-grid zs and fine-grid zs."
        ),
    }


# ----------------------------- h stats ----------------------------- #

def compute_h_log1p_stats(train_rows):
    rs_c = RunningStats()
    rs_f = RunningStats()

    for r in train_rows:
        for x in iter_masked_values(r['coarse_path'], r['mask_coarse_path'], postprocess=log1p_nonneg):
            rs_c.update_batch(x)
        for x in iter_masked_values(r['fine_path'], r['mask_fine_path'], postprocess=log1p_nonneg):
            rs_f.update_batch(x)

    return {
        "mode": "log1p",
        "coarse": {"after": "log1p_zscore", **rs_c.to_dict()},
        "fine": {"after": "log1p_zscore", **rs_f.to_dict()},
        "shared": {"reference_grid": "fine", "after": "log1p_zscore", **rs_f.to_dict()},
    }


def compute_h_wet_ratio_and_pos_weight_fine_raw(train_rows, tau=0.05):
    tau = float(tau)
    total = 0
    wet = 0

    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)

        sel = np.isfinite(arr) & (m == 1)
        if not np.any(sel):
            continue

        x = arr[sel]
        total += int(x.size)
        wet += int(np.sum(x >= tau))

    wet_ratio = float(wet / max(total, 1))
    pos = wet
    neg = total - wet
    pos_weight = float(neg / max(pos, 1))
    return wet_ratio, pos_weight


def compute_h_flood_interval_stats_fine_raw(train_rows, thresholds=(0.05, 0.5, 1.0)):
    t1, t2, t3 = [float(x) for x in thresholds]

    total = 0
    counts = {
        "nonflood": 0,
        "slightflood": 0,
        "severeflood": 0,
        "extremeflood": 0,
    }

    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)
        sel = np.isfinite(arr) & (m == 1)
        if not np.any(sel):
            continue

        x = arr[sel]
        total += int(x.size)

        counts["nonflood"] += int(np.sum(x < t1))
        counts["slightflood"] += int(np.sum((x >= t1) & (x < t2)))
        counts["severeflood"] += int(np.sum((x >= t2) & (x < t3)))
        counts["extremeflood"] += int(np.sum(x >= t3))

    ratios = {k: float(v / max(total, 1)) for k, v in counts.items()}

    return {
        "thresholds_m": {
            "nonflood_upper": t1,
            "slightflood_lower": t1,
            "slightflood_upper": t2,
            "severeflood_lower": t2,
            "severeflood_upper": t3,
            "extremeflood_lower": t3,
        },
        "total_aoi_pixels_fine_raw": int(total),
        "counts": {k: int(v) for k, v in counts.items()},
        "ratios": ratios,
        "note": (
            "Computed on TRAIN only, using fine-grid raw water depth "
            "(mask_fine==1 & isfinite). Ratios are interval pixel counts "
            "divided by total AOI valid pixels."
        ),
    }


def compute_h_asinh_candidates_and_stats(
    train_rows,
    *,
    tau=0.05,
    q_list: List[int],
    bins=8192,
    scale_scope='wet',
):
    tau = float(tau)
    q_list = sorted(list(dict.fromkeys([int(q) for q in q_list])))

    H = StreamingHistogram(bins=bins)
    selected_total = 0

    # pass 1: min/max
    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)

        if scale_scope == 'wet':
            sel = np.isfinite(arr) & (m == 1) & (arr >= tau)
        elif scale_scope == 'all':
            sel = np.isfinite(arr) & (m == 1)
        else:
            raise ValueError(f'Unknown scale_scope: {scale_scope}')

        if not np.any(sel):
            continue
        x = arr[sel].astype(np.float64, copy=False)
        selected_total += int(x.size)
        H.update_minmax(x)

    H.allocate()

    # pass 2: hist
    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)

        if scale_scope == 'wet':
            sel = np.isfinite(arr) & (m == 1) & (arr >= tau)
        elif scale_scope == 'all':
            sel = np.isfinite(arr) & (m == 1)
        else:
            raise ValueError(f'Unknown scale_scope: {scale_scope}')

        if not np.any(sel):
            continue
        x = arr[sel].astype(np.float64, copy=False)
        H.update_hist(x)

    s_candidates: Dict[str, float] = {}
    for q in q_list:
        s = H.percentile(float(q))
        if (not np.isfinite(s)) or (s <= 0):
            s = 1.0
        s_candidates[str(q)] = float(s)

    rs_coarse = {str(q): RunningStats() for q in q_list}
    rs_fine = {str(q): RunningStats() for q in q_list}

    for r in train_rows:
        c = _load_npy_shape(r['coarse_path'], dtype=np.float32)
        mc = _load_npy_shape(r['mask_coarse_path'], expect_shape=c.shape, dtype=np.uint8)
        sel_c = np.isfinite(c) & (mc == 1)
        x_c = c[sel_c].astype(np.float64, copy=False) if np.any(sel_c) else None

        f = _load_npy_shape(r['fine_path'], dtype=np.float32)
        mf = _load_npy_shape(r['mask_fine_path'], expect_shape=f.shape, dtype=np.uint8)
        sel_f = np.isfinite(f) & (mf == 1)
        x_f = f[sel_f].astype(np.float64, copy=False) if np.any(sel_f) else None

        if x_c is None and x_f is None:
            continue

        for q in q_list:
            sq = s_candidates[str(q)]
            if x_c is not None and x_c.size > 0:
                y_c = np.arcsinh(np.clip(x_c, 0.0, None) / (sq + 1e-12))
                rs_coarse[str(q)].update_batch(y_c)
            if x_f is not None and x_f.size > 0:
                y_f = np.arcsinh(np.clip(x_f, 0.0, None) / (sq + 1e-12))
                rs_fine[str(q)].update_batch(y_f)

    asinh_by_q = {}
    for q in q_list:
        qk = str(q)
        coarse_stats = rs_coarse[qk].to_dict()
        fine_stats = rs_fine[qk].to_dict()
        asinh_by_q[qk] = {
            "s": float(s_candidates[qk]),
            "coarse": {"after": "asinh(h/s)_zscore", **coarse_stats},
            "fine": {"after": "asinh(h/s)_zscore", **fine_stats},
            "shared": {"reference_grid": "fine", "after": "asinh(h/s)_zscore", **fine_stats},
        }

    if scale_scope == 'wet':
        percentile_note = (
            "Quantiles computed via histogram on fine-grid RAW wet pixels "
            "(mask_fine==1 & isfinite & h>=tau)."
        )
        total_key = "wet_total_fine_raw_ge_tau"
    else:
        percentile_note = (
            "Quantiles computed via histogram on fine-grid RAW valid pixels "
            "(mask_fine==1 & isfinite)."
        )
        total_key = "fine_valid_total_raw"

    return {
        "mode": "asinh",
        "tau_wet_raw_m": float(tau),
        "h_asinh_scale_scope": scale_scope,
        "h_asinh_scale_candidates": s_candidates,
        "asinh_by_q": asinh_by_q,
        total_key: int(selected_total),
        "percentile_note": percentile_note,
        "stats_note": "All mean/std/min/max are exact streaming stats after transform (Welford).",
    }


# ----------------------------- u/v stats ----------------------------- #

def compute_uv_pos_ratio_and_pos_weight_fine_raw(train_rows, tau=0.1):
    tau = float(tau)
    total = 0
    pos = 0

    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)

        sel = np.isfinite(arr) & (m == 1)
        if not np.any(sel):
            continue

        x = arr[sel]
        total += int(x.size)
        pos += int(np.sum(np.abs(x) >= tau))

    pos_ratio = float(pos / max(total, 1))
    neg = total - pos
    pos_weight = float(neg / max(pos, 1))
    return pos_ratio, pos_weight


def compute_uv_interval_stats_fine_raw(train_rows, thresholds=(0.1, 0.5, 1.0)):
    t1, t2, t3 = [float(x) for x in thresholds]

    total = 0
    counts = {
        "low": 0,
        "moderate": 0,
        "high": 0,
        "extreme": 0,
    }

    for r in train_rows:
        arr = _load_npy_shape(r['fine_path'], dtype=np.float32)
        m = _load_npy_shape(r['mask_fine_path'], expect_shape=arr.shape, dtype=np.uint8)
        sel = np.isfinite(arr) & (m == 1)
        if not np.any(sel):
            continue

        x = np.abs(arr[sel])
        total += int(x.size)

        counts["low"] += int(np.sum(x < t1))
        counts["moderate"] += int(np.sum((x >= t1) & (x < t2)))
        counts["high"] += int(np.sum((x >= t2) & (x < t3)))
        counts["extreme"] += int(np.sum(x >= t3))

    ratios = {k: float(v / max(total, 1)) for k, v in counts.items()}

    return {
        "thresholds_abs": {
            "low_upper": t1,
            "moderate_lower": t1,
            "moderate_upper": t2,
            "high_lower": t2,
            "high_upper": t3,
            "extreme_lower": t3,
        },
        "total_aoi_pixels_fine_raw": int(total),
        "counts": {k: int(v) for k, v in counts.items()},
        "ratios": ratios,
        "note": (
            "Computed on TRAIN only, using fine-grid raw abs(u/v) "
            "(mask_fine==1 & isfinite). Ratios are interval pixel counts "
            "divided by total AOI valid pixels."
        ),
    }


def compute_uv_stats(train_rows, bins=8192):
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
    p99_abs_coarse = Hc.percentile(99.0)
    p99_abs_fine = Hf.percentile(99.0)

    s_shared = p90_abs_fine
    if (not np.isfinite(s_shared)) or (s_shared <= 0):
        s_shared = 1.0

    rs_c = RunningStats()
    rs_f = RunningStats()

    for r in train_rows:
        for x in iter_masked_values(
            r['coarse_path'], r['mask_coarse_path'],
            postprocess=lambda a: asinh_scale_signed(a, s_shared)
        ):
            rs_c.update_batch(x)
        for x in iter_masked_values(
            r['fine_path'], r['mask_fine_path'],
            postprocess=lambda a: asinh_scale_signed(a, s_shared)
        ):
            rs_f.update_batch(x)

    return {
        "mode": "asinh",
        "coarse": {
            "after": "asinh_p90_zscore",
            "asinh_scale_shared": float(s_shared),
            "p90_abs_coarse_ref": float(p90_abs_coarse),
            "p90_abs_fine_ref": float(p90_abs_fine),
            "p99_abs_coarse_ref": float(p99_abs_coarse),
            "p99_abs_fine_ref": float(p99_abs_fine),
            **rs_c.to_dict()
        },
        "fine": {
            "after": "asinh_p90_zscore",
            "asinh_scale_shared": float(s_shared),
            "p90_abs_coarse_ref": float(p90_abs_coarse),
            "p90_abs_fine_ref": float(p90_abs_fine),
            "p99_abs_coarse_ref": float(p99_abs_coarse),
            "p99_abs_fine_ref": float(p99_abs_fine),
            **rs_f.to_dict()
        },
        "shared": {
            "reference_grid": "fine",
            "after": "asinh_p90_zscore",
            "asinh_scale_shared": float(s_shared),
            "p90_abs_coarse_ref": float(p90_abs_coarse),
            "p90_abs_fine_ref": float(p90_abs_fine),
            "p99_abs_coarse_ref": float(p99_abs_coarse),
            "p99_abs_fine_ref": float(p99_abs_fine),
            **rs_f.to_dict()
        },
    }


# ----------------------------- Utilities ----------------------------- #

def parse_three_floats(s: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in s.split(',') if x.strip() != '']
    if len(vals) != 3:
        raise RuntimeError('Expected exactly 3 comma-separated values.')
    return vals[0], vals[1], vals[2]


def parse_aux_vars(s: str) -> List[str]:
    if s is None or str(s).strip() == '':
        return []
    return [x.strip() for x in s.split(',') if x.strip() != '']


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_csv', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_json', required=True, help='Output path of split_stats_*.json')

    # main task
    ap.add_argument('--target_var', required=True, choices=['h', 'u', 'v'])
    ap.add_argument('--aux_vars', type=str, default='',
                    help='Comma-separated auxiliary variables, e.g. zs')

    ap.add_argument('--by', default='scenario', choices=['scenario', 'all'])
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=61)
    ap.add_argument('--bins', type=int, default=8192)

    # h only
    ap.add_argument('--h_transform', default='log1p', choices=['log1p', 'asinh'])
    ap.add_argument('--h_tau', type=float, default=0.05)
    ap.add_argument('--h_q_list', type=str, default='50,75,90,95,99')
    ap.add_argument('--h_asinh_scale_scope', default='wet', choices=['wet', 'all'])
    ap.add_argument('--compute_h_flood_intervals', action='store_true')
    ap.add_argument('--h_flood_interval_thresholds', type=str, default='0.05,0.5,1.0')

    # u/v only
    ap.add_argument('--uv_tau', type=float, default=0.1)
    ap.add_argument('--compute_uv_intervals', action='store_true')
    ap.add_argument('--uv_interval_thresholds', type=str, default='0.1,0.5,1.0')

    args = ap.parse_args()

    aux_vars = parse_aux_vars(args.aux_vars)
    q_list = [int(x.strip()) for x in args.h_q_list.split(',') if str(x).strip() != '']

    rows_all = load_index_csv(args.index_csv)
    rows = [
        r for r in rows_all
        if int(r['filtered_out']) == 0 and str(r.get('var', '')).lower() == args.target_var
    ]

    def _maybe_join(p):
        return p if os.path.isabs(p) else os.path.join(args.root, p)

    need_common = [
        "coarse_path", "fine_path",
        "mask_fine_path", "mask_coarse_path",
        "rough_path", "twi_path", "slope_path",
        "aspect_sin_path", "aspect_cos_path",
    ]

    for r in rows:
        # elevation compatibility
        if 'elev_fine_path' not in r and 'elev_path' in r:
            r['elev_fine_path'] = r['elev_path']

        for k in need_common + ['elev_fine_path']:
            if k not in r:
                raise RuntimeError(f'[ERROR] index.csv missing field: {k}')
            r[k] = _maybe_join(r[k])

        if 'elev_coarse_path' in r and str(r['elev_coarse_path']).strip() != '':
            r['elev_coarse_path'] = _maybe_join(r['elev_coarse_path'])

        # aux vars
        if 'zs' in aux_vars:
            for k in ['zs_coarse_path', 'zs_fine_path']:
                if k not in r:
                    raise RuntimeError(f'[ERROR] index.csv missing field for aux var zs: {k}')
                r[k] = _maybe_join(r[k])

    # split
    if os.path.isfile(args.out_json):
        with open(args.out_json, 'r') as f:
            meta = json.load(f)
        if 'split' not in meta:
            raise RuntimeError(f'[ERROR] Existing out_json missing "split".')
        train_ids = set(meta['split']['train'])
        val_ids = set(meta['split'].get('val', []))
    else:
        train_ids_list, val_ids_list = make_split(rows, args.by, args.val_ratio, args.seed)
        train_ids = set(train_ids_list)
        val_ids = set(val_ids_list)
        meta = {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "split_by": args.by,
            "split": {"train": sorted(list(train_ids)), "val": sorted(list(val_ids))},
        }

    train_rows = [r for r in rows if int(r['_row_id']) in train_ids]

    print(f'Computing stats on {len(train_rows)} train patches, target_var={args.target_var}, aux_vars={aux_vars} ...')

    meta["note"] = (
        "Split with fixed seed; stats computed on TRAIN only with masks. "
        "Static stats use fine-grid elevation as shared reference for both fine/coarse elevation. "
        "For h: log1p -> exact stats on log1p(h); asinh -> s candidates from fine-grid raw pixels "
        "according to h_asinh_scale_scope, then exact stats on asinh(h/s). "
        "For u/v: s = p90(abs(fine raw)) then exact stats on asinh(x/s). "
        "For h and u/v, extra class-imbalance stats are computed from fine-grid raw values. "
        "Auxiliary variables are stored under aux_stats."
    )

    meta["target_var"] = args.target_var
    meta["aux_vars"] = aux_vars

    # static
    meta["stats"] = compute_static_stats(train_rows)

    # main target stats
    if args.target_var == 'h':
        wet_ratio, pos_weight = compute_h_wet_ratio_and_pos_weight_fine_raw(train_rows, tau=args.h_tau)

        if args.h_transform == 'log1p':
            stats_var = compute_h_log1p_stats(train_rows)
        else:
            stats_var = compute_h_asinh_candidates_and_stats(
                train_rows,
                tau=args.h_tau,
                q_list=q_list,
                bins=args.bins,
                scale_scope=args.h_asinh_scale_scope,
            )

        stats_var["tau_wet_raw_m"] = float(args.h_tau)
        stats_var["wet_ratio_fine_raw_tau"] = float(wet_ratio)
        stats_var["pos_weight_fine_raw_tau"] = float(pos_weight)
        stats_var["pos_weight_def"] = (
            "neg/pos on fine-grid raw (mask_fine==1 & isfinite), "
            "pos=(h>=tau), neg=others"
        )

        if args.compute_h_flood_intervals:
            thresholds = parse_three_floats(args.h_flood_interval_thresholds)
            stats_var["flood_interval_stats_fine_raw"] = compute_h_flood_interval_stats_fine_raw(
                train_rows, thresholds=thresholds
            )

        meta["stats_var"] = stats_var
        meta["stats_var_for"] = "h"
        meta["h_transform"] = args.h_transform
        meta["h_tau"] = float(args.h_tau)
        meta["h_q_list"] = [int(x) for x in q_list]
        meta["h_asinh_scale_scope"] = args.h_asinh_scale_scope if args.h_transform == 'asinh' else None
        meta["compute_h_flood_intervals"] = bool(args.compute_h_flood_intervals)
        meta["h_flood_interval_thresholds"] = (
            list(parse_three_floats(args.h_flood_interval_thresholds))
            if args.compute_h_flood_intervals else None
        )

    elif args.target_var in ['u', 'v']:
        stats_uv = compute_uv_stats(train_rows, bins=args.bins)

        pos_ratio, pos_weight = compute_uv_pos_ratio_and_pos_weight_fine_raw(train_rows, tau=args.uv_tau)
        stats_uv["uv_tau_abs_raw"] = float(args.uv_tau)
        stats_uv["pos_ratio_fine_abs_tau"] = float(pos_ratio)
        stats_uv["pos_weight_fine_abs_tau"] = float(pos_weight)
        stats_uv["pos_weight_def"] = (
            "neg/pos on fine-grid raw (mask_fine==1 & isfinite), "
            "pos=(abs(u/v)>=uv_tau), neg=others"
        )

        if args.compute_uv_intervals:
            thresholds = parse_three_floats(args.uv_interval_thresholds)
            stats_uv["uv_interval_stats_fine_raw"] = compute_uv_interval_stats_fine_raw(
                train_rows, thresholds=thresholds
            )

        meta["stats_var"] = stats_uv
        meta["stats_var_for"] = args.target_var
        meta["uv_tau"] = float(args.uv_tau)
        meta["compute_uv_intervals"] = bool(args.compute_uv_intervals)
        meta["uv_interval_thresholds"] = (
            list(parse_three_floats(args.uv_interval_thresholds))
            if args.compute_uv_intervals else None
        )

    else:
        raise RuntimeError(f'Unsupported target_var: {args.target_var}')

    # aux stats
    aux_stats = {}
    if 'zs' in aux_vars:
        aux_stats["zs"] = compute_aux_zs_stats(train_rows, zs_coarse_key='zs_coarse_path', zs_fine_key='zs_fine_path')
    meta["aux_stats"] = aux_stats

    out_dir = os.path.dirname(args.out_json)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Wrote {args.out_json}')


if __name__ == '__main__':
    main()