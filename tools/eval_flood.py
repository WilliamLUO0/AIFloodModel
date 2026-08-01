#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# VAL（看 *_patch_mean 列）
python tools/eval_flood.py \
  --index-csv .../dataset_ds8_filtered_thr0p1_min5100_full/index_with_interval_stats_h.csv \
  --vis-root  .../results/039_FMPFTV8_SRx8_Filter_InbaL1BCE_LW_eval_val/visualization \
  --out-json  .../eval_val_summary.json \
  --out-csv-patch .../eval_val_patch.csv --out-csv-time .../eval_val_time.csv --out-csv-scenario .../eval_val_scenario.csv

# TEST 100y42h（看 *_global 列；dataset_global = He et al. 口径）
python tools/eval_flood.py \
  --index-csv .../testdataset_100y42h0c/index.csv \
  --vis-root  .../results/039_..._eval_test100y42h0c/visualization \
  --out-json  .../eval_test_summary.json --out-csv-time .../eval_test_time.csv --out-csv-scenario .../eval_test_scenario.csv

# 单点查询
python tools/eval_flood.py --index-csv ... --vis-root ... --query "t0047,100y_42h_0c"      # 单整图全局
python tools/eval_flood.py --index-csv ... --vis-root ... --query "3,15,t0047,100y_42h_0c" # 单 patch

# VAL 上采样基线（子集，仍传 vis-root 以对齐模型评估的那 20%）
python tools/eval_flood.py \
  --index-csv .../dataset_ds8_filtered.../index_with_interval_stats_h.csv \
  --vis-root  .../results/039_..._eval_val/visualization \
  --pred-source coarse_upsample \
  --out-json .../eval_val_baseline.json

# TEST 上采样基线（完整集，不用 vis-root 了）
python tools/eval_flood.py \
  --index-csv .../testdataset_100y42h0c/index.csv \
  --pred-source coarse_upsample \
  --out-json .../eval_test_baseline.json --out-csv-time .../eval_test_baseline_time.csv

# 单点也支持基线
python tools/eval_flood.py --index-csv ... --vis-root ... --pred-source coarse_upsample --query "t0047,100y_42h_0c"

# 速度 u/v（--var u/v + --abs）：wet/dry、分档、accuracy、混淆按 |v| 判定，
# RMSE/NSE/correlation 用带符号速度。h 不加 --abs，行为不变。
python tools/eval_flood.py \
  --index-csv .../testdataset_100y42h0c/index.csv \
  --vis-root  .../results/01_..._u_eval_test100y42h0c/visualization \
  --var u --abs \
  --out-json  .../eval_test_u_summary.json --out-csv-time .../eval_test_u_time.csv
"""

import os
import re
import glob
import csv
import json
import argparse
from collections import defaultdict

import numpy as np


# =========================================================
# File/folder parsing
# =========================================================
# scenario is non-greedy so it matches BOTH the design-rain token (e.g.
# "100y_42h_0c", which itself contains underscores) and free-form names like
# "gabrielle"; the fixed "_t<4d>_r<3d>_c<3d>_s<d>" suffix pins where it ends.
CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>.+?)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)
FOLDER_RE = re.compile(r'^(?P<core>.+)_coarse$')


def parse_core_key(core: str) -> dict:
    m = CORE_RE.match(core)
    if not m:
        raise ValueError(f"Bad core key: {core}")
    d = m.groupdict()
    d["patch_row"] = int(d["r"])
    d["patch_col"] = int(d["c"])
    d["downscale"] = int(d["s"])
    return d


def parse_folder_name(folder_basename: str):
    m = FOLDER_RE.match(folder_basename)
    if not m:
        return None
    core = m.group("core")
    try:
        parse_core_key(core)
        return core
    except Exception:
        return None


def pick_pred_file_in_folder(folder: str, core: str) -> str:
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npys:
        raise RuntimeError(f"No .npy found in {folder}")

    cand = []
    for p in npys:
        bn = os.path.basename(p)
        if not bn.startswith(core + "_"):
            continue
        if bn.endswith("_coarse.npy") or bn.endswith("_fine.npy"):
            continue
        cand.append(p)

    if cand:
        return cand[0]

    if len(npys) == 1:
        return npys[0]

    raise RuntimeError(f"Cannot identify predicted .npy in {folder} for core={core}")


# =========================================================
# Index CSV mapping
# =========================================================
def build_index_map(index_csv: str):
    """
    Build mapping core -> row(dict) from dataset/index.csv.
    Required columns:
      var, scenario, t, patch_row, patch_col, downscale,
      fine_path, mask_fine_path
    """
    mp = {}
    with open(index_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                var = row["var"]
                scenario = row["scenario"]
                t = row["t"]
                rr = int(row["patch_row"])
                cc = int(row["patch_col"])
                scale = int(row["downscale"])
            except Exception:
                continue
            core = f"{var}_{scenario}_{t}_r{rr:03d}_c{cc:03d}_s{scale}"
            mp[core] = row
    return mp


# =========================================================
# Array helpers
# =========================================================
def ensure_hw(arr: np.ndarray, name="arr") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0, ...]
    if a.ndim == 4:
        return a[0, 0, ...]
    raise RuntimeError(f"{name} has unsupported shape: {a.shape}")


def load_hw_npy(path: str, name="arr") -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return ensure_hw(np.load(path), name=name)


# =========================================================
# Config
# =========================================================
EPS = 1.0e-12

# Depth event threshold / tolerance
DEPTH_EVENT_THRESHOLD = 0.1
DEPTH_ABS_TOL = 0.02

# Safe NSE parameters
NSE_SAFE_MIN_VAR = 1.0e-6
NSE_SAFE_ABS_TOL_PER_PX = 1.0e-4
NSE_SAFE_LOWER_BOUND = -5.0

# Empty / degenerate handling policy (set from --empty-policy in main()):
#   "nan"  -> (default) flood-free or near-flat patches/maps are UNDEFINED, not 0.
#             * precision/recall/csi with a zero denominator (no event in
#               gt and/or pred) -> NaN
#             * NSE on a map whose target variance < min_var (essentially no
#               flood signal) -> NaN
#             NaNs are dropped from every *_patch_mean / *_mapmean average, so
#             dry patches no longer dilute the score. This matches the
#             training-time validation (empty_as_nan: true) and is the
#             statistically correct behaviour (these metrics are undefined,
#             not zero).
#   "zero" -> legacy behaviour: zero-denominator ratios = 0; degenerate NSE is
#             rescued to 1.0 (near-perfect) or clamped to NSE_SAFE_LOWER_BOUND.
EMPTY_POLICY = "nan"

# Velocity (u/v) mode: set from --abs in main(). When True, all threshold/band
# CLASSIFICATION (event masks, band selection, accuracy/confusion band index) is
# done on |value|; every regression VALUE (rmse/nse/band_sse/correlation/min-max)
# stays on the SIGNED value. When False (default, water depth) behaviour is
# identical to before (pa==pred, ga==gt).
USE_ABS = False

# Four intervals
DEPTH_BANDS = {
    "nonflood": {"ge": None, "lt": 0.1},
    "slight":   {"ge": 0.1,  "lt": 0.5},
    "severe":   {"ge": 0.5,  "lt": 1.0},
    "extreme":  {"ge": 1.0,  "lt": None},
}


# =========================================================
# Small utility helpers
# =========================================================
def _ratio(num, den, eps=EPS, empty_as_nan=False):
    # den is the (non-negative) classification denominator, e.g. tp+fp.
    # den == 0 means the metric is undefined (no event in gt and/or pred).
    if empty_as_nan and den <= 0:
        return float("nan")
    return float(num / (den + eps))


def _rmse_from_sse(sse_sum: float, n_sum: int):
    if n_sum <= 0:
        return float("nan")
    return float(np.sqrt(max(sse_sum, 0.0) / float(max(n_sum, 1))))


def _nse_from_sums(sse_sum: float, t_sum: float, t_sq_sum: float, n_sum: int, eps=EPS,
                   min_var: float = NSE_SAFE_MIN_VAR, empty_as_nan: bool = False):
    if n_sum <= 0:
        return float("nan")
    mean = t_sum / float(n_sum)
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)
    # var ~ 0 => target field is essentially flat (no flood). NSE = 1 - SSE/Var
    # is undefined / wildly unstable there (the source of the -1e15 patches), so
    # under the "nan" policy we drop it instead of dividing by ~0.
    if empty_as_nan and var < min_var:
        return float("nan")
    var = max(var, eps)
    return float(1.0 - (sse_sum / var))


def _nse_safe_from_sums(
    sse_sum: float,
    t_sum: float,
    t_sq_sum: float,
    n_sum: int,
    min_var: float = NSE_SAFE_MIN_VAR,
    abs_tol_per_px: float = NSE_SAFE_ABS_TOL_PER_PX,
    lower_bound: float = NSE_SAFE_LOWER_BOUND,
    eps: float = EPS,
    empty_as_nan: bool = False,
):
    if n_sum <= 0:
        return float("nan")

    mean = t_sum / float(n_sum)
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)

    if var >= min_var:
        var = max(var, eps)
        return float(1.0 - (sse_sum / var))

    # var < min_var: degenerate (essentially flood-free) map.
    # "nan" policy: undefined -> NaN (dropped from averages). This is what
    # removes the -5.0 / 1.0 artefacts on dry timesteps.
    if empty_as_nan:
        return float("nan")

    # legacy "zero" rescue: near-perfect dry map -> 1.0, else clamp.
    tiny_tol = (abs_tol_per_px ** 2) * float(n_sum)
    if sse_sum <= tiny_tol:
        return 1.0

    val = 1.0 - (sse_sum / float(min_var))
    return float(max(val, lower_bound))


def mean_agg(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(v.mean())


def select_band(x: np.ndarray, ge=None, lt=None):
    sel = np.ones_like(x, dtype=bool)
    if ge is not None:
        sel &= (x >= ge)
    if lt is not None:
        sel &= (x < lt)
    return sel


# =========================================================
# Patch-level additive stats
# =========================================================
def compute_patch_stats_depth(pred_hw: np.ndarray, gt_hw: np.ndarray, mask_hw: np.ndarray):
    """
    Compute additive stats for one patch.
    These stats can be summed across patches, then converted into
    global metrics at time/scenario/dataset level.
    """
    pred = np.asarray(pred_hw, dtype=np.float64)
    gt = np.asarray(gt_hw, dtype=np.float64)
    mask = np.asarray(mask_hw, dtype=np.float64)

    m = mask > 0.5
    N = int(m.sum())
    if N <= 0:
        raise RuntimeError("num_valid==0 (mask empty)")

    out = {"N": N}

    # For velocity (USE_ABS) threshold/band CLASSIFICATION is on |v|; regression
    # VALUES (rmse/nse/band_sse/correlation/min-max) stay on signed v. For depth
    # (USE_ABS=False) pa==pred and ga==gt, so behaviour is unchanged.
    pa = np.abs(pred) if USE_ABS else pred
    ga = np.abs(gt) if USE_ABS else gt

    # -------------------------------------------------
    # Global regression stats: RMSE / NSE families
    # -------------------------------------------------
    diff = pred - gt
    diff2 = diff * diff
    sse = float(diff2[m].sum())

    pred_eff = np.where((pa >= DEPTH_EVENT_THRESHOLD) & m, pred, 0.0)
    gt_eff = np.where((ga >= DEPTH_EVENT_THRESHOLD) & m, gt, 0.0)

    diff_thr = pred_eff - gt_eff
    sse_thr = float((diff_thr[m] ** 2).sum())

    abs_err_thr = np.abs(diff_thr)
    large_err_mask = m & (abs_err_thr >= DEPTH_ABS_TOL)
    sse_thr_tol = float((diff_thr[large_err_mask] ** 2).sum())

    gt_vals = gt[m]
    gt_eff_vals = gt_eff[m]

    out["rmse_sse"] = sse
    out["rmse_thr_sse"] = sse_thr
    out["rmse_thr_tol_sse"] = sse_thr_tol

    out["nse_sse"] = sse
    out["nse_t_sum"] = float(gt_vals.sum())
    out["nse_t_sq_sum"] = float((gt_vals * gt_vals).sum())

    out["nse_thr_sse"] = sse_thr
    out["nse_thr_t_sum"] = float(gt_eff_vals.sum())
    out["nse_thr_t_sq_sum"] = float((gt_eff_vals * gt_eff_vals).sum())

    out["nse_thr_tol_sse"] = sse_thr_tol
    out["nse_thr_tol_t_sum"] = float(gt_eff_vals.sum())
    out["nse_thr_tol_t_sq_sum"] = float((gt_eff_vals * gt_eff_vals).sum())

    # -------------------------------------------------
    # Flood/nonflood classification stats
    # -------------------------------------------------
    p_evt = (pa >= DEPTH_EVENT_THRESHOLD) & m
    t_evt = (ga >= DEPTH_EVENT_THRESHOLD) & m

    out["tp"] = int((p_evt & t_evt).sum())
    out["fp"] = int((p_evt & (~t_evt)).sum())
    out["fn"] = int(((~p_evt) & t_evt).sum())

    borderline = (np.abs(pred - gt) <= DEPTH_ABS_TOL) & m
    p_evt_tol = np.where(borderline, t_evt, p_evt)

    out["tp_tol"] = int((p_evt_tol & t_evt).sum())
    out["fp_tol"] = int((p_evt_tol & (~t_evt)).sum())
    out["fn_tol"] = int(((~p_evt_tol) & t_evt).sum())

    out["t_evt_cnt"] = int(t_evt.sum())
    out["p_evt_cnt"] = int(p_evt.sum())

    # -------------------------------------------------
    # Four-band stats for RMSE / Precision / Recall
    # -------------------------------------------------
    for band_name, band_cfg in DEPTH_BANDS.items():
        ge = band_cfg["ge"]
        lt = band_cfg["lt"]

        gt_band = select_band(ga, ge=ge, lt=lt) & m
        pred_band = select_band(pa, ge=ge, lt=lt) & m

        band_n = int(gt_band.sum())
        band_sse = float(((pred - gt) ** 2)[gt_band].sum()) if band_n > 0 else 0.0

        band_tp = int((pred_band & gt_band).sum())
        band_fp = int((pred_band & (~gt_band) & m).sum())
        band_fn = int(((~pred_band) & gt_band).sum())

        out[f"band_{band_name}_n"] = band_n
        out[f"band_{band_name}_sse"] = band_sse
        out[f"band_{band_name}_tp"] = band_tp
        out[f"band_{band_name}_fp"] = band_fp
        out[f"band_{band_name}_fn"] = band_fn

    # -------------------------------------------------
    # He et al. (2023) extras (all additive over AOI cells):
    #   - 4-class accuracy (numerator n_correct; denom = N)
    #   - full 4x4 confusion matrix (rows = gt band, cols = pred band)
    #   - Pearson correlation sums (Sigma p, g, pg, p^2, g^2)
    #   - value range/min/max for dataset value stats
    # Band index: 0=nonflood(<0.1) 1=slight[0.1,0.5) 2=severe[0.5,1) 3=extreme(>=1)
    # -------------------------------------------------
    def _band_index(x):
        idx = np.zeros(x.shape, dtype=np.int64)
        idx[(x >= 0.1) & (x < 0.5)] = 1
        idx[(x >= 0.5) & (x < 1.0)] = 2
        idx[x >= 1.0] = 3
        return idx

    pb = _band_index(pa)
    gb = _band_index(ga)
    out["n_correct"] = int(((pb == gb) & m).sum())
    for gi in range(4):
        for pi in range(4):
            out[f"cm_{gi}_{pi}"] = int((m & (gb == gi) & (pb == pi)).sum())

    pv = pred[m]
    gv = gt[m]
    out["corr_sum_p"] = float(pv.sum())
    out["corr_sum_g"] = float(gv.sum())
    out["corr_sum_pg"] = float((pv * gv).sum())
    out["corr_sum_pp"] = float((pv * pv).sum())
    out["corr_sum_gg"] = float((gv * gv).sum())
    # min/max are NOT summed (aggregated by min()/max() in aggregate_stats)
    out["val_min_p"] = float(pv.min())
    out["val_max_p"] = float(pv.max())
    out["val_min_g"] = float(gv.min())
    out["val_max_g"] = float(gv.max())

    return out


# =========================================================
# Recover metrics from additive stats
# =========================================================
def metrics_from_stats(stats: dict):
    out = {}

    N = int(stats["N"])

    # Empty/degenerate -> NaN under the "nan" policy (see EMPTY_POLICY docstring).
    ean = (EMPTY_POLICY == "nan")

    # Global rmse / nse families
    out["rmse"] = _rmse_from_sse(stats["rmse_sse"], N)
    out["rmse_thr"] = _rmse_from_sse(stats["rmse_thr_sse"], N)
    out["rmse_thr_tol"] = _rmse_from_sse(stats["rmse_thr_tol_sse"], N)

    out["nse"] = _nse_from_sums(stats["nse_sse"], stats["nse_t_sum"], stats["nse_t_sq_sum"], N, empty_as_nan=ean)
    out["nse_thr"] = _nse_from_sums(stats["nse_thr_sse"], stats["nse_thr_t_sum"], stats["nse_thr_t_sq_sum"], N, empty_as_nan=ean)
    out["nse_thr_tol"] = _nse_from_sums(stats["nse_thr_tol_sse"], stats["nse_thr_tol_t_sum"], stats["nse_thr_tol_t_sq_sum"], N, empty_as_nan=ean)

    out["nse_safe"] = _nse_safe_from_sums(
        stats["nse_sse"], stats["nse_t_sum"], stats["nse_t_sq_sum"], N, empty_as_nan=ean
    )
    out["nse_thr_safe"] = _nse_safe_from_sums(
        stats["nse_thr_sse"], stats["nse_thr_t_sum"], stats["nse_thr_t_sq_sum"], N, empty_as_nan=ean
    )
    out["nse_thr_tol_safe"] = _nse_safe_from_sums(
        stats["nse_thr_tol_sse"], stats["nse_thr_tol_t_sum"], stats["nse_thr_tol_t_sq_sum"], N, empty_as_nan=ean
    )

    # Flood/nonflood classification (zero denominator -> NaN under "nan" policy)
    out["precision"] = _ratio(stats["tp"], stats["tp"] + stats["fp"], empty_as_nan=ean)
    out["recall"] = _ratio(stats["tp"], stats["tp"] + stats["fn"], empty_as_nan=ean)
    out["csi"] = _ratio(stats["tp"], stats["tp"] + stats["fp"] + stats["fn"], empty_as_nan=ean)

    out["precision_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fp_tol"], empty_as_nan=ean)
    out["recall_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fn_tol"], empty_as_nan=ean)
    out["csi_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fp_tol"] + stats["fn_tol"], empty_as_nan=ean)

    # Prevalence is a rate over all AOI cells (N > 0 always) -> never empty.
    out["target_prevalence"] = _ratio(stats["t_evt_cnt"], N)
    out["pred_prevalence"] = _ratio(stats["p_evt_cnt"], N)

    # Four-band metrics
    for band_name in DEPTH_BANDS.keys():
        n = int(stats[f"band_{band_name}_n"])
        sse = float(stats[f"band_{band_name}_sse"])
        tp = int(stats[f"band_{band_name}_tp"])
        fp = int(stats[f"band_{band_name}_fp"])
        fn = int(stats[f"band_{band_name}_fn"])

        out[f"rmse_{band_name}"] = _rmse_from_sse(sse, n)
        out[f"precision_{band_name}"] = _ratio(tp, tp + fp, empty_as_nan=ean)
        out[f"recall_{band_name}"] = _ratio(tp, tp + fn, empty_as_nan=ean)
        out[f"support_{band_name}"] = n

    # He et al. extras: 4-class accuracy, Pearson correlation, value stats
    out["accuracy"] = _ratio(stats.get("n_correct", 0), N)
    sp = float(stats.get("corr_sum_p", 0.0))
    sg = float(stats.get("corr_sum_g", 0.0))
    spg = float(stats.get("corr_sum_pg", 0.0))
    spp = float(stats.get("corr_sum_pp", 0.0))
    sgg = float(stats.get("corr_sum_gg", 0.0))
    cov = N * spg - sp * sg
    var_p = N * spp - sp * sp
    var_g = N * sgg - sg * sg
    denom = float(np.sqrt(max(var_p, 0.0) * max(var_g, 0.0)))
    out["correlation"] = float(cov / denom) if denom > EPS else float("nan")
    if N > 0:
        out["pred_mean"] = sp / N
        out["gt_mean"] = sg / N
        out["pred_std"] = float(np.sqrt(max(spp / N - (sp / N) ** 2, 0.0)))
        out["gt_std"] = float(np.sqrt(max(sgg / N - (sg / N) ** 2, 0.0)))
    for k in ("val_min_p", "val_max_p", "val_min_g", "val_max_g"):
        if k in stats:
            out[k] = float(stats[k])

    return out


# =========================================================
# Aggregate additive stats across rows
# =========================================================
def aggregate_stats(rows: list):
    agg = {}

    sum_keys = [
        "N",
        "rmse_sse", "rmse_thr_sse", "rmse_thr_tol_sse",
        "nse_sse", "nse_t_sum", "nse_t_sq_sum",
        "nse_thr_sse", "nse_thr_t_sum", "nse_thr_t_sq_sum",
        "nse_thr_tol_sse", "nse_thr_tol_t_sum", "nse_thr_tol_t_sq_sum",
        "tp", "fp", "fn",
        "tp_tol", "fp_tol", "fn_tol",
        "t_evt_cnt", "p_evt_cnt",
        "n_correct",
        "corr_sum_p", "corr_sum_g", "corr_sum_pg", "corr_sum_pp", "corr_sum_gg",
    ] + [f"cm_{gi}_{pi}" for gi in range(4) for pi in range(4)]

    for band_name in DEPTH_BANDS.keys():
        sum_keys += [
            f"band_{band_name}_n",
            f"band_{band_name}_sse",
            f"band_{band_name}_tp",
            f"band_{band_name}_fp",
            f"band_{band_name}_fn",
        ]

    for k in sum_keys:
        agg[k] = 0.0

    for r in rows:
        st = r["_stats"] if "_stats" in r else r
        for k in sum_keys:
            agg[k] += st.get(k, 0.0)

    # value range: aggregate by min/max (NOT summed)
    def _st(rr):
        return rr["_stats"] if "_stats" in rr else rr
    for key, op in (("val_min_p", min), ("val_min_g", min),
                    ("val_max_p", max), ("val_max_g", max)):
        vals = [float(_st(rr)[key]) for rr in rows if key in _st(rr)]
        agg[key] = (op(vals) if vals else float("nan"))

    int_like = [
        "N",
        "tp", "fp", "fn",
        "tp_tol", "fp_tol", "fn_tol",
        "t_evt_cnt", "p_evt_cnt",
        "n_correct",
    ] + [f"cm_{gi}_{pi}" for gi in range(4) for pi in range(4)]
    for band_name in DEPTH_BANDS.keys():
        int_like += [
            f"band_{band_name}_n",
            f"band_{band_name}_tp",
            f"band_{band_name}_fp",
            f"band_{band_name}_fn",
        ]

    for k in int_like:
        agg[k] = int(round(agg[k]))

    return agg


def global_metrics_from_rows(rows: list):
    agg = aggregate_stats(rows)
    return metrics_from_stats(agg)


def patch_mean_metrics_from_rows(rows: list):
    if not rows:
        return {}

    metric_keys = [
        "rmse", "rmse_thr", "rmse_thr_tol",
        "nse", "nse_thr", "nse_thr_tol",
        "nse_safe", "nse_thr_safe", "nse_thr_tol_safe",
        "precision", "recall", "csi",
        "precision_tol", "recall_tol", "csi_tol",
        "target_prevalence", "pred_prevalence",
        "accuracy", "correlation",
    ]

    for band_name in DEPTH_BANDS.keys():
        metric_keys += [
            f"rmse_{band_name}",
            f"precision_{band_name}",
            f"recall_{band_name}",
            f"support_{band_name}",
        ]

    out = {}
    for k in metric_keys:
        vals = [r.get(k, np.nan) for r in rows]
        out[f"{k}_patch_mean"] = mean_agg(vals)

    return out


# =========================================================
# Output helpers
# =========================================================
METRIC_BETTER = {
    # lower is better
    "rmse_global": "lower",
    "rmse_thr_global": "lower",
    "rmse_thr_tol_global": "lower",
    "rmse_nonflood_global": "lower",
    "rmse_slight_global": "lower",
    "rmse_severe_global": "lower",
    "rmse_extreme_global": "lower",

    "rmse_patch_mean": "lower",
    "rmse_thr_patch_mean": "lower",
    "rmse_thr_tol_patch_mean": "lower",
    "rmse_nonflood_patch_mean": "lower",
    "rmse_slight_patch_mean": "lower",
    "rmse_severe_patch_mean": "lower",
    "rmse_extreme_patch_mean": "lower",

    # higher is better
    "nse_global": "higher",
    "nse_thr_global": "higher",
    "nse_thr_tol_global": "higher",
    "nse_safe_global": "higher",
    "nse_thr_safe_global": "higher",
    "nse_thr_tol_safe_global": "higher",

    "nse_patch_mean": "higher",
    "nse_thr_patch_mean": "higher",
    "nse_thr_tol_patch_mean": "higher",
    "nse_safe_patch_mean": "higher",
    "nse_thr_safe_patch_mean": "higher",
    "nse_thr_tol_safe_patch_mean": "higher",

    "precision_global": "higher",
    "recall_global": "higher",
    "csi_global": "higher",
    "precision_tol_global": "higher",
    "recall_tol_global": "higher",
    "csi_tol_global": "higher",
    "precision_nonflood_global": "higher",
    "precision_slight_global": "higher",
    "precision_severe_global": "higher",
    "precision_extreme_global": "higher",
    "recall_nonflood_global": "higher",
    "recall_slight_global": "higher",
    "recall_severe_global": "higher",
    "recall_extreme_global": "higher",

    "precision_patch_mean": "higher",
    "recall_patch_mean": "higher",
    "csi_patch_mean": "higher",
    "precision_tol_patch_mean": "higher",
    "recall_tol_patch_mean": "higher",
    "csi_tol_patch_mean": "higher",
    "precision_nonflood_patch_mean": "higher",
    "precision_slight_patch_mean": "higher",
    "precision_severe_patch_mean": "higher",
    "precision_extreme_patch_mean": "higher",
    "recall_nonflood_patch_mean": "higher",
    "recall_slight_patch_mean": "higher",
    "recall_severe_patch_mean": "higher",
    "recall_extreme_patch_mean": "higher",

    # neutral
    "target_prevalence_global": "neutral",
    "pred_prevalence_global": "neutral",
    "target_prevalence_patch_mean": "neutral",
    "pred_prevalence_patch_mean": "neutral",
}


def pick_best_worst(rows, metric_key: str, better: str, id_fields: list, extra_fields: list = None):
    extra_fields = extra_fields or []

    cand = []
    for r in rows:
        v = r.get(metric_key, None)
        if v is None or (not np.isfinite(v)):
            continue
        cand.append((float(v), r))
    if not cand:
        return {"best": None, "worst": None}

    if better == "lower":
        best_v, best_r = min(cand, key=lambda x: x[0])
        worst_v, worst_r = max(cand, key=lambda x: x[0])
        best_tag = "min"
        worst_tag = "max"
    elif better == "higher":
        best_v, best_r = max(cand, key=lambda x: x[0])
        worst_v, worst_r = min(cand, key=lambda x: x[0])
        best_tag = "max"
        worst_tag = "min"
    else:
        best_v, best_r = min(cand, key=lambda x: x[0])
        worst_v, worst_r = max(cand, key=lambda x: x[0])
        best_tag = "min"
        worst_tag = "max"

    def pack(v, r, tag):
        out = {k: r.get(k, None) for k in id_fields + extra_fields}
        out["value"] = float(v)
        out["mode"] = tag
        return out

    return {
        "best": pack(best_v, best_r, best_tag),
        "worst": pack(worst_v, worst_r, worst_tag),
    }


# =========================================================
# CSV column ordering (readable layout instead of alphabetical)
# =========================================================
# Identifier / bookkeeping columns come first, then metrics in a logical order:
# overall rmse -> nse -> csi/precision/recall -> prevalence/accuracy/correlation
# -> value distribution -> per depth-band (nonflood/slight/severe/extreme).
# For time/scenario CSVs each base metric's _global/_patch_mean/_mapmean variants
# are kept adjacent so they can be compared at a glance. Any column not listed
# here is appended (alphabetically) at the end, so nothing is ever dropped.
CSV_ID_ORDER = [
    "core", "scenario", "t", "patch_row", "patch_col", "var", "downscale",
    "n_patches", "n_timesteps", "num_valid", "num_valid_sum",
    "pred_path", "fine_path", "mask_path",
]
CSV_METRIC_ORDER = [
    # overall regression
    "rmse", "rmse_thr", "rmse_thr_tol",
    "nse", "nse_safe", "nse_thr", "nse_thr_safe", "nse_thr_tol", "nse_thr_tol_safe",
    # overall classification
    "csi", "precision", "recall",
    "csi_tol", "precision_tol", "recall_tol",
    "target_prevalence", "pred_prevalence",
    "accuracy", "correlation",
    # value distribution
    "pred_mean", "gt_mean", "pred_std", "gt_std",
    "val_min_p", "val_max_p", "val_min_g", "val_max_g",
    # per depth-band: nonflood / slight / severe / extreme
    "precision_nonflood", "recall_nonflood", "rmse_nonflood", "support_nonflood",
    "precision_slight", "recall_slight", "rmse_slight", "support_slight",
    "precision_severe", "recall_severe", "rmse_severe", "support_severe",
    "precision_extreme", "recall_extreme", "rmse_extreme", "support_extreme",
]
CSV_SUFFIXES = ["", "_global", "_patch_mean", "_mapmean"]


def order_csv_fields(all_keys):
    keys = set(all_keys)
    ordered, used = [], set()

    def take(k):
        if k in keys and k not in used:
            ordered.append(k)
            used.add(k)

    for k in CSV_ID_ORDER:
        take(k)
    for base in CSV_METRIC_ORDER:
        for suf in CSV_SUFFIXES:
            take(base + suf)
    # never drop anything: leftover columns appended alphabetically
    for k in sorted(keys - used):
        ordered.append(k)
    return ordered


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    clean_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "_stats"}
        clean_rows.append(rr)

    fieldnames = order_csv_fields(set().union(*[r.keys() for r in clean_rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in clean_rows:
            w.writerow(r)
    print("[OK] saved:", path)


def mapmean_from_time_rows(time_rows):
    """Mean over per-(scenario,t) assembled-map GLOBAL metrics (each timestep-map
    weighted equally). Reads the '*_global' columns of the time rows and returns
    '*_mapmean'. I.e. "assemble each timestep's full map, compute its global
    metrics, then average over maps" -- the scenario/dataset aggregation."""
    if not time_rows:
        return {}
    gkeys = sorted({k for r in time_rows for k in r.keys() if k.endswith("_global")})
    out = {}
    for k in gkeys:
        vals = [float(r[k]) for r in time_rows
                if (k in r and r[k] is not None and np.isfinite(r[k]))]
        base = k[:-len("_global")]
        out[f"{base}_mapmean"] = (float(np.mean(vals)) if vals else float("nan"))
    return out


# =========================================================
# Single-query helpers (one patch, or one assembled map)
# =========================================================
def upsample_coarse_to_fine(coarse_hw, out_hw):
    """Bicubic-upsample a coarse (physical) map to (H, W). Lazy torch import; the
    coarse-upsample baseline is exactly what He et al. call the 'interpolated
    coarse-grid' result. For depth, negative overshoots are clamped to 0
    (depth >= 0); for velocity (USE_ABS) the signed field is kept as-is."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(coarse_hw, dtype=np.float32))[None, None]
    up = F.interpolate(t, size=(int(out_hw[0]), int(out_hw[1])),
                       mode="bicubic", align_corners=False)
    out = up[0, 0].numpy().astype(np.float64)
    return out if USE_ABS else np.clip(out, 0.0, None)


def _load_patch_pred_gt_mask(core, idx, vis_root, pred_source="model"):
    if core not in idx:
        raise RuntimeError(f"core not in index.csv: {core}")
    row = idx[core]
    gt = load_hw_npy(row["fine_path"], "fine")
    mask = load_hw_npy(row["mask_fine_path"], "mask")
    if pred_source == "coarse_upsample":
        pred_path = row["coarse_path"]
        pred = upsample_coarse_to_fine(load_hw_npy(pred_path, "coarse"), gt.shape)
    else:
        folder = os.path.join(vis_root, core + "_coarse")
        pred_path = pick_pred_file_in_folder(folder, core)
        pred = load_hw_npy(pred_path, "pred")
    if pred.shape != gt.shape:
        raise RuntimeError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
    return pred, gt, mask, row, pred_path


def run_query(query, idx, vis_root, var="h", pred_source="model"):
    """`r,c,t,scenario` -> one patch; `t,scenario` -> one assembled-map global."""
    parts = [p.strip() for p in query.split(",")]
    if len(parts) == 4:
        r, c, t, scenario = parts
        core = f"{var}_{scenario}_{t}_r{int(r):03d}_c{int(c):03d}_s8"
        pred, gt, mask, row, pred_path = _load_patch_pred_gt_mask(core, idx, vis_root, pred_source)
        stats = compute_patch_stats_depth(pred, gt, mask)
        met = metrics_from_stats(stats)
        print(f"[QUERY patch] {core}  (N={stats['N']}, pred={pred_path})")
    elif len(parts) == 2:
        t, scenario = parts
        rows = []
        for core, row in idx.items():
            if row.get("var") != var or row.get("scenario") != scenario or row.get("t") != t:
                continue
            try:
                pred, gt, mask, _, _ = _load_patch_pred_gt_mask(core, idx, vis_root, pred_source)
            except Exception:
                continue
            rows.append({"_stats": compute_patch_stats_depth(pred, gt, mask)})
        if not rows:
            raise RuntimeError(f"no patches found for scenario={scenario} t={t}")
        met = global_metrics_from_rows(rows)
        print(f"[QUERY map-global] scenario={scenario} t={t}  (n_patches={len(rows)})")
    else:
        raise SystemExit("[ERROR] --query must be 'r,c,t,scenario' (one patch) or 't,scenario' (one map).")
    for k in sorted(met.keys()):
        print(f"  {k:30s} {met[k]}")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Summarize flood metrics using global aggregation.")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", default="",
                    help="dir with results/<eval>/visualization/<core>_coarse/ model predictions. "
                         "REQUIRED for --pred-source model. For coarse_upsample it just selects the "
                         "patch set (pass it to match a model run e.g. the val subset; omit to use all "
                         "filtered_out==0 index patches, e.g. a complete test set).")

    ap.add_argument("--out-json", default="", help="dataset summary json (required for the full sweep)")
    ap.add_argument("--query", default="",
                    help="single query: 'r,c,t,scenario' (one patch) or 't,scenario' (one assembled map); prints metrics and exits")
    ap.add_argument("--pred-source", default="model", choices=["model", "coarse_upsample"],
                    help="'model' = saved net prediction .npy; 'coarse_upsample' = bicubic-upsampled coarse h "
                         "(He et al. interpolation baseline), evaluated on the SAME patches/folders")
    ap.add_argument("--out-csv-patch", default="", help="patch metrics csv")
    ap.add_argument("--out-csv-time", default="", help="(scenario,t) metrics csv")
    ap.add_argument("--out-csv-scenario", default="", help="scenario metrics csv")

    ap.add_argument("--var", default="h", help="filter variable: h (default), u, or v")
    ap.add_argument("--abs", action="store_true",
                    help="velocity mode (u/v): classify wet/dry, bands, accuracy and confusion by "
                         "|value| (so 0.1/0.5/1.0 are |v| thresholds); RMSE/NSE/correlation stay on "
                         "the signed velocity. Leave OFF for water depth (h).")
    ap.add_argument("--limit", type=int, default=0, help="limit number of patches for quick test")
    ap.add_argument("--empty-policy", choices=["nan", "zero"], default="nan",
                    help="how flood-free / near-flat patches & maps are handled. "
                         "'nan' (default): precision/recall/csi with a zero denominator and "
                         "NSE on a zero-variance (no-flood) map are set to NaN and dropped from "
                         "every *_patch_mean / *_mapmean average (matches training empty_as_nan:true). "
                         "'zero': legacy (zero ratios = 0; degenerate NSE rescued to 1.0 / clamped to -5).")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    global EMPTY_POLICY, USE_ABS
    EMPTY_POLICY = args.empty_policy
    USE_ABS = bool(args.abs)

    idx = build_index_map(args.index_csv)

    # single-query mode: print one patch / one assembled-map metrics and exit
    if args.query:
        run_query(args.query, idx, args.vis_root, var=args.var, pred_source=args.pred_source)
        return

    if not args.out_json:
        raise SystemExit("[ERROR] --out-json is required for the full sweep (only omit it with --query).")

    # Build the list of cores to evaluate:
    #  - with --vis-root: the patches the model was evaluated on (the *_coarse
    #    folders). Required for --pred-source model; for coarse_upsample it pins
    #    the baseline to the SAME patch set (e.g. the val 20% subset).
    #  - without --vis-root (coarse_upsample only): all filtered_out==0 index
    #    patches (e.g. a complete test set).
    if args.vis_root:
        all_coarse_dirs = sorted(glob.glob(os.path.join(args.vis_root, "*_coarse")))
        work_cores = []
        for f in all_coarse_dirs:
            core = parse_folder_name(os.path.basename(f.rstrip("/")))
            if core is not None:
                work_cores.append(core)
        if not work_cores:
            if all_coarse_dirs:
                raise RuntimeError(
                    f"Found {len(all_coarse_dirs)} '*_coarse' folder(s) under {args.vis_root} but NONE "
                    f"parsed as a valid core (e.g. '{os.path.basename(all_coarse_dirs[0])}'). "
                    f"Likely the scenario token in the folder name is not recognised by CORE_RE.")
            raise RuntimeError(f"No *_coarse folders found under: {args.vis_root}")
    else:
        if args.pred_source != "coarse_upsample":
            raise SystemExit("[ERROR] --vis-root is required for --pred-source model.")
        work_cores = [core for core, row in idx.items()
                      if str(row.get("filtered_out", "0")) == "0"]

    patch_rows = []
    skipped = 0
    done = 0

    for core in work_cores:
        try:
            meta = parse_core_key(core)
        except Exception:
            continue
        if args.var and meta["var"] != args.var:
            continue

        if core not in idx:
            skipped += 1
            if args.debug:
                print(f"[skip] core not in index.csv: {core}")
            continue

        try:
            pred, gt, mask, row, pred_path = _load_patch_pred_gt_mask(
                core, idx, args.vis_root, args.pred_source)

            stats = compute_patch_stats_depth(pred, gt, mask)
            patch_metrics = metrics_from_stats(stats)

            pr = {
                "core": core,
                "var": meta["var"],
                "scenario": meta["scenario"],
                "t": meta["t"],
                "patch_row": meta["patch_row"],
                "patch_col": meta["patch_col"],
                "downscale": meta["downscale"],
                "pred_path": pred_path,
                "fine_path": row["fine_path"],
                "mask_path": row["mask_fine_path"],
                "num_valid": int(stats["N"]),
                "_stats": stats,
            }
            pr.update(patch_metrics)
            patch_rows.append(pr)

            done += 1
            if args.limit and done >= args.limit:
                break

            if args.debug and (done % 200 == 0):
                print(f"[info] processed {done} patches ...")

        except Exception as e:
            skipped += 1
            if args.debug:
                print(f"[warn] skip {core}: {e}")

    if not patch_rows:
        raise RuntimeError("No valid patches processed.")

    # -------------------------------------------------
    # Time level: group by (scenario, t)
    # -------------------------------------------------
    st_groups = defaultdict(list)
    for r in patch_rows:
        st_groups[(r["scenario"], r["t"])].append(r)

    time_rows = []
    for (scenario, t), rows in sorted(st_groups.items()):
        out = {
            "scenario": scenario,
            "t": t,
            "n_patches": len(rows),
            "num_valid_sum": int(sum(r["num_valid"] for r in rows)),
        }

        g = global_metrics_from_rows(rows)
        pm = patch_mean_metrics_from_rows(rows)

        out.update({f"{k}_global": v for k, v in g.items()})
        out.update(pm)
        time_rows.append(out)

    # -------------------------------------------------
    # Scenario level
    # -------------------------------------------------
    s_groups = defaultdict(list)
    for r in patch_rows:
        s_groups[r["scenario"]].append(r)

    time_by_scenario = defaultdict(list)
    for tr in time_rows:
        time_by_scenario[tr["scenario"]].append(tr)

    scenario_rows = []
    for scenario, rows in sorted(s_groups.items()):
        out = {
            "scenario": scenario,
            "n_patches": len(rows),
            "n_timesteps": len(time_by_scenario.get(scenario, [])),
            "num_valid_sum": int(sum(r["num_valid"] for r in rows)),
        }

        g = global_metrics_from_rows(rows)
        pm = patch_mean_metrics_from_rows(rows)

        out.update({f"{k}_global": v for k, v in g.items()})
        out.update(pm)
        out.update(mapmean_from_time_rows(time_by_scenario.get(scenario, [])))
        scenario_rows.append(out)

    # -------------------------------------------------
    # Dataset level
    # -------------------------------------------------
    dataset_metrics_global = global_metrics_from_rows(patch_rows)
    dataset_metrics_patch_mean = patch_mean_metrics_from_rows(patch_rows)
    dataset_metrics_mapmean = mapmean_from_time_rows(time_rows)

    # -------------------------------------------------
    # Extremes
    # -------------------------------------------------
    compare_keys = list(time_rows[0].keys()) if time_rows else []
    compare_keys = [
        k for k in compare_keys
        if k not in ("scenario", "t", "n_patches", "num_valid_sum")
    ]

    extremes = {
        "patch": {},
        "time": {},
        "scenario": {},
    }

    # patch level extremes: use original patch-level keys
    patch_metric_keys = [k for k in patch_rows[0].keys() if k not in {
        "core", "var", "scenario", "t", "patch_row", "patch_col", "downscale",
        "pred_path", "fine_path", "mask_path", "num_valid", "_stats"
    }]

    for k in patch_metric_keys:
        better = "neutral"
        if k.startswith("rmse"):
            better = "lower"
        elif k.startswith("nse") or k.startswith("precision") or k.startswith("recall") or k.startswith("csi"):
            better = "higher"
        elif k.startswith("target_prevalence") or k.startswith("pred_prevalence") or k.startswith("support"):
            better = "neutral"

        extremes["patch"][k] = pick_best_worst(
            patch_rows,
            metric_key=k,
            better=better,
            id_fields=["core", "scenario", "t", "patch_row", "patch_col", "var", "downscale"],
            extra_fields=["num_valid", "pred_path", "fine_path", "mask_path"],
        )

    # time/scenario extremes
    for k in compare_keys:
        better = METRIC_BETTER.get(k, "neutral")

        extremes["time"][k] = pick_best_worst(
            time_rows,
            metric_key=k,
            better=better,
            id_fields=["scenario", "t"],
            extra_fields=["n_patches", "num_valid_sum"],
        )

        extremes["scenario"][k] = pick_best_worst(
            scenario_rows,
            metric_key=k,
            better=better,
            id_fields=["scenario"],
            extra_fields=["n_patches", "num_valid_sum"],
        )

    ds = {
        "pred_source": args.pred_source,
        "var": args.var,
        "use_abs": USE_ABS,
        "n_patches": len(patch_rows),
        "n_timesteps": len(time_rows),
        "num_valid_sum": int(sum(r["num_valid"] for r in patch_rows)),
        "empty_policy": EMPTY_POLICY,
        "metrics_dataset_global": dataset_metrics_global,
        "metrics_dataset_mapmean": dataset_metrics_mapmean,
        "metrics_dataset_patch_mean": dataset_metrics_patch_mean,
        "extremes": extremes,
        "notes": {
            "aggregation": {
                "patch": "metrics recovered from one patch's additive stats",
                "time_global": "global metrics over all valid pixels in each (scenario, t) group (= assemble that timestep's full map, then compute)",
                "scenario_global": "global metrics over all valid pixels in each scenario group (pooled)",
                "dataset_global": "global metrics over all valid pixels in the full dataset (pooled = He et al. Table-1 cell-pooled)",
                "mapmean": "mean over per-(scenario,t) assembled-map *_global metrics, each timestep-map weighted equally (the 'per-map then average' aggregation; at scenario and dataset levels)",
                "patch_mean": "simple mean over patch-level metrics for comparison",
                "correlation": "Pearson rho between predicted and simulated water depth over AOI cells (matches He et al. correlation coefficient)",
            },
            "empty_policy": (
                "nan: flood-free / near-flat patches & maps give undefined "
                "precision/recall/csi and NSE, set to NaN and dropped from "
                "*_patch_mean / *_mapmean (matches training empty_as_nan:true). "
                "zero: legacy (0 ratios, NSE rescued to 1.0 / clamped to -5)."
            ),
            "use_abs": (
                "velocity mode: wet/dry, bands, accuracy and confusion use |value| "
                "(0.1/0.5/1.0 are |v| thresholds); RMSE/NSE/correlation stay signed. "
                "OFF = water depth (unchanged)."
            ),
            "depth_event_threshold": DEPTH_EVENT_THRESHOLD,
            "depth_abs_tol": DEPTH_ABS_TOL,
            "depth_bands": DEPTH_BANDS,
            "nse_safe_params": {
                "min_var": NSE_SAFE_MIN_VAR,
                "abs_tol_per_px": NSE_SAFE_ABS_TOL_PER_PX,
                "lower_bound": NSE_SAFE_LOWER_BOUND,
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)
    print("[OK] saved:", args.out_json)

    # optional csv
    write_csv(args.out_csv_patch, patch_rows)
    write_csv(args.out_csv_time, time_rows)
    write_csv(args.out_csv_scenario, scenario_rows)

    print(f"[OK] done. patches={len(patch_rows)} skipped={skipped}")


if __name__ == "__main__":
    main()