#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

# Make sure flood_metrics is imported so all @METRIC_REGISTRY.register() are executed.
# (Even if __init__.py doesn't export every function, registry still works as long as module imported.)
import basicsr.metrics.flood_metrics  # noqa: F401
from basicsr.metrics import calculate_metric


# ---------------------------
# Key parsing (same naming as eval_plot.py)
# ---------------------------
CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>\d+y_\d+h_\d+c)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
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
    d["var"] = d["var"]
    d["scenario"] = d["scenario"]
    d["t"] = d["t"]
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


# ---------------------------
# Index CSV mapping
# ---------------------------
def build_index_map(index_csv: str):
    """
    Build mapping core -> row(dict) from dataset/index.csv.
    Required columns: var, scenario, t, patch_row, patch_col, downscale,
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


# ---------------------------
# Array shape helpers
# ---------------------------
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


def mask_num_valid(mask_hw: np.ndarray) -> int:
    # Treat mask > 0.5 as valid AOI pixels
    m = np.asarray(mask_hw)
    return int((m > 0.5).sum())


# ---------------------------
# Aggregation helpers
# ---------------------------
def mean_agg(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(v.mean())


# ---------------------------
# Best/Worst selection helpers
# ---------------------------
METRIC_BETTER = {
    # lower is better
    "rmse": "lower",
    "rmse_thr": "lower",
    "rmse_thr_tol": "lower",

    # higher is better
    "nse": "higher",
    "nse_thr": "higher",
    "nse_thr_tol": "higher",
    "nse_safe": "higher",
    "nse_thr_safe": "higher",
    "nse_thr_tol_safe": "higher",

    "csi": "higher",
    "csi_tol": "higher",
    "precision": "higher",
    "precision_tol": "higher",
    "recall": "higher",
    "recall_tol": "higher",

    # prevalence is "neutral" (report min/max)
    "target_prevalence": "neutral",
    "pred_prevalence": "neutral",
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
        # neutral: report min/max (do not imply which is "better")
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


# ---------------------------
# Metrics config
# ---------------------------
def get_metrics_config():
    # NOTE: keys here become output keys in csv/json
    cfg = {
        "rmse": {"type": "cal_rmse_pt", "reduction": "mean"},
        "rmse_thr": {"type": "cal_rmse_depth_threshold_pt"},
        "rmse_thr_tol": {"type": "cal_rmse_depth_threshold_tolerant_pt"},

        "nse": {"type": "cal_nse_pt", "reduction": "mean"},
        "nse_thr": {"type": "cal_nse_depth_threshold_pt", "reduction": "mean"},
        "nse_thr_tol": {"type": "cal_nse_depth_threshold_tolerant_pt", "reduction": "mean"},

        "nse_safe": {
            "type": "cal_nse_pt_safe",
            "reduction": "mean",
            "min_var": 1.0e-6,
            "abs_tol_per_px": 1.0e-4,
            "lower_bound": -5.0,
        },
        "nse_thr_safe": {"type": "cal_nse_depth_threshold_pt_safe"},
        "nse_thr_tol_safe": {"type": "cal_nse_depth_threshold_tolerant_pt_safe"},

        "csi": {"type": "cal_csi_pt", "threshold": 0.05, "reduction": "mean"},
        "csi_tol": {"type": "cal_csi_depth_tolerant_pt"},

        "precision": {"type": "cal_precision_pt", "threshold": 0.05, "reduction": "mean"},
        "precision_tol": {"type": "cal_precision_depth_tolerant_pt"},

        "recall": {"type": "cal_recall_pt", "threshold": 0.05, "reduction": "mean"},
        "recall_tol": {"type": "cal_recall_depth_tolerant_pt"},

        "target_prevalence": {"type": "cal_prev_t_pt", "threshold": 0.05, "reduction": "mean"},
        "pred_prevalence": {"type": "cal_prev_p_pt", "threshold": 0.05, "reduction": "mean"},
    }
    order = list(cfg.keys())
    return cfg, order


# ---------------------------
# Metric compute per patch (your original behavior)
# ---------------------------
@torch.no_grad()
def compute_metrics_one_patch(pred_hw: np.ndarray, gt_hw: np.ndarray, mask_hw: np.ndarray, metrics_cfg: dict, device="cpu"):
    pred_t = torch.from_numpy(pred_hw.astype(np.float32)).to(device)
    gt_t = torch.from_numpy(gt_hw.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask_hw.astype(np.float32)).to(device)

    out = {}
    for out_key, opt in metrics_cfg.items():
        data = {"pred": pred_t, "target": gt_t, "mask": mask_t}
        out[out_key] = float(calculate_metric(data, opt))
    return out


# ---------------------------
# Global-stat accumulators (to compute "global" metrics by group)
# ---------------------------
EPS = 1.0e-12
THR = 0.05
ABS_TOL = 0.01


def _safe_params(metrics_cfg: dict):
    # Use nse_safe config if present; otherwise fall back to flood_metrics defaults
    opt = metrics_cfg.get("nse_safe", {})
    return {
        "min_var": float(opt.get("min_var", 1.0e-6)),
        "abs_tol_per_px": float(opt.get("abs_tol_per_px", 1.0e-4)),
        "lower_bound": float(opt.get("lower_bound", -5.0)),
    }


def compute_patch_global_stats(pred_hw: np.ndarray, gt_hw: np.ndarray, mask_hw: np.ndarray, thr=THR, abs_tol=ABS_TOL):
    """
    Return additive statistics per patch to support "global" aggregation:
      - RMSE: need SSE and N
      - NSE: need SSE, sum(t), sum(t^2), N   (t is target masked)
      - CSI/Precision/Recall: need TP/FP/FN
      - Prevalence: need event count and N
    All computations use mask>0.5 as valid pixels.
    """
    pred = np.asarray(pred_hw, dtype=np.float64)
    gt = np.asarray(gt_hw, dtype=np.float64)
    mask = np.asarray(mask_hw, dtype=np.float64)

    m = mask > 0.5
    n = int(m.sum())
    if n <= 0:
        raise RuntimeError("num_valid==0 (mask empty)")

    diff = pred - gt
    diff2 = diff * diff

    # Base SSE over valid pixels
    sse = float(diff2[m].sum())

    # Thresholded pred/target (match your flood_metrics.py logic)
    pred_eff = np.where((pred >= thr) & m, pred, 0.0)
    gt_eff = np.where((gt >= thr) & m, gt, 0.0)

    diff_thr = pred_eff - gt_eff
    sse_thr = float((diff_thr[m] ** 2).sum())

    abs_err_thr = np.abs(diff_thr)
    eff_mask_tol = m & (abs_err_thr >= abs_tol)
    sse_thr_tol = float((diff_thr[eff_mask_tol] ** 2).sum())

    # NSE sums (base)
    t_vals = gt[m]
    t_sum = float(t_vals.sum())
    t_sq_sum = float((t_vals * t_vals).sum())

    # NSE sums (thresholded)
    t_eff_vals = gt_eff[m]
    t_eff_sum = float(t_eff_vals.sum())
    t_eff_sq_sum = float((t_eff_vals * t_eff_vals).sum())

    # NSE tolerant numerator uses only large errors (match your implementation)
    large_err_mask = m & (abs_err_thr >= abs_tol)
    sse_nse_thr_tol = float((diff_thr[large_err_mask] ** 2).sum())

    # Event counts for classification metrics (match your implementations)
    p_evt = (pred >= thr) & m
    t_evt = (gt >= thr) & m
    tp = int((p_evt & t_evt).sum())
    fp = int((p_evt & (~t_evt)).sum())
    fn = int(((~p_evt) & t_evt).sum())

    # tolerant event: borderline pixels (abs_err<=abs_tol) are forced to equal target event
    borderline = (np.abs(pred - gt) <= abs_tol) & m
    p_evt_tol = np.where(borderline, t_evt, p_evt)
    tp_tol = int((p_evt_tol & t_evt).sum())
    fp_tol = int((p_evt_tol & (~t_evt)).sum())
    fn_tol = int(((~p_evt_tol) & t_evt).sum())

    # prevalence
    t_evt_cnt = int(t_evt.sum())
    p_evt_cnt = int(p_evt.sum())

    return {
        "N": n,

        # RMSE stats
        "rmse_SSE": sse,
        "rmse_thr_SSE": sse_thr,
        "rmse_thr_tol_SSE": sse_thr_tol,  # denom is still N (as in your rmse_thr_tol)

        # NSE stats
        "nse_SSE": sse,
        "nse_T_sum": t_sum,
        "nse_T_sq_sum": t_sq_sum,

        "nse_thr_SSE": sse_thr,
        "nse_thr_T_sum": t_eff_sum,
        "nse_thr_T_sq_sum": t_eff_sq_sum,

        "nse_thr_tol_SSE": sse_nse_thr_tol,
        "nse_thr_tol_T_sum": t_eff_sum,
        "nse_thr_tol_T_sq_sum": t_eff_sq_sum,

        # classification counts
        "tp": tp, "fp": fp, "fn": fn,
        "tp_tol": tp_tol, "fp_tol": fp_tol, "fn_tol": fn_tol,

        # prevalence
        "t_evt_cnt": t_evt_cnt,
        "p_evt_cnt": p_evt_cnt,
    }


def _rmse_from_sse(sse_sum: float, n_sum: int):
    if n_sum <= 0:
        return float("nan")
    return float(np.sqrt(max(sse_sum, 0.0) / float(max(n_sum, 1))))


def _nse_from_sums(sse_sum: float, t_sum: float, t_sq_sum: float, n_sum: int, eps=EPS):
    if n_sum <= 0:
        return float("nan")
    mean = t_sum / float(n_sum)
    # var = sum((t-mean)^2) = sum(t^2) - 2*mean*sum(t) + mean^2*n
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)
    var = max(var, eps)
    return float(1.0 - (sse_sum / var))


def _nse_safe_from_sums(sse_sum: float, t_sum: float, t_sq_sum: float, n_sum: int,
                       min_var: float, abs_tol_per_px: float, lower_bound: float, eps=EPS):
    if n_sum <= 0:
        return float("nan")
    mean = t_sum / float(n_sum)
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)

    if var >= min_var:
        var = max(var, eps)
        return float(1.0 - (sse_sum / var))

    # tiny variance: follow your safe logic (group-level)
    tiny_tol = (abs_tol_per_px ** 2) * float(n_sum)
    if sse_sum <= tiny_tol:
        return 1.0

    val = 1.0 - (sse_sum / float(min_var))
    return float(max(val, lower_bound))


def _ratio(num: float, den: float, eps=EPS):
    return float(num / (den + eps))


def compute_globals_from_rows(rows: list, metrics_cfg: dict):
    """
    Compute "global" metrics for a group of patches by summing additive stats.
    Returns dict: {metric_key: value}
    """
    sp = _safe_params(metrics_cfg)
    min_var = sp["min_var"]
    abs_tol_per_px = sp["abs_tol_per_px"]
    lower_bound = sp["lower_bound"]

    N = 0

    # RMSE / NSE sums
    rmse_SSE = 0.0
    rmse_thr_SSE = 0.0
    rmse_thr_tol_SSE = 0.0

    nse_SSE = 0.0
    nse_T_sum = 0.0
    nse_T_sq_sum = 0.0

    nse_thr_SSE = 0.0
    nse_thr_T_sum = 0.0
    nse_thr_T_sq_sum = 0.0

    nse_thr_tol_SSE = 0.0
    nse_thr_tol_T_sum = 0.0
    nse_thr_tol_T_sq_sum = 0.0

    # classification counts
    tp = fp = fn = 0
    tp_tol = fp_tol = fn_tol = 0

    # prevalence
    t_evt_cnt = 0
    p_evt_cnt = 0

    for r in rows:
        st = r.get("_global_stats", None)
        if not st:
            continue

        n = int(st.get("N", 0))
        if n <= 0:
            continue
        N += n

        rmse_SSE += float(st.get("rmse_SSE", 0.0))
        rmse_thr_SSE += float(st.get("rmse_thr_SSE", 0.0))
        rmse_thr_tol_SSE += float(st.get("rmse_thr_tol_SSE", 0.0))

        nse_SSE += float(st.get("nse_SSE", 0.0))
        nse_T_sum += float(st.get("nse_T_sum", 0.0))
        nse_T_sq_sum += float(st.get("nse_T_sq_sum", 0.0))

        nse_thr_SSE += float(st.get("nse_thr_SSE", 0.0))
        nse_thr_T_sum += float(st.get("nse_thr_T_sum", 0.0))
        nse_thr_T_sq_sum += float(st.get("nse_thr_T_sq_sum", 0.0))

        nse_thr_tol_SSE += float(st.get("nse_thr_tol_SSE", 0.0))
        nse_thr_tol_T_sum += float(st.get("nse_thr_tol_T_sum", 0.0))
        nse_thr_tol_T_sq_sum += float(st.get("nse_thr_tol_T_sq_sum", 0.0))

        tp += int(st.get("tp", 0))
        fp += int(st.get("fp", 0))
        fn += int(st.get("fn", 0))

        tp_tol += int(st.get("tp_tol", 0))
        fp_tol += int(st.get("fp_tol", 0))
        fn_tol += int(st.get("fn_tol", 0))

        t_evt_cnt += int(st.get("t_evt_cnt", 0))
        p_evt_cnt += int(st.get("p_evt_cnt", 0))

    out = {}

    # rmse family
    out["rmse"] = _rmse_from_sse(rmse_SSE, N)
    out["rmse_thr"] = _rmse_from_sse(rmse_thr_SSE, N)
    out["rmse_thr_tol"] = _rmse_from_sse(rmse_thr_tol_SSE, N)

    # nse family
    out["nse"] = _nse_from_sums(nse_SSE, nse_T_sum, nse_T_sq_sum, N)
    out["nse_thr"] = _nse_from_sums(nse_thr_SSE, nse_thr_T_sum, nse_thr_T_sq_sum, N)
    out["nse_thr_tol"] = _nse_from_sums(nse_thr_tol_SSE, nse_thr_tol_T_sum, nse_thr_tol_T_sq_sum, N)

    # safe variants computed with safe post-processing at group level
    out["nse_safe"] = _nse_safe_from_sums(nse_SSE, nse_T_sum, nse_T_sq_sum, N, min_var, abs_tol_per_px, lower_bound)
    out["nse_thr_safe"] = _nse_safe_from_sums(nse_thr_SSE, nse_thr_T_sum, nse_thr_T_sq_sum, N, min_var, abs_tol_per_px, lower_bound)
    out["nse_thr_tol_safe"] = _nse_safe_from_sums(nse_thr_tol_SSE, nse_thr_tol_T_sum, nse_thr_tol_T_sq_sum, N, min_var, abs_tol_per_px, lower_bound)

    # csi / precision / recall
    out["csi"] = _ratio(tp, tp + fp + fn)
    out["precision"] = _ratio(tp, tp + fp)
    out["recall"] = _ratio(tp, tp + fn)

    out["csi_tol"] = _ratio(tp_tol, tp_tol + fp_tol + fn_tol)
    out["precision_tol"] = _ratio(tp_tol, tp_tol + fp_tol)
    out["recall_tol"] = _ratio(tp_tol, tp_tol + fn_tol)

    # prevalence
    out["target_prevalence"] = _ratio(t_evt_cnt, N)
    out["pred_prevalence"] = _ratio(p_evt_cnt, N)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)

    ap.add_argument("--out-json", required=True, help="dataset_metrics.json")
    ap.add_argument("--out-csv-patch", default="", help="patch_metrics.csv")
    ap.add_argument("--out-csv-t", default="", help="t_metrics.csv (scenario+t)")
    ap.add_argument("--out-csv-scenario", default="", help="scenario_metrics.csv")

    ap.add_argument("--var", default="h", help="filter var (e.g., h). empty means no filter")
    ap.add_argument("--limit", type=int, default=0, help="limit number of patches for quick test")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("You set --device cuda but CUDA is not available.")

    device = args.device

    idx = build_index_map(args.index_csv)
    metrics_cfg, metric_names = get_metrics_config()

    folders = sorted(glob.glob(os.path.join(args.vis_root, "*_coarse")))
    if not folders:
        raise RuntimeError(f"No *_coarse folders found under: {args.vis_root}")

    patch_rows = []
    skipped = 0
    done = 0

    for f in folders:
        base = os.path.basename(f.rstrip("/"))
        core = parse_folder_name(base)
        if core is None:
            continue

        meta = parse_core_key(core)
        if args.var and meta["var"] != args.var:
            continue

        if core not in idx:
            skipped += 1
            if args.debug:
                print(f"[skip] core not in index.csv: {core}")
            continue

        try:
            pred_path = pick_pred_file_in_folder(f, core)

            row = idx[core]
            fine_path = row.get("fine_path", "")
            mask_path = row.get("mask_fine_path", "")

            gt = load_hw_npy(fine_path, "fine")
            mask = load_hw_npy(mask_path, "mask")
            pred = load_hw_npy(pred_path, "pred")

            if pred.shape != gt.shape:
                raise RuntimeError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")

            num_valid = mask_num_valid(mask)
            if num_valid <= 0:
                raise RuntimeError("num_valid==0 (mask empty)")

            # patch metrics (same as before)
            met = compute_metrics_one_patch(pred, gt, mask, metrics_cfg, device=device)

            # additive stats for global metrics
            gstats = compute_patch_global_stats(pred, gt, mask, thr=THR, abs_tol=ABS_TOL)

            pr = {
                "core": core,
                "var": meta["var"],
                "scenario": meta["scenario"],
                "t": meta["t"],
                "patch_row": meta["patch_row"],
                "patch_col": meta["patch_col"],
                "downscale": meta["downscale"],

                "pred_path": pred_path,
                "fine_path": fine_path,
                "mask_path": mask_path,

                "num_valid": num_valid,
                "_global_stats": gstats,
            }
            pr.update(met)
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
        raise RuntimeError("No valid patches processed. Check vis-root, index.csv mapping, or --var filter.")

    # ---------------------------
    # Aggregate: by (scenario, t)
    # ---------------------------
    st_groups = defaultdict(list)
    for r in patch_rows:
        st_groups[(r["scenario"], r["t"])].append(r)

    st_rows = []
    for (scenario, t), rows in sorted(st_groups.items()):
        out = {
            "scenario": scenario,
            "t": t,
            "n_patches": len(rows),
            "num_valid_sum": int(sum(rr.get("num_valid", 0) for rr in rows)),
        }

        # mean over patches (same as your previous *_mean)
        for k in metric_names:
            vals = [rr.get(k, np.nan) for rr in rows]
            out[f"{k}_mean"] = mean_agg(vals)

        # global over pixels (replace previous *_wmean)
        g = compute_globals_from_rows(rows, metrics_cfg)
        for k in metric_names:
            out[f"{k}_global"] = float(g.get(k, float("nan")))

        st_rows.append(out)

    # ---------------------------
    # Aggregate: by scenario
    # ---------------------------
    s_groups = defaultdict(list)
    for r in patch_rows:
        s_groups[r["scenario"]].append(r)

    s_rows = []
    for scenario, rows in sorted(s_groups.items()):
        out = {
            "scenario": scenario,
            "n_patches": len(rows),
            "num_valid_sum": int(sum(rr.get("num_valid", 0) for rr in rows)),
        }

        for k in metric_names:
            vals = [rr.get(k, np.nan) for rr in rows]
            out[f"{k}_mean"] = mean_agg(vals)

        g = compute_globals_from_rows(rows, metrics_cfg)
        for k in metric_names:
            out[f"{k}_global"] = float(g.get(k, float("nan")))

        s_rows.append(out)

    # ---------------------------
    # Aggregate: dataset
    # ---------------------------
    ds_mean = {}
    for k in metric_names:
        vals = [r.get(k, np.nan) for r in patch_rows]
        ds_mean[k] = mean_agg(vals)

    ds_global = compute_globals_from_rows(patch_rows, metrics_cfg)
    ds_num_valid_sum = int(sum(r.get("num_valid", 0) for r in patch_rows))

    # ---------------------------
    # Extremes (best/worst): patch / t / scenario
    # ---------------------------
    extremes = {
        "patch": {},
        "t_global": {},
        "t_mean": {},
        "scenario_global": {},
        "scenario_mean": {},
    }

    # 1) patch-level (raw metric key)
    for k in metric_names:
        better = METRIC_BETTER.get(k, "neutral")
        extremes["patch"][k] = pick_best_worst(
            patch_rows,
            metric_key=k,
            better=better,
            id_fields=["core", "scenario", "t", "patch_row", "patch_col", "var", "downscale"],
            extra_fields=["num_valid", "pred_path", "fine_path", "mask_path"],
        )

    # 2) (scenario, t) aggregated
    for k in metric_names:
        better = METRIC_BETTER.get(k, "neutral")
        extremes["t_global"][k] = pick_best_worst(
            st_rows,
            metric_key=f"{k}_global",
            better=better,
            id_fields=["scenario", "t"],
            extra_fields=["n_patches", "num_valid_sum"],
        )
        extremes["t_mean"][k] = pick_best_worst(
            st_rows,
            metric_key=f"{k}_mean",
            better=better,
            id_fields=["scenario", "t"],
            extra_fields=["n_patches", "num_valid_sum"],
        )

    # 3) scenario aggregated
    for k in metric_names:
        better = METRIC_BETTER.get(k, "neutral")
        extremes["scenario_global"][k] = pick_best_worst(
            s_rows,
            metric_key=f"{k}_global",
            better=better,
            id_fields=["scenario"],
            extra_fields=["n_patches", "num_valid_sum"],
        )
        extremes["scenario_mean"][k] = pick_best_worst(
            s_rows,
            metric_key=f"{k}_mean",
            better=better,
            id_fields=["scenario"],
            extra_fields=["n_patches", "num_valid_sum"],
        )

    ds = {
        "n_patches": len(patch_rows),
        "num_valid_sum": ds_num_valid_sum,
        "metrics_mean": ds_mean,
        "metrics_global": {k: float(ds_global.get(k, float("nan"))) for k in metric_names},
        "extremes": extremes,
        "notes": {
            "aggregation": {
                "mean": "simple mean over patches (each patch equal weight)",
                "global": "global over all valid pixels (mask>0.5) within the group",
            },
            "metric_better": METRIC_BETTER,
            "global_defs": {
                "rmse_global": "sqrt(sum((pred-target)^2 over valid pixels) / sum(valid pixels))",
                "nse_global": "1 - SSE / Var, where Var uses global masked mean",
                "csi/precision/recall_global": "computed from summed TP/FP/FN over valid pixels",
            },
            "global_params": {"threshold": THR, "abs_tol": ABS_TOL, "eps": EPS},
            "device": device,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)
    print("[OK] saved:", args.out_json)

    # ---------------------------
    # Optional CSV outputs
    # ---------------------------
    def write_csv(path, rows, fieldnames):
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print("[OK] saved:", path)

    # patch csv
    if args.out_csv_patch:
        patch_fields = [
            "core", "var", "scenario", "t", "patch_row", "patch_col", "downscale",
            "num_valid", "pred_path", "fine_path", "mask_path",
        ] + metric_names
        write_csv(args.out_csv_patch, patch_rows, patch_fields)

    # t csv
    if args.out_csv_t:
        t_fields = ["scenario", "t", "n_patches", "num_valid_sum"]
        for k in metric_names:
            t_fields += [f"{k}_mean", f"{k}_global"]
        write_csv(args.out_csv_t, st_rows, t_fields)

    # scenario csv
    if args.out_csv_scenario:
        s_fields = ["scenario", "n_patches", "num_valid_sum"]
        for k in metric_names:
            s_fields += [f"{k}_mean", f"{k}_global"]
        write_csv(args.out_csv_scenario, s_rows, s_fields)

    print(f"[OK] done. patches={len(patch_rows)} skipped={skipped}")


if __name__ == "__main__":
    main()
