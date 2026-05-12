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


# =========================================================
# File/folder parsing
# =========================================================
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
def _ratio(num, den, eps=EPS):
    return float(num / (den + eps))


def _rmse_from_sse(sse_sum: float, n_sum: int):
    if n_sum <= 0:
        return float("nan")
    return float(np.sqrt(max(sse_sum, 0.0) / float(max(n_sum, 1))))


def _nse_from_sums(sse_sum: float, t_sum: float, t_sq_sum: float, n_sum: int, eps=EPS):
    if n_sum <= 0:
        return float("nan")
    mean = t_sum / float(n_sum)
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)
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
):
    if n_sum <= 0:
        return float("nan")

    mean = t_sum / float(n_sum)
    var = t_sq_sum - 2.0 * mean * t_sum + (mean * mean) * float(n_sum)

    if var >= min_var:
        var = max(var, eps)
        return float(1.0 - (sse_sum / var))

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

    # -------------------------------------------------
    # Global regression stats: RMSE / NSE families
    # -------------------------------------------------
    diff = pred - gt
    diff2 = diff * diff
    sse = float(diff2[m].sum())

    pred_eff = np.where((pred >= DEPTH_EVENT_THRESHOLD) & m, pred, 0.0)
    gt_eff = np.where((gt >= DEPTH_EVENT_THRESHOLD) & m, gt, 0.0)

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
    p_evt = (pred >= DEPTH_EVENT_THRESHOLD) & m
    t_evt = (gt >= DEPTH_EVENT_THRESHOLD) & m

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

        gt_band = select_band(gt, ge=ge, lt=lt) & m
        pred_band = select_band(pred, ge=ge, lt=lt) & m

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

    return out


# =========================================================
# Recover metrics from additive stats
# =========================================================
def metrics_from_stats(stats: dict):
    out = {}

    N = int(stats["N"])

    # Global rmse / nse families
    out["rmse"] = _rmse_from_sse(stats["rmse_sse"], N)
    out["rmse_thr"] = _rmse_from_sse(stats["rmse_thr_sse"], N)
    out["rmse_thr_tol"] = _rmse_from_sse(stats["rmse_thr_tol_sse"], N)

    out["nse"] = _nse_from_sums(stats["nse_sse"], stats["nse_t_sum"], stats["nse_t_sq_sum"], N)
    out["nse_thr"] = _nse_from_sums(stats["nse_thr_sse"], stats["nse_thr_t_sum"], stats["nse_thr_t_sq_sum"], N)
    out["nse_thr_tol"] = _nse_from_sums(stats["nse_thr_tol_sse"], stats["nse_thr_tol_t_sum"], stats["nse_thr_tol_t_sq_sum"], N)

    out["nse_safe"] = _nse_safe_from_sums(
        stats["nse_sse"], stats["nse_t_sum"], stats["nse_t_sq_sum"], N
    )
    out["nse_thr_safe"] = _nse_safe_from_sums(
        stats["nse_thr_sse"], stats["nse_thr_t_sum"], stats["nse_thr_t_sq_sum"], N
    )
    out["nse_thr_tol_safe"] = _nse_safe_from_sums(
        stats["nse_thr_tol_sse"], stats["nse_thr_tol_t_sum"], stats["nse_thr_tol_t_sq_sum"], N
    )

    # Flood/nonflood classification
    out["precision"] = _ratio(stats["tp"], stats["tp"] + stats["fp"])
    out["recall"] = _ratio(stats["tp"], stats["tp"] + stats["fn"])
    out["csi"] = _ratio(stats["tp"], stats["tp"] + stats["fp"] + stats["fn"])

    out["precision_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fp_tol"])
    out["recall_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fn_tol"])
    out["csi_tol"] = _ratio(stats["tp_tol"], stats["tp_tol"] + stats["fp_tol"] + stats["fn_tol"])

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
        out[f"precision_{band_name}"] = _ratio(tp, tp + fp)
        out[f"recall_{band_name}"] = _ratio(tp, tp + fn)
        out[f"support_{band_name}"] = n

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
    ]

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

    int_like = [
        "N",
        "tp", "fp", "fn",
        "tp_tol", "fp_tol", "fn_tol",
        "t_evt_cnt", "p_evt_cnt",
    ]
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


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    clean_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "_stats"}
        clean_rows.append(rr)

    fieldnames = sorted(set().union(*[r.keys() for r in clean_rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in clean_rows:
            w.writerow(r)
    print("[OK] saved:", path)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Summarize flood metrics using global aggregation.")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)

    ap.add_argument("--out-json", required=True, help="dataset summary json")
    ap.add_argument("--out-csv-patch", default="", help="patch metrics csv")
    ap.add_argument("--out-csv-time", default="", help="(scenario,t) metrics csv")
    ap.add_argument("--out-csv-scenario", default="", help="scenario metrics csv")

    ap.add_argument("--var", default="h", help="filter variable, e.g. h")
    ap.add_argument("--limit", type=int, default=0, help="limit number of patches for quick test")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    idx = build_index_map(args.index_csv)
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

            gt = load_hw_npy(row["fine_path"], "fine")
            mask = load_hw_npy(row["mask_fine_path"], "mask")
            pred = load_hw_npy(pred_path, "pred")

            if pred.shape != gt.shape:
                raise RuntimeError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")

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

    scenario_rows = []
    for scenario, rows in sorted(s_groups.items()):
        out = {
            "scenario": scenario,
            "n_patches": len(rows),
            "num_valid_sum": int(sum(r["num_valid"] for r in rows)),
        }

        g = global_metrics_from_rows(rows)
        pm = patch_mean_metrics_from_rows(rows)

        out.update({f"{k}_global": v for k, v in g.items()})
        out.update(pm)
        scenario_rows.append(out)

    # -------------------------------------------------
    # Dataset level
    # -------------------------------------------------
    dataset_metrics_global = global_metrics_from_rows(patch_rows)
    dataset_metrics_patch_mean = patch_mean_metrics_from_rows(patch_rows)

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
        "n_patches": len(patch_rows),
        "num_valid_sum": int(sum(r["num_valid"] for r in patch_rows)),
        "metrics_dataset_global": dataset_metrics_global,
        "metrics_dataset_patch_mean": dataset_metrics_patch_mean,
        "extremes": extremes,
        "notes": {
            "aggregation": {
                "patch": "metrics recovered from one patch's additive stats",
                "time_global": "global metrics over all valid pixels in each (scenario, t) group",
                "scenario_global": "global metrics over all valid pixels in each scenario group",
                "dataset_global": "global metrics over all valid pixels in the full dataset",
                "patch_mean": "simple mean over patch-level metrics for comparison",
            },
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