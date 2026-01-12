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


def weighted_mean_agg(values, weights):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    v = v[m]
    w = w[m]
    return float(np.sum(v * w) / np.sum(w))


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

    # prevalence is "neutral" (still report min/max)
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
    else:
        best_v, best_r = max(cand, key=lambda x: x[0])
        worst_v, worst_r = min(cand, key=lambda x: x[0])
        best_tag = "max"
        worst_tag = "min"

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
# Metrics config (as you requested)
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
# Metric compute per patch
# ---------------------------
@torch.no_grad()
def compute_metrics_one_patch(pred_hw: np.ndarray, gt_hw: np.ndarray, mask_hw: np.ndarray, metrics_cfg: dict, device="cpu"):
    # torch tensors; keep float32 to match training
    pred_t = torch.from_numpy(pred_hw.astype(np.float32)).to(device)
    gt_t = torch.from_numpy(gt_hw.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(mask_hw.astype(np.float32)).to(device)

    out = {}
    for out_key, opt in metrics_cfg.items():
        # calculate_metric expects:
        # data = {"pred":..., "target":..., "mask":...}, opt={"type":..., ...}
        data = {"pred": pred_t, "target": gt_t, "mask": mask_t}
        out[out_key] = float(calculate_metric(data, opt))
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
                # still compute? usually meaningless; skip to avoid NaNs
                raise RuntimeError("num_valid==0 (mask empty)")

            met = compute_metrics_one_patch(pred, gt, mask, metrics_cfg, device=device)

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
        weights = [rr["num_valid"] for rr in rows]
        out = {
            "scenario": scenario,
            "t": t,
            "n_patches": len(rows),
            "num_valid_sum": int(np.sum(weights)),
        }
        for k in metric_names:
            vals = [rr.get(k, np.nan) for rr in rows]
            out[f"{k}_mean"] = mean_agg(vals)
            out[f"{k}_wmean"] = weighted_mean_agg(vals, weights)
        st_rows.append(out)

    # ---------------------------
    # Aggregate: by scenario
    # ---------------------------
    s_groups = defaultdict(list)
    for r in patch_rows:
        s_groups[r["scenario"]].append(r)

    s_rows = []
    for scenario, rows in sorted(s_groups.items()):
        weights = [rr["num_valid"] for rr in rows]
        out = {
            "scenario": scenario,
            "n_patches": len(rows),
            "num_valid_sum": int(np.sum(weights)),
        }
        for k in metric_names:
            vals = [rr.get(k, np.nan) for rr in rows]
            out[f"{k}_mean"] = mean_agg(vals)
            out[f"{k}_wmean"] = weighted_mean_agg(vals, weights)
        s_rows.append(out)

    # ---------------------------
    # Aggregate: dataset
    # ---------------------------
    ds_weights = [r["num_valid"] for r in patch_rows]
    ds_mean = {}
    ds_wmean = {}
    for k in metric_names:
        vals = [r.get(k, np.nan) for r in patch_rows]
        ds_mean[k] = mean_agg(vals)
        ds_wmean[k] = weighted_mean_agg(vals, ds_weights)

    # ---------------------------
    # Extremes (best/worst): patch / t / scenario
    # ---------------------------
    extremes = {
        "patch": {},
        "t_wmean": {},
        "t_mean": {},
        "scenario_wmean": {},
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
        extremes["t_wmean"][k] = pick_best_worst(
            st_rows,
            metric_key=f"{k}_wmean",
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
        extremes["scenario_wmean"][k] = pick_best_worst(
            s_rows,
            metric_key=f"{k}_wmean",
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
        "num_valid_sum": int(np.sum(ds_weights)),
        "metrics_mean": ds_mean,
        "metrics_wmean": ds_wmean,
        "extremes": extremes,
        "notes": {
            "aggregation": {
                "mean": "simple mean over patches",
                "wmean": "weighted mean over patches with weight=num_valid (mask>0.5 count)",
            },
            "metric_better": METRIC_BETTER,
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
            t_fields += [f"{k}_mean", f"{k}_wmean"]
        write_csv(args.out_csv_t, st_rows, t_fields)

    # scenario csv
    if args.out_csv_scenario:
        s_fields = ["scenario", "n_patches", "num_valid_sum"]
        for k in metric_names:
            s_fields += [f"{k}_mean", f"{k}_wmean"]
        write_csv(args.out_csv_scenario, s_rows, s_fields)

    print(f"[OK] done. patches={len(patch_rows)} skipped={skipped}")


if __name__ == "__main__":
    main()
