import os
import re
import csv
import glob
import json
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np


CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>\d+y_\d+h_\d+c)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)

EPS = 1e-12


# ---------------------------
# Utilities
# ---------------------------
def ensure_hw(arr: np.ndarray, name="arr") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:   # [B,H,W] or [C,H,W]
        return a[0, ...]
    if a.ndim == 4:   # [B,C,H,W]
        return a[0, 0, ...]
    raise RuntimeError(f"{name} has unsupported shape: {a.shape}")


def load_npy_hw(path: str, name="arr") -> np.ndarray:
    if not path or (not os.path.exists(path)):
        raise FileNotFoundError(f"Missing file: {path}")
    return ensure_hw(np.load(path), name=name)


def parse_core(core: str) -> Dict[str, Any]:
    m = CORE_RE.match(core)
    if not m:
        raise ValueError(f"Bad core key: {core}")
    d = m.groupdict()
    d["patch_row"] = int(d["r"])
    d["patch_col"] = int(d["c"])
    d["downscale"] = int(d["s"])
    return d


def mask_bool(mask_hw: np.ndarray) -> np.ndarray:
    return np.asarray(mask_hw, dtype=np.float64) > 0.5


def downsample_mask_to(mask_f: np.ndarray, Hc: int, Wc: int) -> np.ndarray:
    """
    Downsample a fine mask [Hf,Wf] to [Hc,Wc] by block-max.
    Requires divisibility.
    """
    mf = mask_bool(mask_f)
    Hf, Wf = mf.shape
    if Hf % Hc != 0 or Wf % Wc != 0:
        raise RuntimeError(f"Cannot downsample mask: fine={mf.shape} -> coarse=({Hc},{Wc}) not divisible")
    sh = Hf // Hc
    sw = Wf // Wc
    x = mf.reshape(Hc, sh, Wc, sw)
    mc = x.max(axis=(1, 3))
    return mc.astype(bool)


def finite_in_mask(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    xv = x[m]
    xv = xv[np.isfinite(xv)]
    return xv


def stats_summary(x: np.ndarray, m: Optional[np.ndarray], name: str, bins: int = 50) -> Dict[str, Any]:
    if m is None:
        xv = np.asarray(x, dtype=np.float64).ravel()
        xv = xv[np.isfinite(xv)]
    else:
        xv = finite_in_mask(np.asarray(x, dtype=np.float64), m)

    out: Dict[str, Any] = {"name": name, "count": int(xv.size)}
    if xv.size == 0:
        out.update({"min": None, "max": None, "mean": None, "std": None, "quantiles": {}, "hist": {}})
        return out

    out["min"] = float(np.min(xv))
    out["max"] = float(np.max(xv))
    out["mean"] = float(np.mean(xv))
    out["std"] = float(np.std(xv))

    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    qv = np.percentile(xv, qs).astype(np.float64)
    out["quantiles"] = {str(q): float(v) for q, v in zip(qs, qv)}

    # histogram (use [min, max] range)
    if out["min"] == out["max"]:
        out["hist"] = {"bins": [out["min"], out["max"]], "counts": [int(xv.size)]}
    else:
        counts, edges = np.histogram(xv, bins=bins, range=(out["min"], out["max"]))
        out["hist"] = {
            "bins": [float(e) for e in edges.tolist()],
            "counts": [int(c) for c in counts.tolist()],
        }
    return out


# ---------------------------
# Paper-formula metrics (normal versions)
# ---------------------------
def rmse(sim: np.ndarray, pred: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    s = np.asarray(sim, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mm = mask_bool(m)

    s_v = finite_in_mask(s, mm)
    p_v = finite_in_mask(p, mm)
    n = int(min(s_v.size, p_v.size))
    if n <= 0:
        return {"rmse": float("nan"), "mse": float("nan"), "sse": float("nan"), "n": 0}

    diff = (p_v[:n] - s_v[:n])
    sse = float(np.sum(diff * diff))
    mse = float(sse / max(n, 1))
    return {"rmse": float(np.sqrt(mse)), "mse": mse, "sse": sse, "n": int(n)}


def nse(sim: np.ndarray, pred: np.ndarray, m: np.ndarray) -> Dict[str, float]:
    """
    NSE = 1 - sum((sim - pred)^2) / sum((sim - mean(sim))^2) over masked pixels.
    """
    s = np.asarray(sim, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mm = mask_bool(m)

    s_v = finite_in_mask(s, mm)
    p_v = finite_in_mask(p, mm)
    n = int(min(s_v.size, p_v.size))
    if n <= 0:
        return {"nse": float("nan"), "den": float("nan"), "num": float("nan"), "mean_sim": float("nan"), "n": 0}

    s_v = s_v[:n]
    p_v = p_v[:n]
    num = float(np.sum((s_v - p_v) ** 2))
    mean_s = float(np.mean(s_v))
    den = float(np.sum((s_v - mean_s) ** 2))

    if den <= EPS:
        # paper doesn't define; return NaN + provide components
        return {"nse": float("nan"), "den": den, "num": num, "mean_sim": mean_s, "n": int(n)}

    return {"nse": float(1.0 - num / den), "den": den, "num": num, "mean_sim": mean_s, "n": int(n)}


def classification_metrics(sim: np.ndarray, pred: np.ndarray, m: np.ndarray, delta: float) -> Dict[str, float]:
    s = np.asarray(sim, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mm = mask_bool(m)

    # keep only finite pixels in BOTH
    finite = mm & np.isfinite(s) & np.isfinite(p)
    n = int(finite.sum())
    if n <= 0:
        return {
            "csi": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "tp": 0, "fp": 0, "fn": 0, "n": 0,
            "target_prevalence": float("nan"), "pred_prevalence": float("nan"),
        }

    t_evt = (s >= delta) & finite
    p_evt = (p >= delta) & finite

    tp = int((t_evt & p_evt).sum())
    fp = int((~t_evt & p_evt).sum())
    fn = int((t_evt & ~p_evt).sum())

    csi = float(tp / (tp + fp + fn + EPS))
    precision = float(tp / (tp + fp + EPS))
    recall = float(tp / (tp + fn + EPS))

    target_prev = float(int(t_evt.sum()) / (n + EPS))
    pred_prev = float(int(p_evt.sum()) / (n + EPS))

    return {
        "csi": csi,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": n,
        "target_prevalence": target_prev,
        "pred_prevalence": pred_prev,
    }


# ---------------------------
# File locating
# ---------------------------
def read_index_row(index_csv: str, core: str) -> Optional[Dict[str, str]]:
    if not index_csv:
        return None
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"index_csv not found: {index_csv}")

    meta = parse_core(core)
    want = {
        "var": meta["var"],
        "scenario": meta["scenario"],
        "t": meta["t"],
        "patch_row": str(meta["patch_row"]),
        "patch_col": str(meta["patch_col"]),
        "downscale": str(meta["downscale"]),
    }

    with open(index_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ok = True
            for k, v in want.items():
                if row.get(k, "") != v:
                    ok = False
                    break
            if ok:
                return row
    return None


def _pick_path_from_row(row: Dict[str, str], kind: str) -> Optional[str]:
    """
    Heuristic to pick a path from index.csv row.
    kind in {"fine","mask_fine","coarse"}
    """
    if not row:
        return None

    # common explicit names first
    candidates = []
    if kind == "fine":
        candidates += ["fine_path", "fine_npy_path", "fine_file", "fine"]
    elif kind == "mask_fine":
        candidates += ["mask_fine_path", "mask_path", "fine_mask_path", "mask"]
    elif kind == "coarse":
        candidates += ["coarse_path", "coarse_npy_path", "coarse_file", "coarse"]

    for k in candidates:
        p = row.get(k, "")
        if p and os.path.exists(p):
            return p

    # fallback: scan all columns
    for k, v in row.items():
        kk = k.lower()
        if not v or not isinstance(v, str):
            continue
        if not os.path.exists(v):
            continue
        if kind == "mask_fine":
            if ("mask" in kk) and ("fine" in kk or "hf" in kk or "h_fine" in kk or True):
                return v
        if kind == "fine":
            if ("fine" in kk) and ("mask" not in kk) and ("path" in kk or "file" in kk):
                return v
        if kind == "coarse":
            if ("coarse" in kk) and ("mask" not in kk) and ("path" in kk or "file" in kk):
                return v

    return None


def find_in_folder(folder: str, core: str) -> Dict[str, Optional[str]]:
    """
    Try find:
      - coarse sim: *{core}*_coarse.npy
      - fine sim:   *{core}*_fine.npy
      - mask:       *mask*.npy (optional)
      - pred:       any npy excluding *_coarse.npy, *_fine.npy, mask-like
    """
    out = {"coarse": None, "fine": None, "mask": None, "pred": None}

    if not os.path.isdir(folder):
        return out

    # exact matches preferred
    coarse_exact = os.path.join(folder, f"{core}_coarse.npy")
    fine_exact = os.path.join(folder, f"{core}_fine.npy")
    if os.path.exists(coarse_exact):
        out["coarse"] = coarse_exact
    if os.path.exists(fine_exact):
        out["fine"] = fine_exact

    # glob fallback
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    for p in npys:
        bn = os.path.basename(p).lower()
        if out["coarse"] is None and bn.endswith("_coarse.npy") and bn.startswith(core.lower()):
            out["coarse"] = p
        if out["fine"] is None and bn.endswith("_fine.npy") and bn.startswith(core.lower()):
            out["fine"] = p
        if out["mask"] is None and ("mask" in bn) and bn.endswith(".npy"):
            out["mask"] = p

    # pred: exclude coarse/fine/mask
    for p in npys:
        bn = os.path.basename(p).lower()
        if bn.endswith("_coarse.npy") or bn.endswith("_fine.npy") or ("mask" in bn):
            continue
        # prefer files that start with core_
        if bn.startswith(core.lower() + "_"):
            out["pred"] = p
            break

    if out["pred"] is None:
        # last resort: if only one npy exists besides coarse/fine/mask
        cand = []
        for p in npys:
            bn = os.path.basename(p).lower()
            if bn.endswith("_coarse.npy") or bn.endswith("_fine.npy") or ("mask" in bn):
                continue
            cand.append(p)
        if len(cand) == 1:
            out["pred"] = cand[0]

    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze one core key distributions + paper-formula metrics.")
    ap.add_argument("--vis-root", required=True, help=".../results/.../visualization")
    ap.add_argument("--key", required=True, help="core key, e.g. h_2y_6h_0c_t0000_r001_c007_s16")
    ap.add_argument("--index-csv", default="", help="optional index.csv for fine/mask/coarse fallback")
    ap.add_argument("--delta", type=float, default=0.05, help="threshold delta for CSI/Precision/Recall/Prevalence (paper uses 0.05 for h)")
    ap.add_argument("--bins", type=int, default=50, help="hist bins per distribution")
    ap.add_argument("--out-json", default="", help="optional output json report")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    core = args.key.strip()
    meta = parse_core(core)  # validate

    folder = os.path.join(args.vis_root, f"{core}_coarse")
    found = find_in_folder(folder, core)

    row = read_index_row(args.index_csv, core) if args.index_csv else None
    if row is None and args.index_csv:
        print(f"[warn] key not found in index.csv: {core}")

    # Resolve paths
    coarse_path = found["coarse"]
    fine_path = found["fine"]
    mask_path = found["mask"]
    pred_path = found["pred"]

    if fine_path is None and row is not None:
        fine_path = _pick_path_from_row(row, "fine")
    if mask_path is None and row is not None:
        mask_path = _pick_path_from_row(row, "mask_fine")
    if coarse_path is None and row is not None:
        coarse_path = _pick_path_from_row(row, "coarse")

    missing = [k for k, v in [("coarse", coarse_path), ("fine", fine_path), ("mask", mask_path), ("pred", pred_path)] if not v]
    if missing:
        raise RuntimeError(
            f"Missing required files for {core}: {missing}\n"
            f"  folder={folder}\n"
            f"  found_in_folder={found}\n"
            f"  index_csv={'(none)' if not args.index_csv else args.index_csv}\n"
            f"Tip: ensure visualization/{core}_coarse contains {core}_coarse.npy, {core}_fine.npy, pred.npy, "
            f"or make sure index.csv has fine_path/mask_fine_path/coarse_path columns."
        )

    # Load arrays
    sim_coarse = load_npy_hw(coarse_path, "sim_coarse")
    sim_fine = load_npy_hw(fine_path, "sim_fine")
    pred_fine = load_npy_hw(pred_path, "pred_fine")
    mask_fine = load_npy_hw(mask_path, "mask_fine")

    if sim_fine.shape != pred_fine.shape:
        raise RuntimeError(f"Shape mismatch fine: sim_fine={sim_fine.shape} pred_fine={pred_fine.shape}")

    mf = mask_bool(mask_fine)
    n_valid = int(mf.sum())
    if n_valid <= 0:
        raise RuntimeError("mask_fine has zero valid pixels (mask>0.5).")

    # mask for coarse distribution (downsample from fine if needed)
    mc = None
    if sim_coarse.shape == sim_fine.shape:
        mc = mf
    else:
        try:
            mc = downsample_mask_to(mask_fine, sim_coarse.shape[0], sim_coarse.shape[1])
        except Exception as e:
            if args.debug:
                print(f"[warn] cannot build coarse mask from fine mask: {e}. Use no mask for coarse distribution.")
            mc = None

    # Distributions
    dist = {
        "sim_coarse": stats_summary(sim_coarse, mc, "simulated_coarse", bins=args.bins),
        "sim_fine": stats_summary(sim_fine, mf, "simulated_fine", bins=args.bins),
        "pred_fine": stats_summary(pred_fine, mf, "predicted_fine", bins=args.bins),
        "error_pred_minus_sim": stats_summary(pred_fine - sim_fine, mf, "error(pred-sim)", bins=args.bins),
        "sq_error": stats_summary((pred_fine - sim_fine) ** 2, mf, "squared_error", bins=args.bins),
    }

    # Metrics (paper formulas, normal version)
    rmse_out = rmse(sim_fine, pred_fine, mask_fine)
    nse_out = nse(sim_fine, pred_fine, mask_fine)
    cls_out = classification_metrics(sim_fine, pred_fine, mask_fine, delta=args.delta)

    metrics = {
        "rmse": rmse_out["rmse"],
        "nse": nse_out["nse"],
        "csi": cls_out["csi"],
        "precision": cls_out["precision"],
        "recall": cls_out["recall"],
        "target_prevalence": cls_out["target_prevalence"],
        "pred_prevalence": cls_out["pred_prevalence"],
        "delta": float(args.delta),
    }

    # RMSE breakdown (to answer "RMSE怎么算到这个值")
    rmse_breakdown = {
        "SSE": rmse_out["sse"],
        "MSE": rmse_out["mse"],
        "RMSE": rmse_out["rmse"],
        "N_valid_pixels": rmse_out["n"],
        "definition": "RMSE = sqrt( mean( (pred - sim)^2 ) ) over valid pixels (mask>0.5 & finite).",
    }

    report = {
        "core": core,
        "meta": meta,
        "paths": {
            "folder": folder,
            "sim_coarse": coarse_path,
            "sim_fine": fine_path,
            "pred_fine": pred_path,
            "mask_fine": mask_path,
        },
        "valid_pixels_fine": n_valid,
        "metrics": metrics,
        "rmse_breakdown": rmse_breakdown,
        "nse_breakdown": nse_out,  # includes numerator/denominator/mean_sim
        "classification_counts": {k: cls_out[k] for k in ["tp", "fp", "fn", "n"]},
        "distributions": dist,
        "notes": {
            "paper_formulas": {
                "rmse": "sqrt( sum_i (sim_i - pred_i)^2 / n_p )",
                "nse": "1 - sum_i(sim_i - pred_i)^2 / sum_i(sim_i - mean(sim))^2",
                "csi": "TP/(TP+FP+FN) where event is value >= delta",
            },
            "mask_rule": "valid pixel if mask>0.5 AND finite(sim) AND finite(pred) for classification; for RMSE/NSE, finite are applied after masking.",
        },
    }

    # Pretty print key results
    print("========================================")
    print(f"[KEY] {core}")
    print(f"[PATH] folder={folder}")
    print("--------- Metrics (paper normal) ---------")
    print(f"delta = {args.delta}")
    print(f"RMSE = {metrics['rmse']:.6g}")
    print(f"  SSE = {rmse_breakdown['SSE']:.6g}")
    print(f"  MSE = {rmse_breakdown['MSE']:.6g}")
    print(f"  N   = {rmse_breakdown['N_valid_pixels']}")
    print(f"NSE  = {metrics['nse']:.6g}   (num={nse_out.get('num'):.6g}, den={nse_out.get('den'):.6g}, mean_sim={nse_out.get('mean_sim'):.6g})")
    print(f"CSI  = {metrics['csi']:.6g}   (TP={cls_out['tp']}, FP={cls_out['fp']}, FN={cls_out['fn']}, N={cls_out['n']})")
    print(f"Prec = {metrics['precision']:.6g}")
    print(f"Rec  = {metrics['recall']:.6g}")
    print(f"Prev(target) = {metrics['target_prevalence']:.6g}")
    print(f"Prev(pred)   = {metrics['pred_prevalence']:.6g}")
    print("--------- Distribution snapshots ---------")
    for k in ["sim_coarse", "sim_fine", "pred_fine", "error_pred_minus_sim"]:
        d = dist[k]
        q50 = d["quantiles"].get("50", None)
        q95 = d["quantiles"].get("95", None)
        print(f"{k:>22s}: count={d['count']} mean={d['mean']:.6g} std={d['std']:.6g} q50={q50:.6g} q95={q95:.6g} min={d['min']:.6g} max={d['max']:.6g}")
    print("========================================")

    # Save JSON
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("[OK] saved:", args.out_json)


if __name__ == "__main__":
    main()
