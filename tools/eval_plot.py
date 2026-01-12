#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import csv
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ---------------------------
# GMT .cpt -> Matplotlib cmap (for ELEVATION)
# ---------------------------
def load_gmt_cpt(cpt_path: str, name: str = "cpt") -> LinearSegmentedColormap:
    if not cpt_path or (not os.path.exists(cpt_path)):
        raise RuntimeError(f"CPT not found: {cpt_path}")

    xs, cols = [], []
    with open(cpt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            if line[0] in "BFN":  # background/foreground/nan
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                x1, r1, g1, b1, x2, r2, g2, b2 = parts[:8]
                x1 = float(x1); x2 = float(x2)
                c1 = (float(r1)/255.0, float(g1)/255.0, float(b1)/255.0)
                c2 = (float(r2)/255.0, float(g2)/255.0, float(b2)/255.0)
            except Exception:
                continue
            xs += [x1, x2]
            cols += [c1, c2]

    if len(xs) < 2:
        raise RuntimeError(f"Failed to parse CPT: {cpt_path}")

    xs = np.array(xs, dtype=np.float64)
    xmin, xmax = float(xs.min()), float(xs.max())
    if np.isclose(xmin, xmax):
        raise RuntimeError(f"Invalid CPT range: {cpt_path}")

    xnorm = (xs - xmin) / (xmax - xmin)
    order = np.argsort(xnorm)
    xnorm = xnorm[order]
    cols = np.array(cols, dtype=np.float64)[order]

    ux, uidx = np.unique(xnorm, return_index=True)
    cols = cols[uidx]
    cm = LinearSegmentedColormap.from_list(name, list(zip(ux, cols)))
    try:
        cm.set_bad("white")
    except Exception:
        pass
    return cm


# ---------------------------
# Array helpers
# ---------------------------
def ensure_hw(arr: np.ndarray, name: str = "arr") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0, ...]
    if a.ndim == 4:
        return a[0, 0, ...]
    raise RuntimeError(f"{name} has unsupported shape: {a.shape}")


def apply_aoi_nan(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask == 1, arr, np.nan)


def kron_upsample(coarse: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(coarse, np.ones((scale, scale), dtype=coarse.dtype))


def percentile_range(arr: np.ndarray, qlo: float, qhi: float):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [qlo, qhi])
    if np.isclose(lo, hi):
        lo2, hi2 = float(v.min()), float(v.max())
        if np.isclose(lo2, hi2):
            return lo2, lo2 + 1.0
        return lo2, hi2
    return float(lo), float(hi)


def get_static_vmin_vmax(arr: np.ndarray, mode: str, qlo: float, qhi: float):
    if mode == "minmax":
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = float(v.min()), float(v.max())
        if np.isclose(lo, hi):
            return lo, lo + 1.0
        return lo, hi
    return percentile_range(arr, qlo, qhi)


def make_blues_zero_white():
    """Blues colormap, force 0 to look white."""
    base = plt.cm.get_cmap("Blues", 256)
    cols = base(np.linspace(0, 1, 256))
    cols[0] = np.array([1, 1, 1, 1], dtype=np.float64)  # 0 -> white
    cm = ListedColormap(cols, name="BluesZeroWhite")
    cm.set_bad(color="white")
    return cm


# ---------------------------
# Plot helpers
# ---------------------------
def add_colorbar(fig, im, ax, scientific: bool = False, ticks=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if scientific:
        fmt = mticker.ScalarFormatter(useMathText=False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        cb.formatter = fmt
        cb.update_ticks()
    else:
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        cb.formatter = fmt
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_visible(False)

    if ticks is not None:
        cb.set_ticks(ticks)
        cb.update_ticks()

    return cb


def imshow_with_cb(fig, ax, data, title, cmap, vmin=None, vmax=None,
                   contour_mask=None, scientific=False, cb_ticks=None):
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if contour_mask is not None:
        ax.contour(contour_mask, levels=[0.5], colors="k", linewidths=0.8)
    add_colorbar(fig, im, ax, scientific=scientific, ticks=cb_ticks)
    return im


# ---------------------------
# Key parsing
# ---------------------------
CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>\d+y_\d+h_\d+c)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)
FOLDER_RE = re.compile(r'^(?P<core>.+)_coarse$')


def parse_core_key(core: str):
    m = CORE_RE.match(core)
    if not m:
        raise ValueError(f"Bad core key: {core}")
    return core


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


def build_index_map(index_csv: str):
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


def pick_pred_file_in_folder(folder: str, core: str) -> str:
    npys = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npys:
        raise RuntimeError(f"No .npy found in {folder}")

    def _ok(p):
        bn = os.path.basename(p)
        if not bn.endswith(".npy"):
            return False
        if bn.endswith("_coarse.npy") or bn.endswith("_fine.npy"):
            return False
        return True

    cand1 = []
    for p in npys:
        bn = os.path.basename(p).lower()
        if _ok(p) and bn.startswith((core + "_").lower()) and any(k in bn for k in ["pred", "sr", "output"]):
            cand1.append(p)
    if cand1:
        cand1.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        return cand1[0]

    cand2 = []
    for p in npys:
        bn = os.path.basename(p)
        if _ok(p) and bn.startswith(core + "_"):
            cand2.append(p)
    if cand2:
        cand2.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        return cand2[0]

    if len(npys) == 1:
        return npys[0]

    raise RuntimeError(f"Cannot identify predicted .npy in {folder} for core={core}")


# ---------------------------
# Thresholding rules
# ---------------------------
def flood_apply_threshold_zero(a: np.ndarray, thr: float) -> np.ndarray:
    """AOI-masked array: values < thr -> 0 (outside AOI stays NaN)."""
    if thr is None or thr <= 0:
        return a
    return np.where(np.isfinite(a) & (a < thr), 0.0, a)


def diff_apply_abs_tol_zero(d: np.ndarray, abs_tol: float) -> np.ndarray:
    """AOI-masked array: |d| < abs_tol -> 0 (outside AOI stays NaN)."""
    if abs_tol is None or abs_tol <= 0:
        return d
    return np.where(np.isfinite(d) & (np.abs(d) < abs_tol), 0.0, d)


def make_elev_ticks(vmin: float, vmax: float, step: int = 100):
    """Return ticks like -500,-400,...,500 within [vmin, vmax]."""
    if vmin is None or vmax is None or (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return None
    t0 = int(np.ceil(vmin / step) * step)
    t1 = int(np.floor(vmax / step) * step)
    if t1 < t0:
        return None
    return list(range(t0, t1 + step, step))


# ---------------------------
# Plot core
# ---------------------------
def plot_one(
    core: str,
    row: dict,
    pred_path: str,
    out_png: str,

    layout: str = "3x4",

    # static scaling (rough/slope/twi)
    static_scale: str = "q1q99",
    static_qlo: float = 1.0,
    static_qhi: float = 99.0,

    # elevation colormap + fixed range
    elev_cpt: str = "",
    elev_vmin: float = -500.0,
    elev_vmax: float = 500.0,

    # flood display controls
    flood_thr: float = 0.05,               # <thr -> 0
    flood_vmax_mode: str = "q95",          # q95|max
    flood_q: float = 95.0,                 # for q95

    # diff display controls
    abs_tol: float = 0.01,                 # |err|<tol -> 0
    diff_vabs_mode: str = "q95abs",        # q95abs|maxabs
    diff_q: float = 95.0,

    debug_stats: bool = False,
):
    coarse = ensure_hw(np.load(row["coarse_path"]), "coarse")
    fine   = ensure_hw(np.load(row["fine_path"]), "fine")
    elev   = ensure_hw(np.load(row["elev_path"]), "elev")
    rough  = ensure_hw(np.load(row["rough_path"]), "rough")
    mask   = ensure_hw(np.load(row["mask_fine_path"]), "mask").astype(np.uint8)

    p_slope = row.get("slope_path", "")
    p_twi   = row.get("twi_path", "")
    p_asin  = row.get("aspect_sin_path", "")
    p_acos  = row.get("aspect_cos_path", "")

    slope = ensure_hw(np.load(p_slope), "slope") if (p_slope and os.path.exists(p_slope)) else None
    twi   = ensure_hw(np.load(p_twi),   "twi")   if (p_twi   and os.path.exists(p_twi))   else None
    asin  = ensure_hw(np.load(p_asin),  "asin")  if (p_asin  and os.path.exists(p_asin))  else None
    acos  = ensure_hw(np.load(p_acos),  "acos")  if (p_acos  and os.path.exists(p_acos))  else None

    pred = ensure_hw(np.load(pred_path), "pred")

    Hf, Wf = fine.shape
    Hc, Wc = coarse.shape
    scale = int(round(Hf / Hc))
    if (Hf != Hc * scale) or (Wf != Wc * scale):
        raise RuntimeError(f"Shape mismatch: fine={fine.shape}, coarse={coarse.shape}, inferred scale={scale}")
    if pred.shape != fine.shape:
        raise RuntimeError(f"Pred shape != GT fine shape: pred={pred.shape}, fine={fine.shape}")

    coarse_up = kron_upsample(coarse, scale)

    # AOI mask -> NaN
    elev_m  = apply_aoi_nan(elev, mask)
    rough_m = apply_aoi_nan(rough, mask)
    slope_m = apply_aoi_nan(slope, mask) if slope is not None else None
    twi_m   = apply_aoi_nan(twi, mask)   if twi is not None else None
    asin_m  = apply_aoi_nan(asin, mask)  if asin is not None else None
    acos_m  = apply_aoi_nan(acos, mask)  if acos is not None else None

    fine_m_raw   = apply_aoi_nan(fine, mask)
    coarse_m_raw = apply_aoi_nan(coarse_up, mask)
    pred_m_raw   = apply_aoi_nan(pred, mask)

    # threshold for flood maps (<thr -> 0)
    fine_m   = flood_apply_threshold_zero(fine_m_raw, flood_thr)
    coarse_m = flood_apply_threshold_zero(coarse_m_raw, flood_thr)
    pred_m   = flood_apply_threshold_zero(pred_m_raw, flood_thr)

    # diffs (use thresholded flood)
    diff_fc = fine_m - coarse_m
    diff_pg = pred_m - fine_m

    # abs_tol for diffs
    diff_fc = diff_apply_abs_tol_zero(diff_fc, abs_tol)
    diff_pg = diff_apply_abs_tol_zero(diff_pg, abs_tol)

    if debug_stats:
        def _stat(name, a):
            v = a[np.isfinite(a)]
            if v.size == 0:
                print(f"[stat] {name}: all nan")
            else:
                print(f"[stat] {name}: min={v.min():.6g} max={v.max():.6g} mean={v.mean():.6g} q95={np.percentile(v,95):.6g}")
        print(f"[debug] core={core}")
        print(f"[debug] pred_path={pred_path}")
        _stat("fine_raw(AOI)", fine_m_raw)
        _stat("pred_raw(AOI)", pred_m_raw)
        _stat("coarse_raw(AOI)", coarse_m_raw)
        _stat("diff_fc(AOI)", diff_fc)
        _stat("diff_pg(AOI)", diff_pg)

    # static ranges (except elevation is fixed by elev_vmin/elev_vmax)
    vmin_r, vmax_r = get_static_vmin_vmax(rough_m, static_scale, static_qlo, static_qhi)
    if slope_m is not None:
        vmin_s, vmax_s = get_static_vmin_vmax(slope_m, static_scale, static_qlo, static_qhi)
    if twi_m is not None:
        vmin_t, vmax_t = get_static_vmin_vmax(twi_m, static_scale, static_qlo, static_qhi)

    # aspect fixed
    vmin_as, vmax_as = -1.0, 1.0

    # colormaps
    cm_flood  = make_blues_zero_white()     # 0 -> white
    cm_diff   = plt.cm.get_cmap("seismic"); cm_diff.set_bad("white")
    cm_static = plt.cm.get_cmap("viridis"); cm_static.set_bad("white")
    cm_aspect = plt.cm.get_cmap("RdBu_r");  cm_aspect.set_bad("white")
    cm_mask   = plt.cm.get_cmap("gray_r");  cm_mask.set_bad("white")

    if elev_cpt:
        cm_elev = load_gmt_cpt(elev_cpt, name="wiki-france")
    else:
        cm_elev = plt.cm.get_cmap("terrain"); cm_elev.set_bad("white")

    # Elevation ticks: -500, -400, ..., 500
    elev_ticks = make_elev_ticks(elev_vmin, elev_vmax, step=100)

    # Flood colorbar: compute from GT FINE AFTER threshold (AOI), and ignore zeros
    vmin_flood = 0.0
    fine_vals_thr = fine_m[np.isfinite(fine_m)]  # thresholded
    fine_pos = fine_vals_thr[fine_vals_thr > 0.0]  # ignore 0 after threshold

    if fine_pos.size == 0:
        # no flooded pixels survive the threshold -> keep a sensible scale
        vmax_flood = flood_thr
    else:
        if flood_vmax_mode == "max":
            vmax_flood = float(np.max(fine_pos))
        else:
            vmax_flood = float(np.percentile(fine_pos, flood_q))

    # safety: avoid weird tiny/invalid vmax; also keep vmax >= flood_thr for interpretability
    if (not np.isfinite(vmax_flood)) or (vmax_flood <= vmin_flood + 1e-12):
        vmax_flood = max(flood_thr, 1.0)
    vmax_flood = max(vmax_flood, flood_thr)

    # Diff colorbar: symmetric [-vabs, +vabs]
    def vabs_from_diff(d: np.ndarray):
        v = d[np.isfinite(d)]
        if v.size == 0:
            return 1.0

        # d is already after abs_tol; ignore exact zeros so percentile isn't dominated by zeros
        v = v[v != 0.0]
        if v.size == 0:
            return abs_tol  # everything got thresholded to 0

        av = np.abs(v)
        if diff_vabs_mode == "maxabs":
            return float(np.max(av))
        return float(np.percentile(av, diff_q))  # q95(abs(error)) on surviving errors

    vabs_fc = max(vabs_from_diff(diff_fc), abs_tol, 1e-6)
    vabs_pg = max(vabs_from_diff(diff_pg), abs_tol, 1e-6)

    # layout
    m = re.match(r"^(\d+)x(\d+)$", layout.strip())
    if not m:
        raise RuntimeError(f"Bad --layout: {layout} (use 3x4 or 4x3)")
    nrow, ncol = int(m.group(1)), int(m.group(2))
    if nrow * ncol != 12:
        raise RuntimeError(f"layout must have 12 panels, got {nrow}x{ncol}={nrow*ncol}")

    figsize = (22, 14) if (nrow, ncol) == (3, 4) else (18, 18)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.ravel()

    # Titles (as requested)
    title_elev   = "Elevation"
    title_rough  = "Roughness"
    title_slope  = "Slope"
    title_twi    = "TWI"
    title_asin   = "Aspect Sin (Eastness)"
    title_acos   = "Aspect Cos (Northness)"
    title_mask   = "Fine-grid Mask"
    title_coarse = "Simulated Coarse-grid Flood Map"
    title_fine   = "Simulated Fine-grid Flood Map"
    title_pred   = "Predicted Fine-grid Flood Map"
    title_df_fc  = "Simulated Fine-grid - Simulated Coarse-grid"
    title_df_pg  = "Predicted Fine-grid - Simulated Fine-grid"

    # panels
    imshow_with_cb(fig, axes[0], elev_m,  title_elev,  cm_elev,  elev_vmin, elev_vmax,
                   contour_mask=mask, scientific=False, cb_ticks=elev_ticks)
    imshow_with_cb(fig, axes[1], rough_m, title_rough, cm_static, vmin_r, vmax_r,
                   contour_mask=mask, scientific=False)

    if slope_m is not None:
        imshow_with_cb(fig, axes[2], slope_m, title_slope, cm_static, vmin_s, vmax_s,
                       contour_mask=mask, scientific=False)
    else:
        axes[2].axis("off"); axes[2].set_title("Slope (missing)")

    if twi_m is not None:
        imshow_with_cb(fig, axes[3], twi_m, title_twi, cm_static, vmin_t, vmax_t,
                       contour_mask=mask, scientific=False)
    else:
        axes[3].axis("off"); axes[3].set_title("TWI (missing)")

    if asin_m is not None:
        imshow_with_cb(fig, axes[4], asin_m, title_asin, cm_aspect, vmin_as, vmax_as,
                       contour_mask=mask, scientific=False)
    else:
        axes[4].axis("off"); axes[4].set_title("Aspect Sin (missing)")

    if acos_m is not None:
        imshow_with_cb(fig, axes[5], acos_m, title_acos, cm_aspect, vmin_as, vmax_as,
                       contour_mask=mask, scientific=False)
    else:
        axes[5].axis("off"); axes[5].set_title("Aspect Cos (missing)")

    imshow_with_cb(fig, axes[6], mask, title_mask, cm_mask, 0.0, 1.0,
                   contour_mask=None, scientific=False)

    # flood maps
    imshow_with_cb(fig, axes[7], coarse_m, title_coarse, cm_flood, vmin_flood, vmax_flood,
                   contour_mask=mask, scientific=True)
    imshow_with_cb(fig, axes[8], fine_m,   title_fine,   cm_flood, vmin_flood, vmax_flood,
                   contour_mask=mask, scientific=True)
    imshow_with_cb(fig, axes[9], pred_m,   title_pred,   cm_flood, vmin_flood, vmax_flood,
                   contour_mask=mask, scientific=True)

    # diffs
    imshow_with_cb(fig, axes[10], diff_fc, title_df_fc, cm_diff, -vabs_fc, +vabs_fc,
                   contour_mask=mask, scientific=False)
    imshow_with_cb(fig, axes[11], diff_pg, title_df_pg, cm_diff, -vabs_pg, +vabs_pg,
                   contour_mask=mask, scientific=False)

    fig.suptitle(core, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print("[OK] saved:", out_png)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)

    ap.add_argument("--key", default="")
    ap.add_argument("--out", default="")

    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layout", default="3x4")

    # elevation colormap + fixed range + fixed ticks
    ap.add_argument("--elev-cpt", default="", help="e.g., /path/wiki-france.cpt")
    ap.add_argument("--elev-vmin", type=float, default=-500.0)
    ap.add_argument("--elev-vmax", type=float, default=500.0)

    # static scaling (rough/slope/twi)
    ap.add_argument("--static-scale", choices=["q1q99", "minmax"], default="q1q99")
    ap.add_argument("--static-qlo", type=float, default=1.0)
    ap.add_argument("--static-qhi", type=float, default=99.0)

    # flood controls
    ap.add_argument("--flood-thr", type=float, default=0.05, help="values < thr -> 0 (AOI only)")
    ap.add_argument("--flood-vmax-mode", choices=["q95", "max"], default="q95",
                    help="vmax computed from GT FINE (AOI) after threshold: q95 or max")
    ap.add_argument("--flood-q", type=float, default=95.0)

    # diff controls
    ap.add_argument("--abs-tol", type=float, default=0.01, help="|error| < abs_tol -> 0 (AOI only)")
    ap.add_argument("--diff-vabs-mode", choices=["q95abs", "maxabs"], default="q95abs",
                    help="vabs computed from abs(error) (AOI) after threshold: q95abs or maxabs")
    ap.add_argument("--diff-q", type=float, default=95.0)

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    idx = build_index_map(args.index_csv)

    # single
    if args.key:
        core = args.key.strip()
        parse_core_key(core)
        if core not in idx:
            raise RuntimeError(f"core not found in index.csv: {core}")

        folder = os.path.join(args.vis_root, core + "_coarse")
        if not os.path.isdir(folder):
            raise RuntimeError(f"folder not found: {folder}")

        pred_path = pick_pred_file_in_folder(folder, core)
        out_png = args.out if args.out else f"{core}.png"

        plot_one(
            core, idx[core], pred_path, out_png,
            layout=args.layout,
            static_scale=args.static_scale, static_qlo=args.static_qlo, static_qhi=args.static_qhi,
            elev_cpt=args.elev_cpt, elev_vmin=args.elev_vmin, elev_vmax=args.elev_vmax,
            flood_thr=args.flood_thr, flood_vmax_mode=args.flood_vmax_mode, flood_q=args.flood_q,
            abs_tol=args.abs_tol, diff_vabs_mode=args.diff_vabs_mode, diff_q=args.diff_q,
            debug_stats=args.debug,
        )
        return

    # batch
    if not args.out_dir:
        raise RuntimeError("Batch mode requires --out-dir (since you did not provide --key).")
    os.makedirs(args.out_dir, exist_ok=True)

    folders = sorted(glob.glob(os.path.join(args.vis_root, "*_coarse")))
    n_done = 0
    for f in folders:
        base = os.path.basename(f.rstrip("/"))
        core = parse_folder_name(base)
        if core is None:
            continue
        if core not in idx:
            continue

        try:
            pred_path = pick_pred_file_in_folder(f, core)
            out_png = os.path.join(args.out_dir, f"{core}.png")
            plot_one(
                core, idx[core], pred_path, out_png,
                layout=args.layout,
                static_scale=args.static_scale, static_qlo=args.static_qlo, static_qhi=args.static_qhi,
                elev_cpt=args.elev_cpt, elev_vmin=args.elev_vmin, elev_vmax=args.elev_vmax,
                flood_thr=args.flood_thr, flood_vmax_mode=args.flood_vmax_mode, flood_q=args.flood_q,
                abs_tol=args.abs_tol, diff_vabs_mode=args.diff_vabs_mode, diff_q=args.diff_q,
                debug_stats=args.debug,
            )
            n_done += 1
        except Exception as e:
            print(f"[warn] skip {core}: {e}")

        if args.limit and (n_done >= args.limit):
            break

    print(f"[OK] batch done: {n_done} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
