#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-patch comparison figures: coarse input vs simulated fine (GT) vs predicted fine,
plus error maps; optionally with the full fine-grid static-feature context.

Merges the old tools/eval_plot.py (12-panel: 6 static + mask + coarse/fine/pred + 2 diffs)
and tools/9_4_present_plot/plot_patch_panels.py (compact 2x3: DEM + coarse/fine/pred + 2 diffs).

  --panels full     -> 12-panel (elevation/roughness/slope/twi/aspect_sin/aspect_cos/mask
                       + coarse/fine/pred flood + (coarse-fine) + (pred-fine)); needs the
                       static_path columns in index.csv.
  --panels compact  -> 2x3 (DEM + coarse/fine/pred flood + (coarse-fine) + (pred-fine)).

Selection (pick ONE):
  --key h_100y_42h_0c_t0047_r000_c006_s8                      # single patch by core key
  --scenario 100y_42h_0c --var h --scale 8 --time t0047 --row 0 --col 6   # single by fields
  --out-dir DIR [--scenario ...] [--limit N]                  # batch over results/*_coarse folders

Predictions are read from <vis-root>/<core>_coarse/*.npy (physical space).

Example
-------
python tools/plot_patch_compare.py --panels full \
  --index-csv .../testdataset_100y42h0c/index.csv \
  --vis-root  .../results/039_..._eval_test100y42h0c/visualization \
  --elev-cpt  .../wiki-france.cpt \
  --key h_100y_42h_0c_t0047_r012_c012_s8 --out tools/patch_full.png
"""

import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_utils as pu


def load_one(row, pred_path, want_static, flood_thr, abs_tol):
    """Load + AOI-mask + threshold all arrays needed for a patch comparison figure."""
    coarse = pu.ensure_hw(np.load(row["coarse_path"]), "coarse")
    fine   = pu.ensure_hw(np.load(row["fine_path"]), "fine")
    elev   = pu.ensure_hw(np.load(row["elev_path"]), "elev")
    mask   = pu.ensure_hw(np.load(row["mask_fine_path"]), "mask").astype(np.uint8)
    pred   = pu.ensure_hw(np.load(pred_path), "pred")

    Hf, Wf = fine.shape
    Hc, Wc = coarse.shape
    scale = int(round(Hf / Hc))
    if (Hf != Hc * scale) or (Wf != Wc * scale):
        raise RuntimeError(f"Shape mismatch: fine={fine.shape}, coarse={coarse.shape}, scale={scale}")
    if pred.shape != fine.shape:
        raise RuntimeError(f"Pred shape != GT fine shape: pred={pred.shape}, fine={fine.shape}")

    coarse_up = pu.kron_upsample(coarse, scale)

    d = {"mask": mask, "elev_m": pu.apply_aoi_nan(elev, mask)}
    fine_m   = pu.apply_aoi_nan(fine, mask)
    coarse_m = pu.apply_aoi_nan(coarse_up, mask)
    pred_m   = pu.apply_aoi_nan(pred, mask)

    d["fine"]   = pu.flood_apply_threshold_zero(fine_m, flood_thr)
    d["coarse"] = pu.flood_apply_threshold_zero(coarse_m, flood_thr)
    d["pred"]   = pu.flood_apply_threshold_zero(pred_m, flood_thr)
    d["diff_cf"] = pu.diff_apply_abs_tol_zero(d["coarse"] - d["fine"], abs_tol)
    d["diff_pf"] = pu.diff_apply_abs_tol_zero(d["pred"] - d["fine"], abs_tol)

    if want_static:
        def opt(key):
            p = row.get(key, "")
            return pu.apply_aoi_nan(pu.ensure_hw(np.load(p), key), mask) if (p and os.path.exists(p)) else None
        d["rough_m"] = opt("rough_path")
        d["slope_m"] = opt("slope_path")
        d["twi_m"]   = opt("twi_path")
        d["asin_m"]  = opt("aspect_sin_path")
        d["acos_m"]  = opt("aspect_cos_path")
    return d


def _flood_vmax(fine_thr, mode, q):
    pos = fine_thr[np.isfinite(fine_thr)]
    pos = pos[pos > 0]
    if pos.size == 0:
        return 1.0
    return float(np.max(pos)) if mode == "max" else float(np.percentile(pos, q))


def render_compact(d, core, out_png, args):
    cm_elev = pu.load_gmt_cpt(args.elev_cpt, name="wiki-france")
    cm_flood = pu.make_blues_zero_white()
    cm_diff = plt.get_cmap("seismic").copy(); cm_diff.set_bad("white")
    elev_ticks = pu.make_elev_ticks(args.elev_vmin, args.elev_vmax, step=100)

    vmax_f = max(_flood_vmax(d["fine"], args.flood_vmax_mode, args.flood_q), args.flood_thr)
    vabs_cf = pu.percentile_abs_nonzero(d["diff_cf"], args.err_q, args.abs_tol)
    vabs_pf = pu.percentile_abs_nonzero(d["diff_pf"], args.err_q, args.abs_tol)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.ravel()
    pu.imshow_with_cb(fig, axes[0], d["elev_m"], "DEM Patch", cm_elev, args.elev_vmin, args.elev_vmax,
                      contour_mask=d["mask"], cb_ticks=elev_ticks)
    pu.imshow_with_cb(fig, axes[1], d["coarse"], "Simulated Coarse-grid Flood Map Patch", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[2], d["fine"], "Simulated Fine-grid Flood Map Patch", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[3], d["pred"], "Predicted Fine-grid Flood Map Patch", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[4], d["diff_cf"], "Simulated Coarse - Simulated Fine", cm_diff,
                      -vabs_cf, vabs_cf, contour_mask=d["mask"])
    pu.imshow_with_cb(fig, axes[5], d["diff_pf"], "Predicted Fine - Simulated Fine", cm_diff,
                      -vabs_pf, vabs_pf, contour_mask=d["mask"])
    fig.suptitle(core, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180); plt.close(fig)
    print("[OK] saved:", out_png)


def render_full(d, core, out_png, args):
    cm_flood = pu.make_blues_zero_white()
    cm_diff = plt.get_cmap("seismic").copy(); cm_diff.set_bad("white")
    cm_static = plt.get_cmap("viridis").copy(); cm_static.set_bad("white")
    cm_aspect = plt.get_cmap("RdBu_r").copy(); cm_aspect.set_bad("white")
    cm_mask = plt.get_cmap("gray_r").copy(); cm_mask.set_bad("white")
    cm_elev = pu.load_gmt_cpt(args.elev_cpt, name="wiki-france") if args.elev_cpt \
        else (plt.get_cmap("terrain").copy())
    elev_ticks = pu.make_elev_ticks(args.elev_vmin, args.elev_vmax, step=100)

    vmax_f = max(_flood_vmax(d["fine"], args.flood_vmax_mode, args.flood_q), args.flood_thr)
    vabs_cf = pu.percentile_abs_nonzero(d["diff_cf"], args.err_q, args.abs_tol)
    vabs_pf = pu.percentile_abs_nonzero(d["diff_pf"], args.err_q, args.abs_tol)

    m = __import__("re").match(r"^(\d+)x(\d+)$", args.layout.strip())
    nrow, ncol = (int(m.group(1)), int(m.group(2))) if m else (3, 4)
    if nrow * ncol != 12:
        nrow, ncol = 3, 4
    figsize = (22, 14) if (nrow, ncol) == (3, 4) else (18, 18)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize); axes = axes.ravel()

    pu.imshow_with_cb(fig, axes[0], d["elev_m"], "Elevation", cm_elev, args.elev_vmin, args.elev_vmax,
                      contour_mask=d["mask"], cb_ticks=elev_ticks)

    def static_panel(ax, arr, title):
        if arr is None:
            ax.axis("off"); ax.set_title(f"{title} (missing)"); return
        vmin, vmax = pu.get_static_vmin_vmax(arr, args.static_scale, args.static_qlo, args.static_qhi)
        pu.imshow_with_cb(fig, ax, arr, title, cm_static, vmin, vmax, contour_mask=d["mask"])

    static_panel(axes[1], d.get("rough_m"), "Roughness")
    static_panel(axes[2], d.get("slope_m"), "Slope")
    static_panel(axes[3], d.get("twi_m"), "TWI")
    for ax, arr, title in [(axes[4], d.get("asin_m"), "Aspect Sin (Eastness)"),
                           (axes[5], d.get("acos_m"), "Aspect Cos (Northness)")]:
        if arr is None:
            ax.axis("off"); ax.set_title(f"{title} (missing)")
        else:
            pu.imshow_with_cb(fig, ax, arr, title, cm_aspect, -1.0, 1.0, contour_mask=d["mask"])
    pu.imshow_with_cb(fig, axes[6], d["mask"], "Fine-grid Mask", cm_mask, 0.0, 1.0)

    pu.imshow_with_cb(fig, axes[7], d["coarse"], "Simulated Coarse-grid Flood Map", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[8], d["fine"], "Simulated Fine-grid Flood Map", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[9], d["pred"], "Predicted Fine-grid Flood Map", cm_flood,
                      0.0, vmax_f, contour_mask=d["mask"], scientific=True)
    pu.imshow_with_cb(fig, axes[10], d["diff_cf"], "Simulated Coarse - Simulated Fine", cm_diff,
                      -vabs_cf, vabs_cf, contour_mask=d["mask"])
    pu.imshow_with_cb(fig, axes[11], d["diff_pf"], "Predicted Fine - Simulated Fine", cm_diff,
                      -vabs_pf, vabs_pf, contour_mask=d["mask"])
    fig.suptitle(core, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=160); plt.close(fig)
    print("[OK] saved:", out_png)


def resolve_targets(args, idx):
    """Return list of (core, row)."""
    if args.key:
        core = pu.parse_core_key(args.key.strip())
        if core not in idx:
            raise RuntimeError(f"core not found in index.csv: {core}")
        return [(core, idx[core])]
    if args.scenario and args.time and (args.row is not None) and (args.col is not None):
        t = args.time if args.time.startswith("t") else f"t{int(args.time):04d}"
        core = f"{args.var}_{args.scenario}_{t}_r{int(args.row):03d}_c{int(args.col):03d}_s{int(args.scale)}"
        if core not in idx:
            raise RuntimeError(f"core not found in index.csv: {core}")
        return [(core, idx[core])]
    # batch
    if not args.out_dir:
        raise RuntimeError("Provide --key, or --scenario/--time/--row/--col, or --out-dir for batch.")
    targets = []
    for f in sorted(__import__("glob").glob(os.path.join(args.vis_root, "*_coarse"))):
        core = pu.parse_folder_name(os.path.basename(f.rstrip("/\\")))
        if core is None or core not in idx:
            continue
        if args.scenario and idx[core].get("scenario") != args.scenario:
            continue
        targets.append((core, idx[core]))
        if args.limit and len(targets) >= args.limit:
            break
    return targets


def main():
    ap = argparse.ArgumentParser("Per-patch comparison figures (full 12-panel or compact 2x3).")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)
    ap.add_argument("--panels", choices=["full", "compact"], default="compact")

    # selection
    ap.add_argument("--key", default="")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--var", default="h", choices=["h", "zs", "u", "v"])
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--time", default="")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--col", type=int, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=0)

    # display
    ap.add_argument("--layout", default="3x4", help="full mode only: 3x4 or 4x3")
    ap.add_argument("--elev-cpt", default="")
    ap.add_argument("--elev-vmin", type=float, default=-500.0)
    ap.add_argument("--elev-vmax", type=float, default=500.0)
    ap.add_argument("--static-scale", choices=["q1q99", "minmax"], default="q1q99")
    ap.add_argument("--static-qlo", type=float, default=1.0)
    ap.add_argument("--static-qhi", type=float, default=99.0)
    ap.add_argument("--flood-thr", type=float, default=0.05)
    ap.add_argument("--flood-vmax-mode", choices=["q95", "max"], default="q95")
    ap.add_argument("--flood-q", type=float, default=99.0)
    ap.add_argument("--abs-tol", type=float, default=0.01)
    ap.add_argument("--err-q", type=float, default=99.0)
    args = ap.parse_args()

    if args.panels == "compact" and not args.elev_cpt:
        raise SystemExit("[plot_patch_compare] --elev-cpt is required for the DEM panel.")

    idx = pu.build_index_map(args.index_csv)
    targets = resolve_targets(args, idx)
    if not targets:
        raise SystemExit("[plot_patch_compare] no targets matched.")

    single = bool(args.key) or (args.scenario and args.time and args.row is not None and args.col is not None)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    n = 0
    for core, row in targets:
        try:
            folder = os.path.join(args.vis_root, core + "_coarse")
            pred_path = pu.pick_pred_file_in_folder(folder, core)
            d = load_one(row, pred_path, want_static=(args.panels == "full"),
                         flood_thr=args.flood_thr, abs_tol=args.abs_tol)
            out_png = args.out if (single and args.out) else os.path.join(
                args.out_dir or ".", f"{core}_{args.panels}.png")
            (render_full if args.panels == "full" else render_compact)(d, core, out_png, args)
            n += 1
        except Exception as e:
            print(f"[warn] skip {core}: {e}")
    print(f"[OK] done: {n} figure(s).")


if __name__ == "__main__":
    main()
