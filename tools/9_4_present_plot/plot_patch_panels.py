import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap


CORE_RE = re.compile(
    r'^(?P<var>h|zs|u|v)_(?P<scenario>\d+y_\d+h_\d+c)_(?P<t>t\d{4})_r(?P<r>\d{3})_c(?P<c>\d{3})_s(?P<s>\d+)$'
)
FOLDER_RE = re.compile(r'^(?P<core>.+)_coarse$')


def ensure_hw(arr: np.ndarray, name: str = "arr") -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0, ...]
    if a.ndim == 4:
        return a[0, 0, ...]
    raise RuntimeError(f"{name} has unsupported shape: {a.shape}")


def load_gmt_cpt(cpt_path: str, name: str = "cpt"):
    if not cpt_path or (not os.path.exists(cpt_path)):
        raise RuntimeError(f"CPT not found: {cpt_path}")

    xs, cols = [], []
    with open(cpt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            if line[0] in "BFN":
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                x1, r1, g1, b1, x2, r2, g2, b2 = parts[:8]
                x1 = float(x1)
                x2 = float(x2)
                c1 = (float(r1) / 255.0, float(g1) / 255.0, float(b1) / 255.0)
                c2 = (float(r2) / 255.0, float(g2) / 255.0, float(b2) / 255.0)
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

    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, list(zip(ux, cols)))
    cm.set_bad("white")
    return cm


def make_blues_zero_white():
    base = plt.get_cmap("Blues", 256)
    cols = base(np.linspace(0, 1, 256))
    cols[0] = np.array([1, 1, 1, 1], dtype=np.float64)
    cm = ListedColormap(cols, name="BluesZeroWhite")
    cm.set_bad(color="white")
    return cm


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

    cand = []
    for p in npys:
        bn = os.path.basename(p)
        if _ok(p) and bn.startswith(core + "_"):
            cand.append(p)

    if cand:
        cand.sort(key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        return cand[0]

    if len(npys) == 1:
        return npys[0]

    raise RuntimeError(f"Cannot identify predicted .npy in {folder} for core={core}")


def build_patch_index(index_csv: str, scenario: str, var: str, scale_expected: int):
    mp = {}
    rows_by_time = defaultdict(list)

    with open(index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scenario"] != scenario:
                continue
            if row["var"] != var:
                continue
            if int(row["filtered_out"]) != 0:
                continue
            if int(row["downscale"]) != scale_expected:
                continue

            t = row["t"]
            r = int(row["patch_row"])
            c = int(row["patch_col"])
            key = (t, r, c)
            mp[key] = row
            rows_by_time[t].append(row)

    return mp, dict(rows_by_time)


def percentile_positive(arr: np.ndarray, q: float):
    v = arr[np.isfinite(arr)]
    v = v[v > 0]
    if v.size == 0:
        return 1.0
    out = float(np.percentile(v, q))
    if (not np.isfinite(out)) or (out <= 0):
        out = float(np.max(v))
    return max(out, 1e-8)


def percentile_abs_nonzero(arr: np.ndarray, q: float, fallback: float):
    v = arr[np.isfinite(arr)]
    v = np.abs(v)
    v = v[v > 0]
    if v.size == 0:
        return max(fallback, 1e-6)
    out = float(np.percentile(v, q))
    if (not np.isfinite(out)) or (out <= 0):
        out = float(np.max(v))
    return max(out, fallback, 1e-6)


def flood_apply_threshold_zero(a: np.ndarray, thr: float) -> np.ndarray:
    if thr is None or thr <= 0:
        return a
    return np.where(np.isfinite(a) & (a < thr), 0.0, a)


def diff_apply_abs_tol_zero(d: np.ndarray, abs_tol: float) -> np.ndarray:
    if abs_tol is None or abs_tol <= 0:
        return d
    return np.where(np.isfinite(d) & (np.abs(d) < abs_tol), 0.0, d)


def apply_aoi_nan(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask == 1, arr, np.nan)


def kron_upsample(coarse: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(coarse, np.ones((scale, scale), dtype=coarse.dtype))


def make_elev_ticks(vmin: float, vmax: float, step: int = 100):
    if vmin is None or vmax is None or (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return None
    t0 = int(np.ceil(vmin / step) * step)
    t1 = int(np.floor(vmax / step) * step)
    if t1 < t0:
        return None
    return list(range(t0, t1 + step, step))


def add_colorbar(fig, im, ax, scientific=False, ticks=None):
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
    ax.set_xticks([])
    ax.set_yticks([])
    if contour_mask is not None:
        ax.contour(contour_mask, levels=[0.5], colors="k", linewidths=0.8)
    add_colorbar(fig, im, ax, scientific=scientific, ticks=cb_ticks)
    return im


def plot_one_patch_panel(
    row: dict,
    vis_root: str,
    out_png: str,
    elev_cpt: str,
    elev_vmin: float,
    elev_vmax: float,
    flood_thr: float,
    flood_q: float,
    abs_tol: float,
    err_q: float,
):
    var = row["var"]
    scenario = row["scenario"]
    t = row["t"]
    r = int(row["patch_row"])
    c = int(row["patch_col"])
    scale = int(row["downscale"])

    core = f"{var}_{scenario}_{t}_r{r:03d}_c{c:03d}_s{scale}"

    coarse = ensure_hw(np.load(row["coarse_path"]), "coarse")
    fine = ensure_hw(np.load(row["fine_path"]), "fine")
    elev = ensure_hw(np.load(row["elev_path"]), "elev")
    mask = ensure_hw(np.load(row["mask_fine_path"]), "mask").astype(np.uint8)

    pred_folder = os.path.join(vis_root, core + "_coarse")
    if not os.path.isdir(pred_folder):
        raise RuntimeError(f"Prediction folder not found: {pred_folder}")
    pred_path = pick_pred_file_in_folder(pred_folder, core)
    pred = ensure_hw(np.load(pred_path), "pred")

    Hf, Wf = fine.shape
    Hc, Wc = coarse.shape
    scale2 = int(round(Hf / Hc))
    if (Hf != Hc * scale2) or (Wf != Wc * scale2):
        raise RuntimeError(f"Shape mismatch: fine={fine.shape}, coarse={coarse.shape}, inferred scale={scale2}")
    if pred.shape != fine.shape:
        raise RuntimeError(f"Pred shape != GT fine shape: pred={pred.shape}, fine={fine.shape}")

    coarse_up = kron_upsample(coarse, scale2)

    # AOI outside -> NaN first
    elev_m = apply_aoi_nan(elev, mask)
    fine_m = apply_aoi_nan(fine, mask)
    coarse_m = apply_aoi_nan(coarse_up, mask)
    pred_m = apply_aoi_nan(pred, mask)

    # threshold after AOI masking
    fine_thr = flood_apply_threshold_zero(fine_m, flood_thr)
    coarse_thr = flood_apply_threshold_zero(coarse_m, flood_thr)
    pred_thr = flood_apply_threshold_zero(pred_m, flood_thr)

    # flood vmax: ONLY from simulated fine after threshold, only flooded cells
    flood_vmin = 0.0
    flood_vmax = percentile_positive(fine_thr, flood_q)
    flood_vmax = max(flood_vmax, flood_thr)

    # error maps
    diff_cf = coarse_thr - fine_thr
    diff_pf = pred_thr - fine_thr

    diff_cf = diff_apply_abs_tol_zero(diff_cf, abs_tol)
    diff_pf = diff_apply_abs_tol_zero(diff_pf, abs_tol)

    err_vabs_cf = percentile_abs_nonzero(diff_cf, err_q, abs_tol)
    err_vabs_pf = percentile_abs_nonzero(diff_pf, err_q, abs_tol)

    cm_elev = load_gmt_cpt(elev_cpt, name="wiki-france")
    cm_flood = make_blues_zero_white()
    cm_diff = plt.get_cmap("seismic").copy()
    cm_diff.set_bad("white")

    elev_ticks = make_elev_ticks(elev_vmin, elev_vmax, step=100)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    imshow_with_cb(
        fig, axes[0], elev_m,
        "DEM Patch", cm_elev,
        elev_vmin, elev_vmax,
        contour_mask=mask, scientific=False, cb_ticks=elev_ticks
    )

    imshow_with_cb(
        fig, axes[1], coarse_thr,
        "Simulated Coarse-grid Flood Map Patch", cm_flood,
        flood_vmin, flood_vmax,
        contour_mask=mask, scientific=True
    )

    imshow_with_cb(
        fig, axes[2], fine_thr,
        "Simulated Fine-grid Flood Map Patch", cm_flood,
        flood_vmin, flood_vmax,
        contour_mask=mask, scientific=True
    )

    imshow_with_cb(
        fig, axes[3], pred_thr,
        "Predicted Fine-grid Flood Map Patch", cm_flood,
        flood_vmin, flood_vmax,
        contour_mask=mask, scientific=True
    )

    imshow_with_cb(
        fig, axes[4], diff_cf,
        "Simulated Coarse - Simulated Fine", cm_diff,
        -err_vabs_cf, err_vabs_cf,
        contour_mask=mask, scientific=False
    )

    imshow_with_cb(
        fig, axes[5], diff_pf,
        "Predicted Fine - Simulated Fine", cm_diff,
        -err_vabs_pf, err_vabs_pf,
        contour_mask=mask, scientific=False
    )

    fig.suptitle(core, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


def main():
    ap = argparse.ArgumentParser("Plot patch-level 2x3 panels")

    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--var", default="h", choices=["h", "zs", "u", "v"])
    ap.add_argument("--scale", type=int, default=16)

    ap.add_argument("--elev-cpt", required=True)
    ap.add_argument("--elev-vmin", type=float, default=-500.0)
    ap.add_argument("--elev-vmax", type=float, default=500.0)

    ap.add_argument("--flood-thr", type=float, default=0.05)
    ap.add_argument("--flood-q", type=float, default=99.0)
    ap.add_argument("--abs-tol", type=float, default=0.01)
    ap.add_argument("--err-q", type=float, default=99.0)

    # single patch mode
    ap.add_argument("--time", default="")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--col", type=int, default=None)
    ap.add_argument("--out", default="")

    # batch mode
    ap.add_argument("--time-start", type=int, default=None)
    ap.add_argument("--time-end", type=int, default=None)
    ap.add_argument("--out-dir", default="")

    args = ap.parse_args()

    patch_map, rows_by_time = build_patch_index(
        index_csv=args.index_csv,
        scenario=args.scenario,
        var=args.var,
        scale_expected=args.scale,
    )

    if not patch_map:
        raise RuntimeError("No valid patch rows found for the requested scenario/var/scale.")

    # single patch mode
    if args.time and (args.row is not None) and (args.col is not None):
        t_tag = args.time
        if not t_tag.startswith("t"):
            t_tag = f"t{int(t_tag):04d}"

        key = (t_tag, int(args.row), int(args.col))
        if key not in patch_map:
            raise RuntimeError(f"Patch not found: time={t_tag}, row={args.row}, col={args.col}")

        out_png = args.out if args.out else f"{args.var}_{args.scenario}_{t_tag}_r{args.row:03d}_c{args.col:03d}_panel.png"

        plot_one_patch_panel(
            row=patch_map[key],
            vis_root=args.vis_root,
            out_png=out_png,
            elev_cpt=args.elev_cpt,
            elev_vmin=args.elev_vmin,
            elev_vmax=args.elev_vmax,
            flood_thr=args.flood_thr,
            flood_q=args.flood_q,
            abs_tol=args.abs_tol,
            err_q=args.err_q,
        )
        return

    # batch mode
    if (args.time_start is None) or (args.time_end is None) or (not args.out_dir):
        raise RuntimeError(
            "Use either single-patch mode (--time --row --col [--out]) "
            "or batch mode (--time-start --time-end --out-dir)."
        )

    os.makedirs(args.out_dir, exist_ok=True)

    n_done = 0
    for ti in range(args.time_start, args.time_end + 1):
        t_tag = f"t{ti:04d}"
        if t_tag not in rows_by_time:
            print(f"[warn] skip {t_tag}: no rows found")
            continue

        rows_sorted = sorted(
            rows_by_time[t_tag],
            key=lambda x: (int(x["patch_row"]), int(x["patch_col"]))
        )

        for row in rows_sorted:
            rr = int(row["patch_row"])
            cc = int(row["patch_col"])
            out_png = os.path.join(
                args.out_dir,
                t_tag,
                f'{args.var}_{args.scenario}_{t_tag}_r{rr:03d}_c{cc:03d}_panel.png'
            )

            try:
                plot_one_patch_panel(
                    row=row,
                    vis_root=args.vis_root,
                    out_png=out_png,
                    elev_cpt=args.elev_cpt,
                    elev_vmin=args.elev_vmin,
                    elev_vmax=args.elev_vmax,
                    flood_thr=args.flood_thr,
                    flood_q=args.flood_q,
                    abs_tol=args.abs_tol,
                    err_q=args.err_q,
                )
                n_done += 1
            except Exception as e:
                print(f"[warn] skip {t_tag} r{rr:03d} c{cc:03d}: {e}")

    print(f"[OK] batch done: {n_done} patch panels -> {args.out_dir}")


if __name__ == "__main__":
    main()