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


def make_blues_zero_white():
    """Blues colormap, force 0 to look white. NaN stays white, matching eval_plot.py style."""
    base = plt.get_cmap("Blues", 256)
    cols = base(np.linspace(0, 1, 256))
    cols[0] = np.array([1, 1, 1, 1], dtype=np.float64)  # 0 -> white
    cm = ListedColormap(cols, name="BluesZeroWhite")
    cm.set_bad(color="white")
    return cm


def parse_folder_name(folder_basename: str):
    m = FOLDER_RE.match(folder_basename)
    if not m:
        return None
    core = m.group("core")
    if CORE_RE.match(core):
        return core
    return None


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


def build_time_groups(index_csv: str, scenario: str, var: str, scale_expected: int):
    groups = defaultdict(list)

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
            groups[row["t"]].append(row)

    return dict(groups)


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
    """AOI-masked array: values < thr -> 0 (outside AOI stays NaN)."""
    if thr is None or thr <= 0:
        return a
    return np.where(np.isfinite(a) & (a < thr), 0.0, a)


def diff_apply_abs_tol_zero(d: np.ndarray, abs_tol: float) -> np.ndarray:
    """AOI-masked array: |d| < abs_tol -> 0 (outside AOI stays NaN)."""
    if abs_tol is None or abs_tol <= 0:
        return d
    return np.where(np.isfinite(d) & (np.abs(d) < abs_tol), 0.0, d)


def flood_zero_to_nan(a: np.ndarray, thr: float) -> np.ndarray:
    """
    Presentation mode only:
    values <= threshold become NaN so hillshade can show through.
    """
    out = np.array(a, copy=True, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    out[out <= thr] = np.nan
    return out


def apply_aoi_nan(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask == 1, arr, np.nan)


def kron_upsample(coarse: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(coarse, np.ones((scale, scale), dtype=coarse.dtype))


def compute_hillshade(Z, azdeg=315.0, altdeg=45.0):
    Z = np.asarray(Z, dtype=np.float32)

    gy, gx = np.gradient(Z)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)

    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)

    shaded = (
        np.sin(alt) * np.sin(slope) +
        np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )

    hs = (shaded + 1.0) / 2.0
    hs = np.clip(hs, 0.0, 1.0)
    return hs.astype(np.float32)


def assemble_static_from_rows(rows, field_key, expected_patch_size):
    rows = sorted(rows, key=lambda r: (int(r["patch_row"]), int(r["patch_col"])))

    max_r = max(int(r["patch_row"]) for r in rows)
    max_c = max(int(r["patch_col"]) for r in rows)

    n_rows = max_r + 1
    n_cols = max_c + 1

    H = n_rows * expected_patch_size
    W = n_cols * expected_patch_size

    canvas = np.full((H, W), np.nan, dtype=np.float32)

    for row in rows:
        r = int(row["patch_row"])
        c = int(row["patch_col"])

        y0 = r * expected_patch_size
        y1 = y0 + expected_patch_size
        x0 = c * expected_patch_size
        x1 = x0 + expected_patch_size

        p = row[field_key]
        if not p or (not os.path.exists(p)):
            raise RuntimeError(f"Missing file for {field_key}: {p}")

        block = ensure_hw(np.load(p), field_key)
        if block.shape != (expected_patch_size, expected_patch_size):
            raise RuntimeError(
                f"Bad patch shape for {field_key}: {block.shape}, expected {(expected_patch_size, expected_patch_size)}"
            )

        canvas[y0:y1, x0:x1] = block

    return canvas


def assemble_for_time(
    rows,
    vis_root,
    flood_thr=0.05,
    flood_q=99.0,
    abs_tol=0.01,
    err_q=99.0,
    flood_zero_transparent=False,
):
    if not rows:
        raise RuntimeError("No rows for this timestep.")

    rows = sorted(rows, key=lambda r: (int(r["patch_row"]), int(r["patch_col"])))

    patch_fine = int(rows[0]["patch_size_fine"])
    patch_coarse = int(rows[0]["patch_size_coarse"])
    scale = int(rows[0]["downscale"])

    max_r = max(int(r["patch_row"]) for r in rows)
    max_c = max(int(r["patch_col"]) for r in rows)

    n_rows = max_r + 1
    n_cols = max_c + 1

    Hf = n_rows * patch_fine
    Wf = n_cols * patch_fine
    Hc = n_rows * patch_coarse
    Wc = n_cols * patch_coarse

    sim_fine = np.full((Hf, Wf), np.nan, dtype=np.float32)
    pred_fine = np.full((Hf, Wf), np.nan, dtype=np.float32)
    sim_coarse = np.full((Hc, Wc), np.nan, dtype=np.float32)
    mask_fine = np.zeros((Hf, Wf), dtype=np.uint8)

    for row in rows:
        r = int(row["patch_row"])
        c = int(row["patch_col"])

        y0f = r * patch_fine
        y1f = y0f + patch_fine
        x0f = c * patch_fine
        x1f = x0f + patch_fine

        y0c = r * patch_coarse
        y1c = y0c + patch_coarse
        x0c = c * patch_coarse
        x1c = x0c + patch_coarse

        fine_path = row["fine_path"]
        coarse_path = row["coarse_path"]
        mask_path = row["mask_fine_path"]

        fine_patch = ensure_hw(np.load(fine_path), "fine_patch")
        coarse_patch = ensure_hw(np.load(coarse_path), "coarse_patch")
        mask_patch = ensure_hw(np.load(mask_path), "mask_patch").astype(np.uint8)

        core = f'{row["var"]}_{row["scenario"]}_{row["t"]}_r{r:03d}_c{c:03d}_s{scale}'
        pred_folder = os.path.join(vis_root, core + "_coarse")
        if not os.path.isdir(pred_folder):
            raise RuntimeError(f"Prediction folder not found: {pred_folder}")

        pred_path = pick_pred_file_in_folder(pred_folder, core)
        pred_patch = ensure_hw(np.load(pred_path), "pred_patch")

        if fine_patch.shape != (patch_fine, patch_fine):
            raise RuntimeError(f"Bad fine patch shape: {fine_patch.shape}, expected {(patch_fine, patch_fine)}")
        if pred_patch.shape != (patch_fine, patch_fine):
            raise RuntimeError(f"Bad pred patch shape: {pred_patch.shape}, expected {(patch_fine, patch_fine)}")
        if coarse_patch.shape != (patch_coarse, patch_coarse):
            raise RuntimeError(f"Bad coarse patch shape: {coarse_patch.shape}, expected {(patch_coarse, patch_coarse)}")
        if mask_patch.shape != (patch_fine, patch_fine):
            raise RuntimeError(f"Bad mask patch shape: {mask_patch.shape}, expected {(patch_fine, patch_fine)}")

        sim_fine[y0f:y1f, x0f:x1f] = fine_patch
        pred_fine[y0f:y1f, x0f:x1f] = pred_patch
        sim_coarse[y0c:y1c, x0c:x1c] = coarse_patch
        mask_fine[y0f:y1f, x0f:x1f] = mask_patch

    sim_coarse_up = kron_upsample(sim_coarse, scale).astype(np.float32)

    # Match eval_plot.py logic: AOI outside -> NaN first
    sim_fine_m = apply_aoi_nan(sim_fine, mask_fine)
    pred_fine_m = apply_aoi_nan(pred_fine, mask_fine)
    sim_coarse_m = apply_aoi_nan(sim_coarse_up, mask_fine)

    # Then apply threshold inside AOI only
    sim_fine_thr = flood_apply_threshold_zero(sim_fine_m, flood_thr)
    pred_fine_thr = flood_apply_threshold_zero(pred_fine_m, flood_thr)
    sim_coarse_thr = flood_apply_threshold_zero(sim_coarse_m, flood_thr)

    # Flood vmax: ONLY from simulated fine after threshold, only flooded cells
    flood_vmin = 0.0
    flood_vmax = percentile_positive(sim_fine_thr, flood_q)
    flood_vmax = max(flood_vmax, flood_thr)

    # Error maps follow thresholded flood maps
    err_coarse_minus_fine = sim_coarse_thr - sim_fine_thr
    err_pred_minus_fine = pred_fine_thr - sim_fine_thr

    err_coarse_minus_fine = diff_apply_abs_tol_zero(err_coarse_minus_fine, abs_tol)
    err_pred_minus_fine = diff_apply_abs_tol_zero(err_pred_minus_fine, abs_tol)

    err_vabs_coarse = percentile_abs_nonzero(err_coarse_minus_fine, err_q, abs_tol)
    err_vabs_pred = percentile_abs_nonzero(err_pred_minus_fine, err_q, abs_tol)

    # Optional presentation mode
    if flood_zero_transparent:
        sim_fine_plot = flood_zero_to_nan(sim_fine_thr, flood_thr)
        pred_fine_plot = flood_zero_to_nan(pred_fine_thr, flood_thr)
        sim_coarse_plot = flood_zero_to_nan(sim_coarse_thr, flood_thr)
    else:
        sim_fine_plot = sim_fine_thr
        pred_fine_plot = pred_fine_thr
        sim_coarse_plot = sim_coarse_thr

    return {
        "mask_fine": mask_fine.astype(np.uint8),
        "sim_coarse_plot": sim_coarse_plot.astype(np.float32),
        "sim_fine_plot": sim_fine_plot.astype(np.float32),
        "pred_fine_plot": pred_fine_plot.astype(np.float32),
        "err_coarse_minus_fine": err_coarse_minus_fine.astype(np.float32),
        "err_pred_minus_fine": err_pred_minus_fine.astype(np.float32),
        "flood_vmin": float(flood_vmin),
        "flood_vmax": float(flood_vmax),
        "err_vabs_coarse": float(err_vabs_coarse),
        "err_vabs_pred": float(err_vabs_pred),
    }


def add_colorbar(fig, im, ax, label="", scientific=False):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label)
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

    return cb


def draw_map(
    arr,
    out_png,
    title,
    cmap,
    vmin,
    vmax,
    contour_mask=None,
    cbar_label="",
    scientific=False,
    hillshade=None,
    hillshade_alpha=0.25,
):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)

    if hillshade is not None:
        ax.imshow(hillshade, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, alpha=hillshade_alpha)

    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if contour_mask is not None:
        ax.contour(contour_mask, levels=[0.5], colors="k", linewidths=0.8)

    add_colorbar(fig, im, ax, label=cbar_label, scientific=scientific)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


def main():
    ap = argparse.ArgumentParser("Assemble full flood maps from saved patches")

    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--vis-root", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--var", default="h", choices=["h", "zs", "u", "v"])
    ap.add_argument("--scale", type=int, default=16)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--time-start", type=int, default=0)
    ap.add_argument("--time-end", type=int, default=47)

    ap.add_argument("--flood-thr", type=float, default=0.05)
    ap.add_argument("--flood-q", type=float, default=99.0)

    ap.add_argument("--abs-tol", type=float, default=0.01)
    ap.add_argument("--err-q", type=float, default=99.0)

    ap.add_argument("--show-hillshade", action="store_true")
    ap.add_argument("--flood-zero-transparent", action="store_true")
    ap.add_argument("--hillshade-alpha", type=float, default=0.25)

    args = ap.parse_args()

    groups = build_time_groups(
        index_csv=args.index_csv,
        scenario=args.scenario,
        var=args.var,
        scale_expected=args.scale,
    )
    if not groups:
        raise RuntimeError("No valid rows found in index.csv for the requested scenario/var.")

    all_rows = []
    for t_tag in sorted(groups.keys()):
        all_rows.extend(groups[t_tag])

    if not all_rows:
        raise RuntimeError("No rows available.")

    patch_fine = int(all_rows[0]["patch_size_fine"])

    hillshade = None
    if args.show_hillshade:
        dem_assembled = assemble_static_from_rows(all_rows, "elev_path", patch_fine)
        dem_for_hs = np.nan_to_num(dem_assembled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        hillshade = compute_hillshade(dem_for_hs)

    cm_flood = make_blues_zero_white()
    cm_diff = plt.get_cmap("seismic").copy()
    cm_diff.set_bad("white")

    os.makedirs(args.out_dir, exist_ok=True)

    for ti in range(args.time_start, args.time_end + 1):
        t_tag = f"t{ti:04d}"
        if t_tag not in groups:
            print(f"[warn] skip {t_tag}: no rows found")
            continue

        assembled = assemble_for_time(
            rows=groups[t_tag],
            vis_root=args.vis_root,
            flood_thr=args.flood_thr,
            flood_q=args.flood_q,
            abs_tol=args.abs_tol,
            err_q=args.err_q,
            flood_zero_transparent=args.flood_zero_transparent,
        )

        contour_mask = assembled["mask_fine"]

        subdir = os.path.join(args.out_dir, t_tag)
        os.makedirs(subdir, exist_ok=True)

        draw_map(
            assembled["sim_coarse_plot"],
            os.path.join(subdir, f"{t_tag}_sim_coarse.png"),
            title=f"Simulated Coarse-grid Flood Map ({args.scenario}_{t_tag})",
            cmap=cm_flood,
            vmin=assembled["flood_vmin"],
            vmax=assembled["flood_vmax"],
            contour_mask=contour_mask,
            cbar_label="Water depth [m]",
            scientific=True,
            hillshade=hillshade,
            hillshade_alpha=args.hillshade_alpha,
        )

        draw_map(
            assembled["sim_fine_plot"],
            os.path.join(subdir, f"{t_tag}_sim_fine.png"),
            title=f"Simulated Fine-grid Flood Map ({args.scenario}_{t_tag})",
            cmap=cm_flood,
            vmin=assembled["flood_vmin"],
            vmax=assembled["flood_vmax"],
            contour_mask=contour_mask,
            cbar_label="Water depth [m]",
            scientific=True,
            hillshade=hillshade,
            hillshade_alpha=args.hillshade_alpha,
        )

        draw_map(
            assembled["pred_fine_plot"],
            os.path.join(subdir, f"{t_tag}_pred_fine.png"),
            title=f"Predicted Fine-grid Flood Map ({args.scenario}_{t_tag})",
            cmap=cm_flood,
            vmin=assembled["flood_vmin"],
            vmax=assembled["flood_vmax"],
            contour_mask=contour_mask,
            cbar_label="Water depth [m]",
            scientific=True,
            hillshade=hillshade,
            hillshade_alpha=args.hillshade_alpha,
        )

        draw_map(
            assembled["err_coarse_minus_fine"],
            os.path.join(subdir, f"{t_tag}_err_coarse_minus_fine.png"),
            title=f"Simulated Coarse - Simulated Fine ({args.scenario}_{t_tag})",
            cmap=cm_diff,
            vmin=-assembled["err_vabs_coarse"],
            vmax=assembled["err_vabs_coarse"],
            contour_mask=contour_mask,
            cbar_label="Error [m]",
            scientific=False,
            hillshade=hillshade,
            hillshade_alpha=args.hillshade_alpha,
        )

        draw_map(
            assembled["err_pred_minus_fine"],
            os.path.join(subdir, f"{t_tag}_err_pred_minus_fine.png"),
            title=f"Predicted Fine - Simulated Fine ({args.scenario}_{t_tag})",
            cmap=cm_diff,
            vmin=-assembled["err_vabs_pred"],
            vmax=assembled["err_vabs_pred"],
            contour_mask=contour_mask,
            cbar_label="Error [m]",
            scientific=False,
            hillshade=hillshade,
            hillshade_alpha=args.hillshade_alpha,
        )

        print(f"[OK] finished {t_tag}")


if __name__ == "__main__":
    main()