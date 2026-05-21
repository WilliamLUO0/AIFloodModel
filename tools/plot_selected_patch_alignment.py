#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_index_row(index_csv, scenario, var, timestep, patch_row, patch_col):
    """
    timestep can be:
      - t0000 / t0123
      - 0 / 123, matched against time_index
    """
    timestep = str(timestep)

    matched = []

    with open(index_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["scenario"] != scenario:
                continue
            if row["var"] != var:
                continue
            if int(row["patch_row"]) != int(patch_row):
                continue
            if int(row["patch_col"]) != int(patch_col):
                continue

            same_t_tag = row["t"] == timestep
            same_t_index = row["time_index"] == timestep

            if same_t_tag or same_t_index:
                matched.append(row)

    if len(matched) == 0:
        raise RuntimeError(
            "No matching patch found in index.csv.\n"
            f"scenario={scenario}, var={var}, timestep={timestep}, "
            f"row={patch_row}, col={patch_col}"
        )

    if len(matched) > 1:
        raise RuntimeError(
            "Multiple matching rows found. This should not happen for a single "
            "scenario/var/timestep/row/col."
        )

    row = matched[0]

    if int(row["filtered_out"]) == 1:
        raise RuntimeError(
            "This patch was filtered out by AOI filtering, so its .npy paths are empty.\n"
            f"scenario={scenario}, var={var}, timestep={timestep}, "
            f"row={patch_row}, col={patch_col}, aoi_ratio={row['aoi_ratio']}"
        )

    return row


def masked_stats(arr, mask):
    valid = mask.astype(bool) & np.isfinite(arr)
    if not np.any(valid):
        return {"min": np.nan, "max": np.nan, "mean": np.nan}

    x = arr[valid]
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def binary_scores(pred_wet, target_wet, mask):
    valid = mask.astype(bool)
    p = pred_wet & valid
    t = target_wet & valid

    tp = int(np.sum(p & t))
    fp = int(np.sum(p & (~t)))
    fn = int(np.sum((~p) & t))
    tn = int(np.sum((~p) & (~t) & valid))

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    csi = tp / (tp + fp + fn + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "csi": float(csi),
    }


def downsample_fine_to_coarse_mean(fine, scale, patch_coarse):
    """
    Downsample fine patch [patch_coarse*scale, patch_coarse*scale]
    to [patch_coarse, patch_coarse] by block mean.
    """
    fine = fine.astype(np.float32)

    patch_fine = patch_coarse * scale
    if fine.shape[0] != patch_fine or fine.shape[1] != patch_fine:
        raise RuntimeError(
            f"fine patch shape {fine.shape} does not match expected "
            f"({patch_fine}, {patch_fine}). "
            "Please check scale and patch_coarse from index.csv."
        )

    x = fine.reshape(patch_coarse, scale, patch_coarse, scale)
    return np.nanmean(x, axis=(1, 3)).astype(np.float32)


def shift_array_with_nan(arr, dy, dx):
    out = np.full_like(arr, np.nan, dtype=np.float32)
    h, w = arr.shape

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def best_shift_csi(fine_down, coarse, mask, threshold=0.1, max_shift=3):
    best = None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = shift_array_with_nan(fine_down, dy, dx)
            valid = mask.astype(bool) & np.isfinite(shifted) & np.isfinite(coarse)

            if not np.any(valid):
                continue

            fine_wet = shifted >= threshold
            coarse_wet = coarse >= threshold

            scores = binary_scores(
                pred_wet=coarse_wet,
                target_wet=fine_wet,
                mask=valid,
            )

            item = {"dy": dy, "dx": dx, **scores}

            if best is None or item["csi"] > best["csi"]:
                best = item

    return best


def make_mismatch_map(fine_wet, coarse_wet, mask):
    """
    0 = background / invalid
    1 = both wet
    2 = coarse wet only
    3 = fine wet only
    """
    valid = mask.astype(bool)
    out = np.zeros(fine_wet.shape, dtype=np.float32)

    out[(fine_wet & coarse_wet) & valid] = 1.0
    out[((~fine_wet) & coarse_wet) & valid] = 2.0
    out[(fine_wet & (~coarse_wet)) & valid] = 3.0

    return out


def safe_vmax(a, b, mask, threshold):
    valid = mask.astype(bool)

    values = []
    if np.any(valid):
        values.append(np.nanmax(a[valid]))
        values.append(np.nanmax(b[valid]))

    values.append(threshold)
    vmax = np.nanmax(values)

    if not np.isfinite(vmax):
        vmax = threshold

    return max(float(vmax), threshold)


def plot_selected_patch(row, out_dir, threshold=0.1, max_shift=3):
    scenario = row["scenario"]
    var = row["var"]
    t_tag = row["t"]
    time_index = int(row["time_index"])
    patch_row = int(row["patch_row"])
    patch_col = int(row["patch_col"])

    scale = int(row["downscale"])
    patch_coarse = int(row["patch_size_coarse"])

    coarse_path = row["coarse_path"]
    fine_path = row["fine_path"]
    mask_coarse_path = row["mask_coarse_path"]

    if not os.path.exists(coarse_path):
        raise FileNotFoundError(f"Cannot find coarse file: {coarse_path}")
    if not os.path.exists(fine_path):
        raise FileNotFoundError(f"Cannot find fine file: {fine_path}")
    if not os.path.exists(mask_coarse_path):
        raise FileNotFoundError(f"Cannot find coarse mask file: {mask_coarse_path}")

    coarse = np.load(coarse_path).astype(np.float32)
    fine = np.load(fine_path).astype(np.float32)
    mask = np.load(mask_coarse_path).astype(bool)

    fine_down = downsample_fine_to_coarse_mean(
        fine=fine,
        scale=scale,
        patch_coarse=patch_coarse,
    )

    diff = fine_down - coarse

    fine_wet = fine_down >= threshold
    coarse_wet = coarse >= threshold
    mismatch = make_mismatch_map(fine_wet, coarse_wet, mask)

    scores = binary_scores(
        pred_wet=coarse_wet,
        target_wet=fine_wet,
        mask=mask,
    )

    best = best_shift_csi(
        fine_down=fine_down,
        coarse=coarse,
        mask=mask,
        threshold=threshold,
        max_shift=max_shift,
    )

    cstats = masked_stats(coarse, mask)
    fstats = masked_stats(fine_down, mask)
    dstats = masked_stats(diff, mask)

    valid_diff = mask & np.isfinite(diff)
    if np.any(valid_diff):
        mae = float(np.mean(np.abs(diff[valid_diff])))
        rmse = float(np.sqrt(np.mean(diff[valid_diff] ** 2)))
    else:
        mae = np.nan
        rmse = np.nan

    vmax = safe_vmax(coarse, fine_down, mask, threshold)

    dmax = np.nanmax(np.abs(diff[mask])) if np.any(mask) else 1.0
    if not np.isfinite(dmax):
        dmax = 1.0
    dmax = max(float(dmax), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    im0 = axes[0, 0].imshow(np.where(mask, coarse, np.nan), vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"Coarse {var}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.where(mask, fine_down, np.nan), vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"Fine {var} downsampled")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(np.where(mask, diff, np.nan), vmin=-dmax, vmax=dmax)
    axes[0, 2].set_title("Fine_down - coarse")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(np.where(mask, coarse_wet, np.nan))
    axes[1, 0].set_title(f"Coarse wet >= {threshold} m")

    axes[1, 1].imshow(np.where(mask, fine_wet, np.nan))
    axes[1, 1].set_title(f"Fine_down wet >= {threshold} m")

    im5 = axes[1, 2].imshow(mismatch, vmin=0, vmax=3)
    axes[1, 2].set_title("Mismatch map\n1 both, 2 coarse only, 3 fine only")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    best_text = "None"
    if best is not None:
        best_text = f"dy={best['dy']}, dx={best['dx']}, CSI={best['csi']:.3f}"

    title = (
        f"{scenario} {t_tag} time_index={time_index} "
        f"r={patch_row} c={patch_col} scale={scale}\n"
        f"CSI={scores['csi']:.3f}, Precision={scores['precision']:.3f}, "
        f"Recall={scores['recall']:.3f}, MAE={mae:.4f}, RMSE={rmse:.4f}; "
        f"Best shift: {best_text}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)

    out_png = os.path.join(
        out_dir,
        f"align_{var}_{scenario}_{t_tag}_r{patch_row:03d}_c{patch_col:03d}_s{scale}.png"
    )

    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    summary = {
        "out_png": out_png,
        "scenario": scenario,
        "var": var,
        "t": t_tag,
        "time_index": time_index,
        "patch_row": patch_row,
        "patch_col": patch_col,
        "scale": scale,
        "patch_coarse": patch_coarse,
        "threshold": threshold,
        "csi": scores["csi"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "tp": scores["tp"],
        "fp": scores["fp"],
        "fn": scores["fn"],
        "tn": scores["tn"],
        "mae": mae,
        "rmse": rmse,
        "coarse_min": cstats["min"],
        "coarse_max": cstats["max"],
        "coarse_mean": cstats["mean"],
        "fine_down_min": fstats["min"],
        "fine_down_max": fstats["max"],
        "fine_down_mean": fstats["mean"],
        "diff_min": dstats["min"],
        "diff_max": dstats["max"],
        "diff_mean": dstats["mean"],
        "best_shift_dy": best["dy"] if best is not None else "",
        "best_shift_dx": best["dx"] if best is not None else "",
        "best_shift_csi": best["csi"] if best is not None else "",
    }

    return summary


def write_summary_csv(summary, out_dir):
    csv_path = os.path.join(out_dir, "selected_patch_alignment_summary.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)

    return csv_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", required=True,
                        help="Dataset root, e.g. /path/to/dataset_ds8")
    parser.add_argument("--scenario", required=True,
                        help="Scenario name, e.g. 100y_48h_0c")
    parser.add_argument("--timestep", required=True,
                        help="Either t tag, e.g. t0000, or time_index, e.g. 0")
    parser.add_argument("--row", type=int, required=True,
                        help="Patch row index")
    parser.add_argument("--col", type=int, required=True,
                        help="Patch column index")
    parser.add_argument("--var", default="h", choices=["h"],
                        help="Currently designed for h alignment plot")
    parser.add_argument("--out-dir", default="",
                        help="Output directory. Default: dataset_root/selected_alignment_figures")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Wet/dry threshold for h, default 0.1 m")
    parser.add_argument("--max-shift", type=int, default=3,
                        help="Search range for best shift CSI")

    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    index_csv = os.path.join(dataset_root, "index.csv")

    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"Cannot find index.csv: {index_csv}")

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join(dataset_root, "selected_alignment_figures")
    out_dir = os.path.abspath(out_dir)

    row = load_index_row(
        index_csv=index_csv,
        scenario=args.scenario,
        var=args.var,
        timestep=args.timestep,
        patch_row=args.row,
        patch_col=args.col,
    )

    summary = plot_selected_patch(
        row=row,
        out_dir=out_dir,
        threshold=args.threshold,
        max_shift=args.max_shift,
    )

    csv_path = write_summary_csv(summary, out_dir)

    print(f"Wrote figure to: {summary['out_png']}")
    print(f"Wrote/updated summary CSV: {csv_path}")
    print(
        f"CSI={summary['csi']:.4f}, "
        f"Precision={summary['precision']:.4f}, "
        f"Recall={summary['recall']:.4f}, "
        f"MAE={summary['mae']:.6f}, "
        f"RMSE={summary['rmse']:.6f}"
    )


if __name__ == "__main__":
    main()