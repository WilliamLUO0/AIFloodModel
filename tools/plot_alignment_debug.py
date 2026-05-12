#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


def masked_stats(arr, mask):
    valid = mask.astype(bool) & np.isfinite(arr)
    if not np.any(valid):
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
        }
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


def shift_array_with_nan(arr, dy, dx):
    """
    Shift arr by dy, dx.
    Positive dy shifts downward.
    Positive dx shifts rightward.
    Empty areas are filled with NaN.
    """
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

            fine_wet = (shifted >= threshold)
            coarse_wet = (coarse >= threshold)

            scores = binary_scores(
                pred_wet=coarse_wet,
                target_wet=fine_wet,
                mask=valid,
            )

            item = {
                "dy": dy,
                "dx": dx,
                **scores,
            }

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


def plot_one(npz_path, out_dir, threshold=0.1, max_shift=3):
    data = np.load(npz_path, allow_pickle=True)

    coarse = data["coarse"].astype(np.float32)
    fine_down = data["fine_down"].astype(np.float32)
    diff = data["diff"].astype(np.float32)
    mask = data["mask_coarse"].astype(bool)

    scenario = str(data["scenario"])
    t_tag = str(data["t_tag"])
    patch_row = int(data["patch_row"])
    patch_col = int(data["patch_col"])

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

    vmax = np.nanmax([np.nanmax(coarse[mask]), np.nanmax(fine_down[mask]), threshold])
    vmax = max(float(vmax), threshold)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    im0 = axes[0, 0].imshow(np.where(mask, coarse, np.nan), vmin=0, vmax=vmax)
    axes[0, 0].set_title("Coarse h")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.where(mask, fine_down, np.nan), vmin=0, vmax=vmax)
    axes[0, 1].set_title("Fine h downsampled")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    dmax = np.nanmax(np.abs(diff[mask])) if np.any(mask) else 1.0
    dmax = max(float(dmax), 1e-6)
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
        f"{scenario} {t_tag} r={patch_row} c={patch_col}\n"
        f"CSI={scores['csi']:.3f}, Precision={scores['precision']:.3f}, Recall={scores['recall']:.3f}, "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}; Best shift: {best_text}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_png = os.path.join(out_dir, f"{scenario}_{t_tag}_{base}.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    row = {
        "npz_path": npz_path,
        "scenario": scenario,
        "t_tag": t_tag,
        "patch_row": patch_row,
        "patch_col": patch_col,
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
        "best_shift_dy": best["dy"] if best is not None else None,
        "best_shift_dx": best["dx"] if best is not None else None,
        "best_shift_csi": best["csi"] if best is not None else None,
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-root", required=True,
                        help="Path to _debug_alignment directory or its parent dataset directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=-1)
    args = parser.parse_args()

    debug_root = args.debug_root

    if os.path.basename(debug_root) != "_debug_alignment":
        candidate = os.path.join(debug_root, "_debug_alignment")
        if os.path.isdir(candidate):
            debug_root = candidate

    npz_files = sorted(glob.glob(os.path.join(debug_root, "**", "*.npz"), recursive=True))

    if args.max_files > 0:
        npz_files = npz_files[:args.max_files]

    if len(npz_files) == 0:
        raise RuntimeError(f"No npz files found under {debug_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for i, path in enumerate(npz_files):
        print(f"[{i+1}/{len(npz_files)}] plotting {path}")
        row = plot_one(
            npz_path=path,
            out_dir=args.out_dir,
            threshold=args.threshold,
            max_shift=args.max_shift,
        )
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "alignment_summary.csv")
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote figures to: {args.out_dir}")
    print(f"Wrote summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()