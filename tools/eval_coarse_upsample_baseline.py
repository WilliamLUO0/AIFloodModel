#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import os

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from basicsr.data.paired_floodmap_dataset import PairedFloodMapDataset
from basicsr.metrics import calculate_metric
from basicsr.utils import destand_to_physical, denorm_to_physical


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_metric_opt(opt_):
    # same logic as FMSRModel.nondist_validation
    config_keys = {"better"}
    return {k: v for k, v in opt_.items() if k not in config_keys}


def accumulate_metric(metric_results, metric_counts, name, val):
    # same accumulation logic as FMSRModel.nondist_validation
    if torch.is_tensor(val):
        v = val.detach().view(-1)
        finite = torch.isfinite(v)
        if finite.any():
            metric_results[name] += v[finite].sum().item()
            metric_counts[name] += int(finite.sum().item())
    else:
        fv = float(val)
        if np.isfinite(fv):
            metric_results[name] += fv
            metric_counts[name] += 1


def update_global_confusion(global_conf, pred_phy, target_phy, mask, threshold=0.1):
    pred = pred_phy.detach()
    target = target_phy.detach()
    m = mask > 0.5

    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = torch.sum(p_evt & t_evt).item()
    fp = torch.sum(p_evt & (~t_evt) & m).item()
    fn = torch.sum((~p_evt) & t_evt & m).item()
    tn = torch.sum((~p_evt) & (~t_evt) & m).item()

    global_conf["tp"] += float(tp)
    global_conf["fp"] += float(fp)
    global_conf["fn"] += float(fn)
    global_conf["tn"] += float(tn)


def finalize_global_confusion(global_conf):
    tp = global_conf["tp"]
    fp = global_conf["fp"]
    fn = global_conf["fn"]
    tn = global_conf["tn"]
    eps = 1e-12

    total = tp + fp + fn + tn

    return {
        "global_tp": tp,
        "global_fp": fp,
        "global_fn": fn,
        "global_tn": tn,
        "global_csi_thr01": tp / (tp + fp + fn + eps),
        "global_precision_thr01": tp / (tp + fp + eps),
        "global_recall_thr01": tp / (tp + fn + eps),
        "global_nonflood_precision": tn / (tn + fn + eps),
        "global_nonflood_recall": tn / (tn + fp + eps),
        "global_target_prevalence": (tp + fn) / (total + eps),
        "global_pred_prevalence": (tp + fp) / (total + eps),
    }


def evaluate_one_mode(dataset, metrics_cfg, mode, device="cuda", num_workers=4):
    assert mode in ("nearest", "bilinear")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    stats_var = getattr(dataset, "stats_var", None)
    var_name = getattr(dataset, "target_var", "h")
    norm = str(getattr(dataset, "norm", "zscore")).lower()
    transform = str(getattr(dataset, "transform", "log1p")).lower()
    h_asinh_q = getattr(dataset, "h_asinh_q", 90)

    if stats_var is None:
        raise RuntimeError("[ERROR] dataset.stats_var is required.")

    if norm == "zscore":
        to_phy = destand_to_physical
    elif norm == "minmax":
        to_phy = denorm_to_physical
    else:
        raise RuntimeError(f"[ERROR] Unknown norm: {norm}")

    metric_results = {name: 0.0 for name in metrics_cfg.keys()}
    metric_counts = {name: 0 for name in metrics_cfg.keys()}

    global_conf = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for idx, data in enumerate(loader):
        coarse = data["coarse_flood_map"].to(device)       # [B, C, 64, 64]
        fine = data["fine_flood_map"].to(device)           # [B, 1, 512, 512]
        static = data["fine_static_feature"].to(device)    # [B, 7, 512, 512]
        mask = static[:, -1:, :, :]                        # [B, 1, 512, 512]

        # first channel is target_var h
        coarse_h_norm = coarse[:, 0:1, :, :]

        with torch.no_grad():
            coarse_h_phy = to_phy(coarse_h_norm, var_name, stats_var, transform, h_asinh_q)
            fine_h_phy = to_phy(fine, var_name, stats_var, transform, h_asinh_q)

            if var_name == "h":
                coarse_h_phy = coarse_h_phy.clamp_min(0.0)
                fine_h_phy = fine_h_phy.clamp_min(0.0)

            if mode == "nearest":
                pred_phy = F.interpolate(
                    coarse_h_phy,
                    size=fine_h_phy.shape[-2:],
                    mode="nearest",
                )
            else:
                pred_phy = F.interpolate(
                    coarse_h_phy,
                    size=fine_h_phy.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if var_name == "h":
                pred_phy = pred_phy.clamp_min(0.0)

            for name, opt_ in metrics_cfg.items():
                opt_clean = clean_metric_opt(opt_)
                val = calculate_metric(
                    {"pred": pred_phy, "target": fine_h_phy, "mask": mask},
                    opt_clean,
                )
                accumulate_metric(metric_results, metric_counts, name, val)

            update_global_confusion(
                global_conf=global_conf,
                pred_phy=pred_phy,
                target_phy=fine_h_phy,
                mask=mask,
                threshold=0.1,
            )

        if (idx + 1) % 100 == 0:
            print(f"[{mode}] processed {idx + 1}/{len(loader)} samples")

    final = {}
    for name in metric_results.keys():
        c = metric_counts[name]
        final[name] = metric_results[name] / c if c > 0 else float("nan")

    final.update(finalize_global_confusion(global_conf))

    return final


def print_results(title, results):
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True, help="Training yml path.")
    parser.add_argument("--val-key", default="val_1", help="Validation dataset key, e.g., val_1.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    opt = load_yaml(args.opt)

    if args.val_key not in opt["datasets"]:
        raise RuntimeError(f"[ERROR] Cannot find datasets.{args.val_key} in {args.opt}")

    if "val" not in opt or "metrics" not in opt["val"]:
        raise RuntimeError("[ERROR] opt.val.metrics not found.")

    dataset_opt = copy.deepcopy(opt["datasets"][args.val_key])
    dataset = PairedFloodMapDataset(dataset_opt)

    metrics_cfg = opt["val"]["metrics"]

    print(f"Loaded dataset: {dataset_opt.get('name', args.val_key)}")
    print(f"Number of validation samples: {len(dataset)}")
    print(f"target_var={dataset.target_var}, norm={dataset.norm}, transform={dataset.transform}, h_asinh_q={dataset.h_asinh_q}")

    nearest_results = evaluate_one_mode(
        dataset=dataset,
        metrics_cfg=metrics_cfg,
        mode="nearest",
        device=args.device,
        num_workers=args.num_workers,
    )

    bilinear_results = evaluate_one_mode(
        dataset=dataset,
        metrics_cfg=metrics_cfg,
        mode="bilinear",
        device=args.device,
        num_workers=args.num_workers,
    )

    print_results("Coarse upsample baseline: nearest", nearest_results)
    print_results("Coarse upsample baseline: bilinear", bilinear_results)


if __name__ == "__main__":
    main()