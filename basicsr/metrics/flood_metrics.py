"""
val:
  save_flood_map: false
  metrics:
    rmse:
      type: cal_rmse_pt
      reduction: mean
    nse:
      type: cal_nse_pt
      reduction: mean
    csi:
      type: cal_csi_pt
      threshold: 0.05
      reduction: mean
"""


import numpy as np
import torch
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY


def ensure_4d_pt(x: torch.Tensor):
    if x.dim() == 2:  # [H, W]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
        x = x.unsqueeze(1)
    return x


def sum_over_hw(x: torch.Tensor):
    return x.flatten(1).sum(1)


def ensure_4d_np(x):
    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        x = x[:, None, ...]
    return x


@METRIC_REGISTRY.register()
def cal_rmse_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                reduction: str = "mean", eps: float = 1e-12):
    """
    Root Mean Square Error (RMSE) on tensors in physical domain.

    RMSE = sqrt( sum((pred - target)^2 * mask) / sum(mask))

    Args:
        pred: predicted fine-grid flood map (h, u, v) [B, 1, Hf, Wf]
        target: simulated (ground truth / target) fine-grid flood map (h, u ,v) [B, 1, Hf, Wf]
        mask: fine-grid mask [B, 1, Hf, Wf]
        reduction: "mean" -> average across batch; "none" -> per-sample tensor
        eps: numerical stability

    Returns:
        float if reduction="mean", else torch.Tensor [B]
    """

    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    diff2 = (pred - target) ** 2
    masked_diff2 = sum_over_hw(diff2 * mask)
    num_mask = sum_over_hw(mask).clamp_min(eps)

    rmse = torch.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return rmse.mean().item()


@METRIC_REGISTRY.register()
def cal_rmse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    diff2 = (pred - target) ** 2
    masked_diff2 = (diff2 * mask).reshape(diff2.shape[0], -1).sum(1)
    num_mask = mask.reshape(mask.shape[0], -1).sum(1).clip(min=eps)

    rmse = np.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return float(rmse.mean())


@METRIC_REGISTRY.register()
def cal_nse_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               reduction: str = "mean", eps: float = 1e-12):
    """
    Nash-Sutcliffe Efficiency (NSE) on tensors in physical domain.

    NSE = 1 - sum((pred - target)^2) / sum((target - mean(target_masked))^2)

    Args:
        pred: predicted fine-grid flood map (h, u, v) [B, 1, Hf, Wf]
        target: simulated (ground truth / target) fine-grid flood map (h, u ,v) [B, 1, Hf, Wf]
        mask: fine-grid mask [B, 1, Hf, Wf]
        reduction: "mean" -> average across batch; "none" -> per-sample tensor
        eps: numerical stability

    Returns:
        float if reduction="mean", else torch.Tensor [B]
    """

    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask
    wsum = sum_over_hw(w).clamp_min(eps)  # [B]
    t_sum = sum_over_hw(target * w)  # [B]
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    masked_diff2 = sum_over_hw(((pred - target) ** 2) * w)
    vc = sum_over_hw(((target - t_mean) ** 2) * w)
    nse = 1.0 - (masked_diff2 / vc.clamp_min(eps))
    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_nse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
               reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask
    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    mask_diff2 = (((pred - target) ** 2) * w).reshape(pred.shape[0], -1).sum(1)
    vc = (((target - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)
    nse = 1.0 - mask_diff2 / vc
    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_csi_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    """
    Critical Success Index (CSI) on tensors in physical domain.

    CSI = TP / (TP + FP + FN) with flood event defined by >= threshold.

    Args:
        pred: predicted fine-grid flood map (h, u, v) [B, 1, Hf, Wf]
        target: simulated (ground truth / target) fine-grid flood map (h, u ,v) [B, 1, Hf, Wf]
        mask: fine-grid mask [B, 1, Hf, Wf]
        threshold: flood event threshold (e.g., h: 0.05m; u/v: 0.1 m/s)
        reduction: "mean" -> average across batch; "none" -> per-sample tensor
        eps: numerical stability

    Returns:
        float if reduction="mean", else torch.Tensor [B]
    """

    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    csi = tp / (tp + fp + fn + eps)
    if reduction == "none":
        return csi
    return csi.mean().item()


@METRIC_REGISTRY.register()
def cal_csi_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
               threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    csi = tp / (tp + fp + fn + eps)
    if reduction == "none":
        return csi
    return float(csi.mean())
