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
    nse_safe:
      type: cal_nse_pt_safe
      reduction: mean
      min_var: 1.0e-6
      abs_tol_per_px: 1.0e-4
      lower_bound: -5.0
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
def cal_rmse_threshold_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                          reduction: str = "mean", eps: float = 1e-12, threshold: float = 0.05):
    """
    RMSE with physical threshold.

    Calculate RMSE when target >= threshold
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    diff2 = (pred_eff - target_eff) ** 2
    masked_diff2 = sum_over_hw(diff2 * mask)
    num_mask = sum_over_hw(mask).clamp_min(eps)

    rmse = torch.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return rmse.mean().item()


@METRIC_REGISTRY.register()
def cal_rmse_threshold_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                                   reduction: str = "mean", eps: float = 1e-12,
                                   threshold: float = 0.05, abs_tol: float = 0.01):
    """
    RMSE with physical threshold + error tolerance.

    Calculate RMSE when target >= threshold and |pred - target| >= abs_tol
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    diff = pred_eff - target_eff
    abs_err = diff.abs()

    eff_mask = ((mask > 0) & (abs_err >= abs_tol)).to(pred.dtype)

    diff2 = diff ** 2
    masked_diff2 = sum_over_hw(diff2 * eff_mask)
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
def cal_rmse_threshold_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                          reduction: str = "mean", eps: float = 1e-12, threshold: float = 0.05):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    diff2 = (pred_eff - target_eff) ** 2
    masked_diff2 = (diff2 * mask).reshape(diff2.shape[0], -1).sum(1)
    num_mask = mask.reshape(mask.shape[0], -1).sum(1).clip(min=eps)

    rmse = np.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return float(rmse.mean())


@METRIC_REGISTRY.register()
def cal_rmse_threshold_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                                   reduction: str = "mean", eps: float = 1e-12,
                                   threshold: float = 0.05, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    diff = pred_eff - target_eff
    abs_err = np.abs(diff)

    eff_mask = ((mask > 0) & (abs_err >= abs_tol)).astype(pred.dtype)

    diff2 = diff ** 2
    masked_diff2 = (diff2 * eff_mask).reshape(diff2.shape[0], -1).sum(1)
    num_mask = mask.reshape(mask.shape[0], -1).sum(1).clip(min=eps)

    rmse = np.sqrt(masked_diff2 / num_mask)
    if reduction == "none":
        return rmse
    return float(rmse.mean())


@METRIC_REGISTRY.register()
def cal_rmse_depth_threshold_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_rmse_threshold_pt(pred, target, mask, reduction=reduction, eps=eps, threshold=0.05)


@METRIC_REGISTRY.register()
def cal_rmse_vel_threshold_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_rmse_threshold_pt(pred, target, mask, reduction=reduction, eps=eps, threshold=0.1)


@METRIC_REGISTRY.register()
def cal_rmse_depth_threshold_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_rmse_threshold_tolerant_pt(pred, target, mask, reduction=reduction, eps=eps, threshold=0.05, abs_tol=0.01)


@METRIC_REGISTRY.register()
def cal_rmse_vel_threshold_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_rmse_threshold_tolerant_pt(pred, target, mask, reduction=reduction, eps=eps, threshold=0.1, abs_tol=0.02)


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
def cal_nse_threshold_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         reduction: str = "mean", eps: float = 1e-12, threshold: float = 0.05):
    """
    NSE with physical threshold.

    Calculate NSE when target >= threshold
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target_eff * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    masked_diff2 = sum_over_hw(((pred_eff - target_eff) ** 2) * w)
    vc = sum_over_hw(((target_eff - t_mean) ** 2) * w).clamp_min(eps)

    nse = 1.0 - (masked_diff2 / vc)
    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_nse_threshold_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                                  reduction: str = "mean", eps: float = 1e-12,
                                  threshold: float = 0.05, abs_tol: float = 0.01):
    """
    NSE with physical threshold + error tolerance.

    Numerator only counts cells where mask > 0 and |pred - target| >= abs_tol and target >= threshold.

    Denominator is the variance of target when target >= threshold.
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target_eff * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    diff = pred_eff - target_eff
    abs_err = diff.abs()
    large_err_mask = ((abs_err >= abs_tol) & (mask > 0)).to(pred.dtype)
    masked_diff2 = sum_over_hw((diff ** 2) * large_err_mask)
    vc = sum_over_hw(((target_eff - t_mean) ** 2) * w).clamp_min(eps)

    nse = 1.0 - (masked_diff2 / vc)
    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_nse_pt_safe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                    abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask
    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    masked_diff2 = sum_over_hw(((pred - target) ** 2) * w)
    vc = sum_over_hw(((target - t_mean) ** 2) * w)

    normal = vc >= min_var
    nse = torch.empty_like(vc)
    nse[normal] = 1.0 - (masked_diff2[normal] / vc[normal].clamp_min(eps))

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        tiny_idx = torch.where(tiny)[0]
        nse[tiny_idx[near_perfect]] = 1.0

        bad = ~near_perfect
        if bad.any():
            bad_idx = tiny_idx[bad]
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            lower_bound_tensor = torch.tensor(lower_bound, device=masked_diff2.device, dtype=masked_diff2.dtype)
            nse[bad_idx] = torch.maximum(val, lower_bound_tensor)

    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_nse_threshold_pt_safe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                              reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                              abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0, threshold: float = 0.05):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target_eff * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    masked_diff2 = sum_over_hw(((pred_eff - target_eff) ** 2) * w)
    vc = sum_over_hw(((target_eff - t_mean) ** 2) * w).clamp_min(eps)

    normal = vc >= min_var
    nse = torch.empty_like(vc)
    nse[normal] = 1.0 - (masked_diff2[normal] / vc[normal])

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        tiny_idx = torch.where(tiny)[0]
        nse[tiny_idx[near_perfect]] = 1.0

        bad = ~near_perfect
        if bad.any():
            bad_idx = tiny_idx[bad]
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            lower_bound_tensor = torch.tensor(lower_bound, device=masked_diff2.device, dtype=masked_diff2.dtype)
            nse[bad_idx] = torch.maximum(val, lower_bound_tensor)

    if reduction == "none":
        return nse
    return nse.mean().item()


@METRIC_REGISTRY.register()
def cal_nse_threshold_tolerant_pt_safe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                                       reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                       abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0,
                                       threshold: float = 0.05, abs_tol: float = 0.01):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = torch.where(active_p, pred, torch.zeros_like(pred))
    active_t = (target >= threshold) & (mask > 0)
    target_eff = torch.where(active_t, target, torch.zeros_like(target))

    wsum = sum_over_hw(w).clamp_min(eps)
    t_sum = sum_over_hw(target_eff * w)
    t_mean = (t_sum / wsum).view(-1, 1, 1, 1)

    diff = pred_eff - target_eff
    abs_err = diff.abs()
    large_err_mask = ((abs_err >= abs_tol) & (mask > 0)).to(pred.dtype)

    masked_diff2 = sum_over_hw((diff ** 2) * large_err_mask)
    vc = sum_over_hw(((target_eff - t_mean) ** 2) * w).clamp_min(eps)

    normal = vc >= min_var
    nse = torch.empty_like(vc)
    nse[normal] = 1.0 - (masked_diff2[normal] / vc[normal])

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        tiny_idx = torch.where(tiny)[0]
        nse[tiny_idx[near_perfect]] = 1.0

        bad = ~near_perfect
        if bad.any():
            bad_idx = tiny_idx[bad]
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            lower_bound_tensor = torch.tensor(lower_bound, device=masked_diff2.device, dtype=masked_diff2.dtype)
            nse[bad_idx] = torch.maximum(val, lower_bound_tensor)

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

    masked_diff2 = (((pred - target) ** 2) * w).reshape(pred.shape[0], -1).sum(1)
    vc = (((target - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)
    nse = 1.0 - masked_diff2 / vc
    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_threshold_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                         reduction: str = "mean", eps: float = 1e-12, threshold: float = 0.05):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target_eff * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    masked_diff2 = (((pred_eff - target_eff) ** 2) * w).reshape(pred.shape[0], -1).sum(1)
    vc = (((target_eff - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)

    nse = 1.0 - masked_diff2 / vc
    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_threshold_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                                  reduction: str = "mean", eps: float = 1e-12,
                                  threshold: float = 0.05, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target_eff * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    diff = pred_eff - target_eff
    abs_err = np.abs(diff)
    large_err_mask = ((abs_err >= abs_tol) & (mask > 0)).astype(pred.dtype)

    masked_diff2 = (diff ** 2 * large_err_mask).reshape(diff.shape[0], -1).sum(1)
    vc = (((target_eff - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)
    nse = 1.0 - masked_diff2 / vc
    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_np_safe(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                    reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                    abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask
    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    masked_diff2 = (((pred - target) ** 2) * w).reshape(pred.shape[0], -1).sum(1)
    vc = (((target - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1)

    nse = np.empty_like(vc, dtype=np.float64)
    normal = vc >= min_var
    nse[normal] = 1.0 - (masked_diff2[normal] / np.clip(vc[normal], eps, None))

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        nse[tiny] = 1.0

        bad_idx = np.where(tiny)[0][~near_perfect]
        if bad_idx.size > 0:
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            nse[bad_idx] = np.maximum(val, lower_bound)

    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_threshold_np_safe(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                              reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                              abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0, threshold: float = 0.05):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target_eff * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    masked_diff2 = (((pred_eff - target_eff) ** 2) * w).reshape(pred.shape[0], -1).sum(1)
    vc = (((target_eff - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)

    nse = np.empty_like(vc, dtype=np.float64)
    normal = vc >= min_var
    nse[normal] = 1.0 - (masked_diff2[normal] / vc[normal])

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        nse[tiny] = 1.0

        bad_idx = np.where(tiny)[0][~near_perfect]
        if bad_idx.size > 0:
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            nse[bad_idx] = np.maximum(val, lower_bound)

    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_threshold_tolerant_np_safe(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                                       reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                       abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0,
                                       threshold: float = 0.05, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    w = mask

    active_p = (pred >= threshold) & (mask > 0)
    pred_eff = np.where(active_p, pred, 0.0)
    active_t = (target >= threshold) & (mask > 0)
    target_eff = np.where(active_t, target, 0.0)

    wsum = w.reshape(w.shape[0], -1).sum(1).clip(min=eps)
    t_sum = (target_eff * w).reshape(target.shape[0], -1).sum(1)
    t_mean = (t_sum / wsum)[:, None, None, None]

    diff = pred_eff - target_eff
    abs_err = np.abs(diff)
    large_err_mask = ((abs_err >= abs_tol) & (mask > 0)).astype(pred.dtype)

    masked_diff2 = (diff ** 2 * large_err_mask).reshape(diff.shape[0], -1).sum(1)
    vc = (((target_eff - t_mean) ** 2) * w).reshape(pred.shape[0], -1).sum(1).clip(min=eps)

    nse = np.empty_like(vc, dtype=np.float64)
    normal = vc >= min_var
    nse[normal] = 1.0 - (masked_diff2[normal] / vc[normal])

    tiny = ~normal
    if tiny.any():
        tiny_tol = (abs_tol_per_px ** 2) * wsum[tiny]
        near_perfect = masked_diff2[tiny] <= tiny_tol
        nse[tiny] = 1.0

        bad_idx = np.where(tiny)[0][~near_perfect]
        if bad_idx.size > 0:
            val = 1.0 - (masked_diff2[bad_idx] / min_var)
            nse[bad_idx] = np.maximum(val, lower_bound)

    if reduction == "none":
        return nse
    return float(nse.mean())


@METRIC_REGISTRY.register()
def cal_nse_depth_threshold_pt_safe(pred, target, mask,
                                    reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                    abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    return cal_nse_threshold_pt_safe(pred, target, mask,
                                     reduction=reduction, eps=eps, min_var=min_var,
                                     abs_tol_per_px=abs_tol_per_px, lower_bound=lower_bound, threshold=0.05)


@METRIC_REGISTRY.register()
def cal_nse_depth_threshold_tolerant_pt_safe(pred, target, mask,
                                             reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                             abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    return cal_nse_threshold_tolerant_pt_safe(pred, target, mask,
                                              reduction=reduction, eps=eps, min_var=min_var,
                                              abs_tol_per_px=abs_tol_per_px, lower_bound=lower_bound,
                                              threshold=0.05, abs_tol=0.01)


@METRIC_REGISTRY.register()
def cal_nse_vel_threshold_pt_safe(pred, target, mask,
                                  reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                  abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    return cal_nse_threshold_pt_safe(pred, target, mask,
                                     reduction=reduction, eps=eps, min_var=min_var,
                                     abs_tol_per_px=abs_tol_per_px, lower_bound=lower_bound, threshold=0.1)


@METRIC_REGISTRY.register()
def cal_nse_vel_threshold_tolerant_pt_safe(pred, target, mask,
                                           reduction: str = "mean", eps: float = 1e-12, min_var: float = 1e-6,
                                           abs_tol_per_px: float = 1e-4, lower_bound: float = -5.0):
    return cal_nse_threshold_tolerant_pt_safe(pred, target, mask,
                                              reduction=reduction, eps=eps, min_var=min_var,
                                              abs_tol_per_px=abs_tol_per_px, lower_bound=lower_bound,
                                              threshold=0.1, abs_tol=0.02)


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
def cal_csi_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = (pred - target).abs()
    borderline = (abs_err <= abs_tol) & m
    p_evt = torch.where(borderline, t_evt, p_evt_raw)

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


@METRIC_REGISTRY.register()
def cal_csi_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                        threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = np.abs(pred - target)
    borderline = (abs_err <= abs_tol) & m
    p_evt = np.where(borderline, t_evt, p_evt_raw)

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    csi = tp / (tp + fp + fn + eps)
    if reduction == "none":
        return csi
    return float(csi.mean())


@METRIC_REGISTRY.register()
def cal_csi_depth_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_csi_pt(pred, target, mask,
                      threshold=0.05, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_csi_vel_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_csi_pt(pred, target, mask,
                      threshold=0.1, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_csi_depth_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_csi_tolerant_pt(pred, target, mask,
                               threshold=0.05, reduction=reduction, eps=eps, abs_tol=0.01)


@METRIC_REGISTRY.register()
def cal_csi_vel_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_csi_tolerant_pt(pred, target, mask,
                               threshold=0.1, reduction=reduction, eps=eps, abs_tol=0.02)


@METRIC_REGISTRY.register()
def cal_precision_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = tp / (tp + fp + eps)
    if reduction == "none":
        return prec
    return prec.mean().item()


@METRIC_REGISTRY.register()
def cal_precision_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                              threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = (pred - target).abs()
    borderline = (abs_err <= abs_tol) & m
    p_evt = torch.where(borderline, t_evt, p_evt_raw)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = tp / (tp + fp + eps)
    if reduction == "none":
        return prec
    return prec.mean().item()


@METRIC_REGISTRY.register()
def cal_precision_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                     threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    prec = tp / (tp + fp + eps)
    if reduction == "none":
        return prec
    return float(prec.mean())


@METRIC_REGISTRY.register()
def cal_precision_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                              threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = np.abs(pred - target)
    borderline = (abs_err <= abs_tol) & m
    p_evt = np.where(borderline, t_evt, p_evt_raw)

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    prec = tp / (tp + fp + eps)
    if reduction == "none":
        return prec
    return float(prec.mean())


@METRIC_REGISTRY.register()
def cal_precision_depth_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_precision_pt(pred, target, mask,
                            threshold=0.05, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_precision_vel_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_precision_pt(pred, target, mask,
                            threshold=0.1, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_precision_depth_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_precision_tolerant_pt(pred, target, mask,
                                     threshold=0.05, reduction=reduction, eps=eps, abs_tol=0.01)


@METRIC_REGISTRY.register()
def cal_precision_vel_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_precision_tolerant_pt(pred, target, mask,
                                     threshold=0.1, reduction=reduction, eps=eps, abs_tol=0.02)


@METRIC_REGISTRY.register()
def cal_recall_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = tp / (tp + fn + eps)
    if reduction == "none":
        return rec
    return rec.mean().item()


@METRIC_REGISTRY.register()
def cal_recall_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = (pred - target).abs()
    borderline = (abs_err <= abs_tol) & m
    p_evt = torch.where(borderline, t_evt, p_evt_raw)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = tp / (tp + fn + eps)
    if reduction == "none":
        return rec
    return rec.mean().item()


@METRIC_REGISTRY.register()
def cal_recall_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    rec = tp / (tp + fn + eps)
    if reduction == "none":
        return rec
    return float(rec.mean())


@METRIC_REGISTRY.register()
def cal_recall_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                           threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12, abs_tol: float = 0.01):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt_raw = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    abs_err = np.abs(pred - target)
    borderline = (abs_err <= abs_tol) & m
    p_evt = np.where(borderline, t_evt, p_evt_raw)

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    rec = tp / (tp + fn + eps)
    if reduction == "none":
        return rec
    return float(rec.mean())


@METRIC_REGISTRY.register()
def cal_recall_depth_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_recall_pt(pred, target, mask,
                         threshold=0.05, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_recall_vel_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_recall_pt(pred, target, mask,
                         threshold=0.1, reduction=reduction, eps=eps)


@METRIC_REGISTRY.register()
def cal_recall_depth_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_recall_tolerant_pt(pred, target, mask,
                                  threshold=0.05, reduction=reduction, eps=eps, abs_tol=0.01)


@METRIC_REGISTRY.register()
def cal_recall_vel_tolerant_pt(pred, target, mask, reduction: str = "mean", eps: float = 1e-12):
    return cal_recall_tolerant_pt(pred, target, mask,
                                  threshold=0.1, reduction=reduction, eps=eps, abs_tol=0.02)


@METRIC_REGISTRY.register()
def cal_prev_t_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    _ = pred
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    t_evt = (target >= threshold) & m

    num_t = sum_over_hw(t_evt.to(target.dtype))
    denom = sum_over_hw(m.to(target.dtype)).clamp_min(eps)

    prev_t = num_t / denom
    if reduction == "none":
        return prev_t
    return prev_t.mean().item()


@METRIC_REGISTRY.register()
def cal_prev_t_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    _ = pred
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    t_evt = (target >= threshold) & m

    num_t = t_evt.reshape(target.shape[0], -1).sum(1).astype(np.float64)
    denom = m.reshape(target.shape[0], -1).sum(1).astype(np.float64)

    prev_t = num_t / np.clip(denom, eps, None)
    if reduction == "none":
        return prev_t
    return float(prev_t.mean())


@METRIC_REGISTRY.register()
def cal_prev_p_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    _ = target
    pred = ensure_4d_pt(pred)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m

    num_p = sum_over_hw(p_evt.to(pred.dtype))
    denom = sum_over_hw(m.to(pred.dtype)).clamp_min(eps)

    prev_p = num_p / denom
    if reduction == "none":
        return prev_p
    return prev_p.mean().item()


@METRIC_REGISTRY.register()
def cal_prev_p_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                  threshold: float = 0.05, reduction: str = "mean", eps: float = 1e-12):
    _ = target
    pred = ensure_4d_np(pred)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m

    num_p = p_evt.reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    denom = m.reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    prev_p = num_p / np.clip(denom, eps, None)
    if reduction == "none":
        return prev_p
    return float(prev_p.mean())

