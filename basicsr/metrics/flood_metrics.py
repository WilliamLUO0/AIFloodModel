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
from typing import Optional
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


def _safe_divide_metric_pt(num: torch.Tensor,
                           denom: torch.Tensor,
                           eps: float = 1e-12,
                           empty_as_nan: bool = False):
    """
    Per-sample safe division for metrics.

    If empty_as_nan=True, samples with denom == 0 are set to NaN.
    This is useful for precision / recall / CSI when the metric is undefined.

    Examples:
      precision denominator = TP + FP
      recall denominator    = TP + FN
      CSI denominator       = TP + FP + FN
    """
    if empty_as_nan:
        out = torch.full_like(num, float("nan"))
        valid = denom > 0
        out[valid] = num[valid] / denom[valid].clamp_min(eps)
        return out

    return num / denom.clamp_min(eps)


def _reduce_metric_pt(x: torch.Tensor, reduction: str = "mean"):
    """
    Reduce per-sample metric values.

    If reduction='mean', ignore NaN values.
    """
    if reduction == "none":
        return x

    finite = torch.isfinite(x)
    if finite.any():
        return x[finite].mean().item()
    return float("nan")


def _safe_divide_metric_np(num: np.ndarray,
                           denom: np.ndarray,
                           eps: float = 1e-12,
                           empty_as_nan: bool = False):
    """
    Numpy version of per-sample safe division.
    """
    num = np.asarray(num, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)

    if empty_as_nan:
        out = np.full_like(num, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = num[valid] / np.clip(denom[valid], eps, None)
        return out

    return num / np.clip(denom, eps, None)


def _reduce_metric_np(x: np.ndarray, reduction: str = "mean"):
    """
    Reduce per-sample numpy metric values.

    If reduction='mean', ignore NaN values.
    """
    if reduction == "none":
        return x

    finite = np.isfinite(x)
    if np.any(finite):
        return float(np.mean(x[finite]))
    return float("nan")


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

    Values < threshold are set to 0 for both pred and target,
    then RMSE is calculated over the full masked domain.
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

    Values < threshold are set to 0 for both pred and target,
    and errors < abs_tol are set to 0,
    then RMSE is calculated over the full masked domain.
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
def cal_rmse_conditional_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                            reduction: str = "mean", eps: float = 1e-12,
                            cond_on_target_ge: Optional[float] = None,
                            cond_on_target_lt: Optional[float] = None,
                            abs_tol: Optional[float] = None):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    sel = m

    if cond_on_target_ge is not None:
        sel = sel & (target >= cond_on_target_ge)

    if cond_on_target_lt is not None:
        sel = sel & (target < cond_on_target_lt)

    diff = pred - target
    if abs_tol is not None:
        diff = torch.where(diff.abs() < abs_tol, torch.zeros_like(diff), diff)

    diff2 = diff ** 2
    sse = sum_over_hw(diff2 * sel.to(diff2.dtype))
    n = sum_over_hw(sel.to(diff2.dtype))
    valid = n > 0.5

    rmse = torch.empty_like(sse)
    rmse[valid] = torch.sqrt(sse[valid] / n[valid].clamp_min(eps))
    rmse[~valid] = torch.tensor(float("nan"), device=rmse.device, dtype=rmse.dtype)

    if reduction == "none":
        return rmse

    # reduction == "mean": 只对 valid 求平均; 如果全无效，返回 nan
    if valid.any():
        return rmse[valid].mean().item()
    else:
        return float("nan")


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

    Values < threshold are set to 0 for both pred and target,
    then NSE is calculated over the full masked domain.
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

    Numerator only counts cells where mask == 1 and |pred - target| >= abs_tol and setting values < threshold to 0.

    Denominator is the variance of target when setting values < threshold to 0.
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
def cal_csi_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               threshold: float = 0.05, reduction: str = "mean",
               empty_as_nan: bool = False, eps: float = 1e-12):
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

    csi = _safe_divide_metric_pt(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_csi_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        threshold: float = 0.05, reduction: str = "mean",
                        eps: float = 1e-12, abs_tol: float = 0.01,
                        empty_as_nan: bool = False):
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

    csi = _safe_divide_metric_pt(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_csi_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
               threshold: float = 0.05, reduction: str = "mean",
               empty_as_nan: bool = False, eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    csi = _safe_divide_metric_np(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_csi_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                        threshold: float = 0.05, reduction: str = "mean",
                        eps: float = 1e-12, abs_tol: float = 0.01,
                        empty_as_nan: bool = False):
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

    csi = _safe_divide_metric_np(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     threshold: float = 0.05, reduction: str = "mean",
                     empty_as_nan: bool = False, eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                              threshold: float = 0.05, reduction: str = "mean",
                              eps: float = 1e-12, abs_tol: float = 0.01,
                              empty_as_nan: bool = False):
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

    prec = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                     threshold: float = 0.05, reduction: str = "mean",
                     empty_as_nan: bool = False, eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fp = (p_evt & (~t_evt)).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    prec = _safe_divide_metric_np(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                              threshold: float = 0.05, reduction: str = "mean",
                              eps: float = 1e-12, abs_tol: float = 0.01,
                              empty_as_nan: bool = False):
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

    prec = _safe_divide_metric_np(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_precision_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                          band_ge: Optional[float] = None,
                          band_lt: Optional[float] = None,
                          reduction: str = "mean",
                          empty_as_nan: bool = False,
                          eps: float = 1e-12,):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    p_evt = m
    t_evt = m

    if band_ge is not None:
        p_evt = p_evt & (pred >= band_ge)
        t_evt = t_evt & (target >= band_ge)

    if band_lt is not None:
        p_evt = p_evt & (pred < band_lt)
        t_evt = t_evt & (target < band_lt)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    prec = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(prec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  threshold: float = 0.05, reduction: str = "mean",
                  empty_as_nan: bool = False, eps: float = 1e-12):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(rec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_tolerant_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           threshold: float = 0.05, reduction: str = "mean",
                           eps: float = 1e-12, abs_tol: float = 0.01,
                           empty_as_nan: bool = False):
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

    rec = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(rec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                  threshold: float = 0.05, reduction: str = "mean",
                  empty_as_nan: bool = False, eps: float = 1e-12):
    pred = ensure_4d_np(pred)
    target = ensure_4d_np(target)
    mask = ensure_4d_np(mask)

    m = mask > 0.5
    p_evt = (pred >= threshold) & m
    t_evt = (target >= threshold) & m

    tp = (p_evt & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)
    fn = ((~p_evt) & t_evt).reshape(pred.shape[0], -1).sum(1).astype(np.float64)

    rec = _safe_divide_metric_np(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(rec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_tolerant_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                           threshold: float = 0.05, reduction: str = "mean",
                           eps: float = 1e-12, abs_tol: float = 0.01,
                           empty_as_nan: bool = False):
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

    rec = _safe_divide_metric_np(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_np(rec, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_recall_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                       band_ge: Optional[float] = None,
                       band_lt: Optional[float] = None,
                       reduction: str = "mean",
                       empty_as_nan: bool = False,
                       eps: float = 1e-12,):
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    p_evt = m
    t_evt = m

    if band_ge is not None:
        p_evt = p_evt & (pred >= band_ge)
        t_evt = t_evt & (target >= band_ge)

    if band_lt is not None:
        p_evt = p_evt & (pred < band_lt)
        t_evt = t_evt & (target < band_lt)

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    rec = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(rec, reduction=reduction)


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


@METRIC_REGISTRY.register()
def cal_logit_precision_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                           channel: int = 0,
                           target_threshold: float = 0.1,
                           prob_threshold: float = 0.5,
                           reduction: str = "mean",
                           empty_as_nan: bool = True,
                           eps: float = 1e-12):
    """
    Precision for one flood-logit channel.

    Example:
      channel=0, target_threshold=0.1
      evaluates whether flood_logit[:,0] predicts h >= 0.1 m.

    Precision = TP / (TP + FP)
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if pred.shape[1] <= channel:
        raise ValueError(f"[cal_logit_precision_pt] pred has {pred.shape[1]} channels, but channel={channel}")

    m = mask > 0.5

    prob = torch.sigmoid(pred[:, channel:channel + 1, :, :])
    p_evt = (prob >= prob_threshold) & m
    t_evt = (target >= target_threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))

    precision = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(precision, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_recall_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        channel: int = 0,
                        target_threshold: float = 0.1,
                        prob_threshold: float = 0.5,
                        reduction: str = "mean",
                        empty_as_nan: bool = True,
                        eps: float = 1e-12):
    """
    Recall for one flood-logit channel.

    Recall = TP / (TP + FN)
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if pred.shape[1] <= channel:
        raise ValueError(f"[cal_logit_recall_pt] pred has {pred.shape[1]} channels, but channel={channel}")

    m = mask > 0.5

    prob = torch.sigmoid(pred[:, channel:channel + 1, :, :])
    p_evt = (prob >= prob_threshold) & m
    t_evt = (target >= target_threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    recall = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(recall, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_csi_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     channel: int = 0,
                     target_threshold: float = 0.1,
                     prob_threshold: float = 0.5,
                     reduction: str = "mean",
                     empty_as_nan: bool = True,
                     eps: float = 1e-12):
    """
    CSI for one flood-logit channel.

    CSI = TP / (TP + FP + FN)
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    if pred.shape[1] <= channel:
        raise ValueError(f"[cal_logit_csi_pt] pred has {pred.shape[1]} channels, but channel={channel}")

    m = mask > 0.5

    prob = torch.sigmoid(pred[:, channel:channel + 1, :, :])
    p_evt = (prob >= prob_threshold) & m
    t_evt = (target >= target_threshold) & m

    tp = sum_over_hw((p_evt & t_evt).to(pred.dtype))
    fp = sum_over_hw((p_evt & (~t_evt)).to(pred.dtype))
    fn = sum_over_hw(((~p_evt) & t_evt).to(pred.dtype))

    csi = _safe_divide_metric_pt(tp, tp + fp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(csi, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_prev_p_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        channel: int = 0,
                        prob_threshold: float = 0.5,
                        reduction: str = "mean",
                        eps: float = 1e-12):
    """
    Predicted prevalence for one flood-logit channel.

    pred_prevalence = number of predicted positive pixels / number of valid AOI pixels
    """
    _ = target

    pred = ensure_4d_pt(pred)
    mask = ensure_4d_pt(mask)

    if pred.shape[1] <= channel:
        raise ValueError(f"[cal_logit_prev_p_pt] pred has {pred.shape[1]} channels, but channel={channel}")

    m = mask > 0.5

    prob = torch.sigmoid(pred[:, channel:channel + 1, :, :])
    p_evt = (prob >= prob_threshold) & m

    num_p = sum_over_hw(p_evt.to(pred.dtype))
    denom = sum_over_hw(m.to(pred.dtype)).clamp_min(eps)

    prev = num_p / denom
    return _reduce_metric_pt(prev, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_prev_t_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        target_threshold: float = 0.1,
                        reduction: str = "mean",
                        eps: float = 1e-12):
    """
    Target prevalence for a physical water-depth threshold.

    This is useful for checking class imbalance for each logit threshold.
    """
    _ = pred

    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5
    t_evt = (target >= target_threshold) & m

    num_t = sum_over_hw(t_evt.to(target.dtype))
    denom = sum_over_hw(m.to(target.dtype)).clamp_min(eps)

    prev = num_t / denom
    return _reduce_metric_pt(prev, reduction=reduction)


# -------------------------------------------------------------------------
# Ordinal-logit interval metrics
# -------------------------------------------------------------------------
# Convert three ordinal logits into interval predictions:
#
#   nonflood: prob_tau01 < 0.5
#   slight:   prob_tau01 >= 0.5 and prob_tau05 < 0.5
#   severe:   prob_tau05 >= 0.5 and prob_tau10 < 0.5
#   extreme:  prob_tau10 >= 0.5
#
# Target intervals are defined by physical water depth:
#
#   nonflood: 0.0 <= h < 0.1
#   slight:   0.1 <= h < 0.5
#   severe:   0.5 <= h < 1.0
#   extreme:  h >= 1.0
# -------------------------------------------------------------------------


def _logit_interval_pred_mask(pred: torch.Tensor,
                              interval: str,
                              prob_threshold: float = 0.5):
    """
    Build predicted interval mask from ordinal logits.

    pred: [B, 3, H, W]
    """
    if pred.shape[1] < 3:
        raise ValueError(f"Ordinal interval metrics require pred with 3 channels, got {pred.shape}")

    p1 = torch.sigmoid(pred[:, 0:1, :, :]) >= prob_threshold  # h >= 0.1
    p2 = torch.sigmoid(pred[:, 1:2, :, :]) >= prob_threshold  # h >= 0.5
    p3 = torch.sigmoid(pred[:, 2:3, :, :]) >= prob_threshold  # h >= 1.0

    interval = str(interval).lower().strip()

    if interval in ("nonflood", "nonflood01", "dry"):
        return ~p1

    if interval in ("slight", "slight01_05", "slightflood"):
        return p1 & (~p2)

    if interval in ("severe", "severe05_1", "severeflood"):
        return p2 & (~p3)

    if interval in ("extreme", "extreme1", "extremeflood"):
        return p3

    raise ValueError(f"[logit interval] Unknown interval: {interval}")


def _target_interval_mask(target: torch.Tensor,
                          interval: str,
                          low: Optional[float] = None,
                          high: Optional[float] = None):
    """
    Build target interval mask from physical water depth.

    If low/high are provided, they override the named interval.
    """
    interval = str(interval).lower().strip()

    if low is not None or high is not None:
        m = torch.ones_like(target, dtype=torch.bool)
        if low is not None:
            m = m & (target >= float(low))
        if high is not None:
            m = m & (target < float(high))
        return m

    if interval in ("nonflood", "nonflood01", "dry"):
        return (target >= 0.0) & (target < 0.1)

    if interval in ("slight", "slight01_05", "slightflood"):
        return (target >= 0.1) & (target < 0.5)

    if interval in ("severe", "severe05_1", "severeflood"):
        return (target >= 0.5) & (target < 1.0)

    if interval in ("extreme", "extreme1", "extremeflood"):
        return target >= 1.0

    raise ValueError(f"[target interval] Unknown interval: {interval}")


@METRIC_REGISTRY.register()
def cal_logit_precision_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                                interval: str = "slight",
                                target_low: Optional[float] = None,
                                target_high: Optional[float] = None,
                                prob_threshold: float = 0.5,
                                reduction: str = "mean",
                                empty_as_nan: bool = True,
                                eps: float = 1e-12):
    """
    Interval precision from ordinal flood logits.

    Example:
      interval='slight'
      predicted slight = P(h>=0.1)>=0.5 and P(h>=0.5)<0.5
      target slight    = 0.1 <= h < 0.5

    Precision = TP_interval / predicted_interval
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    p_band = _logit_interval_pred_mask(pred, interval=interval, prob_threshold=prob_threshold) & m
    t_band = _target_interval_mask(target, interval=interval, low=target_low, high=target_high) & m

    tp = sum_over_hw((p_band & t_band).to(pred.dtype))
    fp = sum_over_hw((p_band & (~t_band)).to(pred.dtype))

    precision = _safe_divide_metric_pt(tp, tp + fp, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(precision, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_recall_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                             interval: str = "slight",
                             target_low: Optional[float] = None,
                             target_high: Optional[float] = None,
                             prob_threshold: float = 0.5,
                             reduction: str = "mean",
                             empty_as_nan: bool = True,
                             eps: float = 1e-12):
    """
    Interval recall from ordinal flood logits.

    Recall = TP_interval / target_interval
    """
    pred = ensure_4d_pt(pred)
    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    p_band = _logit_interval_pred_mask(pred, interval=interval, prob_threshold=prob_threshold) & m
    t_band = _target_interval_mask(target, interval=interval, low=target_low, high=target_high) & m

    tp = sum_over_hw((p_band & t_band).to(pred.dtype))
    fn = sum_over_hw(((~p_band) & t_band).to(pred.dtype))

    recall = _safe_divide_metric_pt(tp, tp + fn, eps=eps, empty_as_nan=empty_as_nan)
    return _reduce_metric_pt(recall, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_prev_p_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                             interval: str = "slight",
                             prob_threshold: float = 0.5,
                             reduction: str = "mean",
                             eps: float = 1e-12):
    """
    Predicted prevalence for an interval predicted by ordinal logits.
    """
    _ = target

    pred = ensure_4d_pt(pred)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    p_band = _logit_interval_pred_mask(pred, interval=interval, prob_threshold=prob_threshold) & m

    num_p = sum_over_hw(p_band.to(pred.dtype))
    denom = sum_over_hw(m.to(pred.dtype)).clamp_min(eps)

    prev = num_p / denom
    return _reduce_metric_pt(prev, reduction=reduction)


@METRIC_REGISTRY.register()
def cal_logit_prev_t_band_pt(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                             interval: str = "slight",
                             target_low: Optional[float] = None,
                             target_high: Optional[float] = None,
                             reduction: str = "mean",
                             eps: float = 1e-12):
    """
    Target prevalence for a physical water-depth interval.
    """
    _ = pred

    target = ensure_4d_pt(target)
    mask = ensure_4d_pt(mask)

    m = mask > 0.5

    t_band = _target_interval_mask(target, interval=interval, low=target_low, high=target_high) & m

    num_t = sum_over_hw(t_band.to(target.dtype))
    denom = sum_over_hw(m.to(target.dtype)).clamp_min(eps)

    prev = num_t / denom
    return _reduce_metric_pt(prev, reduction=reduction)