import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskBandLoss(nn.Module):
    """
    Threshold band loss around tau_flood_m to refine boundary.

    L_band = loss_weight * mean( |pred-target| * band_w * mask ) / mean( band_sel * mask )

    band_sel: hard support window [low, high] (std space)
    band_w  : 'hard' (==band_sel) or 'gaussian' (truncated to band_sel if band_truncate)
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau_flood_m: float = 0.05,
        tau_is_physical: bool = True,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,

        band_is_physical: bool = True,
        band_delta_m: float = 0.05,       # meters if band_is_physical else std-space half width
        band_kernel: str = "gaussian",    # 'gaussian' | 'hard'
        band_sigma_ratio: float = 0.5,    # sigma = half_bandwidth_std * ratio
        band_truncate: bool = True,

        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.tau_flood_m = float(tau_flood_m)
        self.tau_is_physical = bool(tau_is_physical)

        self.var = str(var).lower().strip()
        self.transform = str(transform).lower().strip()
        self.norm = str(norm).lower().strip()
        self.h_asinh_q = int(h_asinh_q)

        if self.var in ("u", "v"):
            self.transform = "asinh"

        if self.transform not in ("log1p", "asinh"):
            raise ValueError(f"[MaskBandLoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskBandLoss] norm must be zscore/minmax, got {self.norm}")

        self.band_is_physical = bool(band_is_physical)
        self.band_delta_m = float(band_delta_m)

        self.band_kernel = str(band_kernel).lower().strip()
        if self.band_kernel not in ("gaussian", "hard"):
            raise ValueError(f"[MaskBandLoss] band_kernel must be gaussian/hard, got {self.band_kernel}")

        self.band_sigma_ratio = float(band_sigma_ratio)
        self.band_truncate = bool(band_truncate)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskBandLoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskBandLoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskBandLoss] stats_json must contain key 'stats_var': {stats_json}")
        S_var = meta["stats_var"]

        self.asinh_scale = None

        if self.var == "h":
            if self.transform == "log1p":
                S = S_var["shared"]
            else:
                qk = str(int(self.h_asinh_q))
                node = S_var["asinh_by_q"][qk]
                self.asinh_scale = float(node["s"])
                S = node["shared"]
        else:
            S = S_var["shared"]
            self.asinh_scale = float(S.get("asinh_scale_shared", 1.0))

        self.stats_mean = float(S["mean"])
        self.stats_std = float(S["std"])
        self.stats_min = float(S["min"])
        self.stats_max = float(S["max"])

        tau_std = self._compute_tau_std()
        self.register_buffer("tau_std", torch.tensor(tau_std, dtype=torch.float32))

        low_std, high_std = self._compute_band_low_high_std(tau_std=float(tau_std))
        self.register_buffer("band_low_std", torch.tensor(low_std, dtype=torch.float32))
        self.register_buffer("band_high_std", torch.tensor(high_std, dtype=torch.float32))

        halfw = 0.5 * (self.band_high_std - self.band_low_std).abs().clamp_min(self.eps)
        sigma = (halfw * max(self.band_sigma_ratio, self.eps)).clamp_min(self.eps)
        self.register_buffer("band_sigma_std", sigma.to(dtype=torch.float32))

        logger.info(
            f"[MaskBandLoss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var == 'h' and self.transform == 'asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform == 'asinh' else None}, "
            f"tau_flood_m={self.tau_flood_m}, tau_std={float(self.tau_std.item()):.6f}, "
            f"band_is_physical={self.band_is_physical}, band_delta_m={self.band_delta_m}, "
            f"band_low_std={float(self.band_low_std.item()):.6f}, band_high_std={float(self.band_high_std.item()):.6f}, "
            f"band_kernel={self.band_kernel}, band_truncate={self.band_truncate}, "
            f"band_sigma_ratio={self.band_sigma_ratio}, band_sigma_std={float(self.band_sigma_std.item()):.6f}"
        )

    def _transform_physical_scalar(self, x_m: float) -> float:
        if self.transform == "log1p":
            return math.log1p(max(float(x_m), 0.0))
        s = float(self.asinh_scale) if self.asinh_scale is not None else 1.0
        s = max(s, 1e-12)
        return math.asinh(float(x_m) / s)

    def _norm_scalar(self, x_t: float) -> float:
        if self.norm == "zscore":
            return (float(x_t) - self.stats_mean) / (self.stats_std + self.eps)
        return (float(x_t) - self.stats_min) / (self.stats_max - self.stats_min + self.eps)

    def _compute_tau_std(self) -> float:
        if not self.tau_is_physical:
            return float(self.tau_flood_m)
        tau_t = self._transform_physical_scalar(self.tau_flood_m)
        return float(self._norm_scalar(tau_t))

    def _compute_band_low_high_std(self, tau_std: float):
        if not self.band_is_physical:
            lo = float(tau_std) - float(self.band_delta_m)
            hi = float(tau_std) + float(self.band_delta_m)
            return (min(lo, hi), max(lo, hi))

        low_m = max(0.0, float(self.tau_flood_m) - float(self.band_delta_m))
        high_m = float(self.tau_flood_m) + float(self.band_delta_m)

        low_std = self._norm_scalar(self._transform_physical_scalar(low_m))
        high_std = self._norm_scalar(self._transform_physical_scalar(high_m))

        lo = min(low_std, high_std)
        hi = max(low_std, high_std)
        return (float(lo), float(hi))

    def _band_weight(self, target: torch.Tensor) -> torch.Tensor:
        if self.band_kernel == "hard":
            return ((target >= self.band_low_std) & (target <= self.band_high_std)).to(target.dtype)

        z = (target - self.tau_std) / (self.band_sigma_std + self.eps)
        w = torch.exp(-0.5 * z * z)

        if self.band_truncate:
            inside = ((target >= self.band_low_std) & (target <= self.band_high_std)).to(target.dtype)
            w = w * inside
        return w

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        diff = F.l1_loss(pred, target, reduction="none")

        with torch.no_grad():
            band_sel = ((target >= self.band_low_std) & (target <= self.band_high_std)).to(diff.dtype)
            band_w = self._band_weight(target).to(dtype=diff.dtype)

        num = (diff * band_w * mask).flatten(1).sum(1)
        den = (band_sel * mask).flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = den > 0
            if valid.any():
                band_loss = (num[valid] / (den[valid] + self.eps)).mean()
            else:
                band_loss = pred.new_tensor(0.0)
        else:
            band_loss = (num / (den + self.eps)).mean()

        return self.loss_weight * band_loss