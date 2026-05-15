import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskWeightedL1Loss(nn.Module):
    """
    Masked weighted L1 regression loss (main loss).

    pred/target are in the SAME label space (after transform + norm).
    mask is AOI mask.

    weight map:
      wet = (target >= tau_std)

      w = 1 + wet_lambda * wet + depth_mu * g(target, tau_std)
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

        use_wet_weight: bool = True,
        wet_lambda: float = 2.0,
        use_depth_weight: bool = True,
        depth_mu: float = 1.0,
        depth_mode: str = "exp",          # 'exp' | 'linear' | 'power' | 'none'
        depth_scale: float = 1.0,
        depth_gamma: float = 1.0,

        weight_min: float = 0.0,
        weight_max: float = 20.0,
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
            raise ValueError(f"[MaskWeightedL1Loss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskWeightedL1Loss] norm must be zscore/minmax, got {self.norm}")

        self.use_wet_weight = bool(use_wet_weight)
        self.wet_lambda = float(wet_lambda)

        self.use_depth_weight = bool(use_depth_weight)
        self.depth_mu = float(depth_mu)
        self.depth_mode = str(depth_mode).lower().strip()
        self.depth_scale = float(depth_scale)
        self.depth_gamma = float(depth_gamma)

        if self.depth_mode not in ("exp", "linear", "power", "none"):
            raise ValueError(f"[MaskWeightedL1Loss] depth_mode must be exp/linear/power/none, got {self.depth_mode}")

        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskWeightedL1Loss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskWeightedL1Loss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskWeightedL1Loss] stats_json must contain key 'stats_var': {stats_json}")
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

        logger.info(
            f"[MaskWeightedL1Loss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var == 'h' and self.transform == 'asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform == 'asinh' else None}, "
            f"tau_flood_m={self.tau_flood_m}, tau_std={float(self.tau_std.item()):.6f}, "
            f"use_wet_weight={self.use_wet_weight}, wet_lambda={self.wet_lambda if self.use_wet_weight else None}, "
            f"use_depth_weight={self.use_depth_weight}, depth_mu={self.depth_mu if self.use_depth_weight else None}, "
            f"depth_mode={self.depth_mode if self.use_depth_weight else None}, "
            f"depth_scale={self.depth_scale if self.use_depth_weight else None}, "
            f"depth_gamma={self.depth_gamma if (self.use_depth_weight and self.depth_mode == 'power') else None}, "
            f"weight_clip=[{self.weight_min},{self.weight_max}]"
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

    def _depth_strength(self, target: torch.Tensor) -> torch.Tensor:
        """ g(target, tau_std) """
        excess = (target - self.tau_std).clamp_min(0.0)

        if self.norm == "minmax":
            denom = (1.0 - self.tau_std).clamp_min(self.eps)
            excess = excess / denom

        if self.depth_mode == "none":
            return torch.zeros_like(excess)

        if self.depth_mode == "linear":
            return excess

        if self.depth_mode == "power":
            s = max(self.depth_scale, self.eps)
            return (excess / s).pow(self.depth_gamma)

        # exp saturating
        s = max(self.depth_scale, self.eps)
        return 1.0 - torch.exp(-excess / s)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        diff = F.l1_loss(pred, target, reduction="none")

        with torch.no_grad():
            wet = (target >= self.tau_std).to(dtype=diff.dtype)

        w = diff.new_ones(diff.shape)

        if self.use_wet_weight:
            w = w + self.wet_lambda * wet

        if self.use_depth_weight and (self.depth_mu != 0):
            g = self._depth_strength(target).to(dtype=diff.dtype)
            w = w + self.depth_mu * g

        w = w.clamp(self.weight_min, self.weight_max)

        main_map = diff * w
        main_masked = main_map * mask

        per_sample_num = mask.flatten(1).sum(1)
        per_sample_main = main_masked.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return pred.new_tensor(0.0)
            main_loss = (per_sample_main[valid] / (per_sample_num[valid] + self.eps)).mean()
        else:
            main_loss = (per_sample_main / (per_sample_num + self.eps)).mean()

        return self.loss_weight * main_loss