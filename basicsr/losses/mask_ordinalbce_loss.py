import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskOrdinalBCELoss(nn.Module):
    """
    Masked ordinal BCE loss for 3 ordered thresholds.

    pred_logit:   [B, 3, H, W]
      channel 0 -> h >= tau1
      channel 1 -> h >= tau2
      channel 2 -> h >= tau3

    target_depth: [B, 1, H, W]  (same normalized label space as regression target)
    mask:         [B, 1, H, W] or [B, H, W]
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau1_m: float = 0.1,
        tau2_m: float = 0.5,
        tau3_m: float = 1.0,
        tau_is_physical: bool = True,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,

        use_pos_weight: bool = False,
        pos_weight_min: float = 0.5,
        pos_weight_max: float = 50.0,

        focal_gamma: float = 0.0,
        alpha_pos: float = 0.75,

        # optional per-threshold weights
        w1: float = 1.0,   # for >= 0.1
        w2: float = 1.0,   # for >= 0.5
        w3: float = 1.0,   # for >= 1.0

        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.tau1_m = float(tau1_m)
        self.tau2_m = float(tau2_m)
        self.tau3_m = float(tau3_m)
        self.tau_is_physical = bool(tau_is_physical)

        self.var = str(var).lower().strip()
        self.transform = str(transform).lower().strip()
        self.norm = str(norm).lower().strip()
        self.h_asinh_q = int(h_asinh_q)

        if self.var in ("u", "v"):
            self.transform = "asinh"

        if self.transform not in ("log1p", "asinh"):
            raise ValueError(f"[MaskOrdinalBCELoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskOrdinalBCELoss] norm must be zscore/minmax, got {self.norm}")

        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_min = float(pos_weight_min)
        self.pos_weight_max = float(pos_weight_max)

        self.focal_gamma = float(focal_gamma)
        self.alpha_pos = float(alpha_pos)

        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskOrdinalBCELoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskOrdinalBCELoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskOrdinalBCELoss] stats_json must contain key 'stats_var': {stats_json}")
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

        tau1_std = self._compute_tau_std(self.tau1_m)
        tau2_std = self._compute_tau_std(self.tau2_m)
        tau3_std = self._compute_tau_std(self.tau3_m)

        self.register_buffer("tau1_std", torch.tensor(tau1_std, dtype=torch.float32))
        self.register_buffer("tau2_std", torch.tensor(tau2_std, dtype=torch.float32))
        self.register_buffer("tau3_std", torch.tensor(tau3_std, dtype=torch.float32))

        logger.info(
            f"[MaskOrdinalBCELoss] tau_std=({tau1_std:.6f}, {tau2_std:.6f}, {tau3_std:.6f}), "
            f"weights=({self.w1}, {self.w2}, {self.w3}), "
            f"use_pos_weight={self.use_pos_weight}, focal_gamma={self.focal_gamma}"
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

    def _compute_tau_std(self, tau_m: float) -> float:
        if not self.tau_is_physical:
            return float(tau_m)
        tau_t = self._transform_physical_scalar(tau_m)
        return float(self._norm_scalar(tau_t))

    def _masked_binary_loss(self, logit, target, mask):
        # logit,target,mask: [B,1,H,W]
        if self.use_pos_weight:
            pos = (target * mask).sum()
            neg = ((1.0 - target) * mask).sum()
            pw = (neg / (pos + self.eps)).clamp(self.pos_weight_min, self.pos_weight_max)
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none", pos_weight=pw)
        else:
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")

        if self.focal_gamma > 0:
            p = torch.sigmoid(logit)
            pt = p * target + (1.0 - p) * (1.0 - target)
            alpha_t = self.alpha_pos * target + (1.0 - self.alpha_pos) * (1.0 - target)
            mod = (1.0 - pt).clamp_min(0.0).pow(self.focal_gamma)
            loss_map = alpha_t * mod * bce
        else:
            loss_map = bce

        masked = loss_map * mask
        per_sample_num = mask.flatten(1).sum(1)
        per_sample_loss = masked.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return logit.new_tensor(0.0)
            return (per_sample_loss[valid] / (per_sample_num[valid] + self.eps)).mean()
        else:
            return (per_sample_loss / (per_sample_num + self.eps)).mean()

    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        if pred_logit.dim() != 4 or pred_logit.shape[1] != 3:
            raise ValueError(f"[MaskOrdinalBCELoss] pred_logit must have shape [B,3,H,W], got {pred_logit.shape}")

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred_logit.dtype)

        with torch.no_grad():
            t1 = (target_depth >= self.tau1_std).to(dtype=pred_logit.dtype)
            t2 = (target_depth >= self.tau2_std).to(dtype=pred_logit.dtype)
            t3 = (target_depth >= self.tau3_std).to(dtype=pred_logit.dtype)

        l1 = self._masked_binary_loss(pred_logit[:, 0:1], t1, mask)
        l2 = self._masked_binary_loss(pred_logit[:, 1:2], t2, mask)
        l3 = self._masked_binary_loss(pred_logit[:, 2:3], t3, mask)

        loss = self.w1 * l1 + self.w2 * l2 + self.w3 * l3
        return self.loss_weight * loss