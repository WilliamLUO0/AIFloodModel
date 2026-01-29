"""
Example
prev_opt:
  type: MaskPrevalenceLoss
  loss_weight: 1.0e-2
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  ignore_zero_mask: true
  eps: 1.0e-12
"""

import math
import json
import os

import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskPrevalenceLoss(nn.Module):
    """
    Patch-wise prevalence regularizer on flood head probabilities.

    Inputs:
      pred_logit   : [B,1,H,W] logits (NO sigmoid)
      target_depth : [B,1,H,W] normalized flood map target (same space as regression label)
      mask         : [B,1,H,W] or [B,H,W] AOI/valid mask

    Definitions:
      tau_std  : threshold in normalized label space (computed from physical tau_flood_m)
      pi_t     : mean( 1[target_depth >= tau_std] * mask )   (supervision; detached)
      pi_p     : mean( sigmoid(pred_logit) * mask )          (differentiable)
      loss     : loss_weight * (pi_p - pi_t)^2
    """

    def __init__(
        self,
        loss_weight: float = 1e-3,
        tau_flood_m: float = 0.05,
        tau_is_physical: bool = True,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,
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
            raise ValueError(f"[MaskPrevalenceLoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskPrevalenceLoss] norm must be zscore/minmax, got {self.norm}")

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskPrevalenceLoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskPrevalenceLoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskPrevalenceLoss] stats_json must contain key 'stats_var': {stats_json}")
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
            f"[MaskPrevalenceLoss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var=='h' and self.transform=='asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform=='asinh' else None}, "
            f"tau_flood_m={self.tau_flood_m}, tau_std={tau_std:.6f}, "
            f"loss_weight={self.loss_weight}"
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

    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # pi_t: supervision (detach)
        with torch.no_grad():
            wet_true = (target_depth >= self.tau_std).to(dtype=pred_logit.dtype)
            per_sample_num = mask.flatten(1).sum(1)  # [B]
            pi_t = (wet_true * mask).flatten(1).sum(1) / (per_sample_num + self.eps)

        # pi_p: differentiable
        p = torch.sigmoid(pred_logit)
        per_sample_num = mask.flatten(1).sum(1)
        pi_p = (p * mask).flatten(1).sum(1) / (per_sample_num + self.eps)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return pred_logit.new_tensor(0.0)
            loss = (pi_p[valid] - pi_t[valid]).pow(2).mean()
        else:
            loss = (pi_p - pi_t).pow(2).mean()

        return self.loss_weight * loss
