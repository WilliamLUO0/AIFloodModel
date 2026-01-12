"""
Example (pos_weight):
flood_bce_opt:
  type: MaskBCELoss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_pos_weight: true
  pos_weight_min: 0.5
  pos_weight_max: 20.0
  focal_gamma: 0.0
  alpha_pos: 0.75
  ignore_zero_mask: true
  eps: 1.0e-12

Example (focal_gamma):
flood_bce_opt:
  type: MaskBCELoss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_pos_weight: false
  focal_gamma: 1.0
  alpha_pos: 0.75
  ignore_zero_mask: true
  eps: 1.0e-12

Example (focal + pos):
flood_bce_opt:
  type: MaskBCELoss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_pos_weight: true
  pos_weight_min: 0.5
  pos_weight_max: 20.0
  focal_gamma: 1.0
  alpha_pos: 0.5
  ignore_zero_mask: true
  eps: 1.0e-12
"""

import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskBCELoss(nn.Module):
    """
    Masked flood/non-flood classification loss on logits.

    Inputs:
      pred_logit   : [B,1,H,W] logits (NO sigmoid)
      target_depth : [B,1,H,W] normalized flood map target (same space as regression label)
      mask         : [B,1,H,W] or [B,H,W] AOI mask

    wet_true = (target_depth >= tau_std), where tau_std is computed from physical tau_flood_m
              by applying the SAME (transform + norm) as the dataset labels.
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

        use_pos_weight: bool = False,
        pos_weight_min: float = 0.5,
        pos_weight_max: float = 50.0,

        focal_gamma: float = 0.0,
        alpha_pos: float = 0.75,

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
            raise ValueError(f"[MaskBCELoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskBCELoss] norm must be zscore/minmax, got {self.norm}")

        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_min = float(pos_weight_min)
        self.pos_weight_max = float(pos_weight_max)

        self.focal_gamma = float(focal_gamma)
        self.alpha_pos = float(alpha_pos)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskBCELoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskBCELoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskBCELoss] stats_json must contain key 'stats_var': {stats_json}")
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

        self.pos_weight = None
        if self.use_pos_weight:
            pw = float(S_var["pos_weight_fine_raw_tau"])
            pw = max(self.pos_weight_min, min(self.pos_weight_max, pw))
            self.register_buffer("pos_weight_buf", torch.tensor(pw, dtype=torch.float32))
            self.pos_weight = self.pos_weight_buf

        tau_std = self._compute_tau_std()
        self.register_buffer("tau_std", torch.tensor(tau_std, dtype=torch.float32))

        logger.info(
            f"[MaskBCELoss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var=='h' and self.transform=='asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform=='asinh' else None}, "
            f"tau_flood_m={self.tau_flood_m}, tau_std={tau_std:.6f}, "
            f"use_pos_weight={self.use_pos_weight}, pos_weight={float(self.pos_weight) if self.pos_weight is not None else None}, "
            f"focal_gamma={self.focal_gamma}, alpha_pos={self.alpha_pos if self.focal_gamma>0 else None}"
        )

    def _transform_physical_scalar(self, x_m: float) -> float:
        """Apply transform in physical space (scalar)."""
        if self.transform == "log1p":
            return math.log1p(max(float(x_m), 0.0))
        s = float(self.asinh_scale) if self.asinh_scale is not None else 1.0
        s = max(s, 1e-12)
        return math.asinh(float(x_m) / s)

    def _norm_scalar(self, x_t: float) -> float:
        """Apply norm in transformed space (scalar)."""
        if self.norm == "zscore":
            return (float(x_t) - self.stats_mean) / (self.stats_std + self.eps)
        return (float(x_t) - self.stats_min) / (self.stats_max - self.stats_min + self.eps)

    def _compute_tau_std(self) -> float:
        """tau threshold in the same normalized label space as target_depth."""
        if not self.tau_is_physical:
            return float(self.tau_flood_m)
        tau_t = self._transform_physical_scalar(self.tau_flood_m)
        return float(self._norm_scalar(tau_t))

    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        """
        pred_logit: [B,1,H,W]
        target_depth: [B,1,H,W] (normalized labels)
        mask: [B,1,H,W] or [B,H,W]
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        with torch.no_grad():
            wet_true = (target_depth >= self.tau_std).to(dtype=pred_logit.dtype)

        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(pred_logit, wet_true, reduction="none")
        else:
            bce = F.binary_cross_entropy_with_logits(
                pred_logit, wet_true, reduction="none", pos_weight=self.pos_weight
            )

        if self.focal_gamma > 0:
            p = torch.sigmoid(pred_logit)
            pt = p * wet_true + (1.0 - p) * (1.0 - wet_true)
            alpha_t = self.alpha_pos * wet_true + (1.0 - self.alpha_pos) * (1.0 - wet_true)
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
                return pred_logit.new_tensor(0.0)
            loss = (per_sample_loss[valid] / (per_sample_num[valid] + self.eps)).mean()
        else:
            loss = (per_sample_loss / (per_sample_num + self.eps)).mean()

        return self.loss_weight * loss
