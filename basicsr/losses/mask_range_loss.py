"""
  slight_opt:
    type: MaskRangeLoss
    loss_weight: 1.0
    range_low_m: 0.1
    range_high_m: 0.5
    range_is_physical: true
    use_low: true
    use_high: true
    low_inclusive: true
    high_inclusive: false
    var: 'h'
    transform: asinh
    h_asinh_q: 90
    norm: zscore
    stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
    range_loss_type: smoothl1
    range_beta: 0.1
    ignore_zero_mask: true
    eps: 1.0e-12

  deep_opt:
    type: MaskRangeLoss
    loss_weight: 1.0
    range_low_m: 0.5
    range_high_m: 0.0   # use_high: false
    range_is_physical: true
    use_low: true
    use_high: false
    low_inclusive: true
    high_inclusive: false
    var: 'h'
    transform: asinh
    h_asinh_q: 90
    norm: zscore
    stats_json: /path/to/split_stats_h_asinh.json
    range_loss_type: smoothl1
    range_beta: 0.1
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
class MaskRangeLoss(nn.Module):
    """
    Range-restricted regression loss.

    This loss applies pointwise regression only on GT target values selected
    by a configurable range in label space (or mapped from physical space).

    General form:
        L_range = loss_weight * mean( loss_fn(pred, target) * range_sel * mask ) / mean( range_sel * mask )

    Selection examples:
      - slight-only:   low <= target < high, e.g. 0.1m <= h < 0.5m
      - deep-only:     target >= low, e.g. h >= 0.5m

    Supported pointwise losses:
      - smoothl1 (Huber): F.smooth_l1_loss(..., beta=range_beta)
      - l1
      - l2
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        range_low_m: float = 0.1,
        range_high_m: float = 0.5,
        range_is_physical: bool = True,
        use_low: bool = True,
        use_high: bool = True,
        low_inclusive: bool = True,
        high_inclusive: bool = False,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,

        range_loss_type: str = "smoothl1",  # smoothl1 | l1 | l2
        range_beta: float = 0.1,  # SmoothL1 beta in label space

        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.range_low_m = float(range_low_m)
        self.range_high_m = float(range_high_m)
        self.range_is_physical = bool(range_is_physical)
        self.use_low = bool(use_low)
        self.use_high = bool(use_high)
        self.low_inclusive = bool(low_inclusive)
        self.high_inclusive = bool(high_inclusive)

        self.var = str(var).lower().strip()
        self.transform = str(transform).lower().strip()
        self.norm = str(norm).lower().strip()
        self.h_asinh_q = int(h_asinh_q)

        if self.var in ("u", "v"):
            self.transform = "asinh"

        if self.transform not in ("log1p", "asinh"):
            raise ValueError(f"[MaskRangeLoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskRangeLoss] norm must be zscore/minmax, got {self.norm}")

        self.range_loss_type = str(range_loss_type).lower().strip()
        if self.range_loss_type not in ("smoothl1", "l1", "l2"):
            raise ValueError(f"[MaskRangeLoss] range_loss_type must be smoothl1/l1/l2, got {self.range_loss_type}")

        self.range_beta = float(range_beta)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        if not self.use_low and not self.use_high:
            raise ValueError("[MaskRangeLoss] At least one of use_low/use_high must be True.")

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskRangeLoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskRangeLoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskRangeLoss] stats_json must contain key 'stats_var': {stats_json}")
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

        if self.use_low:
            low_std = self._compute_bound_std(self.range_low_m)
        else:
            low_std = float("-inf")

        if self.use_high:
            high_std = self._compute_bound_std(self.range_high_m)
        else:
            high_std = float("inf")

        if self.use_low and self.use_high and low_std > high_std:
            logger.warning(
                f"[MaskRangeLoss] low bound > high bound after mapping "
                f"(low_std={low_std:.6f}, high_std={high_std:.6f}). Swapping them."
            )
            low_std, high_std = high_std, low_std

        self.register_buffer("range_low_std", torch.tensor(low_std, dtype=torch.float32))
        self.register_buffer("range_high_std", torch.tensor(high_std, dtype=torch.float32))

        logger.info(
            f"[MaskRangeLoss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var == 'h' and self.transform == 'asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform == 'asinh' else None}, "
            f"range_is_physical={self.range_is_physical}, "
            f"use_low={self.use_low}, range_low_m={self.range_low_m if self.use_low else None}, "
            f"low_inclusive={self.low_inclusive if self.use_low else None}, "
            f"range_low_std={float(self.range_low_std.item()) if self.use_low else None}, "
            f"use_high={self.use_high}, range_high_m={self.range_high_m if self.use_high else None}, "
            f"high_inclusive={self.high_inclusive if self.use_high else None}, "
            f"range_high_std={float(self.range_high_std.item()) if self.use_high else None}, "
            f"range_loss_type={self.range_loss_type}, "
            f"range_beta={self.range_beta if self.range_loss_type == 'smoothl1' else None}"
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

    def _compute_bound_std(self, x_val: float) -> float:
        if not self.range_is_physical:
            return float(x_val)
        x_t = self._transform_physical_scalar(x_val)
        return float(self._norm_scalar(x_t))

    def _pointwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.range_loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        if self.range_loss_type == "l2":
            return (pred - target) ** 2
        # smoothl1
        return F.smooth_l1_loss(pred, target, reduction="none", beta=max(self.range_beta, self.eps))

    def _build_range_selector(self, target: torch.Tensor) -> torch.Tensor:
        sel = torch.ones_like(target, dtype=torch.bool)
        if self.use_low:
            if self.low_inclusive:
                sel = sel & (target >= self.range_low_std)
            else:
                sel = sel & (target > self.range_low_std)
        if self.use_high:
            if self.high_inclusive:
                sel = sel & (target <= self.range_high_std)
            else:
                sel = sel & (target < self.range_high_std)
        return sel

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        pw = self._pointwise_loss(pred, target)

        with torch.no_grad():
            range_sel = self._build_range_selector(target).to(dtype=pw.dtype)

        num = (pw * range_sel * mask).flatten(1).sum(1)
        den = (range_sel * mask).flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = den > 0
            if valid.any():
                range_loss = (num[valid] / (den[valid] + self.eps)).mean()
            else:
                range_loss = pred.new_tensor(0.0)
        else:
            range_loss = (num / (den + self.eps)).mean()

        return self.loss_weight * range_loss