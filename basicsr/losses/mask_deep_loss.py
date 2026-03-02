import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskDeepOnlyLoss(nn.Module):
    """
    Deep-only regression loss.

    Select deep pixels by GT target >= deep_tau (in label space, mapped from physical deep_tau_m if deep_tau_is_physical).

    L_deep = loss_weight * mean( loss_fn(pred, target) * deep_sel * mask ) / mean( deep_sel * mask )

    loss_fn:
      - smoothl1 (Huber): F.smooth_l1_loss(..., beta=deep_beta)
      - l1
      - l2
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        deep_tau_m: float = 0.5,
        deep_tau_is_physical: bool = True,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,

        deep_loss_type: str = "smoothl1",  # smoothl1 | l1 | l2
        deep_beta: float = 0.1,            # SmoothL1 beta in label space

        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.deep_tau_m = float(deep_tau_m)
        self.deep_tau_is_physical = bool(deep_tau_is_physical)

        self.var = str(var).lower().strip()
        self.transform = str(transform).lower().strip()
        self.norm = str(norm).lower().strip()
        self.h_asinh_q = int(h_asinh_q)

        if self.var in ("u", "v"):
            self.transform = "asinh"

        if self.transform not in ("log1p", "asinh"):
            raise ValueError(f"[MaskDeepOnlyLoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskDeepOnlyLoss] norm must be zscore/minmax, got {self.norm}")

        self.deep_loss_type = str(deep_loss_type).lower().strip()
        if self.deep_loss_type not in ("smoothl1", "l1", "l2"):
            raise ValueError(f"[MaskDeepOnlyLoss] deep_loss_type must be smoothl1/l1/l2, got {self.deep_loss_type}")

        self.deep_beta = float(deep_beta)

        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskDeepOnlyLoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskDeepOnlyLoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskDeepOnlyLoss] stats_json must contain key 'stats_var': {stats_json}")
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

        deep_tau_std = self._compute_deep_tau_std()
        self.register_buffer("deep_tau_std", torch.tensor(deep_tau_std, dtype=torch.float32))

        logger.info(
            f"[MaskDeepOnlyLoss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var == 'h' and self.transform == 'asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform == 'asinh' else None}, "
            f"deep_tau_m={self.deep_tau_m}, deep_tau_is_physical={self.deep_tau_is_physical}, "
            f"deep_tau_std={float(self.deep_tau_std.item()):.6f}, "
            f"deep_loss_type={self.deep_loss_type}, deep_beta={self.deep_beta if self.deep_loss_type=='smoothl1' else None}"
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

    def _compute_deep_tau_std(self) -> float:
        if not self.deep_tau_is_physical:
            return float(self.deep_tau_m)
        tau_t = self._transform_physical_scalar(self.deep_tau_m)
        return float(self._norm_scalar(tau_t))

    def _pointwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.deep_loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        if self.deep_loss_type == "l2":
            return (pred - target) ** 2
        # smoothl1
        return F.smooth_l1_loss(pred, target, reduction="none", beta=max(self.deep_beta, self.eps))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        pw = self._pointwise_loss(pred, target)

        with torch.no_grad():
            deep_sel = (target >= self.deep_tau_std).to(dtype=pw.dtype)

        num = (pw * deep_sel * mask).flatten(1).sum(1)
        den = (deep_sel * mask).flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = den > 0
            if valid.any():
                deep_loss = (num[valid] / (den[valid] + self.eps)).mean()
            else:
                deep_loss = pred.new_tensor(0.0)
        else:
            deep_loss = (num / (den + self.eps)).mean()

        return self.loss_weight * deep_loss