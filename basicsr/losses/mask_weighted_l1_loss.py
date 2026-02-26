"""
Example (vanilla):
pixel_opt:
  type: MaskWeightedL1Loss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_wet_weight: false
  use_depth_weight: false
  ignore_zero_mask: true
  eps: 1.0e-12
  use_band_loss: false

Example (wet weight only):
pixel_opt:
  type: MaskWeightedL1Loss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_wet_weight: true
  wet_lambda: 2.0
  use_depth_weight: false
  ignore_zero_mask: true
  eps: 1.0e-12
  use_band_loss: false

Example (wet weight + depth weight):
pixel_opt:
  type: MaskWeightedL1Loss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_wet_weight: true
  wet_lambda: 2.0
  use_depth_weight: true
  depth_mu: 1.0
  depth_mode: exp
  depth_scale: 1.5
  weight_max: 20.0
  ignore_zero_mask: true
  eps: 1.0e-12
  use_band_loss: false

Example (wet weight + depth weight + band loss):
pixel_opt:
  type: MaskWeightedL1Loss
  loss_weight: 1.0
  tau_flood_m: 0.05
  tau_is_physical: true
  var: 'h'
  transform: asinh
  h_asinh_q: 90
  norm: zscore
  stats_json: /nesi/nobackup/uoa04425/zluo784/Exp1/AIFloodModel/dataset_backup/split_stats_h_asinh.json
  use_wet_weight: true
  wet_lambda: 2.0
  use_depth_weight: true
  depth_mu: 1.0
  depth_mode: exp
  depth_scale: 1.5
  weight_max: 20.0
  ignore_zero_mask: true
  eps: 1.0e-12
  use_band_loss: true
  band_lambda: 1.0
  band_is_physical: true
  band_delta_m: 0.05
  band_kernel: gaussian
  band_sigma_ratio: 0.5
  band_truncate: true
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
class MaskWeightedL1Loss(nn.Module):
    """
    Masked weighted L1 regression loss.

    Inputs:
      pred   : [B,1,H,W] predicted depth in the SAME label space as target (normalized)
      target : [B,1,H,W] GT depth label (normalized; after transform+norm)
      mask   : [B,1,H,W] or [B,H,W] AOI mask

    Weight map:
      wet = (target >= tau_std), where tau_std computed from physical tau_flood_m
            by applying SAME (transform + norm) as dataset labels.

      w = 1 + wet_lambda * wet + depth_mu * g(target, tau_std)
      g is a depth-strength term (0 at threshold, increases with depth).

    Optional threshold-band loss:
      Focus on near-threshold region to improve shallow-deep boundary (precision/recall/csi).

      L_band = band_lambda * mean(|pred - target| * band_w * mask)

      band_w can be:
        - hard band: 1[low<=target<=high]
        - gaussian: exp(-0.5 * ((target - tau)/sigma)^2)
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
        wet_lambda: float = 2.0,          # adds +2 on wet pixels => wet weight becomes 3
        use_depth_weight: bool = True,
        depth_mu: float = 1.0,            # multiplies g(...) and adds to weight
        depth_mode: str = "exp",          # 'exp' | 'linear' | 'power' | 'none'
        depth_scale: float = 1.0,         # scale in "excess" space (zscore/minmax both ok; minmax auto-normalized)
        depth_gamma: float = 1.0,         # only for 'power' mode

        weight_min: float = 0.0,
        weight_max: float = 20.0,
        ignore_zero_mask: bool = True,
        eps: float = 1e-12,

        use_band_loss: bool = False,
        band_lambda: float = 1.0,
        band_is_physical: bool = True,
        band_delta_m: float = 0.05,
        band_kernel: str = "gaussian",    # 'gaussian' | 'hard'
        band_sigma_ratio: float = 0.5,    # sigma = (half_bandwidth_std) * band_sigma_ratio
        band_truncate: bool = True,
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

        self.use_band_loss = bool(use_band_loss)
        self.band_lambda = float(band_lambda)
        self.band_is_physical = bool(band_is_physical)
        self.band_delta_m = float(band_delta_m)

        self.band_kernel = str(band_kernel).lower().strip()
        if self.band_kernel not in ("gaussian", "hard"):
            raise ValueError(f"[MaskWeightedL1Loss] band_kernel must be gaussian/hard, got {self.band_kernel}")

        self.band_sigma_ratio = float(band_sigma_ratio)
        self.band_truncate = bool(band_truncate)

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

        if self.use_band_loss:
            low_std, high_std = self._compute_band_low_high_std(tau_std=float(tau_std))
            self.register_buffer("band_low_std", torch.tensor(low_std, dtype=torch.float32))
            self.register_buffer("band_high_std", torch.tensor(high_std, dtype=torch.float32))

            # for gaussian, define sigma in std space
            halfw = 0.5 * (self.band_high_std - self.band_low_std).abs().clamp_min(self.eps)
            sigma = (halfw * max(self.band_sigma_ratio, self.eps)).clamp_min(self.eps)
            self.register_buffer("band_sigma_std", sigma.to(dtype=torch.float32))

        band_low = None
        band_high = None
        band_sigma = None
        if self.use_band_loss:
            band_low = float(self.band_low_std.item())
            band_high = float(self.band_high_std.item())
            if self.band_kernel == "gaussian":
                band_sigma = float(self.band_sigma_std.item())

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
            f"weight_clip=[{self.weight_min},{self.weight_max}], "
            f"use_band_loss={self.use_band_loss}, band_lambda={self.band_lambda if self.use_band_loss else None}, "
            f"band_kernel={self.band_kernel if self.use_band_loss else None}, "
            f"band_is_physical={self.band_is_physical if self.use_band_loss else None}, "
            f"band_delta_m={self.band_delta_m if self.use_band_loss else None}, "
            f"band_low_std={band_low}, band_high_std={band_high}, "
            f"band_truncate={self.band_truncate if self.use_band_loss else None}, "
            f"band_sigma_ratio={self.band_sigma_ratio if (self.use_band_loss and self.band_kernel == 'gaussian') else None}, "
            f"band_sigma_std={band_sigma}"
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
        """
        Compute [low_std, high_std] in label space for the threshold band.
        If band_is_physical: band_delta_m is meters, we map [tau-d, tau+d] through transform+norm.
        Else: band_delta_m is std-space half width.
        """
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

    # ---------- depth strength g(target, tau_std) ----------
    def _depth_strength(self, target: torch.Tensor) -> torch.Tensor:
        """
        target: normalized label space
        returns g in [0, ~1] (exp) or >=0 (linear/power)
        """
        # excess above threshold
        excess = (target - self.tau_std).clamp_min(0.0)

        if self.norm == "minmax":
            denom = (1.0 - self.tau_std).clamp_min(self.eps)
            excess = excess / denom

        if self.depth_mode == "none":
            return torch.zeros_like(excess)

        if self.depth_mode == "linear":
            return excess

        if self.depth_mode == "power":
            # g = (excess / scale)^gamma
            s = max(self.depth_scale, self.eps)
            return (excess / s).pow(self.depth_gamma)

        # default: exp saturation: g = 1 - exp(-excess/scale)
        s = max(self.depth_scale, self.eps)
        return 1.0 - torch.exp(-excess / s)

    def _band_weight(self, target: torch.Tensor) -> torch.Tensor:
        """
        Return band weight map in [0,1] (gaussian/hard).
        Assumes buffers band_low_std/band_high_std exist.
        """
        if self.band_kernel == "hard":
            return ((target >= self.band_low_std) & (target <= self.band_high_std)).to(target.dtype)

        # gaussian centered at tau_std
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

        # base L1
        diff = F.l1_loss(pred, target, reduction="none")

        with torch.no_grad():
            wet = (target >= self.tau_std).to(dtype=diff.dtype)

        # weight map
        w = diff.new_ones(diff.shape)

        if self.use_wet_weight:
            w = w + self.wet_lambda * wet

        if self.use_depth_weight and (self.depth_mu != 0):
            g = self._depth_strength(target).to(dtype=diff.dtype)
            w = w + self.depth_mu * g

        # clip for stability
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

        loss = self.loss_weight * main_loss

        if self.use_band_loss and (self.band_lambda != 0):
            with torch.no_grad():
                band_sel = ((target >= self.band_low_std) & (target <= self.band_high_std)).to(diff.dtype)
                band_w = self._band_weight(target).to(dtype=diff.dtype)

            band_num = (diff * band_w * mask).flatten(1).sum(1)
            band_den = (band_sel * mask).flatten(1).sum(1)

            if self.ignore_zero_mask:
                valid = band_den > 0
                if valid.any():
                    band_loss = (band_num[valid] / (band_den[valid] + self.eps)).mean()
                else:
                    band_loss = pred.new_tensor(0.0)
            else:
                band_loss = (band_num / (band_den + self.eps)).mean()

            loss = loss + self.band_lambda * band_loss

        return loss
