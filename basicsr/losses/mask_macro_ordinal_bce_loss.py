import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskMacroOrdinalBCELoss(nn.Module):
    """
    Macro-aggregated ordinal BCE loss for 3 ordered thresholds.

    pred_logit:   [B, 3, H, W]
      channel 0 -> h >= tau1
      channel 1 -> h >= tau2
      channel 2 -> h >= tau3
    target_depth: [B, 1, H, W]  (same normalized label space as regression target)
    mask:         [B, 1, H, W] or [B, H, W]

    Differences vs MaskOrdinalBCELoss:
      - Per threshold channel, the BCE map is split into POS (target>=tau) and
        NEG (target<tau) subsets and each subset's loss is the BATCH-AGGREGATED
        mean (macro) rather than a per-sample micro average. The channel total
        is L_neg + lambda_pos * L_pos, then per-channel weighted sum.
      - `pos_weight` clamping is gone: macro normalization itself gives pos and
        neg equal voice when lambda_pos=1.0, which is structurally stronger
        than any pos_weight up to the clamp ceiling.
      - `alpha_pos` is gone for the same reason; its role is replaced by
        lambda_pos in the macro aggregation. Manual emphasis on positives is
        controlled by lambda_pos directly.
      - `focal_gamma` is kept as an INDEPENDENT axis: it controls
        hard-vs-easy example focus and is orthogonal to class balance. Default
        0 (off); set >0 to enable hard-example mining.

    Empty-subset handling: if a batch has zero pos or zero neg pixels for some
    channel, that subset's mean is the safe 0/eps = 0 (numerator is also 0).
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

        lambda_pos: float = 1.0,

        # optional per-threshold weights (for cross-channel weighted sum)
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,

        # hard-example focal modulation (orthogonal to class balance)
        focal_gamma: float = 0.0,

        log_components: bool = True,
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
            raise ValueError(f"[MaskMacroOrdinalBCELoss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskMacroOrdinalBCELoss] norm must be zscore/minmax, got {self.norm}")

        self.lambda_pos = float(lambda_pos)

        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

        self.focal_gamma = float(focal_gamma)

        self.log_components = bool(log_components)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        # ---------------------------- stats_json ---------------------------- #
        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskMacroOrdinalBCELoss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskMacroOrdinalBCELoss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskMacroOrdinalBCELoss] stats_json must contain key 'stats_var': {stats_json}")
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
        if not (tau1_std < tau2_std < tau3_std):
            raise ValueError(
                f"[MaskMacroOrdinalBCELoss] thresholds must be strictly increasing in label space, "
                f"got tau1_std={tau1_std}, tau2_std={tau2_std}, tau3_std={tau3_std}"
            )

        self.register_buffer("tau1_std", torch.tensor(tau1_std, dtype=torch.float32))
        self.register_buffer("tau2_std", torch.tensor(tau2_std, dtype=torch.float32))
        self.register_buffer("tau3_std", torch.tensor(tau3_std, dtype=torch.float32))

        self.last_components = None

        logger.info(
            f"[MaskMacroOrdinalBCELoss] tau_std=({tau1_std:.6f}, {tau2_std:.6f}, {tau3_std:.6f}), "
            f"channel_weights=({self.w1}, {self.w2}, {self.w3}), "
            f"lambda_pos={self.lambda_pos}, focal_gamma={self.focal_gamma}, "
            f"loss_weight={self.loss_weight}, log_components={self.log_components}"
        )

    # ---------------------------- helpers ---------------------------- #
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

    def _macro_channel(self, logit: torch.Tensor, target_bin: torch.Tensor, mask: torch.Tensor):
        """Compute (L_neg, L_pos, N_pos) for a single threshold channel.

        logit, target_bin, mask: [B, 1, H, W], all same dtype.
        target_bin is the 0/1 indicator of "target >= tau" for this channel.
        Returns three scalar tensors.
        """
        bce_map = F.binary_cross_entropy_with_logits(logit, target_bin, reduction="none")

        if self.focal_gamma > 0:
            p = torch.sigmoid(logit)
            pt = p * target_bin + (1.0 - p) * (1.0 - target_bin)
            mod = (1.0 - pt).clamp_min(0.0).pow(self.focal_gamma)
            bce_map = mod * bce_map

        m_pos = mask * target_bin
        m_neg = mask * (1.0 - target_bin)

        N_pos = m_pos.sum()
        N_neg = m_neg.sum()

        S_pos = (bce_map * m_pos).sum()
        S_neg = (bce_map * m_neg).sum()

        L_pos = S_pos / (N_pos + self.eps)
        L_neg = S_neg / (N_neg + self.eps)
        return L_neg, L_pos, N_pos, N_neg

    # ---------------------------- forward ---------------------------- #
    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        if pred_logit.dim() != 4 or pred_logit.shape[1] != 3:
            raise ValueError(
                f"[MaskMacroOrdinalBCELoss] pred_logit must have shape [B,3,H,W], got {pred_logit.shape}"
            )

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred_logit.dtype)

        with torch.no_grad():
            t1 = (target_depth >= self.tau1_std).to(dtype=pred_logit.dtype)
            t2 = (target_depth >= self.tau2_std).to(dtype=pred_logit.dtype)
            t3 = (target_depth >= self.tau3_std).to(dtype=pred_logit.dtype)

        L1_neg, L1_pos, N1_pos, N1_neg = self._macro_channel(pred_logit[:, 0:1], t1, mask)
        L2_neg, L2_pos, N2_pos, N2_neg = self._macro_channel(pred_logit[:, 1:2], t2, mask)
        L3_neg, L3_pos, N3_pos, N3_neg = self._macro_channel(pred_logit[:, 2:3], t3, mask)

        L_ch1 = L1_neg + self.lambda_pos * L1_pos
        L_ch2 = L2_neg + self.lambda_pos * L2_pos
        L_ch3 = L3_neg + self.lambda_pos * L3_pos

        weighted_total = self.w1 * L_ch1 + self.w2 * L_ch2 + self.w3 * L_ch3

        if self.ignore_zero_mask:
            total_valid = mask.sum()
            if total_valid.item() <= 0:
                weighted_total = pred_logit.new_tensor(0.0, requires_grad=True) * 0.0

        if self.log_components:
            n_total = mask.sum()
            self.last_components = {
                "bce/l_tau1_neg": L1_neg.detach(),
                "bce/l_tau1_pos": L1_pos.detach(),
                "bce/l_tau2_neg": L2_neg.detach(),
                "bce/l_tau2_pos": L2_pos.detach(),
                "bce/l_tau3_neg": L3_neg.detach(),
                "bce/l_tau3_pos": L3_pos.detach(),
                "bce/n_total": n_total.detach(),
                "bce/n_tau1_pos": N1_pos.detach(),
                "bce/n_tau2_pos": N2_pos.detach(),
                "bce/n_tau3_pos": N3_pos.detach(),
            }
        else:
            self.last_components = None

        return self.loss_weight * weighted_total
