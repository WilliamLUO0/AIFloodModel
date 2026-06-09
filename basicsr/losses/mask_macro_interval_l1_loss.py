import math
import json
import os

import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskMacroIntervalL1Loss(nn.Module):
    """
    Macro-aggregated, per-interval weighted L1 regression loss.

    Differences vs MaskWeightedL1Loss:
      - Pixels are partitioned into 4 disjoint intervals by raw target depth
        (in label space, after asinh/log1p + zscore/minmax):
          nonflood : target <  tau_slight_std
          slight   : tau_slight_std  <= target < tau_severe_std
          severe   : tau_severe_std  <= target < tau_extreme_std
          extreme  : target >= tau_extreme_std
      - Each interval contributes the BATCH-AGGREGATED mean L1 error over its
        own pixels (macro mean), not a per-sample micro average. So very rare
        intervals are NOT drowned out by the bulk of nonflood pixels — every
        interval has comparable "voice" in the loss.
      - The 4 per-interval means are combined via a weighted sum with
        per-interval weights w_nonflood, w_slight, w_severe, w_extreme.
        Two weighting modes:
          - "manual":            user supplies the four weights directly.
          - "log_inverse_freq":  weights auto-computed from stats_json's
                                  flood_interval_stats_fine_raw.ratios as
                                  w_k = 1 + log(f_nonflood / f_k), so
                                  w_nonflood is anchored at 1.0 and rarer
                                  intervals get larger weight.

    Empty-interval handling:
      If a batch contains zero pixels in some interval (e.g. no extreme cells
      in a particular batch), that term contributes 0 to the total — see the
      `safe_mean` construction in forward: (sum) / (N + eps) reduces to 0/eps
      when both numerator and denominator are 0 in floating point.

    Logging:
      When `log_components=True`, populates `self.last_components` after each
      forward with per-interval means and counts (all detached). The model
      code is expected to merge this into its loss_dict for log output. Keys
      use the `pix/` prefix to group cleanly under one panel in TensorBoard.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau_slight_m: float = 0.1,
        tau_severe_m: float = 0.5,
        tau_extreme_m: float = 1.0,
        tau_is_physical: bool = True,
        var: str = "h",
        transform: str = "log1p",
        norm: str = "zscore",
        stats_json: str = "",
        h_asinh_q: int = 90,

        weight_mode: str = "manual",
        w_nonflood: float = 1.0,
        w_slight: float = 1.0,
        w_severe: float = 1.0,
        w_extreme: float = 1.0,

        log_components: bool = True,
        ignore_zero_mask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        logger = get_root_logger()

        self.loss_weight = float(loss_weight)
        self.tau_slight_m = float(tau_slight_m)
        self.tau_severe_m = float(tau_severe_m)
        self.tau_extreme_m = float(tau_extreme_m)
        self.tau_is_physical = bool(tau_is_physical)

        self.var = str(var).lower().strip()
        self.transform = str(transform).lower().strip()
        self.norm = str(norm).lower().strip()
        self.h_asinh_q = int(h_asinh_q)

        if self.var in ("u", "v"):
            self.transform = "asinh"

        if self.transform not in ("log1p", "asinh"):
            raise ValueError(f"[MaskMacroIntervalL1Loss] transform must be log1p/asinh, got {self.transform}")
        if self.norm not in ("zscore", "minmax"):
            raise ValueError(f"[MaskMacroIntervalL1Loss] norm must be zscore/minmax, got {self.norm}")

        self.weight_mode = str(weight_mode).lower().strip()
        if self.weight_mode not in ("manual", "log_inverse_freq"):
            raise ValueError(
                f"[MaskMacroIntervalL1Loss] weight_mode must be manual/log_inverse_freq, got {self.weight_mode}"
            )

        self.log_components = bool(log_components)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

        # ---------------------------- stats_json ---------------------------- #
        stats_json = str(stats_json).strip()
        if stats_json == "":
            raise ValueError("[MaskMacroIntervalL1Loss] stats_json is required.")
        if not os.path.exists(stats_json):
            raise FileNotFoundError(f"[MaskMacroIntervalL1Loss] stats_json not found: {stats_json}")

        with open(stats_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "stats_var" not in meta:
            raise KeyError(f"[MaskMacroIntervalL1Loss] stats_json must contain key 'stats_var': {stats_json}")
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

        # ---------------------------- thresholds ---------------------------- #
        tau_slight_std = self._compute_tau_std(self.tau_slight_m)
        tau_severe_std = self._compute_tau_std(self.tau_severe_m)
        tau_extreme_std = self._compute_tau_std(self.tau_extreme_m)
        if not (tau_slight_std < tau_severe_std < tau_extreme_std):
            raise ValueError(
                f"[MaskMacroIntervalL1Loss] thresholds must be strictly increasing in label space, "
                f"got tau_slight_std={tau_slight_std}, tau_severe_std={tau_severe_std}, "
                f"tau_extreme_std={tau_extreme_std}"
            )
        self.register_buffer("tau_slight_std", torch.tensor(tau_slight_std, dtype=torch.float32))
        self.register_buffer("tau_severe_std", torch.tensor(tau_severe_std, dtype=torch.float32))
        self.register_buffer("tau_extreme_std", torch.tensor(tau_extreme_std, dtype=torch.float32))

        # ------------------------- per-interval weights ------------------------- #
        if self.weight_mode == "manual":
            self.w_nonflood = float(w_nonflood)
            self.w_slight = float(w_slight)
            self.w_severe = float(w_severe)
            self.w_extreme = float(w_extreme)
        else:
            # log_inverse_freq mode reads ratios from stats_json
            try:
                ratios = S_var["flood_interval_stats_fine_raw"]["ratios"]
                f_non = float(ratios["nonflood"])
                f_slt = float(ratios["slightflood"])
                f_sev = float(ratios["severeflood"])
                f_ext = float(ratios["extremeflood"])
            except (KeyError, TypeError) as e:
                raise KeyError(
                    f"[MaskMacroIntervalL1Loss] weight_mode=log_inverse_freq requires "
                    f"stats_var.flood_interval_stats_fine_raw.ratios.{{nonflood,slightflood,severeflood,extremeflood}} "
                    f"in stats_json, got error: {e}"
                )
            if min(f_non, f_slt, f_sev, f_ext) <= 0:
                raise ValueError(
                    f"[MaskMacroIntervalL1Loss] log_inverse_freq requires all ratios > 0, "
                    f"got nonflood={f_non}, slight={f_slt}, severe={f_sev}, extreme={f_ext}"
                )
            self.w_nonflood = 1.0
            self.w_slight = 1.0 + math.log(f_non / f_slt)
            self.w_severe = 1.0 + math.log(f_non / f_sev)
            self.w_extreme = 1.0 + math.log(f_non / f_ext)

        # Manual override warning if user also set manual weights but picked log_inverse_freq
        # (not an error — log_inverse_freq just ignores the manual values silently)

        self.last_components = None

        logger.info(
            f"[MaskMacroIntervalL1Loss] var={self.var}, transform={self.transform}, norm={self.norm}, "
            f"h_asinh_q={self.h_asinh_q if (self.var == 'h' and self.transform == 'asinh') else None}, "
            f"asinh_scale={self.asinh_scale if self.transform == 'asinh' else None}, "
            f"tau_slight_m={self.tau_slight_m} -> tau_std={tau_slight_std:.6f}, "
            f"tau_severe_m={self.tau_severe_m} -> tau_std={tau_severe_std:.6f}, "
            f"tau_extreme_m={self.tau_extreme_m} -> tau_std={tau_extreme_std:.6f}, "
            f"weight_mode={self.weight_mode}, "
            f"w=({self.w_nonflood:.4f}, {self.w_slight:.4f}, {self.w_severe:.4f}, {self.w_extreme:.4f}), "
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

    # ---------------------------- forward ---------------------------- #
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype)

        diff = F.l1_loss(pred, target, reduction="none")  # [B,1,H,W]

        with torch.no_grad():
            # Build 4 disjoint interval masks (already restricted by valid mask).
            m_non = mask * (target < self.tau_slight_std).to(dtype=diff.dtype)
            m_slt = mask * (
                (target >= self.tau_slight_std) & (target < self.tau_severe_std)
            ).to(dtype=diff.dtype)
            m_sev = mask * (
                (target >= self.tau_severe_std) & (target < self.tau_extreme_std)
            ).to(dtype=diff.dtype)
            m_ext = mask * (target >= self.tau_extreme_std).to(dtype=diff.dtype)

        # Batch-aggregated counts and sums.
        N_non = m_non.sum()
        N_slt = m_slt.sum()
        N_sev = m_sev.sum()
        N_ext = m_ext.sum()

        S_non = (diff * m_non).sum()
        S_slt = (diff * m_slt).sum()
        S_sev = (diff * m_sev).sum()
        S_ext = (diff * m_ext).sum()

        # Macro mean per interval. When N==0, S is also 0, so 0/eps = 0 (safe).
        L_non = S_non / (N_non + self.eps)
        L_slt = S_slt / (N_slt + self.eps)
        L_sev = S_sev / (N_sev + self.eps)
        L_ext = S_ext / (N_ext + self.eps)

        weighted_total = (
            self.w_nonflood * L_non
            + self.w_slight * L_slt
            + self.w_severe * L_sev
            + self.w_extreme * L_ext
        )

        # Fallback for the entire-batch-empty edge case (would be very weird,
        # e.g. mask all zeros). Keeps gradient graph alive with a zero tensor.
        if self.ignore_zero_mask:
            total_valid = N_non + N_slt + N_sev + N_ext
            if total_valid.item() <= 0:
                weighted_total = pred.new_tensor(0.0, requires_grad=True) * 0.0

        if self.log_components:
            self.last_components = {
                "pix/l_nonflood": L_non.detach(),
                "pix/l_slight": L_slt.detach(),
                "pix/l_severe": L_sev.detach(),
                "pix/l_extreme": L_ext.detach(),
                "pix/n_nonflood": N_non.detach(),
                "pix/n_slight": N_slt.detach(),
                "pix/n_severe": N_sev.detach(),
                "pix/n_extreme": N_ext.detach(),
            }
        else:
            self.last_components = None

        return self.loss_weight * weighted_total
