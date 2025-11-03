"""
train:
  flood_bce_opt:
    type: MaskBCELoss
    loss_weight: 0.05
    tau_flood_m: 0.05
    tau_is_physical: true
    var: 'h'
    stats_mean: 0.05575
    stats_std: 0.17930
    asinh_scale: xxx
    pos_weight_mode: 'batch'
    focal_gamma: 2
    alpha_pos: 0.7
    ignore_zero_mask: true
    eps: 1.0e-12
"""


import math
import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger


@LOSS_REGISTRY.register()
class MaskBCELoss(nn.Module):
    """ Mean over samples of binary cross-entropy for flood/non-flood"""

    def __init__(self,
                 loss_weight: float = 1.0,
                 tau_flood_m: float = 0.05,
                 tau_is_physical: bool = True,
                 var: str = 'h',
                 stats_mean: float = 0.0,
                 stats_std: float = 1.0,
                 asinh_scale: float = 1.0,
                 pos_weight_mode: str = 'batch',
                 focal_gamma: float = 0.0,
                 alpha_pos: float = 0.7,
                 ignore_zero_mask: bool = True,
                 eps: float = 1e-12,
                 pos_weight_min: float = 0.5,
                 pos_weight_max: float = 5.0):
        super(MaskBCELoss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.pos_weight_mode = pos_weight_mode
        self.focal_gamma = float(focal_gamma)
        self.alpha_pos = float(alpha_pos)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)
        self.pos_weight_min = float(pos_weight_min)
        self.pos_weight_max = float(pos_weight_max)

        if tau_is_physical:
            if var == 'h':
                tau_std = (math.log1p(float(tau_flood_m)) - float(stats_mean)) / float(stats_std)
            else:
                s = float(asinh_scale)
                tau_std = (math.asinh(float(tau_flood_m) / (s + self.eps)) - float(stats_mean)) / float(stats_std)
        else:
            tau_std = float(tau_flood_m)
        self.register_buffer('tau_std', torch.tensor(tau_std, dtype=torch.float32))
        logger = get_root_logger()
        logger.info(f'[MaskBCELoss] tau_std = {float(self.tau_std):.6f}')

    def _compute_pos_weight(self, wet_true: torch.Tensor, mask: torch.Tensor):
        pos = (wet_true * mask).sum()
        neg = ((1.0 - wet_true) * mask).sum()
        pos_w = (neg / (pos + self.eps)).clamp(self.pos_weight_min, self.pos_weight_max)
        return pos_w

    def forward(self, pred_logit: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        with torch.no_grad():
            wet_true = (target_depth > self.tau_std).float()

        if isinstance(self.pos_weight_mode, str) and self.pos_weight_mode.lower() == 'batch':
            pos_weight = self._compute_pos_weight(wet_true, mask)
        else:
            pos_weight = torch.tensor(float(self.pos_weight_mode), device=pred_logit.device)

        if self.focal_gamma > 0:
            p = torch.sigmoid(pred_logit)
            bce = -(wet_true * torch.log(p + self.eps) + (1.0 - wet_true) * torch.log(1.0 - p + self.eps))
            weight = torch.where(wet_true > 0.5, self.alpha_pos * (1.0 - p).pow(self.focal_gamma), (1.0 - self.alpha_pos) * p.pow(self.focal_gamma))
            loss_map = weight * bce
        else:
            loss_map = F.binary_cross_entropy_with_logits(pred_logit, wet_true, reduction='none', pos_weight=pos_weight)

        masked = loss_map * mask
        per_sample_num = mask.flatten(1).sum(1)
        per_sample_loss = masked.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_num > 0
            if not valid.any():
                return pred_logit.new_tensor(0.0)
            loss = (per_sample_loss[valid] / (per_sample_num[valid] + self.eps)).mean()
        else:
            loss = (per_sample_loss / per_sample_num).mean()

        return self.loss_weight * loss


