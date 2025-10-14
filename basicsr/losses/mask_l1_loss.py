"""
train:
  pixel_opt:
    type: MaskL1Loss
    loss_weight: 1.0
    ignore_zero_mask: true
    eps: 1.0e-12
    use_wet_mask: true
"""


import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskL1Loss(nn.Module):
    """Mean over samples of ( sum(|pred - target| * mask) / sum(mask) ) """

    def __init__(self, loss_weight: float = 1.0, ignore_zero_mask: bool = True, eps: float = 1e-12):
        super(MaskL1Loss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.ignore_zero_mask = bool(ignore_zero_mask)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # pred, target: [B, 1, Hf, Wf]; mask: [B, 1, Hf, Wf]
        if mask.dim() == 3:
            # [B, Hf, Wf]
            mask = mask.unsqueeze(1)

        diff = F.l1_loss(pred, target, reduction='none')
        masked_diff = diff * mask

        per_sample_masked_diff = masked_diff.flatten(1).sum(1)
        per_sample_mask = mask.flatten(1).sum(1)

        if self.ignore_zero_mask:
            valid = per_sample_mask > 0
            if not valid.any():
                return pred.new_tensor(0.0)
            loss = (per_sample_masked_diff[valid] / per_sample_mask[valid]).mean()
        else:
            loss = (per_sample_masked_diff / (per_sample_mask + self.eps)).mean()

        return self.loss_weight * loss
