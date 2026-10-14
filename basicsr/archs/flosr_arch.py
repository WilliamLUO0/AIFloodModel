"""
flosr_arch.py -- Paper 2 S8: FLO-SR (Choi et al. 2025, J. Hydrol. 661, 133529) under our protocol.

FLO-SR is the EDSR *baseline* network applied to flood maps. Re-implemented from the published description and the
specification paper2_S8/S8_FLOSR.md (the released repository carries no licence, so nothing is copied from it; the
discussion session checks this file layer by layer against the released notebook):
    head            3x3 conv, in -> 64, no activation
    body            16 residual blocks  [3x3 conv -> ReLU -> 3x3 conv] + identity, no BN, residual scale 1.0
    conv_after_body 3x3 conv 64 -> 64, plus the global skip from the head output
    upsampler       x8 = 3 x [3x3 conv 64 -> 256 -> PixelShuffle(2)], no activation between stages
    conv_last       3x3 conv 64 -> 1, linear (no clamp: our target is asinh + z-score and may be negative)
S8 ruling (README paper2_S8 §3): ONE row, unguided as published -> the network sees the coarse depth only
(`static_mode: 'none'`, `coarse_in_chans: 1`, dataset `aux_vars: []`). The fine static tensor is still passed in
(its last channel is the AOI mask used by the loss) and is ignored here. The `unshuffle` terrain variant of the spec
is NOT built. `flood_head` (1x1 conv 64 -> 3) exists for interface parity with the other rows and is untrained in
this row (no ordinal term), like the RSwinUNet native-loss row.
Expected parameters (README §3b): 1,663,172 incl. the 195-parameter flood head.
"""
import torch.nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FloSR(nn.Module):

    def __init__(self, upscale=8, coarse_in_chans=1, static_in_chans=7, static_mode='none', num_feat=64,
                 num_blocks=16, res_scale=1.0, out_chans=1, num_flood_classes=3):
        super().__init__()
        if static_mode != 'none':
            raise ValueError(f"[FloSR] static_mode={static_mode!r}: only 'none' is built (S8 ruling: unguided as "
                             f"published; the 'unshuffle' terrain variant was not adopted)")
        if upscale & (upscale - 1):
            raise ValueError(f'[FloSR] upscale must be a power of 2, got {upscale}')
        self.upscale = upscale
        self.static_in_chans = static_in_chans          # accepted for interface parity; not used by the network
        self.head = nn.Conv2d(coarse_in_chans, num_feat, 3, 1, 1)
        # EDSR baseline residual block (conv-ReLU-conv + identity, no BN); PyTorch default init (pytorch_init=True)
        self.body = nn.Sequential(*[ResidualBlockNoBN(num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
                                    for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)     # log2(upscale) x [conv num_feat -> 4*num_feat, PixelShuffle(2)]
        self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)
        self.flood_head = nn.Conv2d(num_feat, num_flood_classes, 1, 1, 0)

    def forward(self, coarse_fm, static_f):
        _, _, Hc, Wc = coarse_fm.shape
        _, _, Hf, Wf = static_f.shape
        if Hf != Hc * self.upscale or Wf != Wc * self.upscale:
            raise RuntimeError(f'[FloSR] fine/coarse size mismatch: coarse {Hc}x{Wc}, fine {Hf}x{Wf}, scale {self.upscale}')
        x = self.head(coarse_fm)
        x = x + self.conv_after_body(self.body(x))       # EDSR global residual
        x = self.upsample(x)
        return self.conv_last(x), self.flood_head(x)
