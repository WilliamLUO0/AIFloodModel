"""
songdenseunet_arch.py -- Paper 2 S8: DenseUNet of Song et al. (2025, Eng. Appl. Comput. Fluid Mech. 19(1), 2481115)
under our protocol.

Port of `DenseResUnetsimple` from https://github.com/songwk0924/Flood-inundation-mapping
(commit 600e7b07d0276b5d673fba2ef2121c1e11ec1cf5, file DenseResUnetnewsimplemaxprelu.py), released under the MIT License:

    MIT License
    Copyright (c) 2024 songwk0924
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Topology identical to the release (growth rate g = 32; spec paper2_S8/S8_DenseUNet.md §3, §5.7):
    stem   BN-ReLU-Conv3 (in -> g), BN-ReLU-Conv3 (g -> g)
    enc    4 dense blocks g -> 2g -> 4g -> 8g -> 16g, each followed by 2x2 max-pool; the POOLED outputs are the skips
    dec    3 x [BN-ReLU-ConvT(2,2) + cat skip, dense block], then BN-ReLU-ConvT + cat stem output
    head   BN-ReLU-Conv3 (2g -> 2g), BN-ReLU-Conv1 (2g -> 1)
    dense block: x1 = Conv3(ReLU(N(x))); x2 = Conv3(ReLU(N(x1 + Conv1(x)))); out = x1 + x2 + Conv1'(x)
                 (the two 1x1 shortcuts are plain convs, as in the code; Fig. 3b of the paper draws BN+ReLU+Conv1x1)
Changes vs the release (S8 rulings, README paper2_S8 §3):
  * the 3 coarse channels are bilinearly interpolated x`upscale` inside `forward` (they do it in preprocessing) and
    concatenated with the 7 fine statics -> 10 input channels (Paper 1 rule: every row gets the same inputs);
  * normalisation switchable via heunet_arch._make_norm; the row runs `norm: gn` (<= 8 groups) -- BatchNorm diverged
    for the He et al. UNet at batch 4, and BasicSR's EMA averages parameters only (BN buffers would never update);
  * their MaskLayer (x*mask, clamp >= 0 m) re-expressed in asinh + z-score label space: clamp at z0 = z(0 m) and z0
    outside the AOI; `out_floor_stats_json: ~` gives x*mask without the floor;
  * the 13 never-called nn.PReLU stubs are dropped; `flood_head` (1x1 conv 2g -> 3) added for interface parity,
    untrained in this row (native loss = masked L1 only).
Expected parameters (spec §3.3): 8,474,424 (8,474,229 without the flood head), the same for 'gn' and 'bn'.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.heunet_arch import _make_norm
from basicsr.utils.registry import ARCH_REGISTRY


class SongDenseBlock(nn.Module):
    """= ResDenseBlocksimpler (PReLU stub removed). F.relu, not inplace: the block input feeds the shortcuts."""

    def __init__(self, cin, cout, norm):
        super().__init__()
        self.n1 = _make_norm(norm, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2 = _make_norm(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.s1 = nn.Conv2d(cin, cout, 1, 1, 0)
        self.s2 = nn.Conv2d(cin, cout, 1, 1, 0)

    def forward(self, x):
        x1 = self.c1(F.relu(self.n1(x)))
        x2 = self.c2(F.relu(self.n2(x1 + self.s1(x))))
        return x1 + x2 + self.s2(x)


class SongUp(nn.Module):
    """= UpsampleLayer (PReLU stub removed): N-ReLU-ConvTranspose2d(2, 2), then concatenate the skip."""

    def __init__(self, cin, cout, norm):
        super().__init__()
        self.n = _make_norm(norm, cin)
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2, 0)

    def forward(self, x, skip):
        return torch.cat([self.up(F.relu(self.n(x))), skip], dim=1)


@ARCH_REGISTRY.register()
class SongDenseUNetFloodSR(nn.Module):

    def __init__(self, upscale=8, coarse_in_chans=3, static_in_chans=7, growth_rate=32, num_flood_classes=3,
                 interp_mode='bilinear', norm='gn', out_floor_stats_json=None, h_asinh_q=90):
        super().__init__()
        g = growth_rate
        cin = coarse_in_chans + static_in_chans
        self.upscale = upscale
        self.interp_mode = interp_mode
        # module order = the release's named_parameters() order (minus its PReLUs), so weights copy by position
        self.n1 = _make_norm(norm, cin)
        self.c1 = nn.Conv2d(cin, g, 3, 1, 1)
        self.n2 = _make_norm(norm, g)
        self.c2 = nn.Conv2d(g, g, 3, 1, 1)
        self.d1 = SongDenseBlock(g, 2 * g, norm)
        self.d2 = SongDenseBlock(2 * g, 4 * g, norm)
        self.d3 = SongDenseBlock(4 * g, 8 * g, norm)
        self.d4 = SongDenseBlock(8 * g, 16 * g, norm)
        self.u1 = SongUp(16 * g, 8 * g, norm)
        self.d5 = SongDenseBlock(16 * g, 8 * g, norm)
        self.u2 = SongUp(8 * g, 4 * g, norm)
        self.d6 = SongDenseBlock(8 * g, 4 * g, norm)
        self.u3 = SongUp(4 * g, 2 * g, norm)
        self.d7 = SongDenseBlock(4 * g, 2 * g, norm)
        self.u4 = SongUp(2 * g, g, norm)
        self.n3 = _make_norm(norm, 2 * g)
        self.c5 = nn.Conv2d(2 * g, 2 * g, 3, 1, 1)
        self.n6 = _make_norm(norm, 2 * g)
        self.c6 = nn.Conv2d(2 * g, 1, 1, 1, 0)
        self.flood_head = nn.Conv2d(2 * g, num_flood_classes, 1, 1, 0)

        # label-space floor: z0 = z-score of asinh(0 m) with the shared train stats (same json/key as the loss)
        self.z0 = None
        if out_floor_stats_json:
            with open(str(out_floor_stats_json), 'r', encoding='utf-8') as f:
                S = json.load(f)['stats_var']['asinh_by_q'][str(int(h_asinh_q))]['shared']
            self.z0 = (math.asinh(0.0) - float(S['mean'])) / float(S['std'])

    def forward(self, coarse_fm, static_f):
        _, _, Hc, Wc = coarse_fm.shape
        _, _, Hf, Wf = static_f.shape
        if Hf != Hc * self.upscale or Wf != Wc * self.upscale:
            raise RuntimeError(f'[SongDenseUNetFloodSR] fine/coarse size mismatch: coarse {Hc}x{Wc}, fine {Hf}x{Wf}, '
                               f'scale {self.upscale}')
        if Hf % 16 or Wf % 16:
            raise RuntimeError(f'[SongDenseUNetFloodSR] fine size {Hf}x{Wf} must be divisible by 16 (four 2x2 pools)')
        mask = static_f[:, -1:]
        if self.interp_mode in ('bilinear', 'bicubic'):
            coarse_up = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.interp_mode, align_corners=False)
        else:
            coarse_up = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.interp_mode)
        x = torch.cat([coarse_up, static_f], dim=1)
        x = self.c2(F.relu(self.n2(self.c1(F.relu(self.n1(x))))))      # stem, g @ Hf
        x1 = F.max_pool2d(self.d1(x), 2)                                 # 2g  @ Hf/2
        x2 = F.max_pool2d(self.d2(x1), 2)                                # 4g  @ Hf/4
        x3 = F.max_pool2d(self.d3(x2), 2)                                # 8g  @ Hf/8
        x4 = F.max_pool2d(self.d4(x3), 2)                                # 16g @ Hf/16
        y = self.d5(self.u1(x4, x3))
        y = self.d6(self.u2(y, x2))
        y = self.d7(self.u3(y, x1))
        y = self.c5(F.relu(self.n3(self.u4(y, x))))                      # 2g @ Hf
        depth = self.c6(F.relu(self.n6(y)))
        if self.z0 is not None:                                          # MaskLayer in label space
            depth = depth.clamp(min=self.z0) * mask + self.z0 * (1 - mask)
        else:
            depth = depth * mask
        return depth, self.flood_head(y)
