"""
shencbamgan_arch.py -- Paper 2 S8: CBAM-GAN of Shen et al. (2026, J. Hydrol. 672, 135330) under our protocol.

Re-implemented from the description in paper2_S8/S8_ShenCBAMGAN.md (§3, §5.3). The released repository
(github.com/shh20000317/SH_PFM_SR, commit c78477a) carries no licence, so no code is copied from it; the discussion
session checks this file against the release layer by layer (and tools/test_shencbamgan.py reproduces the released
generator's output after a weight copy).

Generator `ShenCBAMGenerator` -- a pre-upsampling residual U-Net (unguided):
    input   coarse depth (1 channel), bicubic x`upscale` inside forward (they interpolate in preprocessing)
    enc     4 levels, widths 32/64/128/256: residual unit -> skip = SpatialAtt(ChannelAtt(feature)) (CBAM on the skip
            only), the UN-attended feature is max-pooled and passed down
    bottom  residual unit 256 -> 512, channel attention, spatial attention
    dec     4 levels: ConvTranspose(2, 2) -> gating signal (1x1 conv + norm + ReLU) -> additive attention gate on the
            skip -> concat [gated skip, upsampled] -> residual unit
    head    1x1 conv 32 -> 1, linear (no output activation; no clamp in z-space, README paper2_S8 §3)
  residual unit: IN -> ReLU -> Conv3 -> IN -> ReLU -> Conv3, plus identity / 1x1 conv shortcut, then ReLU of the sum;
  InstanceNorm2d(affine=False) as in the released instantiation (no running statistics).
Changes vs the release (S8 rulings):
  * the 16 BatchNorm2d of the gating signals / attention gates -> GroupNorm(min(8, C), C) (`gate_norm: gn`);
    `gate_norm: bn` exists only for the fidelity test against the released code;
  * the deep-supervision heads (never used in their training) are not built;
  * the interpolate fall-backs inside the decoder raise instead of silently resizing (sizes always match at 512).
Discriminator `ShenPatchGANDiscriminator` -- unconditional 70x70 PatchGAN: Conv4 s2 (no norm) / 2 x [Conv4 s2 + IN] /
Conv4 s1 + IN, LeakyReLU 0.2, then Conv4 s1 -> 1-channel patch logits; InstanceNorm2d(affine=True) (their registry).
Expected parameters: generator 8,330,535 (in 1 / out 1), discriminator 694,113 (in 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


def _gate_norm(kind, c):
    if kind == 'bn':
        return nn.BatchNorm2d(c)
    if kind == 'gn':
        g = min(8, c)
        if c % g:
            raise ValueError(f'[ShenCBAMGenerator] GroupNorm: {c} channels not divisible by {g} groups')
        return nn.GroupNorm(g, c)
    raise ValueError(f"[ShenCBAMGenerator] gate_norm must be 'gn' or 'bn', got {kind!r}")


class _ChannelAtt(nn.Module):
    """CBAM channel attention: sigmoid(MLP(avg-pool) + MLP(max-pool)), shared bias-free MLP with reduction 16."""

    def __init__(self, c, reduction=16):
        super().__init__()
        hidden = max(1, c // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), nn.ReLU(),
                                 nn.Conv2d(hidden, c, 1, bias=False))

    def forward(self, x):
        w = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(w)


class _SpatialAtt(nn.Module):
    """CBAM spatial attention: sigmoid(Conv7x7([channel mean, channel max]))."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class _ResUnit(nn.Module):
    """Pre-activation residual unit with InstanceNorm (no affine) and a ReLU after the sum."""

    def __init__(self, cin, cout):
        super().__init__()
        self.in1 = nn.InstanceNorm2d(cin)
        self.in2 = nn.InstanceNorm2d(cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.shortcut = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        y = self.conv1(F.relu(self.in1(x)))
        y = self.conv2(F.relu(self.in2(y)))
        return F.relu(y + self.shortcut(x))


class _EncLevel(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.res = _ResUnit(cin, cout)
        self.ca = _ChannelAtt(cout)
        self.sa = _SpatialAtt()

    def forward(self, x):
        f = self.res(x)
        return F.max_pool2d(f, 2), self.sa(self.ca(f))      # (passed down, skip)


class _GateSignal(nn.Module):
    def __init__(self, c, norm):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1)
        self.norm = _gate_norm(norm, c)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class _AttGate(nn.Module):
    """Additive attention gate: skip * sigmoid(N(Conv1(relu(N(Wg g) + N(Ws s)))))."""

    def __init__(self, c, inter, norm):
        super().__init__()
        self.wg = nn.Conv2d(c, inter, 1)
        self.ng = _gate_norm(norm, inter)
        self.ws = nn.Conv2d(c, inter, 1)
        self.ns = _gate_norm(norm, inter)
        self.wpsi = nn.Conv2d(inter, 1, 1)
        self.npsi = _gate_norm(norm, 1)

    def forward(self, g, s):
        a = self.ng(self.wg(g))
        b = self.ns(self.ws(s))
        if a.shape[-2:] != b.shape[-2:]:
            raise RuntimeError(f'[ShenCBAMGenerator] gate/skip size mismatch {tuple(a.shape)} vs {tuple(b.shape)}')
        return s * torch.sigmoid(self.npsi(self.wpsi(F.relu(a + b))))


class _DecLevel(nn.Module):
    def __init__(self, cprev, cskip, cout, norm):
        super().__init__()
        self.up = nn.ConvTranspose2d(cprev, cskip, 2, stride=2)
        self.gate_signal = _GateSignal(cskip, norm)
        self.att = _AttGate(cskip, cskip // 2, norm)
        self.res = _ResUnit(2 * cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(f'[ShenCBAMGenerator] up/skip size mismatch {tuple(x.shape)} vs {tuple(skip.shape)}')
        return self.res(torch.cat([self.att(self.gate_signal(x), skip), x], dim=1))


@ARCH_REGISTRY.register()
class ShenCBAMGenerator(nn.Module):

    def __init__(self, upscale=8, coarse_in_chans=1, static_in_chans=7, out_chans=1, widths=(32, 64, 128, 256, 512),
                 interp_mode='bicubic', gate_norm='gn'):
        super().__init__()
        w = list(widths)
        if len(w) != 5:
            raise ValueError(f'[ShenCBAMGenerator] widths must have 5 entries, got {w}')
        self.upscale = upscale
        self.interp_mode = interp_mode
        self.static_in_chans = static_in_chans          # accepted for interface parity; unguided -> not used
        self.enc1 = _EncLevel(coarse_in_chans, w[0])
        self.enc2 = _EncLevel(w[0], w[1])
        self.enc3 = _EncLevel(w[1], w[2])
        self.enc4 = _EncLevel(w[2], w[3])
        self.bottom = nn.Sequential(_ResUnit(w[3], w[4]), _ChannelAtt(w[4]), _SpatialAtt())
        self.dec1 = _DecLevel(w[4], w[3], w[3], gate_norm)
        self.dec2 = _DecLevel(w[3], w[2], w[2], gate_norm)
        self.dec3 = _DecLevel(w[2], w[1], w[1], gate_norm)
        self.dec4 = _DecLevel(w[1], w[0], w[0], gate_norm)
        self.head = nn.Conv2d(w[0], out_chans, 1)

    def forward(self, coarse_fm, static_f):
        _, _, Hc, Wc = coarse_fm.shape
        _, _, Hf, Wf = static_f.shape
        if Hf != Hc * self.upscale or Wf != Wc * self.upscale:
            raise RuntimeError(f'[ShenCBAMGenerator] fine/coarse size mismatch: coarse {Hc}x{Wc}, fine {Hf}x{Wf}, '
                               f'scale {self.upscale}')
        if Hf % 16 or Wf % 16:
            raise RuntimeError(f'[ShenCBAMGenerator] fine size {Hf}x{Wf} must be divisible by 16 (four 2x2 pools)')
        if self.interp_mode in ('bilinear', 'bicubic'):
            x = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.interp_mode, align_corners=False)
        else:
            x = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.interp_mode)
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottom(x)
        x = self.dec1(x, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        return self.head(x), None                         # no flood-logit head in this model


@ARCH_REGISTRY.register()
class ShenPatchGANDiscriminator(nn.Module):

    def __init__(self, in_chans=1, base_ch=32, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_chans, base_ch, 4, 2, 1), nn.LeakyReLU(0.2)]
        c = base_ch
        for n in range(1, n_layers):                     # stride-2 stages with InstanceNorm (affine)
            c_next = base_ch * min(2 ** n, 8)
            layers += [nn.Conv2d(c, c_next, 4, 2, 1), nn.InstanceNorm2d(c_next, affine=True), nn.LeakyReLU(0.2)]
            c = c_next
        c_next = base_ch * min(2 ** n_layers, 8)         # stride-1 stage, then the patch-logit conv
        layers += [nn.Conv2d(c, c_next, 4, 1, 1), nn.InstanceNorm2d(c_next, affine=True), nn.LeakyReLU(0.2),
                   nn.Conv2d(c_next, 1, 4, 1, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)
