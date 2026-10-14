"""
shen_gan_loss.py -- Paper 2 S8: the CBAM-GAN objective of Shen et al. (2026) in our label space.

Re-implemented from paper2_S8/S8_ShenCBAMGAN.md §4 / §5.3(b) (the released repository has no licence; nothing is
copied). LSGAN branch only.

Generator:  L_G = w_adv * a * clamp(MSE(D(fake'), 1), 0, 10)
                + w_rec * r * SmoothL1_beta(fake*M, real*M)            (plain mean over all elements, as released)
                + w_empty * L1(fake*(1-M), z0*(1-M))                    (mean over all elements; z0 = z-score of 0 m)
                + w_edge * clamp(mean|d_x fake| + mean|d_y fake|, 0, 1) (TV on the full output)
                + w_perc * MSE(VGG16[:16](u(fake)), VGG16[:16](u(real)))
Discriminator: L_D = 0.5 * [MSE(clamp(D(real), +-50), 0.9) + MSE(clamp(D(fake'), +-50), 0.1)]
  M = fine AOI mask; fake' (what D sees) is built by the model class (real pasted outside the AOI).
  Perceptual input u: the first channel, (x*M) mapped from z-space to asinh depth and min-max'd with the training
  asinh range to [0, 1], bicubic-resized to min(128, H), repeated to 3 channels, ImageNet-normalised.
  Adaptive scales (a, r): after 10 D steps, with recent_d / recent_g = mean of the last 20 recorded D / G losses:
  recent_d > 5 -> (0.1, 2.0); 2 < recent_d <= 5 -> (0.5, 1.5); else (1, 1); then recent_g > 10 -> a = min(2a, 2).
Deviations from the released code (stated in the handover): "last 20" is the last 20 recorded steps (the release
reads fixed slots 80-99 of its ring buffer); a missing VGG16 file raises instead of silently dropping the term;
non-finite losses raise in the model class instead of being replaced by a constant (would hang DDP).
"""
import json
import math
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ShenHydrologicalGANLoss(nn.Module):

    def __init__(self, stats_json, vgg16_path, h_asinh_q=90, w_adv=1.0, w_rec=50.0, w_edge=0.5, w_empty=0.5,
                 w_perc=0.1, smooth_l1_beta=0.1, label_real=0.9, label_fake=0.1, adv_clamp=10.0, tv_clamp=1.0,
                 d_logit_clamp=50.0, perc_size=128, adaptive_weights=True, adaptive_min_steps=10,
                 adaptive_window=20):
        super().__init__()
        self.w_adv, self.w_rec, self.w_edge, self.w_empty, self.w_perc = w_adv, w_rec, w_edge, w_empty, w_perc
        self.beta, self.label_real, self.label_fake = smooth_l1_beta, label_real, label_fake
        self.adv_clamp, self.tv_clamp, self.d_logit_clamp = adv_clamp, tv_clamp, d_logit_clamp
        self.perc_size = perc_size
        self.adaptive_weights, self.adaptive_min_steps = adaptive_weights, adaptive_min_steps

        if not os.path.isfile(str(stats_json)):
            raise FileNotFoundError(f'[ShenHydrologicalGANLoss] stats_json not found: {stats_json}')
        with open(str(stats_json), 'r', encoding='utf-8') as f:
            S = json.load(f)['stats_var']['asinh_by_q'][str(int(h_asinh_q))]['shared']
        self.mu, self.sigma = float(S['mean']), float(S['std'])
        self.a_min, self.a_max = float(S['min']), float(S['max'])
        self.z0 = (math.asinh(0.0) - self.mu) / self.sigma      # normalised 0 m (empty-region target)

        self.vgg = None
        if w_perc > 0:
            if not os.path.isfile(str(vgg16_path)):
                raise FileNotFoundError(
                    f'[ShenHydrologicalGANLoss] VGG16 ImageNet weights not found: {vgg16_path} -- download '
                    f'https://download.pytorch.org/models/vgg16-397923af.pth on a login node (no silent fallback)')
            from torchvision.models import vgg16
            net = vgg16(weights=None)
            net.load_state_dict(torch.load(str(vgg16_path), map_location='cpu'))
            self.vgg = net.features[:16].eval()                # conv1_1 ... relu3_3
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.d_hist = deque(maxlen=adaptive_window)
        self.g_hist = deque(maxlen=adaptive_window)
        self.n_d_steps = 0

    def train(self, mode=True):                                 # the VGG stays in eval mode
        super().train(mode)
        if self.vgg is not None:
            self.vgg.eval()
        return self

    # ---------------- adaptive weights ----------------
    def record_d(self, value):
        self.d_hist.append(float(value))
        self.n_d_steps += 1

    def record_g(self, value):
        self.g_hist.append(float(value))

    def adaptive_scales(self):
        if not self.adaptive_weights or self.n_d_steps < self.adaptive_min_steps:
            return 1.0, 1.0
        recent_d = sum(self.d_hist) / max(len(self.d_hist), 1)
        recent_g = sum(self.g_hist) / max(len(self.g_hist), 1)
        if recent_d > 5.0:
            adv, rec = 0.1, 2.0
        elif recent_d > 2.0:
            adv, rec = 0.5, 1.5
        else:
            adv, rec = 1.0, 1.0
        if recent_g > 10.0:
            adv = min(adv * 2.0, 2.0)
        return adv, rec

    # ---------------- terms ----------------
    def tv(self, x):
        t = (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        return t.clamp(0, self.tv_clamp)

    def _to_vgg_input(self, x, mask):
        # release order: bicubic resize first, then the range map + clamp to [0, 1] (the map is affine, so resizing
        # before or after it is the same; only the clamp position matters)
        s = min(self.perc_size, min(x.shape[-2:]))
        v = F.interpolate(x[:, :1] * mask, size=(s, s), mode='bicubic', align_corners=False)
        a = v * self.sigma + self.mu                            # z-space -> asinh depth (outside AOI: fixed constant)
        u = ((a - self.a_min) / (self.a_max - self.a_min)).clamp(0, 1)
        return (u.repeat(1, 3, 1, 1) - self.img_mean) / self.img_std

    def perceptual(self, fake, real, mask):
        if self.vgg is None:
            return fake.new_zeros(())
        with torch.no_grad():
            feat_real = self.vgg(self._to_vgg_input(real, mask))
        return F.mse_loss(self.vgg(self._to_vgg_input(fake, mask)), feat_real)

    def g_loss(self, fake_pred, real, fake, mask):
        adv_s, rec_s = self.adaptive_scales()
        l_adv = F.mse_loss(fake_pred, torch.ones_like(fake_pred)).clamp(0, self.adv_clamp)
        l_rec = F.smooth_l1_loss(fake * mask, real * mask, beta=self.beta)
        l_empty = F.l1_loss(fake * (1 - mask), self.z0 * (1 - mask))
        l_edge = self.tv(fake)
        l_perc = self.perceptual(fake, real, mask)
        total = (self.w_adv * adv_s * l_adv + self.w_rec * rec_s * l_rec + self.w_empty * l_empty
                 + self.w_edge * l_edge + self.w_perc * l_perc)
        parts = dict(l_g_adv=l_adv, l_g_rec=l_rec, l_g_empty=l_empty, l_g_edge=l_edge, l_g_perc=l_perc,
                     adv_scale=fake.new_tensor(adv_s), rec_scale=fake.new_tensor(rec_s))
        return total, parts

    def d_loss(self, real_pred, fake_pred):
        real_pred = real_pred.clamp(-self.d_logit_clamp, self.d_logit_clamp)
        fake_pred = fake_pred.clamp(-self.d_logit_clamp, self.d_logit_clamp)
        l_real = F.mse_loss(real_pred, torch.full_like(real_pred, self.label_real))
        l_fake = F.mse_loss(fake_pred, torch.full_like(fake_pred, self.label_fake))
        total = 0.5 * (l_real + l_fake)
        parts = dict(l_d_real=l_real, l_d_fake=l_fake, d_real_mean=real_pred.detach().mean(),
                     d_fake_mean=fake_pred.detach().mean())
        return total, parts
