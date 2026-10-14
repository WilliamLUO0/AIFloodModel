"""
floodsr3unet_arch.py -- Paper 2 S8: the denoiser of Flood-DM (the pixel-space variant of Flood-LDM, Neo et al. 2026,
WACV), i.e. the SR3 UNet.

Ported from https://github.com/neosunhan/flood-diff (commit d04b75158bb49d65a76f3d4bae87376fadb6c176),
file model/sr3_modules/unet.py (itself adapted from Janspiry's Image-Super-Resolution-via-Iterative-Refinement).

    Copyright the flood-diff authors.
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
    the specific language governing permissions and limitations under the License.

Modifications (Paper 2 S8, 2026-10-15):
  * classes renamed with an `SR3` prefix and registered as `FloodSR3UNet` in BasicSR's ARCH_REGISTRY;
  * the release's orthogonal initialisation (model/networks.py `weights_init_orthogonal`, gain 1, zero bias, applied
    to every Conv*/Linear) is applied in the constructor (`init_type: orthogonal`);
  * optional activation checkpointing of the residual blocks (`grad_checkpoint`, default off; memory fallback);
  * the forward signature is unchanged: forward(x, noise_level) with x = cat[x_t, conditioning] and noise_level the
    continuous sqrt(alpha_bar) of shape (B, 1). The diffusion process (schedule, loss, sampler) lives in
    basicsr/models/flooddiffusion_model.py.
Computational graph, layer order and parameter names are otherwise those of the release. Expected parameters:
316,581,633 (in_channel 3, the paper's DM) / 316,586,241 (in_channel 11, ours).
"""
import math

import torch
from torch import nn
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY


class SR3PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class SR3FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class SR3Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SR3Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class SR3Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class SR3Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            SR3Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1))

    def forward(self, x):
        return self.block(x)


class SR3ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = SR3FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = SR3Block(dim, dim_out, groups=norm_groups)
        self.block2 = SR3Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SR3SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx
        attn = torch.einsum('bnchw, bncyx -> bnhwyx', query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)
        out = torch.einsum('bnhwyx, bncyx -> bnchw', attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input


class SR3ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = SR3ResnetBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SR3SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


def _weights_init_orthogonal(m):
    """= model/networks.py weights_init_orthogonal of the release (Conv*/Linear: orthogonal gain 1, zero bias)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()


@ARCH_REGISTRY.register()
class FloodSR3UNet(nn.Module):

    def __init__(self, in_channel=11, out_channel=1, inner_channel=64, norm_groups=16, channel_mults=(1, 2, 4, 8, 16),
                 attn_res=(32,), res_blocks=3, dropout=0, with_noise_level_emb=True, image_size=512,
                 init_type='orthogonal', grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        attn_res = tuple(attn_res) if isinstance(attn_res, (list, tuple)) else (attn_res,)
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                SR3PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                SR3Swish(),
                nn.Linear(inner_channel * 4, inner_channel))
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(SR3ResnetBlocWithAttn(pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                                                   norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(SR3Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            SR3ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                  norm_groups=norm_groups, dropout=dropout, with_attn=True),
            SR3ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                  norm_groups=norm_groups, dropout=dropout, with_attn=False)])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(SR3ResnetBlocWithAttn(pre_channel + feat_channels.pop(), channel_mult,
                                                 noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                                 dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(SR3Upsample(pre_channel))
                now_res = now_res * 2
        self.ups = nn.ModuleList(ups)
        self.final_conv = SR3Block(pre_channel, out_channel, groups=norm_groups)

        if init_type == 'orthogonal':
            self.apply(_weights_init_orthogonal)
        elif init_type not in (None, 'default'):
            raise ValueError(f"[FloodSR3UNet] init_type must be 'orthogonal' or 'default', got {init_type!r}")

    def _block(self, layer, x, t):
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(layer, x, t, use_reentrant=False)
        return layer(x, t)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if self.noise_level_mlp is not None else None
        feats = []
        for layer in self.downs:
            x = self._block(layer, x, t) if isinstance(layer, SR3ResnetBlocWithAttn) else layer(x)
            feats.append(x)
        for layer in self.mid:
            x = self._block(layer, x, t) if isinstance(layer, SR3ResnetBlocWithAttn) else layer(x)
        for layer in self.ups:
            if isinstance(layer, SR3ResnetBlocWithAttn):
                x = self._block(layer, torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)
