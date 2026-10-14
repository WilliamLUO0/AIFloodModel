"""
fmsatv8_arch.py -- Paper 2 S8: TG-SAT (FloodMapSATV8) and SAT single-fusion (FloodMapSATV7).

TG-SAT  = FloodMapPFTV8 (TG-PFT, fmpftv8_arch.py) with the PFT deep body (patch_embed -> PFTB stack -> norm ->
          patch_unembed) replaced by SAT's released body (Tran et al. 2026, commit cc561ae; basicsr/archs/sat_arch.py):
          LayerNorm -> SAT ResidualGroups (alternating Local Transformer Blocks / Selective Aggregation Transformer
          Blocks with the released SAA) -> LayerNorm. Everything else -- topography encoder + skips, bottleneck fusion,
          AOI gate, U-shaped decoder, bicubic global residual, depth + ordinal flood heads -- is V8 line for line.
SAT-V7  = FloodMapPFTV7 (single fusion: statics fused once, PixelShuffle upsampler, no skips, no global residual)
          with the same body swap; AOI gate off in the S8 row (mirrors Paper 1's final 02_FMPFTV7 row).

Spec and reference implementation: paper2_S8/S8_TGSAT.md (Appendix A; setting A, ruling 2026-10-14). Differences to
Appendix A, all behaviour-neutral: the fine/coarse size check raises RuntimeError like V8 (not assert), and unknown
network_g keys raise ValueError instead of being swallowed by **kwargs (spec risk 8.7: a PFT key such as window_size or
num_topk copied into the yml would otherwise be ignored silently).
Expected parameter counts (spec §9): FloodMapSATV8 setting A = 2,776,322; FloodMapSATV7 = 2,252,546; outside-body part of
FloodMapSATV8 = 1,232,084 = V8's.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.sat_arch import ResidualGroup                      # SAT: LTB/SATB blocks + 1conv residual (released)
from basicsr.archs.fmpftv8_arch import _build_encoder_channels, DecoderUp, DecoderFuseBlock
from basicsr.archs.fmpftv7_arch import _build_static_ch_ladder, Upsample as UpsampleV7
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


def _reject_unknown(cls_name, kwargs):
    if kwargs:
        raise ValueError(f'[{cls_name}] unknown network_g keys {sorted(kwargs)} (PFT keys such as window_size / '
                         f'num_topk / convffn_kernel_size / upsampler do not exist in the SAT body)')


def _sat_init_weights(m):
    """SAT's init (sat_arch.SAT._init_weights): trunc_normal 0.02 on Linear, 1/0 on norms; convs keep PyTorch default."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class SATBody(nn.Module):
    """SAT deep feature extraction = before_RG (flatten + LayerNorm) -> residual groups -> LayerNorm -> unflatten.
    Mirrors sat_arch.SAT.forward_features / SAT.__init__ (before_RG, layers, norm)."""

    def __init__(self, embed_dim, depth, num_heads, img_size=64, mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, split_size=(8, 32), c_ratio=0.5,
                 resi_connection='1conv', use_chk=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_in = norm_layer(embed_dim)                     # = SAT.before_RG[1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.layers = nn.ModuleList()
        for i in range(len(depth)):
            self.layers.append(ResidualGroup(
                dim=embed_dim, num_heads=num_heads[i], reso=img_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])], act_layer=nn.GELU,
                norm_layer=norm_layer, depth=depth[i], use_chk=use_chk,
                resi_connection=resi_connection, rs_id=i, split_size=list(split_size),
                c_ratio=c_ratio))
        self.norm_out = norm_layer(embed_dim)                    # = SAT.norm

    def forward(self, x):                                        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                          # b (h w) c
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x, [H, W])
        x = self.norm_out(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


def _check_body(embed_dim, num_heads, depths, c_ratio, split_size, flood_map_size):
    for h in num_heads:
        if embed_dim % h or int(embed_dim * c_ratio) % h or (embed_dim // 2) % max(h // 2, 1):
            raise ValueError(f'[SAT body] embed_dim {embed_dim} / c_ratio {c_ratio} not divisible by heads {h}')
    if any(d % 2 for d in depths):
        raise ValueError(f'[SAT body] depths {depths} must be even (LTB/SATB alternate; each group ends on a SATB)')
    if flood_map_size % max(split_size):
        raise ValueError(f'[SAT body] flood_map_size {flood_map_size} must be a multiple of max(split_size) '
                         f'{max(split_size)} (shift masks are pre-computed for it)')


def _reflect_pad(coarse_fm, static_f, mod, upscale):
    """V8/V7 reflection padding of both inputs to a multiple of `mod` on the coarse grid (no-op at 64/512)."""
    _, _, Hc0, Wc0 = coarse_fm.shape
    _, _, Hf0, Wf0 = static_f.shape
    Hc_pad = ((Hc0 + mod - 1) // mod) * mod - Hc0
    Wc_pad = ((Wc0 + mod - 1) // mod) * mod - Wc0
    if Hc_pad or Wc_pad:
        coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [2])], 2)[:, :, :Hc0 + Hc_pad, :]
        coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [3])], 3)[:, :, :, :Wc0 + Wc_pad]
        static_f = torch.cat([static_f, torch.flip(static_f, [2])], 2)[:, :, :Hf0 + Hc_pad * upscale, :]
        static_f = torch.cat([static_f, torch.flip(static_f, [3])], 3)[:, :, :, :Wf0 + Wc_pad * upscale]
    return coarse_fm, static_f, Hc0 + Hc_pad, Wc0 + Wc_pad


def _size_check(coarse_fm, static_f, upscale):
    _, _, Hc0, Wc0 = coarse_fm.shape
    _, _, Hf0, Wf0 = static_f.shape
    if (Hf0 != Hc0 * upscale) or (Wf0 != Wc0 * upscale):
        raise RuntimeError(f'[ERROR] fine/coarse size mismatch with upscale: Hc={Hc0}, Wc={Wc0}, '
                           f'Hf={Hf0}, Wf={Wf0}, scale={upscale}')
    return Hf0, Wf0


def _aoi_gate(static_f, Hc, Wc, alpha):
    aoi_c = F.adaptive_avg_pool2d(static_f[:, -1:], (Hc, Wc))
    return ((1.0 - alpha) + alpha * aoi_c).clamp(0.0, 1.0).detach()


def _couple(depth, flood_logit, mode, eps):
    if mode != 'none':
        p = torch.sigmoid(flood_logit[:, 0:1])
        if mode == 'detach':
            p = p.detach()
        depth = depth * (eps + (1 - eps) * p)
    return depth


@ARCH_REGISTRY.register()
class FloodMapSATV8(nn.Module):
    """TG-SAT: FloodMapPFTV8 with the PFT bottleneck replaced by the released SAT body."""

    def __init__(self, flood_map_size=64, coarse_in_chans=3, static_in_chans=7,
                 embed_dim=96, depths=(2, 4, 6, 6), num_heads=6, split_size=(8, 32),
                 mlp_ratio=2., qkv_bias=True, drop_path_rate=0.1, c_ratio=0.5,
                 resi_connection='1conv', use_checkpoint=False, upscale=8,
                 use_shallow_act=True, use_decoder_act=True, use_global_residual=True,
                 use_aoi_gate=True, aoi_alpha=0.5, couple_mode='none', couple_eps=0.2, **kwargs):
        super().__init__()
        _reject_unknown('FloodMapSATV8', kwargs)
        num_feat, num_out_ch = 64, 1
        if isinstance(num_heads, int):
            num_heads = [num_heads] * len(depths)
        _check_body(embed_dim, num_heads, depths, c_ratio, split_size, flood_map_size)
        self.coarse_in_ch, self.static_in_ch = coarse_in_chans, static_in_chans
        self.upscale = upscale
        self.use_global_residual = use_global_residual
        self.use_aoi_gate, self.aoi_alpha = use_aoi_gate, aoi_alpha
        self.couple_mode, self.couple_eps = str(couple_mode).lower(), couple_eps
        self.embed_dim = embed_dim
        self.pad_mod = max(split_size)

        # 1. coarse branch + topography encoder + bottleneck fusion (= fmpftv8_arch.FloodMapPFTV8)
        self.conv_coarse_fm = nn.Conv2d(coarse_in_chans, num_feat, 3, 1, 1)
        n_stages = int(math.log2(upscale))
        self.n_stages = n_stages
        enc_ch = _build_encoder_channels(static_in_chans, num_feat, n_stages)
        self._enc_ch = enc_ch
        stem = [nn.Conv2d(static_in_chans, enc_ch[0], 3, 1, 1)]
        if use_shallow_act:
            stem.append(nn.LeakyReLU(0.2, inplace=True))
        self.static_stem = nn.Sequential(*stem)
        self.static_downs = nn.ModuleList()
        for i in range(n_stages):
            down = [nn.PixelUnshuffle(2), nn.Conv2d(enc_ch[i] * 4, enc_ch[i + 1], 3, 1, 1)]
            if use_shallow_act:
                down.append(nn.LeakyReLU(0.2, inplace=True))
            self.static_downs.append(nn.Sequential(*down))
        self._skip_ch = enc_ch[:n_stages]
        self.gn_coarse = nn.GroupNorm(8, num_feat)
        self.gn_static = nn.GroupNorm(8, num_feat)
        self.conv_first = nn.Conv2d(2 * num_feat, embed_dim, 3, 1, 1)

        # 2. deep feature extraction = SAT body (replaces patch_embed + PFTB stack + norm + patch_unembed)
        self.body = SATBody(embed_dim, list(depths), num_heads, img_size=flood_map_size,
                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path_rate=drop_path_rate,
                            split_size=split_size, c_ratio=c_ratio, resi_connection=resi_connection,
                            use_chk=use_checkpoint)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. U-Net decoder + heads (= V8)
        self.dec_proj = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        skip_ch_dec = list(reversed(self._skip_ch))
        self.dec_ups = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        for i in range(n_stages):
            self.dec_ups.append(DecoderUp(num_feat, num_feat, use_act=use_decoder_act))
            self.dec_fuse.append(DecoderFuseBlock(num_feat + skip_ch_dec[i], num_feat, use_act=use_decoder_act))
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.flood_head = nn.Conv2d(num_feat, 3, 1, 1, 0)

        self.apply(_sat_init_weights)        # SAT's init, as V8 applies PFT's

    def forward(self, coarse_fm, static_f):
        Hf0, Wf0 = _size_check(coarse_fm, static_f, self.upscale)
        coarse_fm, static_f, Hc, Wc = _reflect_pad(coarse_fm, static_f, self.pad_mod, self.upscale)
        coarse_h_pad = coarse_fm[:, 0:1]
        gate = _aoi_gate(static_f, Hc, Wc, self.aoi_alpha) if self.use_aoi_gate else None

        s = self.static_stem(static_f)
        skips = [s]
        for down in self.static_downs:
            s = down(s)
            skips.append(s)
        static_deep, enc_skips = skips[-1], skips[:-1]

        coarse_feat = self.gn_coarse(self.conv_coarse_fm(coarse_fm))
        static_deep = self.gn_static(static_deep)
        x = self.conv_first(torch.cat([coarse_feat, static_deep], 1))
        x_rc = x
        if gate is not None:
            x = x * gate
        x = self.conv_after_body(self.body(x)) + x_rc
        if gate is not None:
            x = x * gate

        d = self.dec_proj(x)
        for i, skip in enumerate(reversed(enc_skips)):
            d_up = self.dec_ups[i](d)
            d = self.dec_fuse[i](torch.cat([d_up, skip], 1)) + d_up
        depth = self.conv_last(d)
        flood_logit = self.flood_head(d)
        if self.use_global_residual:
            depth = depth + F.interpolate(coarse_h_pad, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        depth = depth[..., :Hf0, :Wf0]
        flood_logit = flood_logit[..., :Hf0, :Wf0]
        return _couple(depth, flood_logit, self.couple_mode, self.couple_eps), flood_logit


@ARCH_REGISTRY.register()
class FloodMapSATV7(nn.Module):
    """SAT single-fusion: FloodMapPFTV7 with the PFT body replaced by the SAT body. Statics fused ONCE at shallow
    feature extraction (conv_static_f ladder), PixelShuffle upsampler, NO skips, NO U-shape, NO global residual."""

    def __init__(self, flood_map_size=64, coarse_in_chans=3, static_in_chans=7,
                 embed_dim=96, depths=(2, 4, 6, 6), num_heads=6, split_size=(8, 32),
                 mlp_ratio=2., qkv_bias=True, drop_path_rate=0.1, c_ratio=0.5,
                 resi_connection='1conv', use_checkpoint=False, upscale=8,
                 use_shallow_act=True, use_upsample_act=True,
                 use_aoi_gate=True, aoi_alpha=0.5, couple_mode='none', couple_eps=0.2, **kwargs):
        # Defaults = the committed 02_FMPFTV7 row (AOI gate on, LeakyReLU in the upsampler), so that
        # SAT-V7 vs TG-SAT differs by exactly what PFT-V7 vs TG-PFT differs by (paper2_S8/S8_TGSAT.md §5.0/§6).
        super().__init__()
        _reject_unknown('FloodMapSATV7', kwargs)
        num_feat, num_out_ch = 64, 1
        if isinstance(num_heads, int):
            num_heads = [num_heads] * len(depths)
        _check_body(embed_dim, num_heads, depths, c_ratio, split_size, flood_map_size)
        self.upscale = upscale
        self.use_aoi_gate, self.aoi_alpha = use_aoi_gate, aoi_alpha
        self.couple_mode, self.couple_eps = str(couple_mode).lower(), couple_eps
        self.pad_mod = max(split_size)
        self.conv_coarse_fm = nn.Conv2d(coarse_in_chans, num_feat, 3, 1, 1)
        n_stages = int(math.log2(upscale))
        ladder = _build_static_ch_ladder(static_in_chans, num_feat, n_stages)
        self._static_ch_ladder = ladder
        downs = []
        for i in range(n_stages):
            downs += [nn.PixelUnshuffle(2), nn.Conv2d(ladder[i] * 4, ladder[i + 1], 3, 1, 1)]
            if use_shallow_act:
                downs.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_static_f = nn.Sequential(*downs)
        self.gn_coarse = nn.GroupNorm(8, num_feat)
        self.gn_static = nn.GroupNorm(8, num_feat)
        self.conv_first = nn.Conv2d(2 * num_feat, embed_dim, 3, 1, 1)
        self.body = SATBody(embed_dim, list(depths), num_heads, img_size=flood_map_size,
                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path_rate=drop_path_rate,
                            split_size=split_size, c_ratio=c_ratio, resi_connection=resi_connection,
                            use_chk=use_checkpoint)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = UpsampleV7(upscale, num_feat, use_upsample_act=use_upsample_act)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.flood_head = nn.Conv2d(num_feat, 3, 1, 1, 0)
        self.apply(_sat_init_weights)

    def forward(self, coarse_fm, static_f):
        Hf0, Wf0 = _size_check(coarse_fm, static_f, self.upscale)
        coarse_fm, static_f, Hc, Wc = _reflect_pad(coarse_fm, static_f, self.pad_mod, self.upscale)
        gate = _aoi_gate(static_f, Hc, Wc, self.aoi_alpha) if self.use_aoi_gate else None
        c = self.gn_coarse(self.conv_coarse_fm(coarse_fm))
        s = self.gn_static(self.conv_static_f(static_f))
        x = self.conv_first(torch.cat([c, s], 1))
        x_rc = x
        if gate is not None:
            x = x * gate
        x = self.conv_after_body(self.body(x)) + x_rc
        if gate is not None:
            x = x * gate
        x_up = self.upsample(self.conv_before_upsample(x))
        depth = self.conv_last(x_up)[..., :Hf0, :Wf0]
        flood_logit = self.flood_head(x_up)[..., :Hf0, :Wf0]
        return _couple(depth, flood_logit, self.couple_mode, self.couple_eps), flood_logit
