"""
FloodMapPFTV8Abl — single-axis architecture ablations of FloodMapPFTV8 (TG-PFT).

Three independent flags, each isolating one architectural pillar. Defaults reproduce
V8 exactly, so flipping ONE flag is a clean single-axis ablation:

  * use_ushape_skips (default True):
        False -> the U-Net decoder fuses ONLY the upsampled feature (no multi-scale
        topography skips). The static encoder still collapses to the bottleneck, so
        this reverts to the pre-U-shape design: shallow static fusion + PFT deep
        extraction + upsampling (cf. the original PFT / V7).  -> the "U-shape" pillar.

  * use_decoder_fuse (default True):
      False -> removes the DecoderFuseBlock after each DecoderUp stage. The decoder
      is reduced to plain Conv + PixelShuffle upsampling at each scale. This isolates
      the contribution of the per-scale decoder refinement blocks.

  * bottleneck_type (default 'pft'):
        'conv' -> the bottleneck PFT attention layers are replaced by a residual
        conv stack of matched width (embed_dim). Encoder/decoder are already conv
        in V8, so this single change yields a controlled conv U-Net (cleaner than
        comparing to the external HeUNet).  -> the "PFT block" pillar.

NOTE on fairness: tune `conv_bottleneck_blocks` so the conv bottleneck's parameter
count roughly matches the PFT bottleneck (print param counts on the HPC and adjust),
otherwise the comparison becomes "attention vs fewer params" rather than
"attention vs convolution".

Everything else (topography encoder, global residual, AOI gate, losses, data,
sampler) is inherited verbatim from FloodMapPFTV8. State_dict is NOT compatible
with V8 (train from scratch). Real forward needs smm_cuda on the HPC when
bottleneck_type='pft'; this file is only static-checked locally.
"""

import torch
import torch.nn as nn

from basicsr.archs.fmpftv8_arch import FloodMapPFTV8, DecoderFuseBlock
from basicsr.utils.registry import ARCH_REGISTRY


class ConvResBlock(nn.Module):
    """Conv3x3 -> [LeakyReLU] -> Conv3x3 with a residual add (+ optional out act)."""

    def __init__(self, dim, use_act=True):
        super().__init__()
        layers = [nn.Conv2d(dim, dim, 3, 1, 1)]
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(x + self.body(x))


@ARCH_REGISTRY.register()
class FloodMapPFTV8Abl(FloodMapPFTV8):
    r"""FloodMapPFTV8 with single-axis architecture-ablation flags.

    Extra args (everything else identical to FloodMapPFTV8):
        use_ushape_skips (bool): False removes the multi-scale topography skips.
        use_decoder_fuse (bool): False removes the DecoderFuseBlock.
        bottleneck_type (str): 'pft' (default) | 'conv'. 'conv' replaces the PFT
            attention bottleneck with a residual conv stack.
        conv_bottleneck_blocks (int): number of ConvResBlocks when bottleneck_type
            == 'conv' (tune to match the PFT bottleneck's param count).
    """

    def __init__(self, *args, use_ushape_skips=True, use_decoder_fuse=True,
                 bottleneck_type='pft', conv_bottleneck_blocks=8, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_ushape_skips = bool(use_ushape_skips)
        self.use_decoder_fuse = bool(use_decoder_fuse)
        self.bottleneck_type = str(bottleneck_type).lower().strip()
        assert self.bottleneck_type in ('pft', 'conv'), \
            f"bottleneck_type must be 'pft' or 'conv', got {self.bottleneck_type}"

        num_feat = 64
        embed_dim = self.embed_dim
        use_act = self.use_decoder_act

        # --- pillar 1: no U-shape skips -> decoder fuses only the upsampled feat ---
        if not self.use_decoder_fuse:
            if hasattr(self, 'dec_fuse'):
                delattr(self, 'dec_fuse')

        elif not self.use_ushape_skips:
            self.dec_fuse = nn.ModuleList([
                DecoderFuseBlock(num_feat, num_feat, use_act=use_act)
                for _ in range(self.n_stages)
            ])
            self.dec_fuse.apply(self._init_weights)

        # --- pillar 2: conv bottleneck instead of the PFT attention bottleneck ---
        if self.bottleneck_type == 'conv':
            # Drop the PFT bottleneck modules so the ablated model does not carry
            # unused attention params (keeps the param-count comparison honest).
            for name in ('layers', 'patch_embed', 'patch_unembed', 'conv_after_body', 'norm'):
                if hasattr(self, name):
                    delattr(self, name)
            if hasattr(self, 'absolute_pos_embed'):
                delattr(self, 'absolute_pos_embed')

            self.conv_bottleneck = nn.Sequential(*[
                ConvResBlock(embed_dim, use_act=True)
                for _ in range(int(conv_bottleneck_blocks))
            ])
            self.conv_bottleneck.apply(self._init_weights)

    def forward(self, coarse_fm, static_f):
        # ---- identical to V8: validate, pad to window multiple, masks, gate ----
        B, _, Hc_orig, Wc_orig = coarse_fm.shape
        _, _, Hf_orig, Wf_orig = static_f.shape
        if (Hf_orig != Hc_orig * self.upscale) or (Wf_orig != Wc_orig * self.upscale):
            raise RuntimeError(
                f'[ERROR] fine/coarse size mismatch with upscale: '
                f'Hc={Hc_orig}, Wc={Wc_orig}, Hf={Hf_orig}, Wf={Wf_orig}, scale={self.upscale}')

        mod = self.window_size
        Hc_pad = ((Hc_orig + mod - 1) // mod) * mod - Hc_orig
        Wc_pad = ((Wc_orig + mod - 1) // mod) * mod - Wc_orig
        if (Hc_pad != 0) or (Wc_pad != 0):
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [2])], 2)[:, :, :Hc_orig + Hc_pad, :]
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [3])], 3)[:, :, :, :Wc_orig + Wc_pad]
            Hf_pad = Hc_pad * self.upscale
            Wf_pad = Wc_pad * self.upscale
            static_f = torch.cat([static_f, torch.flip(static_f, [2])], 2)[:, :, :Hf_orig + Hf_pad, :]
            static_f = torch.cat([static_f, torch.flip(static_f, [3])], 3)[:, :, :, :Wf_orig + Wf_pad]

        Hc, Wc = Hc_orig + Hc_pad, Wc_orig + Wc_pad
        coarse_h_pad = coarse_fm[:, 0:1, :, :]

        if self.use_aoi_gate:
            aoi_mask_fine = static_f[:, -1:, :, :]
            aoi_mask_coarse = torch.nn.functional.adaptive_avg_pool2d(aoi_mask_fine, (Hc, Wc))
            gate = (1.0 - self.aoi_alpha) + self.aoi_alpha * aoi_mask_coarse
            gate = gate.clamp(0.0, 1.0).detach()
        else:
            gate = None

        # ---- topography encoder (same as V8) ----
        s = self.static_stem(static_f)
        skips = [s]
        for down in self.static_downs:
            s = down(s)
            skips.append(s)
        static_deep = skips[-1]
        enc_skips = skips[:-1]

        # ---- coarse branch + bottleneck fusion ----
        coarse_feat = self.conv_coarse_fm(coarse_fm)
        coarse_feat = self.gn_coarse(coarse_feat)
        static_deep = self.gn_static(static_deep)
        x = self.conv_first(torch.cat([coarse_feat, static_deep], dim=1))
        x_rc = x
        if gate is not None:
            x = x * gate

        # ---- bottleneck: PFT attention (default) or conv (ablation) ----
        if self.bottleneck_type == 'pft':
            attn_mask = self.calculate_mask([Hc, Wc]).to(coarse_fm.device)
            params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
            x = self.conv_after_body(self.forward_features(x, params)) + x_rc
        else:
            x = self.conv_bottleneck(x) + x_rc

        if gate is not None:
            x = x * gate

        # ---- U-Net decoder: with or without the topography skips ----
        d = self.dec_proj(x)
        skips_dec = list(reversed(enc_skips))
        for i in range(self.n_stages):
            d_up = self.dec_ups[i](d)
            if self.use_decoder_fuse:
                if self.use_ushape_skips:
                    fused = self.dec_fuse[i](torch.cat([d_up, skips_dec[i]], dim=1))
                else:
                    fused = self.dec_fuse[i](d_up)
                d = fused + d_up
            else:
                d = d_up

        depth = self.conv_last(d)
        flood_logit = self.flood_head(d)

        if self.use_global_residual:
            coarse_h_up = torch.nn.functional.interpolate(
                coarse_h_pad, scale_factor=self.upscale, mode='bicubic', align_corners=False)
            depth = depth + coarse_h_up

        depth = depth[..., :Hf_orig, :Wf_orig]
        flood_logit = flood_logit[..., :Hf_orig, :Wf_orig]

        if self.couple_mode != "none":
            p_wet = torch.sigmoid(flood_logit[:, 0:1, :, :])
            if self.couple_mode == "detach":
                p_wet = p_wet.detach()
            depth = depth * (self.couple_eps + (1 - self.couple_eps) * p_wet)

        return depth, flood_logit
