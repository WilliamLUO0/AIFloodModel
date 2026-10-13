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

  * skip_scales (default None = all skips) — Paper 2 M1-B row (i):
        list of ENCODER skip indices that are concatenated in the decoder.
        Index 0 = finest skip (static_stem output, fine resolution), index k =
        after k PixelUnshuffle downs (fine / 2^k); valid range [0, n_stages-1].
        `[0]` keeps only the finest skip (the s = 2 topology reproduced at s = 8).
        Decoder stage i (i = 0 is the coarsest decoder stage) consumes encoder
        index n_stages-1-i. Stages whose skip is dropped keep their
        DecoderFuseBlock (input = upsampled feature only), so decoder capacity
        stays ~constant; the bottleneck topography fusion is untouched.

  * use_fine_topography (default True) — Paper 2 M1-B row (k):
        False -> the fine-grid topography encoder is removed entirely
        (static_stem / static_downs / gn_static deleted) and conv_first takes the
        coarse-map feature alone. Requires static_in_chans == 1 (the dataset's
        `static_vars: [mask]`): the AOI mask is then used ONLY by the AOI gate
        (and by the loss / eval in fmsr_model). Also requires
        use_ushape_skips=False (there is no encoder to provide skips); the
        DecoderFuseBlocks are kept (no-skip form), as in row (j).

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
        skip_scales (list[int] | None): encoder skip indices kept in the decoder
            (0 = finest). None = all (V8 behaviour).
        use_fine_topography (bool): False removes the fine-grid topography
            encoder (static input must then be the mask only).
    """

    def __init__(self, *args, use_ushape_skips=True, use_decoder_fuse=True,
                 bottleneck_type='pft', conv_bottleneck_blocks=8,
                 skip_scales=None, use_fine_topography=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_ushape_skips = bool(use_ushape_skips)
        self.use_decoder_fuse = bool(use_decoder_fuse)
        self.use_fine_topography = bool(use_fine_topography)
        self.bottleneck_type = str(bottleneck_type).lower().strip()
        assert self.bottleneck_type in ('pft', 'conv'), \
            f"bottleneck_type must be 'pft' or 'conv', got {self.bottleneck_type}"

        num_feat = 64
        embed_dim = self.embed_dim
        use_act = self.use_decoder_act

        # --- resolve which encoder skips the decoder concatenates ---
        all_scales = list(range(self.n_stages))
        if skip_scales is not None:
            if not self.use_ushape_skips:
                raise ValueError('[FloodMapPFTV8Abl] skip_scales is set but use_ushape_skips=False; '
                                 'leave skip_scales unset when the skips are disabled.')
            if not self.use_decoder_fuse:
                raise ValueError('[FloodMapPFTV8Abl] skip_scales requires use_decoder_fuse=True '
                                 '(without the fuse blocks the skips are never used).')
            if isinstance(skip_scales, int):
                skip_scales = [skip_scales]
            kept = [int(k) for k in skip_scales]
            if len(kept) == 0:
                raise ValueError('[FloodMapPFTV8Abl] skip_scales is empty; use use_ushape_skips=False instead.')
            if len(set(kept)) != len(kept):
                raise ValueError(f'[FloodMapPFTV8Abl] skip_scales has duplicates: {kept}')
            bad = [k for k in kept if k not in all_scales]
            if bad:
                raise ValueError(f'[FloodMapPFTV8Abl] skip_scales {bad} out of range {all_scales} '
                                 f'(n_stages={self.n_stages}, 0 = finest).')
            kept = sorted(kept)
        elif self.use_ushape_skips:
            kept = list(all_scales)
        else:
            kept = []

        if not self.use_fine_topography:
            if self.use_ushape_skips:
                raise ValueError('[FloodMapPFTV8Abl] use_fine_topography=False requires '
                                 'use_ushape_skips=False (no encoder -> no skips).')
            if self.static_in_ch != 1:
                raise ValueError(f'[FloodMapPFTV8Abl] use_fine_topography=False requires '
                                 f'static_in_chans=1 (static_vars: [mask]); got {self.static_in_ch}.')
        self.skip_scales = kept

        # --- pillar 1: U-shape skips (all / subset / none) -> rebuild the fuse blocks ---
        if not self.use_decoder_fuse:
            if hasattr(self, 'dec_fuse'):
                delattr(self, 'dec_fuse')

        elif kept != all_scales:
            # Decoder stage i consumes encoder skip index n_stages-1-i. A stage whose
            # skip is dropped keeps a DecoderFuseBlock on the upsampled feature only.
            dec_fuse = []
            for i in range(self.n_stages):
                enc_idx = self.n_stages - 1 - i
                in_ch = num_feat + self._skip_ch[enc_idx] if enc_idx in kept else num_feat
                dec_fuse.append(DecoderFuseBlock(in_ch, num_feat, use_act=use_act))
            self.dec_fuse = nn.ModuleList(dec_fuse)
            self.dec_fuse.apply(self._init_weights)

        # --- no fine-grid topography: drop the encoder, conv_first sees coarse only ---
        if not self.use_fine_topography:
            for name in ('static_stem', 'static_downs', 'gn_static'):
                if hasattr(self, name):
                    delattr(self, name)
            self.conv_first = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)
            # parity with the other rows (review A8). _init_weights only touches Linear/LayerNorm, so this is
            # a no-op for this Conv2d: V8's own conv_first also keeps PyTorch's default Conv2d init.
            self.conv_first.apply(self._init_weights)

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

        # ---- coarse branch ----
        coarse_feat = self.conv_coarse_fm(coarse_fm)
        coarse_feat = self.gn_coarse(coarse_feat)

        if self.use_fine_topography:
            # ---- topography encoder (same as V8) + bottleneck fusion ----
            s = self.static_stem(static_f)
            skips = [s]
            for down in self.static_downs:
                s = down(s)
                skips.append(s)
            static_deep = skips[-1]
            enc_skips = skips[:-1]       # index 0 = finest

            static_deep = self.gn_static(static_deep)
            x = self.conv_first(torch.cat([coarse_feat, static_deep], dim=1))
        else:
            # ---- no fine-grid topography: bottleneck sees the coarse feature only ----
            enc_skips = []
            x = self.conv_first(coarse_feat)
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

        # ---- U-Net decoder: all / a subset of / none of the topography skips ----
        # Decoder stage i (coarsest first) pairs with encoder skip n_stages-1-i.
        d = self.dec_proj(x)
        for i in range(self.n_stages):
            d_up = self.dec_ups[i](d)
            if self.use_decoder_fuse:
                enc_idx = self.n_stages - 1 - i
                if enc_idx in self.skip_scales:
                    fused = self.dec_fuse[i](torch.cat([d_up, enc_skips[enc_idx]], dim=1))
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
