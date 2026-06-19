"""
FloodMapPFTV9 — the "deep skip" ablation counterpart of FloodMapPFTV8 (TG-PFT).

Purpose (skip ablation against RSwinUNet's design philosophy):
  V8 (our method): multi-scale skips are SHALLOW CNN features of the PURE fine-grid
    static input; coarse enters only at the bottleneck.
  V9 (this file): the topography encoder is made DEEP (each scale refined by PFT
    block(s)), so the skips fed to the decoder are deep features. A flag controls
    whether the coarse-grid input is EARLY-FUSED into the encoder:
      * fuse_coarse=False (V9-a): encoder on fine static only -> deep skips;
                                  coarse enters at the bottleneck (like V8).
      * fuse_coarse=True  (V9-b): coarse is concatenated into the encoder INPUT
                                  (the whole encoder, incl. skips + the
                                  bottleneck-feeding feature, sees the fused
                                  coarse+fine), and the bottleneck takes ONLY that
                                  fused feature -- NO separate coarse branch.
                                  = the RSwinUNet philosophy (deep + early fusion).

Coarse-grid input is therefore used EXACTLY ONCE in each variant (V9-a: bottleneck
branch; V9-b: encoder fusion) -- no double counting -- so V9-a vs V9-b is a clean
"early fusion" axis. The global residual (coarse_h baseline) is identical across
V8/V9-a/V9-b and is not part of this count.

Controlled single axes:
  * V8 vs V9-a    -> skip DEPTH (shallow vs deep); bottleneck input is the same
                    shallow deepest feature in both, only the skips differ.
  * V9-a vs V9-b  -> EARLY FUSION (coarse at bottleneck vs fused into encoder).
  * V8 vs V9-b    -> the full "U-shape PFT vs RSwinUNet" contrast.

Deep blocks cover ALL scales incl the finest (512 for x8) on purpose so the
"shallow is enough" claim is not attackable; window=16 attention at 512x512 is
~64x the bottleneck (64x64) cost -> set use_checkpoint=true / lower batch if OOM.

State_dict is NOT compatible with V8 (train from scratch). Real forward needs the
smm_cuda kernel on the HPC; this file is only static-/CPU-construction-checked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.fmpftv8_arch import FloodMapPFTV8, PFTB
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FloodMapPFTV9(FloodMapPFTV8):
    r"""TG-PFT with a DEEP topography encoder (optionally early-fused with coarse).

    Extra args (everything else identical to FloodMapPFTV8):
        fuse_coarse (bool): If True, concat the (bicubic-upsampled) coarse-grid
            input into the encoder INPUT, and feed the bottleneck with the fused
            deepest feature only (no separate coarse branch). If False, encoder on
            fine static only and coarse enters at the bottleneck (as in V8).
        skip_deep_depth (int): PFT transformer layers per skip scale.
        skip_deep_topk (int): top-k retained in the skip-scale attention (sparsity).
    """

    def __init__(self, *args, fuse_coarse=False, skip_deep_depth=2, skip_deep_topk=128, **kwargs):
        super().__init__(*args, **kwargs)

        self.fuse_coarse = bool(fuse_coarse)
        self.skip_deep_depth = int(skip_deep_depth)
        self.skip_deep_topk = int(skip_deep_topk)
        self._flood_map_size_nominal = int(kwargs.get('flood_map_size', 64))

        # hyper-params for the skip-scale PFT blocks (mirror V8 defaults; basicsr
        # passes everything by keyword so these live in kwargs).
        embed_dim = self.embed_dim
        window_size = self.window_size
        num_heads = kwargs.get('num_heads', 6)
        convffn_kernel_size = kwargs.get('convffn_kernel_size', 5)
        mlp_ratio = kwargs.get('mlp_ratio', 2.)
        qkv_bias = kwargs.get('qkv_bias', True)
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        resi_connection = kwargs.get('resi_connection', '1conv')
        use_checkpoint = kwargs.get('use_checkpoint', False)
        use_act = self.use_shallow_act
        num_feat = 64

        if embed_dim % num_heads != 0:
            raise ValueError(
                f'[FloodMapPFTV9] embed_dim ({embed_dim}) must be divisible by '
                f'num_heads ({num_heads}) for the skip-scale window attention.')

        # --- Early-fusion restructure (V9-b): coarse goes into the encoder, not
        # the bottleneck. Rebuild the stem to accept coarse channels and the
        # bottleneck fuse-conv to take only the (fused) deepest feature. Drop the
        # now-unused coarse branch so coarse is consumed exactly once. ---
        if self.fuse_coarse:
            enc_ch0 = self._enc_ch[0]
            stem_layers = [nn.Conv2d(self.static_in_ch + self.coarse_in_ch, enc_ch0, 3, 1, 1)]
            if use_act:
                stem_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.static_stem = nn.Sequential(*stem_layers)
            # bottleneck fuse: deepest fused feature (num_feat) -> embed_dim
            self.conv_first = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)
            for name in ('conv_coarse_fm', 'gn_coarse'):
                if hasattr(self, name):
                    delattr(self, name)

        # --- Deep refinement on each encoder skip scale (both variants) ---
        skip_topk = [self.skip_deep_topk] * self.skip_deep_depth  # indexed by layer_id 0..depth-1
        self.skip_in = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.skip_out = nn.ModuleList()
        for i, ch in enumerate(self._skip_ch):
            proj_in = [nn.Conv2d(ch, embed_dim, 3, 1, 1)]
            if use_act:
                proj_in.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.skip_in.append(nn.Sequential(*proj_in))
            self.skip_blocks.append(
                PFTB(
                    dim=embed_dim, idx=i, layer_id=0,
                    input_resolution=(self._flood_map_size_nominal, self._flood_map_size_nominal),
                    depth=self.skip_deep_depth, num_heads=num_heads, num_topk=skip_topk,
                    window_size=window_size, convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                    downsample=None, use_checkpoint=use_checkpoint,
                    flood_map_size=self._flood_map_size_nominal, patch_size=1,
                    resi_connection=resi_connection))
            self.skip_out.append(nn.Conv2d(embed_dim, ch, 3, 1, 1))

        # init only the newly added / rebuilt modules.
        self.skip_in.apply(self._init_weights)
        self.skip_blocks.apply(self._init_weights)
        self.skip_out.apply(self._init_weights)
        if self.fuse_coarse:
            self.static_stem.apply(self._init_weights)
            self.conv_first.apply(self._init_weights)

    def _deep_refine(self, feat, block):
        """Refine one skip feature [B, embed_dim, H, W] with a PFTB at (H, W)."""
        b, c, h, w = feat.shape
        x_size = (h, w)
        attn_mask = self.calculate_mask(x_size).to(feat.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        tokens = feat.flatten(2).transpose(1, 2)              # [B, H*W, C]
        tokens, _ = block(tokens, [[None, None], [None, None]], x_size, params)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

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
        attn_mask = self.calculate_mask([Hc, Wc]).to(coarse_fm.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        coarse_h_pad = coarse_fm[:, 0:1, :, :]

        if self.use_aoi_gate:
            aoi_mask_fine = static_f[:, -1:, :, :]
            aoi_mask_coarse = torch.nn.functional.adaptive_avg_pool2d(aoi_mask_fine, (Hc, Wc))
            gate = (1.0 - self.aoi_alpha) + self.aoi_alpha * aoi_mask_coarse
            gate = gate.clamp(0.0, 1.0).detach()
        else:
            gate = None

        # ---- topography encoder (DEEP). Optionally fuse coarse at the input. ----
        if self.fuse_coarse:
            coarse_up = F.interpolate(
                coarse_fm, size=(static_f.shape[-2], static_f.shape[-1]),
                mode='bicubic', align_corners=False)
            enc_in = torch.cat([coarse_up, static_f], dim=1)
        else:
            enc_in = static_f

        s = self.static_stem(enc_in)
        skips = [s]
        for down in self.static_downs:
            s = down(s)
            skips.append(s)
        static_deep = skips[-1]
        enc_skips = skips[:-1]                 # finest..coarsest

        # deep-refine each skip
        deep_skips = []
        for i, sk in enumerate(enc_skips):
            x_sk = self.skip_in[i](sk)
            x_sk = self._deep_refine(x_sk, self.skip_blocks[i])
            x_sk = self.skip_out[i](x_sk)
            deep_skips.append(x_sk)
        enc_skips = deep_skips

        # ---- bottleneck fusion (coarse already in the encoder when fuse_coarse) ----
        static_deep = self.gn_static(static_deep)
        if self.fuse_coarse:
            x = self.conv_first(static_deep)
        else:
            coarse_feat = self.gn_coarse(self.conv_coarse_fm(coarse_fm))
            x = self.conv_first(torch.cat([coarse_feat, static_deep], dim=1))
        x_rc = x
        if gate is not None:
            x = x * gate
        x = self.conv_after_body(self.forward_features(x, params)) + x_rc
        if gate is not None:
            x = x * gate

        # ---- U-Net decoder: fuse the DEEP skips ----
        d = self.dec_proj(x)
        skips_dec = list(reversed(enc_skips))
        for i in range(self.n_stages):
            d_up = self.dec_ups[i](d)
            fused = self.dec_fuse[i](torch.cat([d_up, skips_dec[i]], dim=1))
            d = fused + d_up

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
