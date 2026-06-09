import math
import torch
import torch.nn as nn

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


def window_partition(x, window_size):
    """
    Args:
        x: [B, H, W, C]
    Returns:
        windows: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: [num_windows * B, window_size, window_size, C]
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    This is the standard Swin/SwinIR-style attention, not PFT sparse attention.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()

        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_position_bias_table_size = (
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_position_bias_table_size, num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: [num_windows * B, N, C]
            mask: [num_windows, N, N] or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                B_ // num_windows,
                num_windows,
                self.num_heads,
                N,
                N,
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer block.
    If shift_size = 0, it is W-MSA.
    If shift_size = window_size // 2, it is SW-MSA.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
            self.shift_size = 0

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """
        Args:
            x: [B, H*W, C]
            x_size: (H, W)
        """
        H, W = x_size
        B, L, C = x.shape

        assert L == H * W, "Input feature has wrong spatial size."

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,
        )

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C,
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.contiguous().view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, norm_layer=None, embed_dim=128):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        H, W = x_size
        B, HW, C = x.shape
        assert HW == H * W
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block.
    This follows the SwinIR/SwinFlood-style structure:
    multiple Swin blocks -> patch reconstruction -> conv -> residual addition.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        resi_connection='1conv',
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        self.patch_embed = PatchEmbed(norm_layer=None, embed_dim=dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

    def forward(self, x, x_size):
        residual = x

        for blk in self.blocks:
            x = blk(x, x_size)

        x_img = self.patch_unembed(x, x_size)
        x_img = self.conv(x_img)
        x = self.patch_embed(x_img)

        return x + residual


class ConvBlockI(nn.Module):
    """
    Static-feature downsampling block.
    Used to reduce fine-grid static features from Hf x Wf to Hm x Wm.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlockII(nn.Module):
    """
    Standard convolution block.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlockIII(nn.Module):
    """
    Conv. Block III in SwinFlood:
    Conv -> PixelShuffle.
    No activation is used, following Fig. 2(d) and Supplementary S3.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual CNN block used in the coarse branch.
    """

    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()

        self.body = nn.Sequential(
            ConvBlockII(in_ch, out_ch),
            ConvBlockII(out_ch, out_ch),
            ConvBlockII(out_ch, out_ch),
        )

        self.shortcut = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


@ARCH_REGISTRY.register()
class SwinFlood(nn.Module):
    """
    SwinFlood-style baseline for flood map downscaling.

    Expected inputs:
        coarse_fm: [B, coarse_in_chans, Hc, Wc]
        static_f:  [B, static_in_chans, Hf, Wf]

    For our Gisborne setting:
        Hc = Wc = 64
        Hf = Wf = 512
        upscale = 8
        Hm = Wm = 64

    Outputs:
        depth:       [B, 1, Hf, Wf]
        flood_logit: [B, num_flood_classes, Hf, Wf]
    """

    def __init__(
        self,
        upscale=8,
        flood_map_size=64,
        coarse_in_chans=1,
        static_in_chans=7,
        embed_dim=128,
        depths=(6, 6, 6),
        num_heads=8,
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        resi_connection='1conv',
        num_flood_classes=3,
        **kwargs,
    ):
        super().__init__()

        assert upscale in (2, 4, 8, 16), "This implementation expects upscale to be a power of 2."
        assert flood_map_size % window_size == 0, (
            f"flood_map_size ({flood_map_size}) should be divisible by "
            f"window_size ({window_size})."
        )
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})."
        )

        self.upscale = upscale
        self.flood_map_size = flood_map_size
        self.coarse_in_chans = coarse_in_chans
        self.static_in_chans = static_in_chans
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_flood_classes = num_flood_classes

        num_feat = 64
        input_resolution = (flood_map_size, flood_map_size)
        num_down = int(math.log2(upscale))

        # ---------------------------------------------------------------------
        # 1. Multi-resolution input feature fusion module
        # ---------------------------------------------------------------------
        # Coarse dynamic branch:
        # [B, Cc, 64, 64] -> [B, 64, 64, 64]
        self.coarse_preprocess = nn.Conv2d(
            coarse_in_chans,
            num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.coarse_rb = ResidualBlock(
            in_ch=num_feat,
            out_ch=num_feat,
        )

        # Fine static branch:
        # [B, Cs, 512, 512] -> [B, 64, 64, 64] for scale=8
        static_layers = []
        in_ch = static_in_chans

        for _ in range(num_down):
            static_layers.append(ConvBlockI(in_ch, num_feat))
            in_ch = num_feat

        self.static_down = nn.Sequential(*static_layers)

        # Concatenate coarse dynamic features and downsampled static features:
        # [B, 64 + 64, 64, 64] -> [B, embed_dim, 64, 64]
        assert embed_dim == 2 * num_feat, (
            f"For this paper-style SwinFlood implementation, embed_dim should be "
            f"2 * num_feat = {2 * num_feat}, but got embed_dim={embed_dim}."
        )

        # ---------------------------------------------------------------------
        # 2. Deep feature extraction module: 3 RSTBs + conv + long skip
        # ---------------------------------------------------------------------
        #   FD_i = RSTB(FD_{i-1}), i = 1, 2, 3
        #   FD = Conv(FD_3) + FD_0
        self.patch_embed = PatchEmbed(
            norm_layer=None,
            embed_dim=embed_dim,
        )

        self.patch_unembed = PatchUnEmbed(
            embed_dim=embed_dim,
        )

        self.layers = nn.ModuleList()
        for depth in depths:
            self.layers.append(
                RSTB(
                    dim=embed_dim,
                    input_resolution=input_resolution,
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    resi_connection=resi_connection,
                )
            )

        self.norm = norm_layer(embed_dim)

        self.conv_after_body = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ---------------------------------------------------------------------
        # 3. Upsampling and output module
        # ---------------------------------------------------------------------
        # FUp_0 = ConvBlockII(FD)
        self.conv_before_upsample = ConvBlockII(
            embed_dim,
            num_feat,
        )

        # FUp_i = ConvBlockIII(FUp_{i-1}), i = 1, ..., log2(upscale)
        up_layers = []
        for _ in range(num_down):
            up_layers.append(ConvBlockIII(num_feat, num_feat))

        self.upsample = nn.Sequential(*up_layers)

        self.conv_last = nn.Conv2d(
            num_feat,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.flood_head = nn.Conv2d(
            num_feat,
            num_flood_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, coarse_fm, static_f):
        """
        Args:
            coarse_fm: [B, Cc, Hc, Wc]
            static_f:  [B, Cs, Hf, Wf]
        Returns:
            depth:       [B, 1, Hf, Wf]
            flood_logit: [B, num_flood_classes, Hf, Wf]
        """

        B, _, Hc, Wc = coarse_fm.shape
        _, _, Hf, Wf = static_f.shape

        if Hf != Hc * self.upscale or Wf != Wc * self.upscale:
            raise RuntimeError(
                f"[SwinFlood] fine/coarse size mismatch: "
                f"Hc={Hc}, Wc={Wc}, Hf={Hf}, Wf={Wf}, "
                f"upscale={self.upscale}"
            )

        if Hc != self.flood_map_size or Wc != self.flood_map_size:
            raise RuntimeError(
                f"[SwinFlood] this implementation expects coarse patch size "
                f"{self.flood_map_size} x {self.flood_map_size}, "
                f"but got {Hc} x {Wc}."
            )

        # ------------------------------------------------------------------
        # 1. Multi-resolution input feature fusion
        # ------------------------------------------------------------------
        # Coarse dynamic branch: [B, Cc, 64, 64] -> [B, 64, 64, 64]
        fc = self.coarse_preprocess(coarse_fm)
        fc = self.coarse_rb(fc)

        # Static branch: [B, Cs, 512, 512] -> [B, 64, 64, 64]
        fs = self.static_down(static_f)

        if fs.shape[-2:] != fc.shape[-2:]:
            raise RuntimeError(
                f"[SwinFlood] static branch and coarse branch size mismatch: "
                f"static={fs.shape[-2:]}, coarse={fc.shape[-2:]}"
            )

        # Fusion: [B, 128, 64, 64] -> [B, embed_dim, 64, 64]
        x = torch.cat([fc, fs], dim=1)

        # ------------------------------------------------------------------
        # 2. Deep feature extraction with long skip connection
        # ------------------------------------------------------------------
        x_skip = x
        x = self.forward_features(x)
        x = self.conv_after_body(x) + x_skip

        # ------------------------------------------------------------------
        # 3. Upsampling and output
        # ------------------------------------------------------------------
        x = self.conv_before_upsample(x)
        x = self.upsample(x)

        depth = self.conv_last(x)
        flood_logit = self.flood_head(x)

        return depth, flood_logit