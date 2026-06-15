# -----------------------------------------------------------------------------------
# Residual Swin-UNet (RSwinUNet) baseline for flood-map downscaling.
#
# This is a FAITHFUL port of the official implementation released with:
#   Wenke Song & Mingfu Guan (2026), "Enhancing cross-regional transferability of
#   super-resolution-based flood surrogate models to data-scarce catchments",
#   Water Research 298, 125799.   (their loss = masked mean absolute error / L1, Eq. 2)
#
# The model body below (drop_path / Mlp / WindowAttention / SwinTransformerBlock /
# PatchMerging / PatchExpand / FinalPatchExpand_X4 / BasicLayer(_up) / PatchEmbed /
# PatchUnEmbed / RSTB / RSTB_up / SwinTransformerSys) is copied VERBATIM from the
# authors' `tools/archive/Residual_SwinUNet.py` so that the architecture matches the paper
# exactly. Only the following repo-integration adaptations are made, and they do NOT
# alter the depth-prediction computation graph:
#
#   1. INPUT. `forward(coarse_fm, static_f)` matches this repo's FMSRModel contract
#      (same as heunet_arch / swinflood_arch). The coarse input is upsampled (x`upscale`)
#      to the fine grid and concatenated with the fine static features along channel,
#      so in_chans = coarse_in_chans + static_in_chans. The paper stacks
#      {t-1,t,t+1 coarse flood map, DEM, DEM_LR, SLOPE, ASPECT, TWI, DRAINAGE}; here we
#      substitute OUR inputs (coarse h/dem/zs + fine dem/slope/twi/aspect_sin/aspect_cos/
#      roughness/mask) so the baseline is compared on the SAME data as TG-PFT/HeUNet/SwinFlood.
#
#   2. OUTPUT. Depth is produced by the paper's own final output conv (num_classes=1).
#      For the matched-setup experiment (+ ordinal BCE) an OPTIONAL parallel
#      `flood_head` = Conv2d(embed_dim, num_flood_classes, 1) is added on the same
#      pre-output feature — identical pattern to swinflood_arch.py. `forward` returns
#      (depth, flood_logit). The depth path is byte-for-byte the paper's.
#
#   3. The paper's `clamp(min=0)` + study-area "masking layer" are DISABLED by default
#      because our targets are asinh+z-score normalized (0 m maps to a negative z-score,
#      so clamping at 0 would be wrong) and masking is already applied by the loss /
#      eval mask — consistent with how SwinFlood/HeUNet are ported in this repo. They
#      remain available via `clamp_min` / `use_output_mask` if a [0,1]-normalized run is used.
#
# Original implementation credits (see tools/archive/Residual_SwinUNet.py header): SwinIR
# (Liang et al. 2021), Swin Transformer (Liu et al. 2021), Swin-Unet (Cao et al. 2023).
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, x_size=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)  # [B, HW, C] -> [B, HW, 2C]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)  # [B, HW, 2C] -> [B, H, W, 2C]

        B, H, W, C = x.shape
        x = x.view(B, H, W, 2, 2, C // 4)  # [B, H, W, C] -> [B, H, W, 2, 2, C//4]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, H, W, 2, 2, C//4] -> [B, H, 2, W, 2, C//4]
        x = x.reshape(B, H * 2, W * 2, C // 4)  # [B, H, 2, W, 2, C//4] -> [B, H*2, W*2, C//4]

        x = x.view(B, -1, C // 4)  # [B, H*2, W*2, C//4] -> [B, HW*4, C//4]
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        B, H, W, C = x.shape
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, C // (self.dim_scale ** 2))
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, C // (self.dim_scale ** 2))

        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage (encoder). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()

        return flops


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage (decoder). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding """

    def __init__(self, img_size=256, patch_size=4, in_chans=11, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.patch_size[0] > 1 or self.patch_size[1] > 1:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        else:
            pass

        x_proj = self.proj(x)
        x_flatten = x_proj.flatten(2)
        x_trans = x_flatten.transpose(1, 2)

        if self.norm is not None:
            x_trans = self.norm(x_trans)
        return x_trans

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """ Patch to Image (sequence -> feature map). """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """
        x: B, L, C
        """
        B, L, C = x.shape
        assert L == x_size[0] * x_size[1], "Input feature size mismatch"

        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1])  # B, C, H, W
        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB) — encoder path. """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, x_size):

        identity = x
        out = self.residual_group(x)
        out = self.patch_unembed(out, x_size)
        out = self.conv(out)
        out = self.patch_embed(out)

        if out.shape[1] != x.shape[1] or out.shape[2] != x.shape[2]:
            out = out.flatten(2).transpose(1, 2)

        out = identity + out

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class RSTB_up(nn.Module):
    """ Residual Swin Transformer Block for Decoder Path (RSTB_up). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4):
        super(RSTB_up, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer_up(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer) if upsample else None

    def forward(self, x, x_size):

        identity = x
        out = self.residual_group(x)
        residual_spatial = self.patch_unembed(identity, x_size)
        residual_conv = self.conv(residual_spatial)
        residual_seq = self.patch_embed(residual_conv)
        out = out + residual_seq

        if self.upsample is not None:
            out = self.upsample(out)

        return out, self.upsample is not None


class SwinTransformerSys(nn.Module):
    r""" Residual Swin-UNet backbone (encoder-decoder with skip connections).

    Faithful to tools/archive/Residual_SwinUNet.py. The only change is that `forward` returns
    the pre-output feature map (B, embed_dim, H, W) together with the depth produced by
    the paper's own output conv, so a parallel flood head can be attached by the wrapper.
    The paper's `clamp(min=0)` + masking layer are handled (optionally) by the wrapper.
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=11, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=int(embed_dim * 2 ** i_layer),
                         input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                           patches_resolution[1] // (2 ** i_layer)),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                          self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = RSTB_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                   input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                     patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                   depth=depths[(self.num_layers - 1 - i_layer)],
                                   num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                       depths[:(self.num_layers - 1 - i_layer) + 1])],
                                   norm_layer=norm_layer,
                                   upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint,
                                   img_size=img_size,
                                   patch_size=patch_size,
                                   )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,
                                    out_channels=self.num_classes,
                                    kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        x_size = (self.patches_resolution[0], self.patches_resolution[1])

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x, x_size)
            if layer.downsample is not None:
                x_size = (x_size[0] // 2, x_size[1] // 2)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        x_size = (self.patches_resolution[0] // (2 ** (self.num_layers - 1)),
                  self.patches_resolution[1] // (2 ** (self.num_layers - 1)))

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x, x_size)
                x_size = (x_size[0] * 2, x_size[1] * 2)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x, has_upsampled = layer_up(x, x_size)
                if has_upsampled:
                    x_size = (x_size[0] * 2, x_size[1] * 2)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4_feature(self, x):
        """Final x4 patch expand -> pre-output feature map (B, embed_dim, H, W)."""
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W   (C = embed_dim)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        feat = self.up_x4_feature(x)          # (B, embed_dim, H, W)
        depth = self.output(feat)             # paper's output conv -> (B, num_classes, H, W)
        return depth, feat

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


@ARCH_REGISTRY.register()
class RSwinUNet(nn.Module):
    """Residual Swin-UNet (Song & Guan 2026) adapted to this repo's flood-downscaling pipeline.

    Expected inputs (same contract as heunet_arch / swinflood_arch):
        coarse_fm: [B, coarse_in_chans, Hc, Wc]   (e.g. coarse h / dem / zs at 64x64)
        static_f:  [B, static_in_chans, Hf, Wf]   (e.g. fine dem/slope/twi/aspect_sin/
                                                    aspect_cos/roughness/mask at 512x512)
    Outputs:
        depth:       [B, 1, Hf, Wf]
        flood_logit: [B, num_flood_classes, Hf, Wf]  (None if use_flood_head=False)

    The coarse input is upsampled x`upscale` to the fine grid and concatenated with the
    fine static features along the channel dim, then fed to the (verbatim) RSwinUNet body.
    """

    def __init__(
        self,
        upscale=8,
        img_size=512,
        coarse_in_chans=3,
        static_in_chans=7,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        depths_decoder=(1, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        num_flood_classes=3,
        use_flood_head=True,
        coarse_upsample_mode='bilinear',
        clamp_min=None,
        use_output_mask=False,
        **kwargs,
    ):
        super().__init__()

        self.upscale = int(upscale)
        self.img_size = int(img_size)
        self.coarse_in_chans = int(coarse_in_chans)
        self.static_in_chans = int(static_in_chans)
        self.embed_dim = int(embed_dim)
        self.num_flood_classes = int(num_flood_classes)
        self.use_flood_head = bool(use_flood_head)
        self.coarse_upsample_mode = str(coarse_upsample_mode)
        self.clamp_min = clamp_min
        self.use_output_mask = bool(use_output_mask)

        in_chans = self.coarse_in_chans + self.static_in_chans

        self.body = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=1,                       # depth
            embed_dim=embed_dim,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample="expand_first",
        )

        if self.use_flood_head:
            # Parallel flood head on the same pre-output feature (identical to swinflood_arch).
            self.flood_head = nn.Conv2d(embed_dim, self.num_flood_classes, kernel_size=1, stride=1, padding=0)
        else:
            self.flood_head = None

    def forward(self, coarse_fm, static_f):
        Hf, Wf = static_f.shape[-2], static_f.shape[-1]
        Hc, Wc = coarse_fm.shape[-2], coarse_fm.shape[-1]

        if Hf != Hc * self.upscale or Wf != Wc * self.upscale:
            raise RuntimeError(
                f"[RSwinUNet] fine/coarse size mismatch: "
                f"Hc={Hc}, Wc={Wc}, Hf={Hf}, Wf={Wf}, upscale={self.upscale}"
            )
        if Hf != self.img_size or Wf != self.img_size:
            raise RuntimeError(
                f"[RSwinUNet] fine patch size {Hf}x{Wf} != img_size {self.img_size}."
            )

        # Coarse -> fine, then channel-concat with fine static features.
        if self.coarse_upsample_mode in ("bilinear", "bicubic"):
            coarse_up = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.coarse_upsample_mode, align_corners=False)
        else:
            coarse_up = F.interpolate(coarse_fm, size=(Hf, Wf), mode=self.coarse_upsample_mode)

        x = torch.cat([coarse_up, static_f], dim=1)

        depth, feat = self.body(x)
        flood_logit = self.flood_head(feat) if (self.flood_head is not None) else None

        # Paper's clamp(min=0) + masking layer: OFF by default (asinh+zscore space). Optional.
        if self.clamp_min is not None:
            depth = depth.clamp(min=float(self.clamp_min))
        if self.use_output_mask:
            mask = static_f[:, -1:, ...]
            depth = depth * mask

        return depth, flood_logit
