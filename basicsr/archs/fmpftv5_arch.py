"""
FloodMapPFTV5

Changes vs FloodMapPFTV4 (see fmpftv4_arch.py):

1) WindowAttention.forward
   - Removed the inference-mode `torch.cuda.empty_cache()` + `del q, k, v, ...`
     block that was previously executed for every transformer layer * every
     window. Calling `empty_cache()` per window is extremely expensive and
     was slowing validation down by orders of magnitude. Tensors will be
     released naturally by Python's reference counting / autograd.

2) FloodMapPFTV5.forward (padding logic)
   - Fixed the self-contradictory padding branch in V4 where any non-zero
     pad amount would immediately raise RuntimeError, making the padding
     code path effectively dead.
   - The padding is now fully implemented:
       * coarse_fm is reflection-padded to a multiple of window_size on Hc/Wc.
       * static_f (which lives at fine resolution = upscale * coarse) is
         reflection-padded by `pad * upscale` on the corresponding axes, so
         that after `conv_static_f` it aligns spatially with the padded
         coarse stream.
       * Final outputs `depth` and `flood_logit` are cropped back to the
         original fine spatial size before being returned.
   - The strict scale-consistency check (Hf == upscale * Hc, Wf == upscale * Wc)
     is now performed BEFORE padding and is the only fail-fast path. The
     legacy `Hc_pad != 0` clause that broke everything is gone.

Example config (same as V4 except for `type`):

network_g:
  type: FloodMapPFTV5
  upscale: 16
  coarse_in_chans: 1
  static_in_chans: 7
  flood_map_size: 64
  use_shallow_act: true
  use_aoi_gate: true
  aoi_alpha: 0.8
  embed_dim: 240
  depths: [4, 4, 4, 6, 6, 6]
  num_heads: 6
  num_topk: [ 1024, 1024, 1024, 1024,
              256, 256, 256, 256,
              128, 128, 128, 128,
              64, 64, 64, 64, 64, 64,
              32, 32, 32, 32, 32, 32,
              16, 16, 16, 16, 16, 16,]
  window_size: 32
  convffn_kernel_size: 7
  mlp_ratio: 2.
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
  use_checkpoint: false
"""

import math
import torch
import torch.nn as nn
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import smm_cuda


class SMM_QmK(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM) of
    query (Q) and key (K) matrices, based on given sparse indices.

    This function leverages a CUDA-implemented kernel for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Query matrix).
            B: Input tensor B (Key matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save input tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function for sparse matrix multiplication
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class SMM_AmV(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM)
    between an activation matrix (A) and a value matrix (V), guided by sparse indices.

    This function utilizes a CUDA-optimized implementation for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Activation matrix).
            B: Input tensor B (Value matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.hidden_features, x_size[0], x_size[1])
        # x [B, N, C] batch B * token N * channel C, N = Ph*Pw
        # .transpose(1, 2): [B, N, C] -> [B, C, N]
        # .view: [B, C, N] -> [B, C, Ph, Pw]
        x = self.depthwise_conv(x)
        # H_out = [(H_in + 2*padding - dilation*(kernel_size-1)-1)/stride + 1]
        x = x.flatten(2).transpose(1, 2).contiguous()
        # .flatten(2): [B, C, Ph, Pw] -> [B, C, N]; .transpose() -> [B, N, C]
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of input
        w (int): Width of input

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention (MSA).

    Args:
        dim (int): Number of input channels.
        layer_id (int): Index of the current layer in the network.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values retained for sparsity.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value tensors. Default: True.
    """

    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 1/sqrt(d)
        self.eps = 1e-20  # stabilize value for PFA normalization

        # define a parameter table of relative position bias
        if dim > 100:
            # for classical SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        else:
            # for lightweight SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        r"""
        Args:
        qkvp (Tensor): Input tensor containing query, key, value tokens, and LePE positional encoding matrix with shape (num_windows * b, n, c * 4),
        pfa_values (Tensor or None): Precomputed attention values for Progressive Focusing Attention (PFA). If None, standard attention is applied.
        pfa_indices (Tensor or None): Index tensor for Progressive Focusing Attention (PFA), indicating which attention values should be retained or discarded.
        rpi (Tensor): Relative position index tensor, encoding positional information for tokens.
        mask (Tensor or None, optional): Attention mask tensor.
        shift (int, optional): Indicates whether window shifting is applied (e.g., 0 for no shift, 1 for shifted windows). Default: 0.
        """
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # .reshape: [b_, n, c4], n=Wh*Ww, b_=b*num_windows -> [b_, n, 4, heads, c//heads]
        # .permute: -> [4, b_, h, n, d], d = c//heads
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # Standard Attention Computation, q/sqrt(d)
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))
            # [b_, h, n, d] @ [b_ h, d, n] -> [b_, h, n, n]
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            # rpi: [n, n], .view(-1) -> n*n; .view: [n*n, h] -> [n, n, h] = [Wh*Ww, Wh*Ww, h]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            # .permute -> [h, n, n]; .unsqueeze -> [1, h, n, n]
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

            if shift:
                nw = mask.shape[0]
                # mask: [num_windows, n, n]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                # .view -> [b, num_windows, h, n, n]; .unsqueeze(1).unsqueeze(0) -> [1, num_windows, 1, n, n]
                attn = attn.view(-1, self.num_heads, n, n)
                # .view -> [b_, h, n, n]
        # # Sparse Attention Computation using SMM_QmK
        else:
            topk = pfa_indices[shift].shape[-1]
            # pfa_indices[shift]: [b_, h, n, k]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            # .view: [b_, h, n, d] -> [b_*h, n, d]
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            # .transpose -> [b_*h, d, n]
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            # .view().int() -> [b_*h, n, k]
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)
            # .view() -> [b_, h, n, k]

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            # .permute().unsqueeze().expand() -> [b_, h, n, n]
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            # .gather() -> [b_, h, n, k]
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

        # Use in-place operations where possible (only in inference mode)
        if not self.training:  # Check if in inference mode
            attn = torch.softmax(attn, dim=-1, out=attn)  # 原地softmax
        else:
            attn = self.softmax(attn)  # Non-inplace if training

        # Apply Hadamard product for PFA and normalize.
        if pfa_values[shift] is not None:
            if not self.training:  # only in inference
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        # If sparsification is enabled, select top-k attention values and save the corresponding indexes
        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        # Save the current attention results as PFA maps.
        pfa_values[shift] = attn

        # # Save the attention map as a .npy file for visualization or further analysis
        # # Scatter the attention values back to their original indices
        # # attn_npy has shape (batch_size * num_windows, num_heads, n, n)
        # if pfa_indices[shift] is None:
        #     attn_npy = attn
        # else:
        #     attn_npy = torch.zeros((b_, self.num_heads, n, n), device=attn.device).scatter(-1, pfa_indices[shift], attn)
        # # Define the path where the attention map will be saved
        # attention_save_path = f"./results/Attention_map/PFT_light_attention_map_w32_L{self.layer_id}.npy"
        # os.makedirs("./results/Attention_map", exist_ok=True)
        # # Save the attention map only if the file does not already exist to avoid overwriting
        # if not os.path.exists(attention_save_path):
        #     np.save(attention_save_path, attn_npy.cpu().detach().numpy())

        # Check whether sparsification has been applied; if so, use SMM_AmV for computation, otherwise perform standard matrix multiplication A @ V.
        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
            # [b_, h, n, n] @ [b_, h, n, d] -> [b_, h, n, d]; .transpose() -> [b_, n, h, d]; .reshape() -> [b_, n, c]
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads)+ v_lepe).transpose(1, 2).reshape(b_, n, c)
            # SMM_AmV -> [b_*h, n, d]; .view() -> [b_, h, n, d]; .transpose() -> [b_, n, h, d]; .reshape() -> [b_, n, c]

        # NOTE (V5 fix #1): The V4 version called `torch.cuda.empty_cache()` here in
        # inference mode, once per transformer layer per window. That made validation
        # an order of magnitude slower with no real memory benefit (Python reference
        # counting / autograd already release these intermediates). Removed.

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        if self.layer_id < 2:
            # attn = (q @ k.transpose(-2, -1)) [b_, h, n, d] @ [b_ h, d, n] -> [b_, h, n, n]
            flops += self.num_heads * n * (self.dim // self.num_heads) * n
            #  x = (attn @ v)
            flops += self.num_heads * n * n * (self.dim // self.num_heads)
        else:
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * n * (self.dim // self.num_heads) * self.num_topk[self.layer_id-2]
            #  x = (attn @ v)
            flops += self.num_heads * n * self.num_topk[self.layer_id] * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class PFTransformerLayer(nn.Module):
    r"""
    PFT Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.layer_id = layer_id
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        # NOTE (V5): removed `self.softmax / self.lrelu / self.sigmoid`. They
        # were declared in V4 but never referenced in `forward` (dead modules).
        # Removal is safe: they are stateless (no parameters or buffers), so
        # checkpoint state_dict compatibility is unaffected.

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,act_layer=act_layer)

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x

        x = self.norm1(x)
        # -> [b, n, c]
        x_qkv = self.wqkv(x)
        # -> [b, n, 3c]

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)
        # -> [b, n, 4c]

        # SW-MSA
        # cyclic shift
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, h, w, c4), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # .roll() -> [b, Ph, Pw, c4]
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)
            # .reshape() -> [b, Ph, Pw, c4]
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        # [b, Ph, Pw, c4] -> [b*num_windows (b_), Wh, Ww, c4]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        # -> [b_, n, c4]
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, pfa_values, pfa_indices = self.attn_win(x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices, rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift)
        # rpi_sa: [n, n]; attn_mask: [num_windows, n, n]; attn_windows: [b_, n, c]
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        # .view(): [b_, n, c] -> [b_, Wh, Ww, c]
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        # -> [b, Ph, Pw, c]
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        x = shortcut + x_win.view(b, n, c)
        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # wqkv, n*c*3c
        flops += self.dim * 3 * self.dim * h * w

        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)

        # mlp
        # 2 linear layer: dim -> mlp_ratio*dim -> dim
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        # depthwide conv layer
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio
        # lepe
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)
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
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic PFT Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Local window size.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 layer_id,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                PFTransformerLayer(
                    dim=dim,
                    block_id=idx,
                    layer_id= layer_id + i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, x_size, params):
        for layer in self.layers:
            # checkpoint_wrapper is not yet supported for PFT
            # idx_checkpoint = 4
            # if self.use_checkpoint and self.idx < idx_checkpoint:
            #     layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x, pfa_list = layer(x, pfa_list, x_size, params)

        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class PFTB(nn.Module):
    """Adaptive Token Dictionary Block (PFTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        flood_map_size: Input flood map size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 idx,
                 layer_id,
                 input_resolution,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 flood_map_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 ):
        super(PFTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            flood_map_size=flood_map_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            flood_map_size=flood_map_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            layer_id=layer_id,
            depth=depth,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, pfa_list, x_size, params):
        x_Basicblock, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_Basicblock, x_size))) + x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        flood_map_size (int): Flood map size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input flood map channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, flood_map_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        flood_map_size = to_2tuple(flood_map_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [flood_map_size[0] // patch_size[0], flood_map_size[1] // patch_size[1]]
        self.flood_map_size = flood_map_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.flood_map_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        flood_map_size (int): Flood map size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input flood map channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, flood_map_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        flood_map_size = to_2tuple(flood_map_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [flood_map_size[0] // patch_size[0], flood_map_size[1] // patch_size[1]]
        self.flood_map_size = flood_map_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                # m.append(nn.Conv2d(num_feat, 4 * num_feat, 5, 1, 2))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            # m.append(nn.Conv2d(num_feat, 9 * num_feat, 5, 1, 2))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
            # flops += self.num_feat * 4 * self.num_feat * 25 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
            # flops += self.num_feat * 9 * self.num_feat * 25 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.scale = scale
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        # h, w = self.patches_resolution if input_resolution is None else input_resolution
        h, w = input_resolution
        # flops = h * w * self.num_feat * 3 * 9
        flops = h * w* self.num_feat * (self.scale ** 2) * self.num_out_ch * 9
        return flops


@ARCH_REGISTRY.register()
class FloodMapPFTV5(nn.Module):
    r""" FloodMapPFTV5
        A PyTorch impl of : `Progressive Focused Transformer for Single Image Super-Resolution`.

        Differences vs FloodMapPFTV4 (see fmpftv4_arch.py):
          - Removed expensive per-window `torch.cuda.empty_cache()` in
            WindowAttention.forward (V5 fix #1).
          - Padding path in `forward` is now actually executable, and also
            applies a matching reflection-pad to `static_f` so that the
            fine-resolution branch stays aligned with the (padded) coarse
            branch. Outputs are cropped back to the original fine size.

    Args:
        flood_map_size (int | tuple(int)): Input flood map size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input flood map channels. Default: 1
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
                        This controls the sparsity of the attention map, keeping only the most relevant attention scores for further processing.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8/16 for flood map SR, 1 for denoising and compress artifact reduction
        upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 flood_map_size=64,
                 patch_size=1,
                 coarse_in_chans=1,
                 static_in_chans=7,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 num_topk=[256, 256,  128, 128, 128, 128,  64, 64, 64, 64, 64, 64,  32, 32, 32, 32, 32, 32,  16, 16, 16, 16, 16, 16],
                 window_size=8,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 upsampler='',
                 resi_connection='1conv',
                 use_shallow_act=True,
                 use_aoi_gate=True,
                 aoi_alpha=0.8,
                 couple_mode="detach",
                 couple_eps=0.2,
                 **kwargs):
        super().__init__()
        num_in_ch = coarse_in_chans
        num_out_ch = 1
        self.coarse_in_ch = coarse_in_chans
        self.static_in_ch = static_in_chans
        num_feat = 64
        self.upscale = upscale
        self.upsampler = upsampler

        self.use_shallow_act = use_shallow_act
        self.use_aoi_gate = use_aoi_gate
        self.aoi_alpha = aoi_alpha

        self.couple_mode = str(couple_mode).lower().strip()
        self.couple_eps = couple_eps
        assert self.couple_mode in ("detach", "coupled", "none")

        # -------------------- 1. Multi-Resolution Shallow Feature Fusion and Extraction Module -------------------- #
        # self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_coarse_fm = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        downs = []
        if self.upsampler == 'pixelshuffle':
            for i in range(int(math.log2(self.upscale))):
                downs.append(nn.PixelUnshuffle(2))
                downs.append(nn.Conv2d(4 * self.static_in_ch, self.static_in_ch, 3, 1, 1))
                if i > 0 and self.use_shallow_act:
                    downs.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif self.upsampler == 'pixelshuffledirect':
            downs.append(nn.PixelUnshuffle(self.upscale))
            downs.append(nn.Conv2d(self.static_in_ch * (self.upscale ** 2), self.static_in_ch, 3, 1, 1))
            if self.use_shallow_act:
                downs.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise RuntimeError(f'[ERROR] yml.upsample ({self.upsampler}) needs to be "pixelshuffle" or '
                               f'"pixelshuffledirect"')
        self.conv_static_f = nn.Sequential(*downs, nn.Conv2d(self.static_in_ch, num_feat, 3, 1, 1))

        self.gn_coarse = nn.GroupNorm(num_groups=8, num_channels=num_feat)
        self.gn_static = nn.GroupNorm(num_groups=8, num_channels=num_feat)

        self.conv_first = nn.Conv2d(2 * num_feat, embed_dim, 3, 1, 1)

        # ------------------------- 2. Deep Feature Extraction Module ------------------------- #
        self.num_layers = len(depths)
        self.layer_id = 0
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split flood map into non-overlapping patches
        self.patch_embed = PatchEmbed(
            flood_map_size=flood_map_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into flood map
        self.patch_unembed = PatchUnEmbed(
            flood_map_size=flood_map_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # build Residual Adaptive Token Dictionary Blocks (PFTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PFTB(
                dim=embed_dim,
                idx=i_layer,
                layer_id=self.layer_id,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads,
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                flood_map_size=flood_map_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
            self.layer_id = self.layer_id + depths[i_layer]

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 5, 1, 2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3. Up-sampling Module ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
                # nn.Conv2d(embed_dim, num_feat, 5, 1, 2), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 5, 1, 2)
            # flood head for nse + precision + recall
            self.flood_head = nn.Conv2d(num_feat, 3, kernel_size=1, stride=1, padding=0)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
            # flood head for nse + precision + recall
            self.flood_head = nn.Conv2d(num_out_ch, 3, kernel_size=1, stride=1, padding=0)
        else:
            raise RuntimeError(f'[ERROR] yml.upsample ({self.upsampler}) needs to be "pixelshuffle" or '
                               f'"pixelshuffledirect"')

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

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        # Define progressive focusing attention (PFA) values and their corresponding indices
        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        # generate x-y coordinates for all tokens
        coords_h = torch.arange(self.window_size)
        # [Wh], 0..Wh-1
        coords_w = torch.arange(self.window_size)
        # [Ww], 0..Ww-1
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # # [2, Wh, Ww]  coords[0] -> y (row), coords[1] -> x (col)
        coords_flatten = torch.flatten(coords, 1)
        # [2, n], n=Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # calculate relative position between any two tokens
        # broadcasting: [2, n, 1] - [2, 1, n] -> [2, n, n]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # [n, n, 2]
        relative_coords[:, :, 0] += self.window_size - 1
        # shift to non-zero interval [0..(2*W-2)]; dy += Wh-1
        relative_coords[:, :, 1] += self.window_size - 1
        # dx += Ww-1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        # map (dy, dx) to one-dimensional index; idx = dy * (2*Ww-1) + dx; [n, n]
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        flood_map_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                flood_map_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(flood_map_mask, self.window_size)  # nw, window_size, window_size, 1
        # [1, h, w, 1] -> [num_windows, Wh, Ww, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # -> [num_windows, n]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # -> [num_windows, n, n]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, coarse_fm, static_f):
        # coarse-grid flood map: [B, coarse_in_chans, Hc, Hc], fine-grid static feature: [B, static_in_chans, Hf, Hf]
        B, _, Hc_orig, Wc_orig = coarse_fm.shape
        _, _, Hf_orig, Wf_orig = static_f.shape

        # V5 fix #2: validate the fine/coarse scale relationship BEFORE padding.
        # The old V4 check intermingled this with the pad-mismatch error and made
        # the pad branch unreachable.
        if (Hf_orig != Hc_orig * self.upscale) or (Wf_orig != Wc_orig * self.upscale):
            raise RuntimeError(
                f'[ERROR] fine/coarse size mismatch with upscale: '
                f'Hc={Hc_orig}, Wc={Wc_orig}, Hf={Hf_orig}, Wf={Wf_orig}, scale={self.upscale}'
            )

        mod = self.window_size
        Hc_pad = ((Hc_orig + mod - 1) // mod) * mod - Hc_orig
        Wc_pad = ((Wc_orig + mod - 1) // mod) * mod - Wc_orig

        if (Hc_pad != 0) or (Wc_pad != 0):
            # Pad coarse_fm by reflection on the bottom/right.
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [2])], 2)[:, :, :Hc_orig + Hc_pad, :]
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [3])], 3)[:, :, :, :Wc_orig + Wc_pad]
            # Pad static_f by reflection by `pad * upscale` to keep the
            # fine-resolution branch aligned with the (padded) coarse branch
            # after conv_static_f downsamples by `upscale`.
            Hf_pad = Hc_pad * self.upscale
            Wf_pad = Wc_pad * self.upscale
            static_f = torch.cat([static_f, torch.flip(static_f, [2])], 2)[:, :, :Hf_orig + Hf_pad, :]
            static_f = torch.cat([static_f, torch.flip(static_f, [3])], 3)[:, :, :, :Wf_orig + Wf_pad]

        Hc, Wc = Hc_orig + Hc_pad, Wc_orig + Wc_pad

        attn_mask = self.calculate_mask([Hc, Wc]).to(coarse_fm.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # AOI mask gate
        if self.use_aoi_gate:
            aoi_mask_fine = static_f[:, -1:, :, :]
            aoi_mask_coarse = torch.nn.functional.adaptive_avg_pool2d(aoi_mask_fine, (Hc, Wc))
            gate = (1.0 - self.aoi_alpha) + self.aoi_alpha * aoi_mask_coarse
            gate = gate.clamp(0.0, 1.0).detach()
        else:
            gate = None

        # Multi-Resolution Shallow Feature Fusion and Extraction Module
        coarse_fm = self.conv_coarse_fm(coarse_fm)  # [B, 64, Hc, Hc]
        coarse_fm = self.gn_coarse(coarse_fm)
        static_f = self.conv_static_f(static_f)  # [B, 64, Hc, Hc]
        static_f = self.gn_static(static_f)
        x_fm_st = torch.cat([coarse_fm, static_f], dim=1)
        x = self.conv_first(x_fm_st)
        x_rc = x

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            if gate is not None:
                x = x * gate
            x = self.conv_after_body(self.forward_features(x, params)) + x_rc
            if gate is not None:
                x = x * gate
            x = self.conv_before_upsample(x)
            x_up = self.upsample(x)
            depth = self.conv_last(x_up)
            flood_logit = self.flood_head(x_up)
            # x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            if gate is not None:
                x = x * gate
            x = self.conv_after_body(self.forward_features(x, params)) + x_rc
            if gate is not None:
                x = x * gate
            x = self.upsample(x)
            depth = x
            flood_logit = self.flood_head(x)
        else:
            raise RuntimeError(f'[ERROR] yml.upsample ({self.upsampler}) needs to be "pixelshuffle" or '
                               f'"pixelshuffledirect"')

        # V5 fix #2: crop the outputs back to the ORIGINAL fine-resolution size,
        # not the padded size. In V4 this used the padded `Hc * upscale`, which
        # was effectively a no-op when nothing was padded but would have
        # exposed the padded border if padding had ever worked.
        depth = depth[..., :Hf_orig, :Wf_orig]
        flood_logit = flood_logit[..., :Hf_orig, :Wf_orig]

        if self.couple_mode != "none":
            p_wet = torch.sigmoid(flood_logit[:, 0:1, :, :])
            if self.couple_mode == "detach":
                p_wet = p_wet.detach()
            depth = depth * (self.couple_eps + (1 - self.couple_eps) * p_wet)

        return depth, flood_logit

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        # conv first
        # flops += h * w * 3 * self.embed_dim * 9
        # coarse branch
        flops += h * w * self.coarse_in_ch * 64 * 9
        # static branch
        if self.upsampler == 'pixelshuffle':
            Hf, Wf = h * self.upscale, w * self.upscale
            curC = self.static_in_ch
            curH, curW = Hf, Wf
            for _ in range(int(math.log2(self.upscale))):
                inC = 4 * curC
                outC = curC
                curH //= 2
                curW //= 2
                flops += curH * curW * inC * outC * 9
        else:
            flops += h * w * (self.static_in_ch * (self.upscale ** 2)) * self.static_in_ch * 9
        # last layer of conv_static_f
        flops += h * w * self.static_in_ch * 64 * 9
        # conv_first
        flops += h * w * 128 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        # conv after body
        # flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += h * w * self.embed_dim * self.embed_dim * 9
        if self.upsampler == 'pixelshuffle':
            # conv before upsample
            flops += h * w * self.embed_dim * 64 * 9
            flops += self.upsample.flops(resolution)
            # conv last
            flops += (h * self.upscale) * (w * self.upscale) * 64 * 1 * 9
            # flood head
            flops += (h * self.upscale) * (w * self.upscale) * 64 * 1 * 1
        else:
            flops += self.upsample.flops(resolution)
            # flood head
            flops += (h * self.upscale) * (w * self.upscale) * 1 * 1 * 1

        return flops


if __name__ == '__main__':
    upscale = 2
    model = FloodMapPFTV5(
        upscale=2,
        flood_map_size=64,
        embed_dim=240,
        depths=[4, 4, 4, 6, 6, 6],
        num_heads=6,
        num_topk=[1024, 1024, 1024, 1024,
                      256, 256, 256, 256,
                      128, 128, 128, 128,
                      64, 64, 64, 64, 64, 64,
                      32, 32, 32, 32, 32, 32,
                      16, 16, 16, 16, 16, 16,],

        window_size=32,
        convffn_kernel_size=7,
        mlp_ratio=2,
        upsampler='pixelshuffle',
    )

    # Parameters and computational complexity
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    print(640, 360, model.flops([640, 360]) / 1e9, 'G')
    print(426, 240, model.flops([426, 240]) / 1e9, 'G')
    print(320, 180, model.flops([320, 180]) / 1e9, 'G')

    # # Test
    # _input = torch.randn([1, 3, 640, 360])
    # output = model(_input)[0]
    # print(output.shape)