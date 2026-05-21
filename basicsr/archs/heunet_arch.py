import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class HeResUnit(nn.Module):
    """
    Residual unit used in a He et al.-style U-Net.

    The original paper uses:
      BN -> ReLU -> Conv 3x3
      BN -> ReLU -> Conv 3x3
      shortcut 1x1 Conv

    Here we use padding=1 to keep the spatial size unchanged.
    This is more convenient for patch-based flood map downscaling.
    """

    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()

        if norm:
            self.body = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            )
        else:
            self.body = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            )

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


@ARCH_REGISTRY.register()
class HeUNetFloodSR(nn.Module):
    """
    He et al.-style U-Net baseline for flood map downscaling.

    Input:
        coarse_fm: [B, coarse_in_chans, Hc, Wc]
        static_f:  [B, static_in_chans, Hf, Wf]

    Output:
        depth:       [B, 1, Hf, Wf]
        flood_logit: [B, num_flood_classes, Hf, Wf]

    Notes:
        - The coarse flood map is first interpolated to fine resolution.
        - It is then concatenated with fine-grid static features.
        - final_relu should be False when the target is asinh + zscore.
    """

    def __init__(
        self,
        upscale=8,
        coarse_in_chans=1,
        static_in_chans=7,
        base_ch=64,
        out_chans=1,
        num_flood_classes=3,
        interp_mode='bicubic',
        final_relu=False,
        use_batchnorm=True,
        **kwargs
    ):
        super().__init__()

        self.upscale = upscale
        self.coarse_in_chans = coarse_in_chans
        self.static_in_chans = static_in_chans
        self.interp_mode = interp_mode
        self.final_relu = final_relu

        in_ch = coarse_in_chans + static_in_chans

        # Encoder
        self.enc1 = HeResUnit(in_ch, base_ch, norm=use_batchnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = HeResUnit(base_ch, base_ch * 2, norm=use_batchnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = HeResUnit(base_ch * 2, base_ch * 4, norm=use_batchnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = HeResUnit(base_ch * 4, base_ch * 8, norm=use_batchnorm)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = HeResUnit(base_ch * 8, base_ch * 4, norm=use_batchnorm)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = HeResUnit(base_ch * 4, base_ch * 2, norm=use_batchnorm)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = HeResUnit(base_ch * 2, base_ch, norm=use_batchnorm)

        # Output heads
        self.depth_head = nn.Conv2d(base_ch, out_chans, kernel_size=1, stride=1, padding=0)
        self.flood_head = nn.Conv2d(base_ch, num_flood_classes, kernel_size=1, stride=1, padding=0)

    def _interp_coarse(self, coarse_fm, size):
        if self.interp_mode in ['bilinear', 'bicubic']:
            return F.interpolate(
                coarse_fm,
                size=size,
                mode=self.interp_mode,
                align_corners=False
            )
        else:
            return F.interpolate(
                coarse_fm,
                size=size,
                mode=self.interp_mode
            )

    def _match_size(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def _check_same_size(self, x, ref, stage=''):
        if x.shape[-2:] != ref.shape[-2:]:
            raise RuntimeError(
                f'[ERROR] HeUNetFloodSR skip connection size mismatch{stage}: '
                f'x spatial size={tuple(x.shape[-2:])}, '
                f'ref spatial size={tuple(ref.shape[-2:])}. '
                f'Please check fine patch size, number of pooling layers, '
                f'and input scale consistency.'
            )
        return x

    def forward(self, coarse_fm, static_f):
        B, _, Hc, Wc = coarse_fm.shape
        _, _, Hf, Wf = static_f.shape

        if (Hf != Hc * self.upscale) or (Wf != Wc * self.upscale):
            raise RuntimeError(
                f'[ERROR] fine/coarse size mismatch: '
                f'Hc={Hc}, Wc={Wc}, Hf={Hf}, Wf={Wf}, scale={self.upscale}'
            )

        coarse_up = self._interp_coarse(coarse_fm, size=(Hf, Wf))
        x = torch.cat([coarse_up, static_f], dim=1)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        # d3 = self._match_size(d3, e3)
        d3 = self._check_same_size(d3, e3, stage=' at d3/e3')
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        # d2 = self._match_size(d2, e2)
        d2 = self._check_same_size(d2, e2, stage=' at d2/e2')
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        # d1 = self._match_size(d1, e1)
        d1 = self._check_same_size(d1, e1, stage=' at d1/e1')
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        depth = self.depth_head(d1)

        if self.final_relu:
            depth = F.relu(depth)

        flood_logit = self.flood_head(d1)

        return depth, flood_logit