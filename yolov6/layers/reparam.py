"""Reparameterizable building blocks for the YOLOv6 Pico backbone.

Multi-scale depthwise convolutions and stride-2 downsample modules
with structural reparameterization — multi-branch training collapses
to single convolutions at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class FusedDWConv(nn.Module):
    """Reparameterizable small depthwise convolution.

    Training:  3x3 + (1x3) + (3x1) + dilated-2x2  (4 parallel branches)
    Inference: single 3x3 DW conv  (via structural reparameterization)
    """

    def __init__(self, channels: int):
        super().__init__()
        kw = dict(in_channels=channels, out_channels=channels,
                  groups=channels, bias=False)
        self.conv_3x3 = nn.Conv2d(kernel_size=3, padding=1, **kw)
        self.conv_1x3 = nn.Conv2d(kernel_size=(1, 3), padding=(0, 1), **kw)
        self.conv_3x1 = nn.Conv2d(kernel_size=(3, 1), padding=(1, 0), **kw)
        self.conv_d22 = nn.Conv2d(kernel_size=2, dilation=2, padding=1, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_1x3"):
            return (self.conv_3x3(x) + self.conv_1x3(x)
                    + self.conv_3x1(x) + self.conv_d22(x))
        return self.conv_3x3(x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        if not hasattr(self, "conv_1x3"):
            return self.conv_3x3
        w = self.conv_3x3.weight.clone()
        w += F.pad(self.conv_1x3.weight, [0, 0, 1, 1])
        w += F.pad(self.conv_3x1.weight, [1, 1, 0, 0])
        w[..., ::2, ::2] += self.conv_d22.weight
        self.conv_3x3.weight.data.copy_(w)
        del self.conv_1x3, self.conv_3x1, self.conv_d22
        return self.conv_3x3


class MultiScaleConv(nn.Module):
    """Multi-scale depthwise convolution via channel chunking.

    Splits input channels into 4 equal groups and applies:
      0) Identity  -- free skip, gradient highway
      1) Fused DW 3x3 -- local features (fuses to single 3x3)
      2) DW 5x5    -- medium receptive field
      3) DW 7x7    -- large receptive field via (1x7)*(7x1) decomposition

    At inference, the small parallel branches fuse into one DW convolution;
    the sequential 1x7 -> 7x1 branch fuses into one 7x7 DW convolution.
    """

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, f"channels ({channels}) must be divisible by 4"
        c = channels // 4
        self._c = c
        self.dw_s = FusedDWConv(c)
        self.dw_m = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw_l_h = nn.Conv2d(c, c, (1, 7), 1, (0, 3), groups=c, bias=False)
        self.dw_l_v = nn.Conv2d(c, c, (7, 1), 1, (3, 0), groups=c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i, s, m, l = x.chunk(4, dim=1)
        if hasattr(self, "dw_l_h"):
            l_out = self.dw_l_v(self.dw_l_h(l))
        else:
            l_out = self.dw_l(l)
        return torch.cat([i, self.dw_s(s), self.dw_m(m), l_out], dim=1)

    @torch.no_grad()
    def fuse(self):
        """Fuse small branches to 3x3 and the separable large path to 7x7."""
        if isinstance(self.dw_s, FusedDWConv):
            self.dw_s = self.dw_s.fuse()
        if hasattr(self, 'dw_l_h'):
            horizontal = self.dw_l_h.weight
            vertical = self.dw_l_v.weight
            self.dw_l = nn.Conv2d(
                self._c, self._c, 7, 1, 3, groups=self._c, bias=False,
            ).to(device=horizontal.device, dtype=horizontal.dtype)
            # Depthwise kernels compose independently per channel.
            self.dw_l.weight.copy_(vertical * horizontal)
            self.dw_l.weight.requires_grad_(horizontal.requires_grad or vertical.requires_grad)
            self.dw_l.train(self.training)
            del self.dw_l_h, self.dw_l_v
        return self


class PicoBlock(nn.Module):
    """Residual block with multi-scale depthwise convolution.

    Architecture:  x -> MultiScaleConv -> BN -> PW(1x1) -> BN -> Act -> (+x)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.chunk_conv = MultiScaleConv(dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn2(self.pw(self.bn1(self.chunk_conv(x)))))

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse depthwise paths and PW+BN, preserving the residual path."""
        if getattr(self, 'deploy', False):
            # Upgrade older deployed blocks that retained the factorized path.
            self.chunk_conv.fuse()
            return self
        if self.training:
            raise RuntimeError('Call eval() before PicoBlock.switch_to_deploy()')
        self.chunk_conv.fuse()
        self.pw = fuse_conv_bn_eval(self.pw, self.bn2)
        self.bn2 = nn.Identity()
        self.deploy = True
        return self


class StrideDown(nn.Module):
    """Stride-2 downsample that doubles channels via parallel DW paths.

    Two depthwise convolutions with different kernel sizes (3x3 || 5x5)
    are concatenated then projected, capturing multi-scale information
    during the spatial reduction step.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        out = in_channels * 2
        self.dw_s = nn.Conv2d(
            in_channels, in_channels, 3, 2, 1,
            groups=in_channels, bias=False,
        )
        self.dw_l = nn.Conv2d(
            in_channels, in_channels, 5, 2, 2,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out)
        self.pw = nn.Conv2d(out, out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(torch.cat([self.dw_s(x), self.dw_l(x)], dim=1)))
        return x + self.act(self.bn2(self.pw(x)))
