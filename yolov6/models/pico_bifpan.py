"""Pico PAN with lightweight three-input BiFusion blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.layers.conv_module import ConvModule, DepthwiseConvModule
from yolov6.models.pico_neck import DySample, LiteBlocks, SNI


class PicoBiFusion(nn.Module):
    """Three-scale fusion with a dense 3C-to-C projection.

    Upsampling uses PicoNeck's nearest/SNI/DySample choices. Downsample the
    fine input with a depthwise 3x3 before its 1x1 projection, saving weights
    and avoiding the high-resolution pointwise convolution. All convolutions
    use ConvModule with the neck's activation configuration.
    """

    def __init__(self, in_channels, out_channels, upsample_type="nearest",
                 activation="LeakyReLU"):
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError('PicoBiFusion requires lateral and fine input channels')
        if upsample_type not in ("nearest", "sni", "dysample"):
            raise ValueError('upsample_type must be "nearest", "sni" or "dysample"')
        if activation not in ("LeakyReLU", "ReLU", "SiLU", None):
            raise ValueError('activation must be "LeakyReLU", "ReLU", "SiLU" or None')

        self.cv1 = ConvModule(in_channels[0], out_channels, 1, activation=activation)
        self.cv2 = ConvModule(in_channels[1], out_channels, 1, activation=activation)
        self.cv3 = ConvModule(out_channels * 3, out_channels, 1, activation=activation)
        self.downsample = ConvModule(
            in_channels[1], in_channels[1], 3, stride=2,
            groups=in_channels[1], activation=activation)
        if upsample_type == "dysample":
            self.upsample = DySample(out_channels, scale=2, groups=4, style="lp")
        elif upsample_type == "sni":
            self.upsample = SNI(out_channels, out_channels, up_f=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs):
        coarse, lateral, fine = inputs
        up = self.upsample(coarse)
        lateral = self.cv1(lateral)
        fine = self.cv2(self.downsample(fine))
        if up.shape[2:] != lateral.shape[2:]:
            if isinstance(self.upsample, DySample):
                up = F.interpolate(up, size=lateral.shape[2:], mode="bilinear",
                                   align_corners=False)
            else:
                up = F.interpolate(coarse, size=lateral.shape[2:], mode="nearest")
                if isinstance(self.upsample, SNI):
                    up = self.upsample.alpha * up
        if fine.shape[2:] != lateral.shape[2:]:
            fine = F.interpolate(fine, size=lateral.shape[2:], mode="nearest")
        return self.cv3(torch.cat([up, lateral, fine], dim=1))


class PicoRefine(nn.Module):
    """Residual depthwise/pointwise refinement"""

    def __init__(self, channels, kernel_size, activation="LeakyReLU"):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('Refine kernel_size must be a positive odd integer')
        self.dw = ConvModule(channels, channels, kernel_size, groups=channels,
                             activation=activation)
        self.pw = ConvModule(channels, channels, 1, activation=None)

    def forward(self, x):
        return x + self.pw(self.dw(x))


class PicoBiFPANNeck(nn.Module):
    """Fuse backbone P2/P3/P4/P5 and return detection levels P3/P4/P5.

    P4 receives [reduced P5, backbone P4, backbone P3], then P3 receives
    [top-down P4, backbone P3, backbone P2]. Each PicoBiFusion uses configurable
    upsampling, depthwise fine-input downsampling, lateral/fine 1x1 projections
    and a dense 3C-to-C projection. Pico LiteBlocks refine
    these C-channel outputs; the bottom-up path follows PicoNeck.

    upsample_type accepts the same nearest/SNI/DySample modes as PicoNeck.
    activation configures both BiFusion and the other Pico modules;
    use_depthwise configures PAN downsampling.
    use_refine adds residual depthwise/pointwise blocks after the top-down
    LiteBlocks, before subsequent fusion/PAN. refine_kernel_sizes is ordered
    as (P3, P4); refinement is disabled by default for existing configurations.
    """

    def __init__(
        self,
        in_channels: list,
        out_channels: int = 64,
        kernel_size: int = 5,
        expand: int = 1,
        num_blocks: int = 1,
        use_res: bool = False,
        num_extra_level: int = 0,
        use_depthwise: bool = True,
        activation: str = "LeakyReLU",
        upsample_type: str = "nearest",
        use_refine: bool = False,
        refine_kernel_sizes: tuple = (5, 3),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError('PicoBiFPANNeck requires P2/P3/P4/P5 input channels')
        if num_extra_level != 0:
            raise ValueError('PicoBiFPANNeck outputs only P3/P4/P5')

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.upsample_type = upsample_type
        self.reduce_layer = ConvModule(in_channels[-1], out_channels, 1,
                                       activation=activation)
        self.bifusions = nn.ModuleList([
            PicoBiFusion([in_channels[2], in_channels[1]], out_channels, upsample_type,
                         activation=activation),
            PicoBiFusion([in_channels[1], in_channels[0]], out_channels, upsample_type,
                         activation=activation),
        ])
        self.top_down_blocks = nn.ModuleList([
            LiteBlocks(out_channels, out_channels, expand,
                       kernel_size=kernel_size, num_blocks=num_blocks,
                       use_res=use_res, activation=activation)
            for _ in range(2)
        ])

        Conv = DepthwiseConvModule if use_depthwise else ConvModule
        self.downsamples = nn.ModuleList([
            Conv(out_channels, out_channels, kernel_size, stride=2,
                 activation=activation)
            for _ in range(2)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            LiteBlocks(out_channels * 2, out_channels, expand,
                       kernel_size=kernel_size, num_blocks=num_blocks,
                       use_res=use_res, activation=activation)
            for _ in range(2)
        ])
        if use_refine:
            if len(refine_kernel_sizes) != 2:
                raise ValueError('refine_kernel_sizes must contain (P3, P4) kernel sizes')
            self.p3_refine = PicoRefine(out_channels, refine_kernel_sizes[0], activation)
            self.p4_refine = PicoRefine(out_channels, refine_kernel_sizes[1], activation)
        else:
            self.p3_refine = nn.Identity()
            self.p4_refine = nn.Identity()

    def forward(self, inputs: list) -> list:
        if len(inputs) != 4:
            raise ValueError('PicoBiFPANNeck expects [P2, P3, P4, P5]')
        p2, p3, p4, p5 = inputs
        p5 = self.reduce_layer(p5)
        p4 = self.top_down_blocks[0](self.bifusions[0]([p5, p4, p3]))
        if hasattr(self, 'p4_refine'):
            p4 = self.p4_refine(p4)
        p3 = self.top_down_blocks[1](self.bifusions[1]([p4, p3, p2]))
        if hasattr(self, 'p3_refine'):
            p3 = self.p3_refine(p3)

        n4 = self.bottom_up_blocks[0](torch.cat([self.downsamples[0](p3), p4], dim=1))
        n5 = self.bottom_up_blocks[1](torch.cat([self.downsamples[1](n4), p5], dim=1))
        return [p3, n4, n5]
