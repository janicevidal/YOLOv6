"""
PicoNeck feature pyramid network for YOLOv6.

Uses nearest-neighbor upsampling by default, with optional DySample (ICCV 2023)
from FlashDownUP for learned point sampling in the top-down path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.layers.conv_module import ConvModule, DepthwiseConvModule


# ---------------------------------------------------------------------------
# DySample — Dynamic Point-Sampling Upsampler (ICCV 2023)
# From FlashVision/FlashDownUP: https://github.com/FlashVision/FlashDownUP
# Paper: "Learning to Upsample by Learning to Sample" (arXiv:2308.15085)
# ---------------------------------------------------------------------------

class DySample(nn.Module):
    """Content-aware upsampler that learns sampling offsets dynamically.

    Instead of fixed bilinear kernels, generates per-pixel sampling offsets
    via a lightweight 1x1 conv, then uses grid_sample for the actual
    resampling.  Only ~2.6K params for 80-ch input (vs 0 for bilinear),
    but significantly better feature quality in FPN paths.
    """

    def __init__(self, in_channels: int, scale: int = 2, groups: int = 4,
                 style: str = "lp"):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ("lp", "pl")
        if style == "pl":
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            offset_in = in_channels // (scale ** 2)
            out_ch = 2 * groups
        else:
            offset_in = in_channels
            out_ch = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(offset_in, out_ch, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2,
                         (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
            .transpose(1, 2).unsqueeze(1).unsqueeze(0).to(x.device)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords,
            mode="bilinear", align_corners=False, padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "pl":
            x_ = F.pixel_shuffle(x, self.scale)
            offset = (F.pixel_unshuffle(self.offset(x_), self.scale)
                      * 0.25 + self.init_pos)
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)


class LiteModule(nn.Module):
    """Lightweight module for efficient feature extraction.

    Generates additional feature maps via cheap linear operations
    on a subset of primary features, reducing computation while
    maintaining representational capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        if activation is None:
            act = nn.Identity()
        elif activation == "LeakyReLU":
            act = nn.LeakyReLU(0.1, inplace=True)
        else:
            act = nn.ReLU(inplace=True)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            act,
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act,
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class LiteBlock(nn.Module):
    """Lightweight bottleneck block.

    Uses two LiteModule stages (expansion + projection) with an optional
    depthwise convolution for strided downsampling. Efficient alternative
    to standard bottleneck blocks for resource-constrained models.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.stride = stride

        self.ghost1 = LiteModule(in_channels, mid_channels, activation=activation)

        if stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels, mid_channels, dw_kernel_size,
                stride=stride, padding=(dw_kernel_size - 1) // 2,
                groups=mid_channels, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        self.ghost2 = LiteModule(mid_channels, out_channels, activation=None)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, dw_kernel_size,
                    stride=stride, padding=(dw_kernel_size - 1) // 2,
                    groups=in_channels, bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        x = self.ghost2(x)

        x = x + self.shortcut(residual)
        return x


class LiteBlocks(nn.Module):
    """Stack of LiteBlock modules used in PicoNeck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: int = 1,
        kernel_size: int = 5,
        num_blocks: int = 1,
        use_res: bool = False,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        self.use_res = use_res

        if use_res:
            self.reduce_conv = ConvModule(
                in_channels, out_channels, 1, activation=activation
            )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                LiteBlock(
                    in_channels, int(out_channels * expand), out_channels,
                    dw_kernel_size=kernel_size, activation=activation
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class PicoNeck(nn.Module):
    """Lightweight Path Aggregation Network (PAN) for feature fusion.

    Supports nearest-neighbor interpolation and optional learned DySample.

    Args:
        in_channels: Input channel list from backbone.
        out_channels: Output channels for all levels.
        kernel_size: Convolution kernel size.
        num_extra_level: Number of extra output levels.
        use_depthwise: Whether to use depthwise separable conv.
        activation: Activation function name.
        upsample_type: "nearest" (default) or "dysample" for top-down fusion.
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
    ):
        super().__init__()
        if upsample_type not in ("nearest", "dysample"):
            raise ValueError('upsample_type must be "nearest" or "dysample"')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_type = upsample_type

        Conv = DepthwiseConvModule if use_depthwise else ConvModule

        # One upsampler per top-down transition.
        self.upsamplers = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.upsamplers.append(
                DySample(out_channels, scale=2, groups=4, style="lp")
                if upsample_type == "dysample"
                else nn.Upsample(scale_factor=2, mode="nearest")
            )

        # Reduce layers (lateral connections)
        self.reduce_layers = nn.ModuleList()
        for ch in in_channels:
            self.reduce_layers.append(
                ConvModule(ch, out_channels, 1, activation=activation)
            )

        # Top-down blocks
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.top_down_blocks.append(
                LiteBlocks(
                    out_channels * 2, out_channels, expand,
                    kernel_size=kernel_size, num_blocks=num_blocks,
                    use_res=use_res, activation=activation
                )
            )

        # Bottom-up path
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsamples.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )
            self.bottom_up_blocks.append(
                LiteBlocks(
                    out_channels * 2, out_channels, expand,
                    kernel_size=kernel_size, num_blocks=num_blocks,
                    use_res=use_res, activation=activation
                )
            )

        # Extra levels for additional feature scales
        self.num_extra_level = num_extra_level
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )
            self.extra_lvl_out_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2, activation=activation)
            )

    def forward(self, inputs: list) -> list:
        """Forward pass.

        Args:
            inputs: List of feature maps from backbone.

        Returns:
            List of fused feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # Reduce channels
        inputs = [reduce(x) for reduce, x in zip(self.reduce_layers, inputs)]

        # Top-down path.
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]

            up_idx = len(self.in_channels) - 1 - idx
            upsampler = self.upsamplers[up_idx]
            upsample_feat = upsampler(feat_high)
            if upsample_feat.shape[2:] != feat_low.shape[2:]:
                # Inspect the module to support older pickled necks that do
                # not have upsample_type. Keep nearest mode for odd sizes.
                if isinstance(upsampler, DySample):
                    upsample_feat = F.interpolate(upsample_feat, size=feat_low.shape[2:], mode="bilinear", align_corners=False)
                else:
                    upsample_feat = F.interpolate(feat_high, size=feat_low.shape[2:], mode="nearest")

            inner_out = self.top_down_blocks[up_idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsamples[idx](feat_low)

            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        # Extra levels: combine downsampled backbone feature and PAN output
        for i, (extra_in, extra_out) in enumerate(zip(self.extra_lvl_in_conv, self.extra_lvl_out_conv)):
            outs.append(extra_in(inputs[-1]) + extra_out(outs[-1]))

        return outs
