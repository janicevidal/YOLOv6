"""Common convolution modules with configurable activation.

Shared across neck, head, and other components.
"""

import torch.nn as nn


class ConvModule(nn.Module):
    """Standard convolution with BN and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        bias: bool = False,
        activation: str = "LeakyReLU"
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution with BN and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        activation: str = "LeakyReLU",
        **kwargs
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "ReLU":
            self.act = nn.ReLU(inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x
