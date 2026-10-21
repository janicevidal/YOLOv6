"""Pico backbone for YOLOv6.

Lightweight backbone with multi-scale depthwise convolutions and
structural reparameterization:
  - Multi-scale depthwise convolution: 4-way channel split with identity, 3x3,
    5x5, 7x7 depthwise convolutions for diverse spatial feature extraction.
  - Stride-2 downsampling: two parallel DW paths (3x3 || 5x5) concat
    to double channels at each stride-2 transition.
  - Structural reparameterization: multi-branch training paths fuse into
    single depthwise convolutions at inference (zero extra cost).

Independent implementation based on:
    Mao et al., "RepNeXt: A Fast Multi-Scale CNN using Structural
    Reparameterization", arXiv 2024.

Drop-in replacement for LiteBackbone-0.5x:
  Output channels [48, 96, 192] at strides [8, 16, 32].
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from yolov6.layers.reparam import PicoBlock, StrideDown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class PicoBackbone(nn.Module):
    """Pico backbone — ultra-lightweight drop-in replacement for LiteBackbone-0.5x.

    +----------+
    |  Stem    |  Conv3x3 stride-2 -> BN -> Act
    | MaxPool  |  stride-2
    +----------+  stride 4, 24 ch
    | Stage 0  |  StrideDown(24->48) + 2x PicoBlock(48)
    +----------+  stride 8, 48 ch
    | Stage 1  |  StrideDown(48->96) + 3x PicoBlock(96)
    +----------+  stride 16, 96 ch
    | Stage 2  |  StrideDown(96->192) + 1x PicoBlock(192)
    +----------+  stride 32, 192 ch

    Outputs [48, 96, 192] at strides [8, 16, 32] — identical interface
    to LiteBackbone(model_size="0.5x", out_stages=(2, 3, 4)).

    Parameter budget: ~135K (comparable to LiteBackbone-0.5x backbone).
    """

    def __init__(
        self,
        stem_channels: int = 24,
        stage_channels: Tuple[int, ...] = (48, 96, 192),
        stage_depths: Tuple[int, ...] = (2, 3, 1),
        out_stages: Tuple[int, ...] = (0, 1, 2),
        activation: str = "LeakyReLU",
        pretrained: bool = False,
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        if len(stage_channels) != len(stage_depths):
            raise ValueError("stage_channels and stage_depths must have equal lengths")
        previous_channels = stem_channels
        for ch, depth in zip(stage_channels, stage_depths):
            if ch != previous_channels * 2 or ch % 4:
                raise ValueError("Each Pico stage must double channels and be divisible by 4")
            if depth < 0:
                raise ValueError("Pico stage depths must be non-negative")
            previous_channels = ch
        if not out_stages or tuple(sorted(set(out_stages))) != tuple(out_stages):
            raise ValueError("out_stages must contain unique stage indices in ascending order")
        if min(out_stages) < 0 or max(out_stages) >= len(stage_channels):
            raise ValueError("out_stages contains an invalid Pico stage index")
        self.out_stages = out_stages
        self.out_channels = [stage_channels[s] for s in out_stages]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        in_ch = stem_channels
        self.stages = nn.ModuleList()
        for ch, depth in zip(stage_channels, stage_depths):
            layers: List[nn.Module] = [StrideDown(in_ch)]
            for _ in range(depth):
                layers.append(PicoBlock(ch))
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch

        self._init_weights()
        self.pretrained_loaded = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.pool(self.stem(x))
        outputs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                outputs.append(x)
        return outputs

    @torch.no_grad()
    def switch_to_deploy(self):
        """Convert PicoBlocks in place after loading weights and calling eval()."""
        if self.training:
            raise RuntimeError('Call eval() before PicoBackbone.switch_to_deploy()')
        for module in list(self.modules()):
            if isinstance(module, PicoBlock):
                module.switch_to_deploy()
        logger.info("PicoBackbone: fused small branches, 1x7/7x1 to 7x7, and PW+BN")
        return self

    def fuse(self):
        """Backward-compatible alias for switch_to_deploy()."""
        return self.switch_to_deploy()
