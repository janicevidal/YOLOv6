"""FlashDet LiteBackbone port, based on ShuffleNet V2 (Ma et al., ECCV 2018).

Source: FlashDet/flashdet/models/backbone/lite_backbone.py.
The 0.5x variant outputs 48/96/192 channels at strides 8/16/32.
"""

# MIT License
# Copyright (c) 2024-2026 FlashVision
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

MODEL_URLS = {
    "0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "1.5x": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
    "2.0x": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
}


def channel_mix(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels between groups without changing spatial dimensions."""
    batch, channels, height, width = x.size()
    x = x.view(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().view(batch, channels, height, width)


def _make_act(name: str) -> nn.Module:
    if name == "LeakyReLU":
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)


class LiteUnit(nn.Module):
    """Channel-split depthwise block with two branches for stride-2 reduction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 activation: str = "LeakyReLU"):
        super().__init__()
        if stride not in (1, 2) or out_channels <= 0 or out_channels % 2:
            raise ValueError('LiteUnit requires stride 1/2 and positive even output channels')
        if stride == 1 and in_channels != out_channels:
            raise ValueError('Stride-1 LiteUnit must preserve channels')
        self.stride = stride
        branch_channels = out_channels // 2
        self.branch1 = nn.Sequential()
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                _make_act(activation),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_channels,
                      branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            _make_act(activation),
            nn.Conv2d(branch_channels, branch_channels, 3, stride, 1,
                      groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            _make_act(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_mix(out, 2)


class LiteBackbone(nn.Module):
    """FlashDet backbone with optional ImageNet weights and no classifier.

    Stage numbers 2/3/4 correspond to strides 8/16/32. Pretraining defaults
    to False so a backbone comparison starts both models from random weights.
    Enabling it downloads torchvision ShuffleNet V2 weights on first use.
    """

    STAGE_CONFIGS = {
        "0.5x": ([4, 8, 4], [24, 48, 96, 192, 1024]),
        "1.0x": ([4, 8, 4], [24, 116, 232, 464, 1024]),
        "1.5x": ([4, 8, 4], [24, 176, 352, 704, 1024]),
        "2.0x": ([4, 8, 4], [24, 244, 488, 976, 2048]),
    }

    def __init__(self, model_size: str = "0.5x", out_stages: tuple = (2, 3, 4),
                 pretrained: bool = False, activation: str = "LeakyReLU",
                 in_channels: int = 3):
        super().__init__()
        if model_size not in self.STAGE_CONFIGS:
            raise ValueError(f'Unsupported LiteBackbone model_size: {model_size}')
        if not out_stages or tuple(sorted(set(out_stages))) != tuple(out_stages):
            raise ValueError('out_stages must contain unique stage indices in ascending order')
        if any(stage not in (2, 3, 4) for stage in out_stages):
            raise ValueError('LiteBackbone out_stages must be selected from (2, 3, 4)')
        if pretrained and in_channels != 3:
            raise ValueError('ImageNet pretrained LiteBackbone requires three input channels')
        self.out_stages = tuple(out_stages)
        repeats, channels = self.STAGE_CONFIGS[model_size]
        self.out_channels = [channels[stage - 1] for stage in out_stages]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            _make_act(activation),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        in_ch = channels[0]
        self.stage_names = []
        for i, (repeat, out_ch) in enumerate(zip(repeats, channels[1:-1])):
            stage = []
            for j in range(repeat):
                stage.append(LiteUnit(in_ch, out_ch, 2 if j == 0 else 1, activation))
                in_ch = out_ch
            name = f'stage{i + 2}'
            setattr(self, name, nn.Sequential(*stage))
            self.stage_names.append(name)
        self.pretrained_loaded = False
        if pretrained:
            self._load_pretrained(model_size)

    def _load_pretrained(self, model_size: str):
        # Keep torchvision-compatible keys. Its conv5 and classifier are unused.
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URLS[model_size], map_location='cpu', progress=True, check_hash=True,
        )
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f'Missing LiteBackbone pretrained keys: {incompatible.missing_keys}')
        self.pretrained_loaded = True
        logger.info('Loaded ImageNet pretrained LiteBackbone %s', model_size)

    def forward(self, x: torch.Tensor) -> list:
        x = self.maxpool(self.conv1(x))
        outputs = []
        for stage, name in enumerate(self.stage_names, start=2):
            x = getattr(self, name)(x)
            if stage in self.out_stages:
                outputs.append(x)
        return outputs
