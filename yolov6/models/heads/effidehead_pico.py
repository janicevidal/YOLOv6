"""Depthwise-separable Pico head and YOLOv6 training/inference adapter."""

import math
from typing import Tuple

import torch
import torch.nn as nn

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class E2EDetHead(nn.Module):
    """Predict class logits and direct LTRB offsets with gated residual branches."""

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)
        self.cls_gate = nn.Parameter(torch.zeros(1))
        self.reg_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_feat = self.cls_convs(x) + self.cls_gate * x
        reg_feat = self.reg_convs(x) + self.reg_gate * x
        return self.cls_pred(cls_feat), self.reg_pred(reg_feat)


class Detect(nn.Module):
    """Adapt three independent E2EDetHead blocks to YOLOv6's AF interface.

    Like the Lite head, training receives class probabilities and raw LTRB
    offsets in grid units. Inference returns pixel-space xywh boxes, and scores.
    Assignment and NMS remain the standard YOLOv6 pipeline.
    """

    export = False

    def __init__(self, num_classes=80, in_channels=(64, 64, 64), strides=(8, 16, 32)):
        super().__init__()
        if len(in_channels) != 3 or list(strides) != [8, 16, 32]:
            raise ValueError('PicoHead requires three levels with strides [8, 16, 32]')
        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = len(in_channels)
        self.stride = torch.tensor(strides, dtype=torch.float32)
        self.grid = [torch.zeros(1) for _ in in_channels]
        self.use_dfl = False
        self.reg_max = 0
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.prior_prob = 1e-2
        self.heads = nn.ModuleList(E2EDetHead(num_classes, channels) for channels in in_channels)

    def initialize_biases(self):
        with torch.no_grad():
            for head in self.heads:
                head.cls_pred.weight.zero_()
                head.cls_pred.bias.fill_(math.log(self.prior_prob / (1 - self.prior_prob)))
                head.reg_pred.weight.zero_()
                head.reg_pred.bias.fill_(1.0)

    def forward(self, x):
        cls_scores, reg_distances = [], []
        for head, feature in zip(self.heads, x):
            cls_logits, reg_output = head(feature)
            scores = cls_logits.sigmoid()
            cls_scores.append(scores)
            # Non-DFL regression follows Lite: pass raw offsets to dist2bbox.
            reg_distances.append(reg_output)

        if not self.training and self.export:
            return tuple(torch.cat((scores, distances), dim=1)
                         for scores, distances in zip(cls_scores, reg_distances))

        scores = torch.cat([value.flatten(2).permute(0, 2, 1) for value in cls_scores], dim=1)
        distances = torch.cat([value.flatten(2).permute(0, 2, 1) for value in reg_distances], dim=1)
        if self.training:
            return x, scores, distances

        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset,
            device=x[0].device, is_eval=True, mode='af')
        boxes = dist2bbox(distances, anchor_points, box_format='xywh') * stride_tensor
        return torch.cat((boxes, torch.ones_like(boxes[..., :1]), scores), dim=-1)
