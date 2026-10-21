#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn


class WIoU_Scale(nn.Module):
    """Wise-IoU focusing gain with an independent running mean of 1 - IoU.

    monotonous: None for v1, True for monotonic v2, False for non-monotonic v3.
    """
    def __init__(self, monotonous=False, momentum=1 - 0.5 ** (1 / 7000),
                 gamma=1.9, delta=3.0, eps=1e-7):
        super().__init__()
        if monotonous is not None and not isinstance(monotonous, bool):
            raise ValueError('WIoU monotonous must be None, True or False')
        if not 0 < momentum <= 1:
            raise ValueError('WIoU momentum must satisfy 0 < momentum <= 1')
        if not math.isfinite(gamma) or gamma <= 1:
            raise ValueError('WIoU gamma must be finite and greater than 1')
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError('WIoU delta must be finite and positive')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('WIoU eps must be finite and positive')
        self.monotonous = monotonous
        self.momentum = momentum
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, iou_loss):
        # No Python .item() here: keep the mean update on the loss device.
        if self.training and iou_loss.numel():
            with torch.no_grad():
                mean = iou_loss.detach().mean().to(self.iou_mean)
                self.iou_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        return self._scaled_loss(iou_loss)

    def _scaled_loss(self, iou_loss):
        if self.monotonous is None:
            return torch.ones_like(iou_loss)
        mean = self.iou_mean.to(iou_loss).clamp_min(self.eps)
        beta = iou_loss.detach() / mean
        if self.monotonous:
            return beta.sqrt()
        alpha = self.delta * torch.pow(self.gamma, beta - self.delta)
        return beta / alpha


class IOUloss(nn.Module):
    """ Calculate IoU loss.
    """
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7,
                 fociou_m=0.0, fociou_n=0.95, fociou_beta=2.0,
                 wiou_monotonous=False, wiou_momentum=1 - 0.5 ** (1 / 7000),
                 wiou_gamma=1.9, wiou_delta=3.0):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), 'ciou', 'diou', 'giou', 'siou', 'fociou' or 'wiou'.
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
            fociou_m, fociou_n: (float), FoCIoU mapping bounds, 0 <= m < n <= 1.
                Defaults follow the supplied bbox_iou implementation; the YOLO-ULM
                paper does not specify experimental values for these bounds.
            fociou_beta: (float), finite exponent >= 1 for finite boundary gradients.
                The YOLO-ULM paper uses 2.
            wiou_monotonous: None for WIoU v1, True for v2, False for v3 (default).
            wiou_momentum: (float), update weight for the running mean of 1 - IoU.
            wiou_gamma, wiou_delta: (float), non-monotonic focusing parameters.
        """
        super().__init__()
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps
        self.fociou_m = fociou_m
        self.fociou_n = fociou_n
        self.fociou_beta = fociou_beta
        if self.iou_type == 'fociou':
            if not 0 <= fociou_m < fociou_n <= 1:
                raise ValueError('FoCIoU requires 0 <= fociou_m < fociou_n <= 1')
            if not math.isfinite(fociou_beta) or fociou_beta < 1:
                raise ValueError('FoCIoU requires finite fociou_beta >= 1')
        if self.iou_type == 'wiou':
            self.wiou_scale = WIoU_Scale(wiou_monotonous, wiou_momentum,
                                         wiou_gamma, wiou_delta, eps)

    def forward(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        if self.iou_type in ['fociou', 'wiou']:
            # Keep eps and box areas representable when called with AMP outputs.
            if box1.dtype in (torch.float16, torch.bfloat16):
                box1 = box1.float()
            if box2.dtype in (torch.float16, torch.bfloat16):
                box2 = box2.float()
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou', 'fociou', 'wiou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'wiou':
                # WIoU v1 distance attention; detach only the enclosing diagonal.
                valid = (b1_x2 > b1_x1) & (b1_y2 > b1_y1) & (b2_x2 > b2_x1) & (b2_y2 > b2_y1)
                # Zero-area boxes have IoU zero, without an unbounded overlap gradient.
                loss_iou = torch.where(valid, 1.0 - iou, torch.ones_like(iou))
                distance_weight = torch.exp(rho2 / c2.detach())
                loss = self.wiou_scale(loss_iou) * distance_weight * loss_iou
            elif self.iou_type == 'diou':
                iou = iou - rho2 / c2
            else:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    if self.iou_type == 'fociou':
                        # Add eps last: (1 + eps) can round to 1 in FP32.
                        alpha = v / (1 - iou + v + self.eps)
                    else:
                        alpha = v / (v - iou + (1 + self.eps))
                if self.iou_type == 'fociou':
                    # YOLO-ULM, Eqs. (7)-(8): 1 - CIoU + IoU - IoU_MF.
                    # Keep raw IoU for alpha; map only the overlap term.
                    iou = ((iou - self.fociou_m) / (self.fociou_n - self.fociou_m)).clamp(0, 1)
                    iou = iou.pow(self.fociou_beta)
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        if self.iou_type != 'wiou':
            loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def pairwise_bbox_iou(box1, box2, box_format='xywh'):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    if box_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)

    elif box_format == 'xywh':
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)
