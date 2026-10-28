"""Log-area sample weights for supervised detection losses.

Inspired by Sec. 4.1, Eqs. (1) and (4), of "On the Importance of Large
Objects in CNN Based Object Detection Algorithms", arXiv:2311.11714.
"""

import math

import torch


class AreaLossWeight:
    """Reweight positives by GT pixel area, retaining TAL's quality weights.

    With quality q and raw weight r = max(log(area), 1) ** power, return
    w = r * sum(q) / sum(q * r). Then sum(q * w * loss) / sum(q) is the
    weighted mean sum(q * r * loss) / sum(q * r). The logarithm floor keeps
    subpixel or degenerate boxes finite. Background anchors retain weight 1.

    Use augmented, matched GT boxes in input pixels, BEFORE stride scaling.
    Classification applies w only to the positive class, preserving negative
    terms and soft-label values. No prediction or assignment is modified.
    """

    def __init__(self, enabled=False, power=1.0, apply_to=('class', 'iou', 'dfl')):
        if not isinstance(enabled, bool):
            raise ValueError('area_weight.enabled must be a bool')
        if isinstance(power, bool) or not math.isfinite(power) or power < 0:
            raise ValueError('area_weight.power must be finite and non-negative')
        if not isinstance(apply_to, (tuple, list)) or any(
                name not in ('class', 'iou', 'dfl') for name in apply_to):
            raise ValueError('area_weight.apply_to must contain class, iou and/or dfl')
        self.enabled = enabled
        self.power = float(power)
        self.apply_to = tuple(dict.fromkeys(apply_to))

    @torch.no_grad()
    def __call__(self, target_bboxes, target_scores, fg_mask):
        if not self.enabled or not self.apply_to:
            return {}

        # FP32 prevents pixel-area overflow with AMP (e.g. 416 * 416 > 65504).
        wh = (target_bboxes.detach().float()[..., 2:] -
              target_bboxes.detach().float()[..., :2]).clamp_min(0)
        raw_weight = wh.prod(-1).clamp_min(1).log().clamp_min(1).pow(self.power)
        foreground = fg_mask.bool()
        quality = target_scores.detach().float().sum(-1) * foreground
        quality_sum = quality.sum()
        weighted_sum = (quality * raw_weight).sum()
        weights = raw_weight * quality_sum / weighted_sum.clamp_min(
            torch.finfo(raw_weight.dtype).tiny)
        # Empty/zero-quality batches must retain the original background loss.
        weights = torch.where(foreground & (weighted_sum > 0), weights,
                              torch.ones_like(weights)).unsqueeze(-1)
        return {name: weights for name in self.apply_to}


def apply_bbox_area_weight(bbox_weight, area_weights, fg_mask, loss_type):
    """Apply optional [B,N,1] area factors to selected [P,1] box weights."""
    if not area_weights or loss_type not in area_weights:
        return bbox_weight
    return bbox_weight * area_weights[loss_type][fg_mask.bool()]
