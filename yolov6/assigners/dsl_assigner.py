# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from MMYOLO 8c4d9dc503dc8e327bec8147e8dc97124052f693:
# mmyolo/models/task_modules/assigners/batch_dsl_assigner.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSoftLabelAssigner(nn.Module):
    """MMYOLO DSL with the same batch input/output contract as YOLOv6 TAL.

    Boxes and points are in pixels. Scores are sigmoid probabilities by
    default. Valid GTs precede padding, as in YOLOv6's loss preprocessing.
    Matching preserves MMYOLO's GT-class cost, all-point dynamic-k and sort.
    """

    def __init__(self, num_classes=80, soft_center_radius=3.0, topk=13,
                 iou_weight=3.0, batch_iou=True, scores_are_logits=False):
        super().__init__()
        for name, value in (('num_classes', num_classes), ('topk', topk)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('{} must be a positive integer'.format(name))
        if not math.isfinite(soft_center_radius) or soft_center_radius < 0:
            raise ValueError('soft_center_radius must be nonnegative and finite')
        if not math.isfinite(iou_weight) or iou_weight < 0:
            raise ValueError('iou_weight must be nonnegative and finite')
        if not isinstance(batch_iou, bool) or not isinstance(scores_are_logits, bool):
            raise ValueError('batch_iou and scores_are_logits must be bools')
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.batch_iou = batch_iou
        self.scores_are_logits = scores_are_logits
        self.eps = 1e-7

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride_tensor=None):
        """Return labels [B,N], boxes [B,N,4], scores [B,N,C], mask [B,N].

        Inputs follow TaskAlignedAssigner: scores [B,N,C], boxes [B,N,4],
        points [N,2], labels [B,G,1], GT boxes [B,G,4], flags [B,G,1].
        DSL additionally needs strides [N,1] or [N,2] for its center cost.
        """
        if stride_tensor is None or stride_tensor.shape not in (
                (anc_points.size(0), 1), (anc_points.size(0), 2)):
            raise ValueError('DSL requires stride_tensor with shape [N,1] or [N,2]')
        with torch.cuda.amp.autocast(enabled=False):
            scores = pd_scores.float()
            if not self.scores_are_logits:
                scores = torch.logit(scores, eps=self.eps)
            boxes, points = pd_bboxes.float(), anc_points.float()
            gt_boxes, strides = gt_bboxes.float(), stride_tensor.float()
            # YOLOv6 pads labels with -1. MMYOLO's dense class gather needs
            # legal padded indices; use its zero-padding representation.
            labels = torch.where(mask_gt.bool(), gt_labels, torch.zeros_like(gt_labels))
            if gt_boxes.size(1) == 0 or boxes.size(1) == 0:
                return (torch.full(scores.shape[:2], self.bg_idx, dtype=torch.long, device=scores.device),
                        torch.zeros_like(boxes), torch.zeros_like(scores),
                        torch.zeros(scores.shape[:2], dtype=torch.bool, device=scores.device))

            cost, overlaps = self.get_box_metrics(scores, boxes, points, labels, gt_boxes, mask_gt, strides)
            matched_ious, matched_gt, fg_mask = self.dynamic_k_matching(cost, overlaps, mask_gt)
            return self.get_targets(labels, gt_boxes, matched_gt, matched_ious, fg_mask)

    def get_box_metrics(self, logits, pd_bboxes, anc_points, gt_labels,
                        gt_bboxes, mask_gt, strides):
        """Build [B,N,G] costs without changing MMYOLO candidate semantics."""
        lt = anc_points[:, None, None] - gt_bboxes[..., :2]
        rb = gt_bboxes[..., 2:] - anc_points[:, None, None]
        is_in_gts = torch.cat((lt, rb), dim=-1).min(-1).values > 0
        is_in_gts = (is_in_gts * mask_gt[..., 0][None]).permute(1, 0, 2)
        valid_mask = is_in_gts.sum(-1) > 0
        centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0
        distance = (anc_points[None, :, None] - centers[:, None]).pow(2).sum(-1).sqrt()
        distance = distance / strides[None, :, 0, None]
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        if self.batch_iou:
            overlaps = self.pairwise_iou(pd_bboxes, gt_bboxes)
        else:
            overlaps = torch.stack([
                self.pairwise_iou(box, gt) for box, gt in zip(pd_bboxes, gt_bboxes)], dim=0)
        iou_cost = -torch.log(overlaps + self.eps) * self.iou_weight

        # Retain upstream indexing and GT-class-only classification cost.
        batch_size, num_gt = gt_labels.shape[:2]
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_scores = logits.permute(0, 2, 1)[idx[0], idx[1]].permute(0, 2, 1)
        scale_factor = overlaps - pairwise_scores.sigmoid()
        cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_scores, overlaps, reduction='none') * scale_factor.abs().pow(2.0)
        cost = cls_cost + iou_cost + soft_center_prior
        max_pad_value = torch.ones_like(cost) * 100000000
        cost = torch.where(valid_mask[..., None].repeat(1, 1, num_gt), cost, max_pad_value)
        return cost, overlaps

    @staticmethod
    def pairwise_iou(boxes, gt_boxes):
        """Upstream horizontal IoU; YOLOv6 TAL uses a different union epsilon."""
        area = (boxes[..., 2:] - boxes[..., :2]).prod(-1)
        gt_area = (gt_boxes[..., 2:] - gt_boxes[..., :2]).prod(-1)
        lt = torch.maximum(boxes[..., :, None, :2], gt_boxes[..., None, :, :2])
        rb = torch.minimum(boxes[..., :, None, 2:], gt_boxes[..., None, :, 2:])
        overlap = (rb - lt).clamp(min=0).prod(-1)
        union = area[..., None] + gt_area[..., None, :] - overlap
        return overlap / union.clamp(min=1e-6)

    def dynamic_k_matching(self, cost, overlaps, mask_gt):
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_topk = min(self.topk, overlaps.size(1))
        topk_ious, _ = torch.topk(overlaps, candidate_topk, dim=1)
        # Keep ALL-point IoUs, including those outside every GT.
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        num_gts = mask_gt.sum((1, 2)).int()
        _, sorted_indices = torch.sort(cost, dim=1)
        for batch_idx in range(mask_gt.size(0)):
            for gt_idx in range(num_gts[batch_idx]):
                pos_idx = sorted_indices[batch_idx, :dynamic_ks[batch_idx, gt_idx], gt_idx]
                matching[batch_idx, :, gt_idx][pos_idx] = 1

        conflicts = matching.sum(2) > 1
        if conflicts.sum() > 0:
            _, min_cost_gt = torch.min(cost[conflicts, :], dim=1)
            matching[conflicts, :] *= 0
            matching[conflicts, min_cost_gt] = 1
        fg_mask = matching.sum(2) > 0
        matched_ious = (matching * overlaps).sum(2)[fg_mask]
        matched_gt = matching[fg_mask, :].argmax(1)
        return matched_ious, matched_gt, fg_mask

    def get_targets(self, gt_labels, gt_bboxes, matched_gt, matched_ious, fg_mask):
        """Translate MMYOLO labels/boxes/metrics into the YOLOv6 four-tuple."""
        batch_idx = (fg_mask > 0).nonzero(as_tuple=True)[0]
        target_labels = gt_labels.new_full(fg_mask.shape, self.bg_idx, dtype=torch.long)
        target_labels[fg_mask] = gt_labels[batch_idx, matched_gt].squeeze(-1).long()
        target_bboxes = gt_bboxes.new_zeros((*fg_mask.shape, 4))
        target_bboxes[fg_mask] = gt_bboxes[batch_idx, matched_gt]
        target_scores = gt_bboxes.new_zeros((*fg_mask.shape, self.num_classes))
        target_scores[fg_mask] = F.one_hot(target_labels[fg_mask], self.num_classes) * matched_ious[:, None]
        # Zero-IoU assignments remain foreground, as in MMYOLO.
        return target_labels, target_bboxes, target_scores, fg_mask
