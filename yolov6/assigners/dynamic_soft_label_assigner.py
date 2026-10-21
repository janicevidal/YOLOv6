# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from MMDetection's DynamicSoftLabelAssigner (horizontal box path).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSoftLabelAssigner(nn.Module):
    """RTMDet assignment with YOLOv6's batched target interface.

    Scores are probabilities by default; native MMDetection logits can be used
    with ``scores_are_logits=True``. Boxes and anchor points are in pixels.
    ``stride_tensor`` is [N, 1] or [N, 2]; the first stride normalizes distance,
    matching MMDetection. ``max_cost_elements`` limits estimated BCE elements
    per batch chunk, with one image as the minimum chunk.
    """

    def __init__(self, num_classes=80, soft_center_radius=3.0, topk=13,
                 iou_weight=3.0, eps=1e-7, scores_are_logits=False,
                 max_cost_elements=4000000):
        super().__init__()
        for name, value in (('num_classes', num_classes), ('topk', topk),
                            ('max_cost_elements', max_cost_elements)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('DynamicSoftLabelAssigner {} must be a positive integer'.format(name))
        if not math.isfinite(soft_center_radius) or soft_center_radius < 0:
            raise ValueError('soft_center_radius must be nonnegative and finite')
        if not math.isfinite(iou_weight) or iou_weight < 0:
            raise ValueError('iou_weight must be nonnegative and finite')
        if not math.isfinite(eps) or not 0 < eps < 0.5:
            raise ValueError('eps must be finite and between 0 and 0.5')
        if not isinstance(scores_are_logits, bool):
            raise ValueError('scores_are_logits must be a bool')
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.eps = eps
        self.scores_are_logits = scores_are_logits
        self.max_cost_elements = max_cost_elements

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride_tensor=None):
        """Return labels [B,N], boxes [B,N,4], scores [B,N,C], mask [B,N]."""
        if stride_tensor is None or stride_tensor.shape not in (
                (anc_points.size(0), 1), (anc_points.size(0), 2)):
            raise ValueError('DynamicSoftLabelAssigner requires stride_tensor with shape [N,1] or [N,2]')
        # Center powers and BCE must not be evaluated in FP16 under AMP.
        with torch.cuda.amp.autocast(enabled=False):
            logits = pd_scores.float()
            if not self.scores_are_logits:
                # YOLOv6 heads already apply sigmoid. Recover logits once,
                # clamping saturated probabilities to keep BCE finite.
                logits = torch.logit(logits, eps=self.eps)
            boxes, points = pd_bboxes.float(), anc_points.float()
            gt_boxes, strides = gt_bboxes.float(), stride_tensor.float()
            batch_size, num_anchors, _ = logits.shape
            target_labels = torch.full((batch_size, num_anchors), self.num_classes,
                                       dtype=torch.long, device=logits.device)
            target_boxes = torch.zeros_like(boxes)
            target_scores = torch.zeros_like(logits)
            foreground = torch.zeros_like(target_labels, dtype=torch.bool)
            num_gt = gt_boxes.size(1)
            if num_gt == 0 or num_anchors == 0:
                return target_labels, target_boxes, target_scores, foreground
            chunk_size = max(1, self.max_cost_elements // (num_anchors * num_gt * self.num_classes))
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                result = self._assign_batch(
                    logits[start:end], boxes[start:end], points, gt_labels[start:end],
                    gt_boxes[start:end], mask_gt[start:end].squeeze(-1).bool(), strides)
                target_labels[start:end], target_boxes[start:end], \
                    target_scores[start:end], foreground[start:end] = result
        return target_labels, target_boxes, target_scores, foreground

    @staticmethod
    def _candidates(points, gt_boxes, valid_gt):
        deltas = torch.cat((points[None, :, None] - gt_boxes[:, None, :, :2],
                            gt_boxes[:, None, :, 2:] - points[None, :, None]), dim=-1)
        # The union across GTs is the candidate set. Do not apply a separate
        # hard inside-GT mask to each candidate/GT cost.
        return ((deltas.min(-1).values > 0) & valid_gt[:, None]).any(-1)

    @staticmethod
    def _pairwise_iou(boxes, gt_boxes):
        area1 = (boxes[..., 2:] - boxes[..., :2]).prod(-1)
        area2 = (gt_boxes[..., 2:] - gt_boxes[..., :2]).prod(-1)
        lt = torch.maximum(boxes[..., :, None, :2], gt_boxes[..., None, :, :2])
        rb = torch.minimum(boxes[..., :, None, 2:], gt_boxes[..., None, :, 2:])
        intersection = (rb - lt).clamp(min=0).prod(-1)
        union = area1[..., None] + area2[..., None, :] - intersection
        return intersection / union.clamp(min=1e-6)

    def _cost(self, logits, boxes, points, labels, gt_boxes, strides):
        ious = self._pairwise_iou(boxes, gt_boxes)
        centers = (gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2
        distance = (points[:, :, None] - centers[:, None]).pow(2).sum(-1).sqrt()
        distance = distance / strides[..., 0, None]
        center_cost = torch.pow(10, distance - self.soft_center_radius)
        soft_labels = F.one_hot(labels, self.num_classes).float()[:, None] * ious[..., None]
        expanded_logits = logits[:, :, None, :].expand_as(soft_labels)
        scale = soft_labels - expanded_logits.sigmoid()
        # Sum ALL classes, as in the linked MMDetection implementation.
        cls_cost = (F.binary_cross_entropy_with_logits(expanded_logits, soft_labels, reduction='none') *
                    scale.abs().pow(2)).sum(-1)
        cost = cls_cost - self.iou_weight * torch.log(ious + self.eps) + center_cost
        return cost, ious

    def _assign_batch(self, logits, boxes, points, labels, gt_boxes, valid_gt, strides):
        batch_size, num_anchors, _ = logits.shape
        candidates = self._candidates(points, gt_boxes, valid_gt)
        counts = candidates.sum(-1)
        num_candidates = int(counts.max().item())
        if num_candidates == 0:
            return (labels.new_full((batch_size, num_anchors), self.num_classes, dtype=torch.long),
                    torch.zeros_like(boxes), torch.zeros_like(logits), candidates)

        # Pack in original anchor order; padding writes only to a discarded
        # extra column. This avoids per-image boolean indexing on the fast path.
        ranks = candidates.long().cumsum(-1) - 1
        indices = torch.zeros((batch_size, num_candidates + 1), dtype=torch.long, device=boxes.device)
        anchor_indices = torch.arange(num_anchors, device=boxes.device).expand(batch_size, -1)
        indices.scatter_(1, torch.where(candidates, ranks, num_candidates), anchor_indices)
        indices = indices[:, :num_candidates]
        candidate_valid = torch.arange(num_candidates, device=boxes.device)[None] < counts[:, None]
        packed_boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        packed_logits = logits.gather(1, indices[..., None].expand(-1, -1, self.num_classes))
        safe_labels = torch.where(valid_gt, labels.squeeze(-1).long(), 0)
        cost, ious = self._cost(
            packed_logits, packed_boxes, points[indices], safe_labels, gt_boxes, strides[indices])
        pair_valid = candidate_valid[..., None] & valid_gt[:, None]
        cost = cost.masked_fill(~pair_valid, float('inf'))
        ious = ious * pair_valid

        topk = min(self.topk, num_candidates)
        iou_sums = ious.topk(topk, dim=1).values.sum(1)
        dynamic_ks = torch.minimum(iou_sums.int().clamp(min=1), counts[:, None]) * valid_gt
        selected_k = min(topk + 1, num_candidates)
        values, selected = cost.topk(selected_k, dim=1, largest=False)
        selected_ranks = torch.arange(selected_k, device=boxes.device)[None, :, None]
        matching = torch.zeros_like(cost).scatter_(
            1, selected, ((selected_ranks < dynamic_ks[:, None]) & pair_valid.gather(1, selected)).float())
        selected_count = matching.sum(-1)
        # Resolve conflicts by minimum cost over ALL valid GTs, as upstream.
        matched_gt = torch.where(selected_count > 1, cost.argmin(-1), matching.argmax(-1))
        matched_iou = ious.gather(2, matched_gt[..., None]).squeeze(-1)
        unpack = ranks.clamp(min=0)
        foreground = ((selected_count > 0) & candidate_valid).gather(1, unpack) & candidates
        matched_gt = matched_gt.gather(1, unpack)
        matched_iou = matched_iou.gather(1, unpack)
        target_labels = safe_labels.gather(1, matched_gt)
        target_boxes = gt_boxes.gather(1, matched_gt[..., None].expand(-1, -1, 4))
        target_boxes = torch.where(foreground[..., None], target_boxes, torch.zeros_like(target_boxes))
        target_scores = torch.zeros_like(logits).scatter_(
            2, target_labels[..., None], (matched_iou * foreground)[..., None])
        target_labels = torch.where(foreground, target_labels, self.num_classes)

        # Preserve upstream topk choices at ties and dynamic-k rounding
        # boundaries, which can otherwise depend on k or candidate padding.
        boundary = (dynamic_ks.long() - 1).clamp(min=0, max=selected_k - 1)
        following = dynamic_ks.long().clamp(max=selected_k - 1)
        ties = (values.gather(1, boundary[:, None]) == values.gather(1, following[:, None])).squeeze(1)
        ties = (ties & valid_gt & (dynamic_ks > 0) & (dynamic_ks < counts[:, None])).any(-1)
        tolerance = topk * torch.finfo(ious.dtype).eps * iou_sums.abs()
        near_integer = ((iou_sums - iou_sums.round()).abs() <= tolerance) & (iou_sums > 1)
        replay = (ties | ((counts > 0) & (counts < self.topk)) |
                  (near_integer & valid_gt).any(-1))
        for batch_idx in replay.nonzero(as_tuple=False).flatten().tolist():
            valid = valid_gt[batch_idx]
            result = self._assign_single(
                logits[batch_idx], boxes[batch_idx], points,
                labels[batch_idx, valid].reshape(-1).long(), gt_boxes[batch_idx, valid], strides)
            target_labels[batch_idx], target_boxes[batch_idx], \
                target_scores[batch_idx], foreground[batch_idx] = result
        return target_labels, target_boxes, target_scores, foreground

    def _assign_single(self, logits, boxes, points, labels, gt_boxes, strides):
        """Reference-compatible path for ambiguous top-k boundaries."""
        target_labels = labels.new_full((boxes.size(0),), self.num_classes)
        target_boxes, target_scores = torch.zeros_like(boxes), torch.zeros_like(logits)
        foreground = torch.zeros_like(target_labels, dtype=torch.bool)
        valid_gt = torch.ones((1, gt_boxes.size(0)), dtype=torch.bool, device=boxes.device)
        candidates = self._candidates(points, gt_boxes[None], valid_gt)[0]
        cost, ious = self._cost(
            logits[candidates][None], boxes[candidates][None], points[candidates][None],
            labels[None], gt_boxes[None], strides[candidates][None])
        cost, ious = cost[0], ious[0]
        matching = torch.zeros_like(cost, dtype=torch.uint8)
        dynamic_ks = ious.topk(min(self.topk, ious.size(0)), dim=0).values.sum(0).int().clamp(min=1)
        for gt_idx in range(gt_boxes.size(0)):
            selected = cost[:, gt_idx].topk(dynamic_ks[gt_idx].item(), largest=False).indices
            matching[selected, gt_idx] = 1
        conflicts = matching.sum(1) > 1
        if conflicts.any():
            matched_gt = cost[conflicts].argmin(1)
            matching[conflicts] = 0
            matching[conflicts, matched_gt] = 1
        selected = matching.sum(1) > 0
        matched_gt = matching[selected].argmax(1)
        positive_indices = candidates.nonzero(as_tuple=False).squeeze(1)[selected]
        positive_labels = labels[matched_gt]
        foreground[positive_indices] = True
        target_labels[positive_indices] = positive_labels
        target_boxes[positive_indices] = gt_boxes[matched_gt]
        target_scores[positive_indices, positive_labels] = (matching * ious).sum(1)[selected]
        return target_labels, target_boxes, target_scores, foreground
